# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only MiniCPM-o 4.5 unified model (Thinker + Talker + Code2Wav)."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
    MiniCPMOConfig,
)
from vllm_omni.model_executor.models.minicpm_o.minicpm_o_code2wav import (
    MiniCPMOCode2Wav,
)
from vllm_omni.model_executor.models.minicpm_o.minicpm_o_talker import (
    MiniCPMOTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpm_o.minicpm_o_thinker import (
    MiniCPMOThinkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# MiniCPM-o 4.5 special token IDs — used to identify TTS control tokens
# in the thinker's output sequence so the talker stage can find them.
# TODO(L2): Verify exact IDs from openbmb/MiniCPM-o-4_5 tokenizer_config.json.
AUDIO_BOS_TOKEN_ID: int = 151859  # <|audio_bos|>  (placeholder)
AUDIO_EOS_TOKEN_ID: int = 151860  # <|audio_eos|>  (placeholder)


# TODO(Phase 6): Add @MULTIMODAL_REGISTRY.register_processor() decorator here
# with MiniCPMOThinkerMultiModalProcessor, MiniCPMOThinkerProcessingInfo, and
# MiniCPMOThinkerDummyInputsBuilder once those classes are implemented.
class MiniCPMOForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    CustomProcessMixin,
):
    """Unified MiniCPM-o 4.5 model: Thinker → Talker → Code2Wav.

    3-stage vllm-omni pipeline:
      Stage 1 (Thinker):  vision + audio + text → LLM hidden states (Qwen3)
      Stage 2 (Talker):   TTS conditioning → audio codec tokens (MiniCPMTTS)
      Stage 3 (Code2Wav): codec tokens → waveform (CosyVoice2 + HiFi-GAN)

    Usage:
        Set ``model_stage`` in vllm_config to one of:
        ``"thinker"``, ``"talker"``, ``"code2wav"``.

    Architecture summary:
      - Thinker: SigLIP2 (vision) + Whisper-medium (audio) + Qwen3 (LLM)
      - Talker:  Llama AR backbone (hidden=768, layers=20) with MiniCPMTTS
                 hidden_text_merge conditioning from thinker outputs
      - Code2Wav: CosyVoice2 flow-matching DiT + HiFi-GAN vocoder

    Weight loading:
      - Thinker: main HF safetensors checkpoint (vpm.*, resampler.*, apm.*,
                 audio_projection_layer.*, llm.*)
      - Talker:  same checkpoint under tts.* prefix
      - Code2Wav: separate flow.pt + hift.pt files in the model directory
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.config = config
        self.model_stage: str = vllm_config.model_config.model_stage

        # ---- Stage 1: Thinker ----
        if self.model_stage == "thinker":
            self.thinker = MiniCPMOThinkerForConditionalGeneration(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
            )
            self.model = self.thinker
            self.talker = None
            self.code2wav = None

        # ---- Stage 2: Talker ----
        elif self.model_stage == "talker":
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.talker_preprocess)
            self.set_custom_postprocess(self.talker_postprocess)

            self.thinker = None
            # Talker's __init__ reads MiniCPMTTSConfig from hf_config
            talker_vllm_config = vllm_config.with_hf_config(
                config.tts_config,
                architectures=["MiniCPMOTalkerModel"],
            )
            self.talker = MiniCPMOTalkerForConditionalGeneration(
                vllm_config=talker_vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
            )
            self.model = self.talker
            self.code2wav = None
            self.requires_raw_input_tokens = True

        # ---- Stage 3: Code2Wav ----
        elif self.model_stage == "code2wav":
            self.enable_update_additional_information = True
            self.thinker = None
            self.talker = None
            # Code2Wav reads model_dir from vllm_config.model_config.model
            self.code2wav = MiniCPMOCode2Wav(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
            )
            self.model = self.code2wav
            self.requires_raw_input_tokens = True

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage!r}. "
                "Must be one of: 'thinker', 'talker', 'code2wav'."
            )

        # Intermediate tensor factory (used by vllm PP pipeline)
        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors
            if self.model_stage == "thinker"
            else lambda: None
        )

    # ==================== Device utility ====================

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Return the device of a module (CPU fallback if no parameters)."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    @cached_property
    def sampler(self) -> Sampler:
        """Token sampler shared across stages."""
        return Sampler()

    # ==================== Embedding ====================

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Phase 6 (MultiModalProcessor) provides the real placeholder strings.
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "thinker":
            return self.thinker.embed_input_ids(
                input_ids=input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        if self.model_stage == "talker":
            # Talker's embed_input_ids only accepts input_ids (codec tokens).
            return self.talker.embed_input_ids(input_ids)
        # code2wav: no token embeddings; return a zero dummy tensor.
        return torch.zeros_like(input_ids).reshape(-1, 1).repeat(
            1, self.vllm_config.model_config.get_hidden_size()
        )

    def embed_multimodal(self, **kwargs: object):
        """Delegate multimodal embedding to the thinker (vision + audio only)."""
        if self.model_stage != "thinker":
            raise RuntimeError(
                "embed_multimodal is only valid for the thinker stage; "
                f"current stage: {self.model_stage!r}"
            )
        return self.thinker.embed_multimodal(**kwargs)

    # ==================== Forward ====================

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | list[torch.Tensor]:
        """Unified forward for all three model stages."""

        # ---- Thinker ----
        if self.model_stage == "thinker":
            return self.thinker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        # ---- Talker ----
        elif self.model_stage == "talker":
            # inputs_embeds pre-built by talker_preprocess (CustomProcessMixin)
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.talker.embed_input_ids(input_ids)

            return self.talker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        # ---- Code2Wav ----
        elif self.model_stage == "code2wav":
            seq_token_counts: list[int] | None = kwargs.get("seq_token_counts")  # type: ignore[assignment]
            codes = input_ids
            if codes.ndim == 1:
                codes = codes.unsqueeze(0)

            return self.code2wav.chunked_decode(
                codes=codes,
                chunk_size=300,
                left_context_size=25,
                seq_token_counts=seq_token_counts,
            )

        # Unreachable (ValueError raised in __init__)
        raise AssertionError(f"Unexpected model_stage: {self.model_stage!r}")

    # ==================== OmniOutput ====================

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | tuple | list[torch.Tensor] | OmniOutput,
        **kwargs: object,
    ) -> OmniOutput:
        """Wrap model outputs into OmniOutput for downstream pipeline."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if self.model_stage == "thinker":
            # Thinker forward returns (hidden_states, inputs_embeds) tuple
            # so that talker can use both for hidden_text_merge conditioning.
            hidden, text_embeds = model_outputs

            # Pipeline-parallel non-final rank: language_model returns
            # IntermediateTensors, not a flat tensor.  Pass through as-is.
            if isinstance(hidden, IntermediateTensors):
                return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

            flat_hidden = hidden.reshape(-1, hidden.shape[-1])
            mmo: dict = {"thinker_hidden_states": flat_hidden}
            if text_embeds is not None:
                # Shape: [total_batch_tokens, thinker_hidden_dim]
                # gpu_ar_model_runner slices these per-request via [start:end].
                mmo["thinker_text_embeds"] = text_embeds
            return OmniOutput(
                text_hidden_states=flat_hidden,
                multimodal_outputs=mmo,
            )

        if self.model_stage == "talker":
            return OmniOutput(
                text_hidden_states=model_outputs,
                multimodal_outputs=None,
            )

        if self.model_stage == "code2wav":
            audio_tensors: list[torch.Tensor] = model_outputs  # type: ignore[assignment]
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [t.reshape(1, -1) for t in audio_tensors]
                },
            )

        return model_outputs  # type: ignore[return-value]

    # ==================== Talker pre/postprocess ====================

    def talker_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Build talker input embeddings from thinker conditioning.

        Implements MiniCPMTTS hidden_text_merge:
            talker_input[t] = text_projection(thinker_text_embeds[t])
                            + hidden_projection(thinker_hidden_states[t])
                            + codec_embedding(input_ids[t])

        Prefill (span_len > 1):
            - Project full thinker sequence → ``full_conditioning``
            - Use first ``span_len`` positions for this prefill step
            - Store remaining positions in ``trailing_text_hidden`` queue

        Decode (span_len == 1):
            - Pop one entry from ``trailing_text_hidden`` queue
            - Add codec embedding for the generated codec token

        Args:
            input_ids:    Codec token IDs (placeholder zeros during prefill).
            input_embeds: Pre-built embeddings (unused; we build from scratch).
            **info_dict:  Per-request buffer carrying thinker outputs and
                          decode-step state:
                          ``thinker_text_embeds``   — [N, thinker_hidden]
                          ``thinker_hidden_states``  — [N, thinker_hidden]
                          ``trailing_text_hidden``   — [remaining, talker_hidden]

        Returns:
            (input_ids, input_embeds, update_dict)
        """
        span_len = input_ids.shape[0]
        device = self._module_device(self.talker)
        update_dict: dict = {}

        if span_len > 1:
            # ---- Prefill ----
            thinker_text_embeds: torch.Tensor | None = info_dict.get(  # type: ignore[assignment]
                "thinker_text_embeds"
            )
            thinker_hidden_states: torch.Tensor | None = info_dict.get(  # type: ignore[assignment]
                "thinker_hidden_states"
            )

            if thinker_text_embeds is not None or thinker_hidden_states is not None:
                t_text = (
                    thinker_text_embeds.to(device=device, dtype=torch.bfloat16)
                    if isinstance(thinker_text_embeds, torch.Tensor)
                    else None
                )
                t_hid = (
                    thinker_hidden_states.to(device=device, dtype=torch.bfloat16)
                    if isinstance(thinker_hidden_states, torch.Tensor)
                    else None
                )
                # Project all N thinker positions to talker hidden dim
                full_conditioning = self.talker.project_thinker_outputs(t_text, t_hid)  # [N, talker_hidden]

                # Support chunked prefill: track position within thinker sequence.
                # num_processed_tokens counts how many talker positions have been
                # processed in previous prefill chunks for this request.
                start_pos: int = info_dict.get("num_processed_tokens", 0)  # type: ignore[assignment]
                end_pos = start_pos + span_len
                input_conditioning = full_conditioning[start_pos:end_pos]

                # Store remaining positions as trailing queue for decode steps.
                if full_conditioning.shape[0] > end_pos:
                    update_dict["trailing_text_hidden"] = (
                        full_conditioning[end_pos:].detach()
                    )
                update_dict["num_processed_tokens"] = end_pos

                # hidden_text_merge: conditioning + codec embedding
                codec_embeds = self.talker.embed_input_ids(input_ids.to(device))
                input_embeds = input_conditioning + codec_embeds
            else:
                # No thinker conditioning — use codec embedding only
                if input_embeds is None:
                    input_embeds = self.talker.embed_input_ids(input_ids.to(device))

        else:
            # ---- Decode (one token at a time) ----
            codec_embeds = self.talker.embed_input_ids(input_ids.to(device))
            trailing: torch.Tensor | None = info_dict.get(  # type: ignore[assignment]
                "trailing_text_hidden"
            )

            if isinstance(trailing, torch.Tensor) and trailing.numel() > 0 and trailing.shape[0] > 0:
                text_step = trailing[:1].to(device=device, dtype=codec_embeds.dtype)
                new_trailing = (
                    trailing[1:].detach()
                    if trailing.shape[0] > 1
                    else torch.zeros(
                        0,
                        trailing.shape[-1],
                        device=trailing.device,
                        dtype=trailing.dtype,
                    )
                )
                input_embeds = text_step + codec_embeds
                update_dict["trailing_text_hidden"] = new_trailing
            else:
                # Queue exhausted — use codec embedding only
                input_embeds = codec_embeds

        return input_ids, input_embeds, update_dict

    def talker_postprocess(
        self,
        hidden_states: torch.Tensor,
        **info_dict: object,
    ) -> dict:
        """Persist last talker hidden state for streaming decode steps."""
        return {"last_talker_hidden": hidden_states[-1].detach()}

    # ==================== Logits + Sampling ====================

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: object = None,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states for the active stage."""
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if self.model_stage == "thinker":
            return self.thinker.language_model.compute_logits(hidden_states)

        if self.model_stage == "talker":
            return self.talker.compute_logits(hidden_states)

        # code2wav has no logits
        return None

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: object,
    ):
        """Sample next tokens from logits."""
        return self.sampler(logits, sampling_metadata)

    # ==================== Weight Loading ====================

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Dispatch weights to the appropriate stage module.

        HF checkpoint key prefixes:
          tts.*      → Talker
          code2wav.* → Code2Wav (+ flow.pt / hift.pt separately)
          *          → Thinker (vpm.*, resampler.*, apm.*,
                                audio_projection_layer.*, llm.*)
        """
        thinker_weights: list[tuple[str, torch.Tensor]] = []
        talker_weights: list[tuple[str, torch.Tensor]] = []
        code2wav_weights: list[tuple[str, torch.Tensor]] = []

        for k, v in weights:
            if k.startswith("tts."):
                talker_weights.append((k, v))
            elif k.startswith("code2wav."):
                code2wav_weights.append((k, v))
            else:
                # Remaining keys (vpm.*, resampler.*, apm.*, llm.*) → thinker
                thinker_weights.append((k, v))

        loaded: set[str] = set()

        if self.thinker is not None:
            loaded.update(self.thinker.load_weights(iter(thinker_weights)))

        if self.talker is not None:
            loaded.update(self.talker.load_weights(iter(talker_weights)))

        if self.code2wav is not None:
            loaded.update(self.code2wav.load_weights(iter(code2wav_weights)))

        logger.info(
            "MiniCPMO: loaded %d weight keys (stage=%s)",
            len(loaded),
            self.model_stage,
        )
        return loaded
