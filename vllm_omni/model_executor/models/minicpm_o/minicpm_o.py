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
from vllm.model_executor.models.minicpmv import (
    MiniCPMVDummyInputsBuilder,
    MiniCPMVMultiModalProcessor,
    MiniCPMVProcessingInfo,
)


class MiniCPMOProcessingInfo(MiniCPMVProcessingInfo):
    """Extends MiniCPMV to support audio input via Whisper encoder."""

    def get_supported_mm_limits(self):
        limits = dict(super().get_supported_mm_limits())
        limits["audio"] = None  # Allow audio input
        return limits
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
    MiniCPMOConfig,
    MiniCPMTTSConfig,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMVMultiModalProcessor,
    info=MiniCPMOProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
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

    # Required by BitsAndBytes quantization loader (_verify_model_compatibility).
    # Reflects the packed modules used by Thinker (Qwen3) and Talker (Llama) —
    # the two stages that contain fused linear layers.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.config = config
        self.model_stage: str = vllm_config.model_config.model_stage

        # Sub-configs (MiniCPM-o HF config uses tts_config, not talker_config)
        tts_config: MiniCPMTTSConfig = config.tts_config

        if self.model_stage == "thinker":
            thinker_vllm_config = vllm_config.with_hf_config(
                config,
                architectures=["MiniCPMOThinkerForConditionalGeneration"],
            )
            self.thinker = init_vllm_registered_model(
                vllm_config=thinker_vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                architectures=["MiniCPMOThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.talker = None
            self.code2wav = None

        elif self.model_stage == "talker":
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.talker_preprocess)
            self.set_custom_postprocess(self.talker_postprocess)

            self.thinker = None
            talker_vllm_config = vllm_config.with_hf_config(
                tts_config,
                architectures=["MiniCPMOTalkerForConditionalGeneration"],
            )
            self.talker = init_vllm_registered_model(
                vllm_config=talker_vllm_config,
                prefix=maybe_prefix(prefix, "talker"),
                hf_config=tts_config,
                architectures=["MiniCPMOTalkerForConditionalGeneration"],
            )
            self.model = self.talker
            self.code2wav = None
            self.requires_raw_input_tokens = True
            # Keys that should stay on GPU to avoid CPU↔GPU round-trips
            self.gpu_resident_buffer_keys: set[str] = {
                "trailing_conditioning",
            }

        elif self.model_stage == "code2wav":
            self.enable_update_additional_information = True
            self.thinker = None
            self.talker = None
            code2wav_vllm_config = vllm_config.with_hf_config(
                config, architectures=["MiniCPMOCode2Wav"]
            )
            self.code2wav = init_vllm_registered_model(
                vllm_config=code2wav_vllm_config,
                prefix=maybe_prefix(prefix, "code2wav"),
                hf_config=config,
                architectures=["MiniCPMOCode2Wav"],
            )
            self.model = self.code2wav
            self.requires_raw_input_tokens = True

        else:
            raise ValueError(
                f"Invalid model_stage: {self.model_stage}. "
                "Must be one of: 'thinker', 'talker', 'code2wav'"
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
        # Placeholder strings are provided by MiniCPMVMultiModalProcessor.
        return None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "code2wav":
            # Code2Wav: no token embeddings; return a zero dummy tensor.
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(
                1, self.vllm_config.model_config.get_hidden_size()
            )
        # Thinker and Talker: delegate to the active model's embed_input_ids.
        # Talker returns 768-wide embeddings (codec_embedding), not 4096.
        return self.model.embed_input_ids(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def embed_multimodal(self, **kwargs: object):
        """Delegate multimodal embedding to the thinker (vision + audio only).

        Talker and Code2Wav stages have no multimodal encoder; return empty so
        that callers gracefully skip encoding.
        Use --skip-mm-profiling when serving non-thinker stages to avoid the
        profiling sanity check (which expects a non-zero item count).
        """
        if self.model_stage != "thinker":
            return []
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
            # During real inference, talker_preprocess (CustomProcessMixin) builds
            # correct 768-wide embeddings (codec + thinker conditioning) and passes
            # them via inputs_embeds.
            # During profile_run/_dummy_run, the model runner passes its pre-allocated
            # inputs_embeds buffer, which is sized for the main model (4096-wide) and
            # is incompatible with Talker's LlamaAR (hidden_size=768).
            # Fall back to plain codec embeddings whenever shape is wrong or missing.
            talker_hidden = self.config.tts_config.hidden_size
            if (
                inputs_embeds is None
                or input_ids is None
                or inputs_embeds.shape[-1] != talker_hidden
            ):
                inputs_embeds = self.talker.embed_input_ids(input_ids)

            return self.talker(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

        # ---- Code2Wav ----
        elif self.model_stage == "code2wav":
            codes = input_ids
            if codes.ndim == 1:
                codes = codes.unsqueeze(0)

            return self.code2wav.decode(codes)

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
            # Thinker forward returns (hidden_states, inputs_embeds) tuple.
            # thinker_hidden_states is used by talker for semantic_projection.
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
                    # Key "audio" matches serving_chat.py _create_audio_choice
                    # which reads: multimodal_output.get("audio")
                    "audio": [t.reshape(1, -1) for t in audio_tensors]
                },
            )

        raise ValueError(
            f"make_omni_output: unhandled model_stage={self.model_stage!r}"
        )

    # ==================== Talker pre/postprocess ====================

    def talker_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: object,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Build talker input embeddings from thinker conditioning.

        Implements original MiniCPM-o hidden_text_merge conditioning:
            tts_embeds = emb_text(token_ids) + normalize(semantic_projection(hidden_states))

        During prefill, the full conditioning sequence is:
            [tts_embeds, text_eos_embed, audio_bos_embed]
        Then AR decoding uses codec_embedding(prev_token) for each step.

        Prefill (span_len > 1):
            - Build conditioning from thinker_token_ids + thinker_hidden_states
            - Append text_eos and audio_bos boundary tokens
            - Use first ``span_len`` positions; store rest as trailing queue

        Decode (span_len == 1):
            - Pop one entry from trailing queue if available
            - Otherwise use codec_embedding(input_ids) for AR decoding

        Args:
            input_ids:    Codec token IDs (placeholder zeros during prefill).
            input_embeds: Pre-built embeddings (unused; we build from scratch).
            **info_dict:  Per-request buffer:
                          ``thinker_token_ids``    — [N] LLM token IDs
                          ``thinker_hidden_states`` — [N, 4096] hidden states
                          ``trailing_conditioning`` — [remaining, 768]

        Returns:
            (input_ids, input_embeds, update_dict)
        """
        span_len = input_ids.shape[0]
        device = self._module_device(self.talker)
        update_dict: dict = {}

        if span_len > 1:
            # ---- Prefill ----
            thinker_token_ids: torch.Tensor | None = info_dict.get(  # type: ignore[assignment]
                "thinker_token_ids"
            )
            thinker_hidden_states: torch.Tensor | None = info_dict.get(  # type: ignore[assignment]
                "thinker_hidden_states"
            )

            if thinker_hidden_states is not None and thinker_token_ids is not None:
                t_ids = thinker_token_ids.to(device=device, dtype=torch.long)
                # Serialization cast hidden to float32; restore model dtype
                model_dtype = next(self.talker.parameters()).dtype
                t_hid = thinker_hidden_states.to(device=device, dtype=model_dtype)

                # Build conditioning: emb_text(tokens) + normalize(semantic_projection(hidden))
                full_conditioning = self.talker.build_conditioning(t_ids, t_hid)

                # Append boundary tokens: [tts_embeds, text_eos, audio_bos]
                tts_config = self.config.tts_config
                boundary_ids = torch.tensor(
                    [tts_config.text_eos_token_id, tts_config.audio_bos_token_id],
                    device=device, dtype=torch.long,
                )
                boundary_embeds = self.talker.emb_text(boundary_ids)
                full_conditioning = torch.cat(
                    [full_conditioning, boundary_embeds], dim=0
                )

                # Chunked prefill support
                start_pos: int = info_dict.get("num_processed_tokens", 0)  # type: ignore[assignment]
                end_pos = start_pos + span_len
                input_embeds = full_conditioning[start_pos:end_pos]

                # Store remaining positions as trailing queue for decode steps
                if full_conditioning.shape[0] > end_pos:
                    update_dict["trailing_conditioning"] = (
                        full_conditioning[end_pos:].detach()
                    )
                update_dict["num_processed_tokens"] = end_pos
            else:
                # No thinker conditioning — use codec embedding only (fallback)
                if input_embeds is None:
                    input_embeds = self.talker.embed_input_ids(input_ids.to(device))

        else:
            # ---- Decode (one token at a time) ----
            codec_embeds = self.talker.embed_input_ids(input_ids.to(device))
            trailing: torch.Tensor | None = info_dict.get(  # type: ignore[assignment]
                "trailing_conditioning"
            )

            if isinstance(trailing, torch.Tensor) and trailing.shape[0] > 0:
                # Still have conditioning positions to consume (boundary tokens etc.)
                # Use conditioning alone — matches prefill path where conditioning
                # replaces input embeddings without adding codec embeddings.
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
                input_embeds = text_step
                update_dict["trailing_conditioning"] = new_trailing
            else:
                # All conditioning consumed — pure AR codec decoding
                input_embeds = codec_embeds

        return input_ids, input_embeds, update_dict

    def talker_postprocess(
        self,
        hidden_states: torch.Tensor,
        **info_dict: object,
    ) -> dict:
        """No-op: MiniCPM-o uses num_vq=1 (no code_predictor), so talker
        hidden states are not consumed by downstream stages."""
        return {}

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
            # AutoWeightsLoader returns keys relative to the sub-module
            # (e.g. "llm.layers.*"). vllm's post-load validator checks keys
            # against the ROOT model's named_parameters() which include the
            # sub-module prefix (e.g. "thinker.llm.layers.*").  Add it here.
            thinker_loaded = self.thinker.load_weights(iter(thinker_weights))
            loaded.update(f"thinker.{k}" for k in thinker_loaded)

        if self.talker is not None:
            # Same prefix logic: talker-relative keys → root-relative keys.
            talker_loaded = self.talker.load_weights(iter(talker_weights))
            loaded.update(f"talker.{k}" for k in talker_loaded)

        if self.code2wav is not None:
            loaded.update(self.code2wav.load_weights(iter(code2wav_weights)))
            # Code2Wav has no registered nn.Module parameters (weights loaded
            # from assets/token2wav/ separately). named_parameters() is empty,
            # so vllm's state_dict check finds nothing to validate.

        logger.info(
            "MiniCPMO: loaded %d weight keys (stage=%s)",
            len(loaded),
            self.model_stage,
        )
        return loaded
