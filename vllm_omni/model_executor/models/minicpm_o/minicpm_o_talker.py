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
"""Inference-only MiniCPM-o 4.5 Talker model (MiniCPMTTS Llama AR backbone)."""

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
    MiniCPMTTSConfig,
)

logger = init_logger(__name__)


def _build_llama_config(tts_config: MiniCPMTTSConfig) -> LlamaConfig:
    """Build a LlamaConfig from MiniCPMTTSConfig for the Talker backbone."""
    return LlamaConfig(
        hidden_size=tts_config.hidden_size,
        num_hidden_layers=tts_config.num_hidden_layers,
        num_attention_heads=tts_config.num_attention_heads,
        num_key_value_heads=tts_config.num_key_value_heads,
        intermediate_size=tts_config.intermediate_size,
        hidden_act=tts_config.hidden_act,
        max_position_embeddings=tts_config.max_position_embeddings,
        # LlamaModel backbone uses its own 32000-token text vocab
        # (tts.model.embed_tokens.weight shape [32000, 768] in HF checkpoint).
        # Codec tokens (num_audio_tokens) are handled by codec_embedding separately.
        vocab_size=32000,
    )


class MiniCPMOTalkerResizeMLP(nn.Module):
    """2-layer MLP projecting thinker dim → talker hidden dim (ReLU activation).

    Matches openbmb MultiModalProjector: Linear(in→out) → ReLU → Linear(out→out).
    Used by projector_semantic (hidden state projection) in hidden_text_merge.

    Args:
        llm_dim: Thinker hidden size (e.g. 4096).
        intermediate_size: MLP intermediate dim (llm_intermediate_size).
        hidden_size: Talker hidden size (e.g. 768).
    """

    def __init__(self, llm_dim: int, intermediate_size: int, hidden_size: int):
        super().__init__()
        # Layer names match HF checkpoint: tts.projector_semantic.linear1/linear2
        self.linear1 = nn.Linear(llm_dim, intermediate_size, bias=True)
        self.linear2 = nn.Linear(intermediate_size, hidden_size, bias=True)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act_fn(self.linear1(x)))


class MiniCPMOTalkerLLM(LlamaForCausalLM):
    """Llama AR backbone for MiniCPM-o Talker.

    Subclasses LlamaForCausalLM but removes the text LM head (unused;
    codec_head lives in the parent MiniCPMOTalkerForConditionalGeneration).

    The model's embed_tokens (vocab=32000) is loaded from ``tts.model.*``
    in the HF checkpoint but is NOT called during inference — codec_embedding
    in the parent class is used instead, fed as ``inputs_embeds``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        tts_config: MiniCPMTTSConfig,
        prefix: str = "",
    ):
        llama_vllm_config = vllm_config.with_hf_config(
            _build_llama_config(tts_config),
            architectures=["LlamaForCausalLM"],
        )
        super().__init__(vllm_config=llama_vllm_config, prefix=prefix)

        # Remove the text LM head — codec_head in parent class is used instead.
        if hasattr(self, "lm_head"):
            del self.lm_head
        if hasattr(self, "logits_processor"):
            del self.logits_processor


class MiniCPMOTalkerForConditionalGeneration(nn.Module, SupportsPP):
    """MiniCPM-o 4.5 Talker: TTS conditioning → audio codec tokens.

    Stage 2 of the 3-stage vllm-omni pipeline:
      Stage 1 (Thinker): vision + audio + text → token hidden states
      Stage 2 (Talker):  hidden_text_merge conditioning → audio codec tokens
      Stage 3 (Code2Wav): codec tokens → waveform

    Architecture (matches openbmb/MiniCPM-o-4_5 MiniCPMTTS):
      - emb_text:            Embedding(152064, 768) — TTS text token embeddings
      - semantic_projection: MLP(4096 → 768, ReLU) — projects thinker hidden states
      - language_model:      Llama (hidden_size=768, num_layers=20, num_vq=1)
      - codec_embedding:     Embedding(6562, 768) — audio codec token embeddings
      - codec_head:          Linear(768 → 6562) — audio token logits

    Conditioning (hidden_text_merge, from original modeling_minicpmo.py):
      tts_embeds = emb_text(token_ids) + normalize(semantic_projection(hidden_states))

    Note: projector_spk (tts.projector_spk.*) is defined in the original model
    but NEVER called during inference. We load its weights but do not use it.

    Weight key mapping from HuggingFace checkpoint (tts.* prefix):
      tts.model.*              → language_model.model.*
      tts.emb_code.0.*         → codec_embedding.*
      tts.emb_text.*           → emb_text.*
      tts.head_code.0.*        → codec_head.*  (spectral norm resolved)
      tts.projector_semantic.* → semantic_projection.*
      tts.projector_spk.*      → spk_projection.* (loaded but unused in inference)
      tts.*                    → (catch-all strip)
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "tts.model.": "language_model.model.",
            "tts.emb_code.0.": "codec_embedding.",
            "tts.emb_text.": "emb_text.",
            "tts.head_code.0.": "codec_head.",
            "tts.projector_semantic.": "semantic_projection.",
            "tts.projector_spk.": "spk_projection.",
            "tts.": "",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        tts_config: MiniCPMTTSConfig = vllm_config.model_config.hf_config
        self.config = tts_config

        # TTS text embedding table (tts.emb_text.* in HF checkpoint)
        # Maps LLM token IDs → 768-dim talker space. Primary text conditioning.
        self.emb_text = nn.Embedding(
            tts_config.num_text_tokens,
            tts_config.hidden_size,
        )

        # Projection: thinker hidden states → talker hidden dim (ReLU)
        # (tts.projector_semantic.* in HF checkpoint)
        self.semantic_projection = MiniCPMOTalkerResizeMLP(
            llm_dim=tts_config.llm_dim,
            intermediate_size=tts_config.llm_intermediate_size,
            hidden_size=tts_config.hidden_size,
        )

        # Speaker projection — loaded but NOT used during inference.
        # (tts.projector_spk.* in HF checkpoint — defined but never called
        # in openbmb/MiniCPM-o-4_5 modeling_minicpmo.py inference paths)
        self.spk_projection = MiniCPMOTalkerResizeMLP(
            llm_dim=tts_config.llm_dim,
            intermediate_size=tts_config.llm_intermediate_size,
            hidden_size=tts_config.hidden_size,
        )

        # Codec token embedding for AR decoding steps
        # (tts.emb_code.0.* in HF checkpoint; separate from language_model.embed_tokens)
        self.codec_embedding = nn.Embedding(
            tts_config.num_audio_tokens,
            tts_config.hidden_size,
        )

        # Llama AR backbone (embed_tokens vocab=32000; loaded but not called during inference)
        self.language_model = MiniCPMOTalkerLLM(
            vllm_config=vllm_config,
            tts_config=tts_config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        # Output projection: talker hidden → audio token logits (layer 0 of num_vq=1)
        self.codec_head = nn.Linear(
            tts_config.hidden_size,
            tts_config.num_audio_tokens,
            bias=False,
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def build_conditioning(
        self,
        thinker_token_ids: torch.Tensor,
        thinker_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Build hidden_text_merge conditioning from thinker outputs.

        Matches original MiniCPM-o conditioning:
            tts_embeds = emb_text(token_ids) + normalize(semantic_projection(hidden_states))

        Args:
            thinker_token_ids:     [seq] LLM token IDs from thinker
            thinker_hidden_states: [seq, llm_dim] hidden states from thinker LLM

        Returns:
            conditioning: [seq, talker_hidden_size]
        """
        # Text conditioning: emb_text maps LLM vocab token IDs → 768-dim
        llm_embeds = self.emb_text(thinker_token_ids)

        # Hidden state conditioning: project 4096-dim → 768-dim
        hidden_embeds = self.semantic_projection(thinker_hidden_states)

        # L2 normalization (when normalize_projected_hidden=True in config)
        if self.config.normalize_projected_hidden:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

        return llm_embeds + hidden_embeds

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed codec token IDs via codec_embedding (used during AR decoding)."""
        return self.codec_embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass through the Talker Llama model."""
        return self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute audio codec token logits."""
        return self.codec_head(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        prefix_map = self.hf_to_vllm_mapper.orig_to_new_prefix

        def _preprocess(
            ws: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            """Apply prefix mapper then resolve spectral-norm parametrizations.

            ``tts.head_code.0`` uses ``torch.nn.utils.spectral_norm``, which
            stores the weight as ``parametrizations.weight.original1`` (the
            actual weight matrix) and ``parametrizations.weight.original0``
            (the singular-value vector, not needed for inference).  We unwrap
            ``original1`` → ``.weight`` and discard ``original0``.
            """
            for orig_name, param in ws:
                # Apply prefix mapping
                new_name = orig_name
                for orig_pfx, new_pfx in prefix_map.items():
                    if orig_name.startswith(orig_pfx):
                        new_name = new_pfx + orig_name[len(orig_pfx):]
                        break
                # Resolve spectral-norm storage format
                if new_name.endswith(".parametrizations.weight.original1"):
                    base = new_name[: -len(".parametrizations.weight.original1")]
                    yield f"{base}.weight", param
                elif ".parametrizations.weight.original0" in new_name:
                    continue  # singular-value vector; not needed at inference
                else:
                    yield new_name, param

        loader = AutoWeightsLoader(
            self,
            # Thinker and Code2Wav weights are handled by other stages.
            skip_prefixes=["thinker.", "code2wav."],
        )
        return loader.load_weights(_preprocess(weights))
