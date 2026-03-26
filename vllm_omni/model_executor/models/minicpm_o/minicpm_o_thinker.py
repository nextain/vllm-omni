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
"""Inference-only MiniCPM-o 4.5 Thinker model (vision + audio + Qwen3 LLM)."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import Qwen3Config, WhisperConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.siglip import SiglipVisionTransformer
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.model_executor.models.whisper import WhisperEncoder
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
    MiniCPMOConfig,
)

logger = init_logger(__name__)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]) -> np.ndarray:
    """Generate 2D sinusoidal position embeddings.

    Args:
        embed_dim: Embedding dimension.
        grid_size: (height, width) of the position grid.

    Returns:
        pos_embed: [grid_h * grid_w, embed_dim]
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, H, W)
    grid = np.stack(grid, axis=0).reshape(2, -1)  # (2, H*W)

    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
    emb_h = _sincos_1d(grid[0], embed_dim // 2)  # (H*W, D/2)
    emb_w = _sincos_1d(grid[1], embed_dim // 2)  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)


def _sincos_1d(pos: np.ndarray, dim: int) -> np.ndarray:
    assert dim % 2 == 0
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / (10000**omega)
    out = np.outer(pos, omega)  # (N, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # (N, D)


class MiniCPMOResampler(nn.Module):
    """2D perceiver-resampler with cross-attention.

    Projects variable-length visual tokens to a fixed number of output tokens
    (query_num) via cross-attention over learnable query vectors.

    Reference: openbmb/MiniCPM-o-4_5 modeling_minicpmo.py::Resampler
    """

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        max_size: tuple[int, int] = (70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_size = list(max_size)

        self.query = nn.Parameter(torch.zeros(num_queries, embed_dim))

        self.kv_proj = (
            nn.Linear(kv_dim, embed_dim, bias=False) if kv_dim != embed_dim else nn.Identity()
        )

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_kv = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_post = nn.LayerNorm(embed_dim, eps=1e-6)
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size: list[int], device: str = "cpu") -> None:
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float()
        self.register_buffer("pos_embed", pos_embed.to(device), persistent=False)

    def _adjust_pos_cache(self, tgt_sizes: torch.Tensor, device: torch.device) -> None:
        max_h = int(tgt_sizes[:, 0].max().item())
        max_w = int(tgt_sizes[:, 1].max().item())
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device=str(device))

    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual features [batch, max_patches, kv_dim]
            tgt_sizes: Target grid sizes [batch, 2] (height, width)

        Returns:
            Resampled features [batch, num_queries, embed_dim]
        """
        bs = x.shape[0]
        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        self._adjust_pos_cache(tgt_sizes, device)

        max_patch_len = int(patch_len.max().item())
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            h, w = int(tgt_sizes[i, 0].item()), int(tgt_sizes[i, 1].item())
            pos_embed.append(
                self.pos_embed[: h * w, :].to(dtype)
            )
            key_padding_mask[i, h * w :] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True)

        x = self.ln_kv(self.kv_proj(x[:, :max_patch_len]))
        x = x + pos_embed

        q = self.ln_q(self.query).unsqueeze(1).expand(-1, bs, -1)  # (Q, B, D)
        kv = x.permute(1, 0, 2)  # (S, B, D)

        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        out = out.permute(1, 0, 2)  # (B, Q, D)
        out = self.ln_post(out)
        out = out @ self.proj
        return out  # (B, Q, embed_dim)


class MiniCPMOVisionEncoder(nn.Module):
    """SigLIP vision transformer + Resampler for MiniCPM-o.

    Processes images at native resolution (slice-based) through SigLIP,
    then projects to LLM embedding dimension via cross-attention resampler.
    """

    def __init__(
        self,
        vision_config: SiglipVisionConfig,
        llm_hidden_size: int,
        query_num: int = 64,
        drop_vision_last_layer: bool = True,
        quant_config: Any = None,
        prefix: str = "",
    ):
        super().__init__()
        # Optionally drop the last SigLIP encoder layer
        num_hidden_layers_override = (
            vision_config.num_hidden_layers - 1 if drop_vision_last_layer else None
        )
        self.encoder = SiglipVisionTransformer(
            vision_config,
            quant_config=quant_config,
            num_hidden_layers_override=num_hidden_layers_override,
            prefix=maybe_prefix(prefix, "encoder"),
        )
        self.resampler = MiniCPMOResampler(
            num_queries=query_num,
            embed_dim=llm_hidden_size,
            num_heads=llm_hidden_size // 128,
            kv_dim=vision_config.hidden_size,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        tgt_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, C, H, W] - pre-cropped image patches
            tgt_sizes: [batch, 2] - (patch_h, patch_w) for each image

        Returns:
            Visual embeddings [batch, query_num, llm_hidden_size]
        """
        # SiglipVisionTransformer.forward returns [batch, num_patches, hidden]
        visual_features = self.encoder(pixel_values)
        return self.resampler(visual_features, tgt_sizes)


class MiniCPMOAudioProjectionMLP(nn.Module):
    """2-layer MLP projecting Whisper d_model → LLM hidden dim (ReLU activation).

    Matches openbmb MultiModalProjector: Linear(in→out) → ReLU → Linear(out→out).
    HF checkpoint keys: audio_projection_layer.linear1/linear2, both with bias.
    """

    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.act_fn = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act_fn(self.linear1(x)))


class MiniCPMOAudioEncoder(nn.Module):
    """Whisper encoder + 2-layer MLP projection for MiniCPM-o.

    Encodes mel-spectrogram audio features and projects to LLM embedding
    dimension with average pooling (audio_pool_step).
    """

    def __init__(
        self,
        audio_config: WhisperConfig,
        llm_hidden_size: int,
        audio_pool_step: int = 5,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.audio_pool_step = audio_pool_step
        audio_vllm_config = vllm_config.with_hf_config(
            audio_config,
            architectures=["WhisperForConditionalGeneration"],
        )
        self.encoder = WhisperEncoder(
            vllm_config=audio_vllm_config,
            prefix=maybe_prefix(prefix, "encoder"),
        )
        # HF checkpoint: audio_projection_layer.linear1/linear2 (both with bias).
        # Mapper: audio_projection_layer.* → audio_encoder.projection.*
        self.projection = MiniCPMOAudioProjectionMLP(
            d_model=audio_config.d_model,
            hidden_size=llm_hidden_size,
        )

    def forward(self, input_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            input_features: List of mel-spectrogram tensors [num_mels, T]

        Returns:
            List of projected audio embeddings [T', llm_hidden_size]
        """
        hidden = self.encoder(input_features)  # [B, T, d_model]
        projected = self.projection(hidden)

        # Average pooling with audio_pool_step
        if self.audio_pool_step > 1:
            T = projected.shape[1]
            padded_T = (T + self.audio_pool_step - 1) // self.audio_pool_step * self.audio_pool_step
            if padded_T > T:
                projected = torch.nn.functional.pad(projected, (0, 0, 0, padded_T - T))
            projected = projected.reshape(
                projected.shape[0], -1, self.audio_pool_step, projected.shape[-1]
            )
            projected = projected.mean(dim=2)

        # Return as list (one tensor per batch element)
        return list(projected.unbind(dim=0))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights, stacking WhisperEncoder q/k/v into qkv_proj."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


class MiniCPMOThinkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    """MiniCPM-o 4.5 Thinker: multimodal input → text + TTS hidden states.

    Implements the first stage of the 3-stage vllm-omni pipeline:
      Stage 1 (Thinker): vision + audio + text → token hidden states
      Stage 2 (Talker):  TTS hidden states → audio codec tokens
      Stage 3 (Code2Wav): codec tokens → waveform

    Architecture:
      - Vision: SigLIP (image_size=980, patch=14) + Resampler (query_num=64)
      - Audio:  Whisper-medium (d_model=1024) + linear projection (→4096)
      - LLM:    Qwen3ForCausalLM (hidden_size=4096, num_layers=36)

    Weight key mapping from HuggingFace checkpoint:
      vpm.*                  → visual.encoder.*
      resampler.*            → visual.resampler.*
      apm.*                  → audio_encoder.encoder.*
      audio_projection_layer.* → audio_encoder.projection.*
      tts.*                  → (skipped — loaded by Talker stage)
      llm.lm_head.*          → language_model.lm_head.*
      llm.model.*            → language_model.model.*
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "vpm.": "visual.encoder.",
            "resampler.": "visual.resampler.",
            "apm.": "audio_encoder.encoder.",
            "audio_projection_layer.": "audio_encoder.projection.",
            "llm.lm_head.": "language_model.lm_head.",
            "llm.model.": "language_model.model.",
            "llm.": "language_model.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Placeholder strings are provided by MiniCPMVMultiModalProcessor.
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        quant_config = vllm_config.quant_config

        # Vision encoder (SigLIP + Resampler)
        with self._mark_tower_model(vllm_config, {"image"}):
            self.visual = MiniCPMOVisionEncoder(
                vision_config=config.vision_config,
                llm_hidden_size=config.hidden_size,
                query_num=config.query_num,
                drop_vision_last_layer=config.drop_vision_last_layer,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        # Audio encoder (Whisper + projection)
        with self._mark_tower_model(vllm_config, "audio"):
            self.audio_encoder = MiniCPMOAudioEncoder(
                audio_config=config.audio_config,
                llm_hidden_size=config.hidden_size,
                audio_pool_step=config.audio_pool_step,
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "audio_encoder"),
            )

        # LLM backbone (Qwen3ForCausalLM)
        qwen3_cfg = Qwen3Config(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=config.tie_word_embeddings,
        )
        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3ForCausalLM(
                vllm_config=vllm_config.with_hf_config(
                    qwen3_cfg,
                    architectures=["Qwen3ForCausalLM"],
                ),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="visual.resampler",
            tower_model=["visual.encoder.", "audio_encoder."],
        )

    def _embed_pixel_values(
        self,
        pixel_values: torch.Tensor | list[torch.Tensor],
        tgt_sizes: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Run SigLIP + resampler on a batch of image patches.

        pixel_values may be:
          - torch.Tensor [batch, C, H, W] — all patches same spatial size
          - list[torch.Tensor] — per-item list when spatial dims differ
            Each element is [num_slices, C, H, W]; tgt_sizes is [total_slices, 2]
        """
        if isinstance(pixel_values, (list, tuple)):
            results: list[torch.Tensor] = []
            slice_offset = 0
            for pv in pixel_values:
                if pv.dim() == 3:
                    pv = pv.unsqueeze(0)  # [1, C, H, W]
                num_slices = pv.shape[0]
                if tgt_sizes is not None:
                    ts = tgt_sizes[slice_offset : slice_offset + num_slices]
                else:
                    h = w = int(pv.shape[-1] ** 0.5)
                    ts = torch.tensor([[h, w]] * num_slices, device=pv.device)
                visual_embed = self.visual(pv, ts)  # [num_slices, query_num, llm_hidden]
                results.append(visual_embed.flatten(0, 1))  # [num_slices*query_num, llm_hidden]
                slice_offset += num_slices
            return tuple(results)

        # Batched tensor path (same spatial dims)
        if tgt_sizes is None:
            n_patches = pixel_values.shape[1]
            h = w = int(n_patches**0.5)
            tgt_sizes = torch.tensor(
                [[h, w]] * pixel_values.shape[0],
                device=pixel_values.device,
            )
        visual_embeds = self.visual(pixel_values, tgt_sizes)
        return tuple(visual_embeds.unbind(dim=0))

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Process vision and audio inputs into embeddings.

        Handles three kwargs layouts emitted by MiniCPMVMultiModalProcessor:
          - image: pixel_values + tgt_sizes
          - video: video_pixel_values + video_tgt_sizes (same tensor format)
          - audio: input_audio_features
        """
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # Images
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is not None:
            tgt_sizes: torch.Tensor | None = kwargs.get("tgt_sizes")  # type: ignore
            multimodal_embeddings += self._embed_pixel_values(
                pixel_values,  # type: ignore[arg-type]
                tgt_sizes,
            )

        # Videos — each item is a batch of frames; processed identically to images
        video_pixel_values = kwargs.get("video_pixel_values")
        if video_pixel_values is not None:
            video_tgt_sizes: torch.Tensor | None = kwargs.get("video_tgt_sizes")  # type: ignore
            multimodal_embeddings += self._embed_pixel_values(
                video_pixel_values,  # type: ignore[arg-type]
                video_tgt_sizes,
            )

        # Audio
        input_audio_features = kwargs.get("input_audio_features")
        if input_audio_features is not None:
            audio_embeds = self.audio_encoder(input_audio_features)
            multimodal_embeddings += tuple(audio_embeds)

        return list(multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Embed input IDs and merge multimodal embeddings."""
        inputs_embeds = self.language_model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        if is_multimodal is None:
            return inputs_embeds

        return _merge_multimodal_embeddings(inputs_embeds, multimodal_embeddings, is_multimodal)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor | IntermediateTensors, torch.Tensor | None]:
        """Run the thinker LLM and return (hidden_states, inputs_embeds).

        Returns a 2-tuple so that ``make_omni_output`` can emit both
        ``thinker_hidden_states`` (for talker semantic_projection) and
        ``thinker_text_embeds`` (for per-request slicing in gpu_ar_model_runner).
        """
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states, inputs_embeds

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _preprocess(
            ws: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            """Scope fc1/fc2 → mlp.fc1/mlp.fc2 remapping to APM keys only.

            HF Whisper checkpoint stores FFN as bare ``fc1``/``fc2``
            (e.g. ``apm.layers.0.fc1.weight``), but vllm's WhisperEncoder
            wraps the FFN in an ``mlp`` submodule (``layers.0.mlp.fc1``).

            Applying this globally would corrupt VPM keys that already use
            ``mlp.fc1`` (SigLIP's FFN is already named ``mlp.fc1`` in the
            checkpoint).  So we restrict the rename to ``apm.layers.*`` only.
            """
            for name, param in ws:
                if name.startswith("apm.layers."):
                    name = name.replace(".fc1.", ".mlp.fc1.")
                    name = name.replace(".fc2.", ".mlp.fc2.")
                yield name, param

        loader = AutoWeightsLoader(
            self,
            # TTS weights are loaded by the Talker stage
            skip_prefixes=["tts."],
        )
        return loader.load_weights(_preprocess(weights), mapper=self.hf_to_vllm_mapper)
