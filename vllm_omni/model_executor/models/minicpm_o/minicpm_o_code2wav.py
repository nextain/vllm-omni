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
"""Inference-only MiniCPM-o 4.5 Code2Wav stage (CosyVoice2 + HiFi-GAN)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import (
    CosyVoice3Code2Wav,
)

logger = init_logger(__name__)


def _build_minicpmo_cosyvoice_config() -> CosyVoice3Config:
    """Build a CosyVoice3Config compatible with MiniCPM-o 4.5's s3tokenizer.

    MiniCPM-o uses CosyVoice2's flow model + HiFi-GAN, which shares the same
    architecture family as CosyVoice3.  The s3tokenizer codec has a vocabulary
    of ≈6562 tokens; all other defaults from CosyVoice3Config apply.

    Note: Exact hyperparameter values are verified in Phase 7 (L2 tests) by
    comparing with the MiniCPM-o 4.5 flow.pt / hift.pt weight shapes.
    """
    cfg = CosyVoice3Config()
    # s3tokenizer vocabulary (may differ by ±1 from CosyVoice3's 6561)
    cfg.flow["vocab_size"] = 6562
    cfg.llm["speech_token_size"] = 6562
    cfg.llm["eos_token_id"] = 6563
    return cfg


class MiniCPMOCode2Wav(nn.Module):
    """MiniCPM-o 4.5 Code2Wav: audio codec tokens → waveform.

    Stage 3 of the 3-stage vllm-omni pipeline:
      Stage 1 (Thinker): vision + audio + text → token hidden states
      Stage 2 (Talker):  TTS hidden states → audio codec tokens (s3tokenizer)
      Stage 3 (Code2Wav): codec tokens → waveform via CosyVoice2 + HiFi-GAN

    MiniCPM-o uses s3tokenizer (CosyVoice codec family).  This stage reuses
    CosyVoice3Code2Wav since CosyVoice2 and CosyVoice3 share the same
    flow-matching + HiFi-GAN architecture.

    Weight loading:
      Weights are stored in separate files within the HF model snapshot:
        <model_dir>/flow.pt   — flow-matching (DiT + CFM) weights
        <model_dir>/hift.pt   — HiFi-GAN vocoder weights
      These are loaded lazily in load_from_directory(), called by the main
      model's load_weights() after the model directory is known.
    """

    input_modalities = "audio"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        cosyvoice_config = _build_minicpmo_cosyvoice_config()
        self.flow_model = CosyVoice3Code2Wav(cosyvoice_config)

        # model_dir is needed for load_weights(); extracted from vllm_config.
        self._model_dir: str | None = (
            vllm_config.model_config.model if vllm_config is not None else None
        )

    def forward(
        self,
        codes: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """Convert s3tokenizer codes to audio waveform.

        Args:
            codes:       [batch, seq_len] — audio token IDs from Talker
                         (num_vq=1, so codes has shape [B, T] not [B, 1, T])
            n_timesteps: Diffusion steps for the flow model (default 10)

        Returns:
            waveform: [batch, 1, audio_len] clipped to [-1, 1]
        """
        device = codes.device
        batch_size = codes.shape[0]

        # Build dummy zero speaker embedding (default voice, no cloning).
        # Dimension is spk_embed_dim (192 for CosyVoice3), the raw speaker
        # embedding size *before* the affine projection inside flow_model.
        embedding = torch.zeros(
            batch_size,
            self.flow_model.config.spk_embed_dim,
            device=device,
        )

        # Empty prompt: no reference audio for zero-shot cloning
        prompt_token = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        prompt_feat = torch.zeros(
            batch_size,
            0,
            self.flow_model.output_size,
            device=device,
        )

        return self.flow_model(
            token=codes,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
            n_timesteps=n_timesteps,
        )

    def chunked_decode(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
        seq_token_counts: list[int] | None = None,
    ) -> list[torch.Tensor]:
        """Decode long codec sequences in chunks to avoid OOM.

        Args:
            codes:            [batch, seq_len] — audio token IDs
            chunk_size:       Codec frames per chunk
            left_context_size: Overlap frames for boundary smoothing
            seq_token_counts: Per-request token count (for variable-length batch)

        Returns:
            list[torch.Tensor]: One waveform tensor per batch element [1, audio_len]
        """
        wavs: list[torch.Tensor] = []
        start_index = 0
        # Audio samples produced per codec token:
        #   sample_rate / input_frame_rate  (e.g. 24000 / 25 = 960)
        total_upsample = int(
            self.flow_model.config.sample_rate / self.flow_model.input_frame_rate
        )

        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index >= left_context_size else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)

            # Remove left-context portion from the output
            context_samples = context_size * total_upsample
            wavs.append(wav_chunk[..., context_samples:])
            start_index = end_index

        batch_wav = torch.cat(wavs, dim=-1)

        if seq_token_counts is None:
            return list(batch_wav.unbind(dim=0))

        result: list[torch.Tensor] = []
        for idx, token_count in enumerate(seq_token_counts):
            audio_len = token_count * total_upsample
            result.append(batch_wav[idx, :, :audio_len])
        return result

    def load_from_directory(self, model_dir: str, device: torch.device) -> None:
        """Load flow.pt and hift.pt weights from the HF model directory.

        This method should be called by the main model's load_weights()
        after the model directory has been resolved.
        """
        self.flow_model.load_weights(model_dir, device)
        logger.info("MiniCPMOCode2Wav: loaded flow.pt + hift.pt from %s", model_dir)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights (standard vllm-omni interface).

        MiniCPM-o's CosyVoice2 weights reside in separate files (flow.pt +
        hift.pt) rather than the HF safetensors checkpoint.  This method
        drains the iterator to satisfy the interface, then triggers file-based
        loading from the model directory stored during __init__.

        TODO(L2): Verify actual weight storage location by inspecting the
        MiniCPM-o-4_5 HF snapshot and updating if weights are embedded.
        """
        # Drain the iterator (weights may contain other stage keys)
        loaded: set[str] = set()
        for name, _ in weights:
            if name.startswith("code2wav."):
                loaded.add(name)

        if self._model_dir is not None:
            device = next(self.parameters(), torch.tensor(0.0)).device
            if not isinstance(device, torch.device):
                device = torch.device("cpu")
            try:
                self.load_from_directory(self._model_dir, device)
            except FileNotFoundError:
                logger.warning(
                    "MiniCPMOCode2Wav: flow.pt/hift.pt not found in %s. "
                    "Code2Wav stage will use randomly initialized weights.",
                    self._model_dir,
                )
        else:
            logger.warning(
                "MiniCPMOCode2Wav: model_dir not set; skipping weight loading."
            )

        return loaded
