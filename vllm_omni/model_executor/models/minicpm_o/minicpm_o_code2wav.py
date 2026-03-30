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
"""Inference-only MiniCPM-o 4.5 Code2Wav stage (CosyVoice2 + HiFi-GAN).

MiniCPM-o 4.5 bundles CosyVoice2 weights in:
    {model_dir}/assets/token2wav/flow.yaml   — model architecture config
    {model_dir}/assets/token2wav/flow.pt     — CosyVoice2 flow model weights
    {model_dir}/assets/token2wav/hift.pt     — HiFi-GAN vocoder weights

These are loaded via minicpmo-utils (pip install minicpmo-utils) which
provides stepaudio2.cosyvoice2.* CosyVoice2 implementation.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class MiniCPMOCode2Wav(nn.Module):
    """MiniCPM-o 4.5 Code2Wav: audio codec tokens → waveform.

    Stage 3 of the 3-stage vllm-omni pipeline:
      Stage 1 (Thinker): vision + audio + text → token hidden states
      Stage 2 (Talker):  TTS hidden states → audio codec tokens (s3tokenizer)
      Stage 3 (Code2Wav): codec tokens → waveform via CosyVoice2 + HiFi-GAN

    The CosyVoice2 flow model and HiFi-GAN are stored as non-Module attributes
    (bypassing nn.Module parameter tracking) because they are loaded from
    separate .pt files with yaml-driven architecture, not from the main
    HF safetensors checkpoint.  vllm's post-load parameter check passes
    because this module has no registered nn.Module parameters.

    Requires: pip install minicpmo-utils
    """

    input_modalities = "audio"

    def __init__(
        self,
        *,
        vllm_config: VllmConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self._model_dir: str | None = None
        if vllm_config is not None:
            model_id = vllm_config.model_config.model
            # Resolve HF model ID to local snapshot directory
            if os.path.isdir(model_id):
                self._model_dir = model_id
            else:
                try:
                    from huggingface_hub import snapshot_download
                    self._model_dir = snapshot_download(
                        model_id,
                        local_files_only=True,  # Don't re-download
                    )
                except Exception:
                    self._model_dir = model_id  # Fallback
        # Flow and HiFi-GAN loaded lazily in load_from_directory().
        # Stored via __dict__ to bypass nn.Module's __setattr__ and keep
        # them out of state_dict() / named_parameters().
        self.__dict__["_flow"] = None
        self.__dict__["_hift"] = None
        self.__dict__["_spk_embed_dim"] = None

    def _get_token2wav_dir(self, model_dir: str) -> str:
        return os.path.join(model_dir, "assets", "token2wav")

    def load_from_directory(self, model_dir: str, device: torch.device) -> None:
        """Load CosyVoice2 flow model and HiFi-GAN from assets/token2wav/.

        Uses minicpmo-utils (stepaudio2) for the CosyVoice2 implementation.
        """
        try:
            from hyperpyyaml import load_hyperpyyaml
            from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator
            from stepaudio2.token2wav import _setup_cosyvoice2_alias
        except ImportError as e:
            raise ImportError(
                "MiniCPM-o Code2Wav requires minicpmo-utils. "
                "Install with: pip install minicpmo-utils"
            ) from e

        # Register cosyvoice2 module aliases required by hyperpyyaml
        _setup_cosyvoice2_alias()

        token2wav_dir = self._get_token2wav_dir(model_dir)
        flow_yaml = os.path.join(token2wav_dir, "flow.yaml")
        flow_pt = os.path.join(token2wav_dir, "flow.pt")
        hift_pt = os.path.join(token2wav_dir, "hift.pt")

        # Load CosyVoice2 flow model (architecture from yaml, weights from pt)
        with open(flow_yaml) as f:
            configs = load_hyperpyyaml(f)
            flow = configs["flow"]
        flow.load_state_dict(
            torch.load(flow_pt, map_location=device, weights_only=True),
            strict=True,
        )
        flow.to(device).eval()
        self.__dict__["_flow"] = flow

        # Extract speaker embedding dimension from flow model
        if hasattr(flow, "spk_embed_affine_layer"):
            self.__dict__["_spk_embed_dim"] = flow.spk_embed_affine_layer.in_features
        else:
            self.__dict__["_spk_embed_dim"] = 192
            logger.warning(
                "MiniCPMOCode2Wav: could not determine spk_embed_dim from "
                "flow model, falling back to 192."
            )

        # Load HiFi-GAN vocoder (strips 'generator.' prefix from checkpoint keys)
        hift = HiFTGenerator()
        hift_ckpt = torch.load(hift_pt, map_location=device, weights_only=True)
        hift_state_dict = {
            k.replace("generator.", ""): v for k, v in hift_ckpt.items()
        }
        hift.load_state_dict(hift_state_dict, strict=True)
        # HiFi-GAN uses float32 internally (SineGenerator produces float32 sine
        # waves that must match linear layer dtypes).  Keep in float32 regardless
        # of the main model dtype.
        hift.float().to(device).eval()
        self.__dict__["_hift"] = hift

        logger.info(
            "MiniCPMOCode2Wav: loaded CosyVoice2 flow + HiFi-GAN from %s",
            token2wav_dir,
        )

    def forward(
        self,
        codes: torch.Tensor,
        n_timesteps: int = 10,
    ) -> torch.Tensor:
        """Convert s3tokenizer codec tokens to audio waveform.

        Args:
            codes:       [batch, seq_len] — audio token IDs from Talker
            n_timesteps: Diffusion steps for the flow model (default 10)

        Returns:
            waveform: [batch, 1, audio_len]
        """
        # CosyVoice2 flow uses Python-level diffusion loops and CPU-GPU sync ops
        # that are incompatible with CUDA graph capture.  Return a dummy tensor
        # during capture so vllm's graph compilation phase can proceed.
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return codes.new_zeros(codes.shape[0], 1, 1, dtype=torch.float32)

        # Guard against empty codec sequences (e.g. talker produced only EOS)
        if codes.numel() == 0 or codes.shape[-1] == 0:
            return codes.new_zeros(codes.shape[0], 1, 1, dtype=torch.float32)

        flow = self.__dict__["_flow"]
        hift = self.__dict__["_hift"]

        if flow is None or hift is None:
            raise RuntimeError(
                "MiniCPMOCode2Wav: model not loaded. Call load_from_directory() first."
            )

        device = codes.device
        dtype = next(flow.parameters()).dtype
        spk_dim = self.__dict__["_spk_embed_dim"]

        # CosyVoice2 flow.inference() requires batch_size == 1
        results: list[torch.Tensor] = []
        for i in range(codes.shape[0]):
            token = codes[i : i + 1].to(dtype=torch.int32)
            token_len = torch.tensor([token.shape[1]], dtype=torch.int32, device=device)

            # Empty reference prompt — no voice cloning, uses default voice.
            # 80 = CosyVoice2 mel bins (from flow.yaml); irrelevant here since
            # prompt length is 0 (no reference audio).
            prompt_token = torch.zeros(1, 0, dtype=torch.int32, device=device)
            prompt_token_len = torch.tensor([0], dtype=torch.int32, device=device)
            prompt_feat = torch.zeros(1, 0, 80, dtype=dtype, device=device)
            prompt_feat_len = torch.tensor([0], dtype=torch.int32, device=device)

            # Zero speaker embedding — no voice cloning, uses default voice
            embedding = torch.zeros(1, spk_dim, dtype=dtype, device=device)

            with torch.inference_mode():
                mel = flow.inference(
                    token=token,
                    token_len=token_len,
                    prompt_token=prompt_token,
                    prompt_token_len=prompt_token_len,
                    prompt_feat=prompt_feat,
                    prompt_feat_len=prompt_feat_len,
                    embedding=embedding,
                    n_timesteps=n_timesteps,
                )
                wav_out = hift(mel.float())  # HiFi-GAN requires float32
                wav = wav_out[0] if isinstance(wav_out, tuple) else wav_out

            results.append(wav)

        return torch.cat(results, dim=0)

    def decode(self, codes: torch.Tensor) -> list[torch.Tensor]:
        """Decode codec token sequences to waveforms.

        Args:
            codes: [batch, seq_len] — audio token IDs from Talker

        Returns:
            list[torch.Tensor]: One waveform [1, audio_len] per batch element
        """
        wav_batch = self.forward(codes)  # [batch, 1, audio_len]
        return list(wav_batch.unbind(dim=0))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights (standard vllm-omni interface).

        Code2Wav weights are in assets/token2wav/ (separate from HF checkpoint).
        Drain the weights iterator (no checkpoint keys for Code2Wav), then
        trigger file-based loading from the model directory.

        This module has no registered nn.Module parameters, so vllm's
        post-load state_dict check finds an empty dict — no error.
        """
        for _ in weights:
            pass

        if self._model_dir is not None:
            token2wav_dir = self._get_token2wav_dir(self._model_dir)
            flow_yaml = os.path.join(token2wav_dir, "flow.yaml")
            if not os.path.isfile(flow_yaml):
                # Dummy mode or incomplete checkpoint — skip file-based loading.
                logger.info(
                    "MiniCPMOCode2Wav: %s not found; skipping weight loading "
                    "(expected with load_format=dummy).",
                    flow_yaml,
                )
            else:
                # Use current CUDA device (set by vllm worker based on stage config)
                device = torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                    if torch.cuda.is_available()
                    else "cpu"
                )
                self.load_from_directory(self._model_dir, device)
        else:
            logger.warning(
                "MiniCPMOCode2Wav: model_dir not set; skipping weight loading."
            )

        return set()
