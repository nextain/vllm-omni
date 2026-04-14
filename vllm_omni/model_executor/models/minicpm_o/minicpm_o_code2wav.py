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

import math
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
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
        self.__dict__["_spk_embed"] = None  # Cached speaker embedding (loaded once)

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
        flow.eval()  # GPU move deferred to first forward() call
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
        hift.float().eval()  # GPU move deferred to first forward() call
        self.__dict__["_hift"] = hift

        # Load speaker embedding once at weight-load time (not on every forward).
        # Stored via __dict__ to bypass nn.Module parameter tracking.
        spk_dim = self.__dict__["_spk_embed_dim"]
        _spk_file = Path(__file__).parent / "female_spk_embedding.npy"
        try:
            _spk_np = np.load(str(_spk_file)).astype(np.float32)
            _spk_embed = torch.from_numpy(_spk_np).unsqueeze(0)
            if _spk_embed.shape[1] != spk_dim:
                logger.warning(
                    "MiniCPMOCode2Wav: female_spk_embedding dim %d != expected %d, using zeros",
                    _spk_embed.shape[1], spk_dim,
                )
                _spk_embed = torch.zeros(1, spk_dim)
        except Exception:  # FileNotFoundError, IOError, corrupted .npy, etc.
            _spk_embed = torch.zeros(1, spk_dim)
        self.__dict__["_spk_embed"] = _spk_embed

        logger.info(
            "MiniCPMOCode2Wav: loaded CosyVoice2 flow + HiFi-GAN from %s",
            token2wav_dir,
        )

    def forward(
        self,
        codes: torch.Tensor,
        n_timesteps: int = 10,
        left_context_size: int = 0,
    ) -> torch.Tensor:
        """Convert s3tokenizer codec tokens to audio waveform.

        Args:
            codes:       [batch, seq_len] — audio token IDs from Talker
            n_timesteps: Diffusion steps for the flow model (default 10)
            left_context_size: Number of frames to preserve from previous chunk (for streaming)

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

        # Lazy move from CPU to the correct GPU on first forward call.
        # CosyVoice2 flow is loaded from external YAML, not nn.Module
        # registration — check device via parameters or buffers.
        p = next(flow.parameters(), None)
        if p is not None:
            flow_device = p.device
            dtype = p.dtype
        else:
            b = next(flow.buffers(), None)
            flow_device = b.device if b is not None else torch.device("cpu")
            dtype = b.dtype if b is not None else torch.float32

        if flow_device != device:
            flow.to(device)
            hift.to(device)

        spk_dim = self.__dict__["_spk_embed_dim"]

        # Speaker embedding cached at load_from_directory() time u2014 no disk I/O here.
        _spk_embed = self.__dict__["_spk_embed"]
        if _spk_embed is None:
            # Fallback if load_from_directory was not called (e.g. dummy mode)
            _spk_embed = torch.zeros(1, spk_dim if spk_dim is not None else 192)

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

            embedding = _spk_embed.to(dtype=dtype, device=device)

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
                # _trim_silence contract: requires 2D [1, audio_len].
                # HiFTGenerator._istft returns [1, audio_len] via torch.istft;
                # guard against hypothetical 1D vocoder variants.
                if wav.ndim == 1:
                    wav = wav.unsqueeze(0)  # [audio_len] -> [1, audio_len]

            # Trim left-context audio: left context tokens are prepended to
            # help the flow model generate smooth chunk boundaries, but their
            # corresponding audio was already emitted in the previous chunk.
            # Keep only the audio fraction corresponding to the NEW tokens
            # (the tail of the sequence) to avoid 2x duration output.
            # Use token.shape[1] (per-item length) rather than codes.shape[-1]
            # (batch max-length) to handle variable-length batches correctly.
            if left_context_size > 0 and wav.shape[-1] > 0:
                item_total_tokens = token.shape[1]
                new_token_count = item_total_tokens - left_context_size
                if new_token_count <= 0:
                    # All tokens are left context (already emitted) — no new audio.
                    wav = wav[..., :0]
                else:
                    samples_to_keep = math.ceil(
                        wav.shape[-1] * new_token_count / item_total_tokens
                    )
                    if samples_to_keep > 0:
                        wav = wav[..., -samples_to_keep:]
                    else:
                        # new_token_count rounds to 0 samples — emit nothing.
                        wav = wav[..., :0]

            # _trim_silence returns [1, audio_len] (2D); unsqueeze to
            # [1, 1, audio_len] (3D) so torch.cat produces [batch, 1, audio_len].
            results.append(self._trim_silence(wav).unsqueeze(1))

        # Filter out empty tensors before concatenation: if any batch item
        # produced no new audio (all-context or rounds-to-zero), torch.cat
        # would fail on mismatched trailing dimensions.
        non_empty = [r for r in results if r.numel() > 0]
        if not non_empty:
            return codes.new_zeros(codes.shape[0], 1, 1, dtype=torch.float32)
        return torch.cat(non_empty, dim=0)  # [batch, 1, audio_len]

    def _trim_silence(self, wav: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
        """Trim trailing silence/noise from waveform.

        The Talker may generate codec tokens up to max_tokens without
        producing the stop token (6561).  CosyVoice2 then synthesises
        audio for the full sequence including silence/noise for the
        trailing portion.  This detects where speech ends by energy
        analysis and trims the trailing silence.

        Args:
            wav: [1, audio_len] waveform tensor
            threshold: Energy threshold as fraction of peak energy
        """
        if wav.numel() == 0 or wav.ndim < 2 or wav.shape[-1] == 0:
            return wav
        frame_samples = int(0.04 * 16000)  # 40ms at 16kHz
        n_frames = wav.shape[-1] // frame_samples
        if n_frames == 0:
            return wav

        energies = torch.zeros(n_frames, device=wav.device)
        for i in range(n_frames):
            frame = wav[0, i * frame_samples:(i + 1) * frame_samples]
            energies[i] = torch.mean(frame ** 2).float()

        peak_energy = energies.max().item()
        if peak_energy < 1e-8:
            return wav

        # Scan from end to find last frame above threshold
        energy_thresh = peak_energy * threshold
        speech_end_frame = n_frames
        for i in range(n_frames - 1, -1, -1):
            if energies[i] >= energy_thresh:
                speech_end_frame = i + 1
                break

        # Trim with small padding
        trim_sample = min((speech_end_frame + 2) * frame_samples, wav.shape[-1])
        return wav[:, :trim_sample]

    def decode(self, codes: torch.Tensor, left_context_size: int = 0) -> list[torch.Tensor]:
        """Decode codec token sequences to waveforms.

        Args:
            codes: [batch, seq_len] — audio token IDs from Talker
            left_context_size: Number of leading codec tokens that are
                left-context (already emitted in a previous chunk).
                Must be 0 for offline/non-streaming use.

        Returns:
            list[torch.Tensor]: One waveform [1, audio_len] per batch element
        """
        wav_batch = self.forward(codes, left_context_size=left_context_size)  # [batch, 1, audio_len]
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
                # Load to CPU first, then move to correct GPU in forward().
                # vllm's multi-stage executor may not have set CUDA device yet
                # during load_weights(), so loading directly to GPU can hit
                # the wrong device or OOM.
                self.load_from_directory(self._model_dir, torch.device("cpu"))
        else:
            logger.warning(
                "MiniCPMOCode2Wav: model_dir not set; skipping weight loading."
            )

        return set()
