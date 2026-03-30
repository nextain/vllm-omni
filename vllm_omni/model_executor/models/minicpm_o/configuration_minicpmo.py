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
"""Configuration classes for MiniCPM-o 4.5 inference in vllm-omni."""

from transformers import SiglipVisionConfig, WhisperConfig
from transformers.configuration_utils import PretrainedConfig


class MiniCPMVSliceConfig(PretrainedConfig):
    """Configuration for MiniCPM-o vision slicing (image tiling)."""

    model_type = "minicpmv"

    def __init__(
        self,
        patch_size: int = 14,
        max_slice_nums: int = 1,
        scale_resolution: int = 448,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution


class MiniCPMTTSConfig(PretrainedConfig):
    """Configuration for MiniCPMTTS — the Talker stage (Llama AR backbone).

    Generates audio tokens from LLM hidden states via hidden_text_merge
    conditioning. Uses s3tokenizer as the audio codec (num_vq=1).
    """

    model_type = "minicpmtts"

    def __init__(
        self,
        # LLM interface
        llm_dim: int = 4096,
        llm_intermediate_size: int = 768,
        llm_down_scale: bool = False,
        llm_dim_model_base: int = 256,
        projector_type: str = "mlp",
        normalize_projected_hidden: bool = True,
        # Talker LLM (Llama backbone)
        hidden_act: str = "silu",
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 12,
        num_hidden_layers: int = 20,
        max_position_embeddings: int = 4096,
        # Audio codec
        num_audio_tokens: int = 6562,
        num_text_tokens: int = 152064,
        num_mel_bins: int = 100,
        num_vq: int = 1,
        audio_bos_token_id: int = 151687,
        text_eos_token_id: int = 151692,
        tts_bos_token_id: int = 151703,
        tts_eos_token_id: int = 151704,
        audio_tokenizer_type: str = "s3tokenizer",
        audio_tokenizer_sample_rate: int = 16000,
        # Conditioning
        condition_type: str = "hidden_text_merge",
        backbone_model: str = "llama",
        use_llm_hidden_state: bool = False,
        use_text: bool = True,
        interleaved: bool = False,
        # Attention
        attention_type: str = "full_attention",
        recomputed_chunks: int = 1,
        window_size: int = 2,
        # Streaming (inference-only; training flags excluded)
        streaming: bool = False,
        streaming_text_chunk_min: int = 3,
        streaming_text_chunk_max: int = 7,
        streaming_text_reserved_len: int = 300,
        streaming_audio_chunk_size: int = 50,
        s3_stream_chunk_size: int = 25,
        s3_stream_n_timesteps: int = 10,
        s3_stream_prelook_size: int = 3,
        s3_stream_generate: bool = False,
        streaming_sliding_window: bool = False,
        streaming_sliding_window_max_text_len: int = 500,
        streaming_sliding_window_average_speed: int = 5,
        streaming_sliding_window_fast_speed: int = 7,
        streaming_sliding_window_slow_speed: int = 3,
        streaming_sliding_window_audio_frame_rate: int = 50,
        streaming_sliding_window_audio_init_text_length: int = 10,
        streaming_sliding_window_audio_window_size: int = 300,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_dim = llm_dim
        self.llm_hidden_size = llm_dim
        self.llm_intermediate_size = llm_intermediate_size
        self.llm_down_scale = llm_down_scale
        self.llm_dim_model_base = llm_dim_model_base
        self.projector_type = projector_type
        self.normalize_projected_hidden = normalize_projected_hidden

        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings

        self.num_audio_tokens = num_audio_tokens
        self.num_text_tokens = num_text_tokens
        self.num_mel_bins = num_mel_bins
        self.num_vq = num_vq
        self.audio_bos_token_id = audio_bos_token_id
        self.text_eos_token_id = text_eos_token_id
        self.tts_bos_token_id = tts_bos_token_id
        self.tts_eos_token_id = tts_eos_token_id
        self.audio_tokenizer_type = audio_tokenizer_type
        self.audio_tokenizer_sample_rate = audio_tokenizer_sample_rate

        self.condition_type = condition_type
        self.backbone_model = backbone_model
        self.use_llm_hidden_state = use_llm_hidden_state
        self.use_text = use_text
        self.interleaved = interleaved

        self.attention_type = attention_type
        self.recomputed_chunks = recomputed_chunks
        self.window_size = window_size

        self.streaming = streaming
        self.streaming_text_chunk_min = streaming_text_chunk_min
        self.streaming_text_chunk_max = streaming_text_chunk_max
        self.streaming_text_reserved_len = streaming_text_reserved_len
        self.streaming_audio_chunk_size = streaming_audio_chunk_size
        self.s3_stream_chunk_size = s3_stream_chunk_size
        self.s3_stream_n_timesteps = s3_stream_n_timesteps
        self.s3_stream_prelook_size = s3_stream_prelook_size
        self.s3_stream_generate = s3_stream_generate
        self.streaming_sliding_window = streaming_sliding_window
        self.streaming_sliding_window_max_text_len = streaming_sliding_window_max_text_len
        self.streaming_sliding_window_average_speed = streaming_sliding_window_average_speed
        self.streaming_sliding_window_fast_speed = streaming_sliding_window_fast_speed
        self.streaming_sliding_window_slow_speed = streaming_sliding_window_slow_speed
        self.streaming_sliding_window_audio_frame_rate = streaming_sliding_window_audio_frame_rate
        self.streaming_sliding_window_audio_init_text_length = (
            streaming_sliding_window_audio_init_text_length
        )
        self.streaming_sliding_window_audio_window_size = streaming_sliding_window_audio_window_size


# Default vision config matches openbmb/MiniCPM-o-4_5 vision_config in config.json
_DEFAULT_VISION_CONFIG = {
    "hidden_size": 1152,
    "image_size": 980,
    "intermediate_size": 4304,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "patch_size": 14,
}


class MiniCPMOConfig(PretrainedConfig):
    """Configuration for MiniCPM-o 4.5 unified model.

    Contains the full LLM config (Qwen3 backbone) plus sub-configs for
    vision (SigLIP2), audio (Whisper), and TTS (MiniCPMTTS).

    In vllm-omni this is split into three stages:
      - Thinker: MiniCPMOThinkerForConditionalGeneration
      - Talker:  MiniCPMOTalkerForConditionalGeneration
      - Code2Wav: MiniCPMOCode2Wav
    """

    model_type = "minicpmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        # Qwen3 LLM backbone params (mirrored from Qwen3Config)
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 12288,
        hidden_act: str = "silu",
        vocab_size: int = 151748,
        max_position_embeddings: int = 40960,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-06,
        tie_word_embeddings: bool = False,
        torch_dtype: str = "bfloat16",
        # Vision
        query_num: int = 64,
        image_size: int = 448,
        drop_vision_last_layer: bool = False,
        batch_vision_input: bool = True,
        use_image_id: bool = True,
        vision_batch_size: int = 16,
        slice_config: dict | None = None,
        vision_config: dict | None = None,
        # Audio
        audio_config: dict | None = None,
        audio_pool_step: int = 5,
        audio_chunk_length: float = 1.0,
        stream_input: bool = True,
        listen_speak_type: str = "asr",
        # TTS / Talker
        tts_config: dict | None = None,
        # Init flags
        init_vision: bool = True,
        init_audio: bool = True,
        init_tts: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # LLM backbone
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype

        # Vision
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.use_image_id = use_image_id
        self.vision_batch_size = vision_batch_size

        if slice_config is None:
            self.slice_config = MiniCPMVSliceConfig()
        else:
            self.slice_config = MiniCPMVSliceConfig(**slice_config)
        self.slice_mode = True

        if vision_config is None:
            self.vision_config = SiglipVisionConfig(**_DEFAULT_VISION_CONFIG)
        elif isinstance(vision_config, dict):
            self.vision_config = SiglipVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.patch_size = self.vision_config.patch_size

        # Audio
        if audio_config is None:
            self.audio_config = WhisperConfig()
        elif isinstance(audio_config, dict):
            self.audio_config = WhisperConfig(**audio_config)
        else:
            self.audio_config = audio_config

        self.audio_pool_step = audio_pool_step
        self.audio_chunk_length = audio_chunk_length
        self.stream_input = stream_input
        self.listen_speak_type = listen_speak_type

        # TTS / Talker
        if tts_config is None:
            self.tts_config = MiniCPMTTSConfig()
        elif isinstance(tts_config, dict):
            self.tts_config = MiniCPMTTSConfig(**tts_config)
        else:
            self.tts_config = tts_config

        # Init flags
        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts
