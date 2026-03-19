# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for MiniCPM-o 4.5 configuration classes.

Tests run on CPU with no GPU required.  They verify:
- Config classes can be instantiated with default values
- Sub-configs are correctly nested
- Key hyperparameters match the MiniCPM-o 4.5 reference architecture
"""

import pytest

pytestmark = [pytest.mark.cpu]


class TestMiniCPMTTSConfig:
    """Tests for the Talker stage configuration."""

    @pytest.fixture
    def cfg(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMTTSConfig

        return MiniCPMTTSConfig()

    def test_default_hidden_size(self, cfg):
        assert cfg.hidden_size == 768

    def test_default_num_hidden_layers(self, cfg):
        assert cfg.num_hidden_layers == 20

    def test_default_num_audio_tokens(self, cfg):
        """CosyVoice3 codec vocab: 6561 tokens + 1 EOS = 6562."""
        assert cfg.num_audio_tokens == 6562

    def test_default_num_vq(self, cfg):
        """MiniCPM-o uses num_vq=1 (single-layer codec)."""
        assert cfg.num_vq == 1

    def test_llm_dim(self, cfg):
        """llm_dim matches MiniCPM-o 4.5's Qwen3-7B thinker hidden size."""
        assert cfg.llm_dim == 4096

    def test_condition_type(self, cfg):
        assert cfg.condition_type == "hidden_text_merge"

    def test_backbone_model(self, cfg):
        assert cfg.backbone_model == "llama"

    def test_model_type(self, cfg):
        assert cfg.model_type == "minicpmtts"

    def test_custom_kwargs(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMTTSConfig

        cfg = MiniCPMTTSConfig(hidden_size=512, num_hidden_layers=10)
        assert cfg.hidden_size == 512
        assert cfg.num_hidden_layers == 10


class TestMiniCPMVSliceConfig:
    """Tests for the vision slice config."""

    def test_defaults(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMVSliceConfig

        cfg = MiniCPMVSliceConfig()
        assert cfg.patch_size == 14
        assert cfg.max_slice_nums == 9
        assert cfg.scale_resolution == 448

    def test_model_type(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMVSliceConfig

        cfg = MiniCPMVSliceConfig()
        assert cfg.model_type == "minicpmv"


class TestMiniCPMOConfig:
    """Tests for the unified MiniCPM-o 4.5 configuration."""

    @pytest.fixture
    def cfg(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMOConfig

        return MiniCPMOConfig()

    def test_model_type(self, cfg):
        assert cfg.model_type == "minicpmo"

    def test_llm_backbone_defaults(self, cfg):
        """Verify Qwen3 LLM backbone defaults match MiniCPM-o 4.5."""
        assert cfg.hidden_size == 4096
        assert cfg.num_hidden_layers == 36
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 8
        assert cfg.vocab_size == 151748

    def test_vision_config_is_siglip(self, cfg):
        from transformers import SiglipVisionConfig

        assert isinstance(cfg.vision_config, SiglipVisionConfig)
        assert cfg.vision_config.hidden_size == 1152

    def test_audio_config_is_whisper(self, cfg):
        from transformers import WhisperConfig

        assert isinstance(cfg.audio_config, WhisperConfig)

    def test_tts_config_is_minicpmtts(self, cfg):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMTTSConfig

        assert isinstance(cfg.tts_config, MiniCPMTTSConfig)

    def test_slice_config_is_minicpmv(self, cfg):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMVSliceConfig

        assert isinstance(cfg.slice_config, MiniCPMVSliceConfig)
        assert cfg.slice_config.max_slice_nums == 9  # MiniCPMVSliceConfig default

    def test_query_num(self, cfg):
        assert cfg.query_num == 64

    def test_init_flags(self, cfg):
        assert cfg.init_vision is True
        assert cfg.init_audio is True
        assert cfg.init_tts is True

    def test_tts_config_from_dict(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMOConfig

        cfg = MiniCPMOConfig(tts_config={"hidden_size": 512, "num_hidden_layers": 8})
        assert cfg.tts_config.hidden_size == 512
        assert cfg.tts_config.num_hidden_layers == 8

    def test_slice_config_from_dict(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMOConfig

        cfg = MiniCPMOConfig(slice_config={"max_slice_nums": 4})
        assert cfg.slice_config.max_slice_nums == 4
