# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for MiniCPM-o 4.5 stage input processors and talker preprocess.

All tests run on CPU with no GPU required.  They verify:
- _find_tts_bound: TTS boundary detection with edge cases
- thinker2talker: hidden state slicing and config-based token ID reading
- talker_preprocess: prefill vs decode branching, trailing conditioning queue
"""

import pytest
import torch

pytestmark = [pytest.mark.cpu]


# ---------------------------------------------------------------------------
# _find_tts_bound
# ---------------------------------------------------------------------------


class TestFindTtsBound:
    """Tests for TTS boundary detection in thinker token sequences."""

    @pytest.fixture
    def find_bound(self):
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            _find_tts_bound,
        )
        return _find_tts_bound

    # Default token IDs matching MiniCPM-o 4.5
    BOS = 151703
    EOS = 151704

    def test_normal_boundary(self, find_bound):
        """Standard case: tts_bos and tts_eos present."""
        tokens = [10, 20, self.BOS, 30, 40, 50, self.EOS, 60]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        # start = position AFTER tts_bos (index 3)
        assert start == 3
        # end = position OF tts_eos (index 6)
        assert end == 6

    def test_no_markers(self, find_bound):
        """No TTS markers → use entire sequence (start=0, end=None)."""
        tokens = [10, 20, 30, 40]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        assert start == 0
        assert end is None

    def test_bos_only(self, find_bound):
        """Only tts_bos, no tts_eos → start after bos, end=None."""
        tokens = [10, self.BOS, 30, 40, 50]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        assert start == 2  # after bos at index 1
        assert end is None

    def test_eos_only(self, find_bound):
        """Only tts_eos, no tts_bos → start=0, end at eos position."""
        tokens = [10, 20, 30, self.EOS, 50]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        assert start == 0  # no bos found, fallback
        assert end == 3

    def test_bos_at_end(self, find_bound):
        """tts_bos at last position → start = len(tokens), no content."""
        tokens = [10, 20, self.BOS]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        assert start == 3  # after last element
        assert end is None

    def test_adjacent_bos_eos(self, find_bound):
        """tts_bos immediately followed by tts_eos → empty content."""
        tokens = [10, self.BOS, self.EOS, 20]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        assert start == 2
        assert end == 2
        # tokens[2:2] is empty — matches expected behavior

    def test_multiple_bos(self, find_bound):
        """Multiple tts_bos → uses the LAST one (overwrites)."""
        tokens = [self.BOS, 10, self.BOS, 20, 30, self.EOS]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        assert start == 3  # after second bos at index 2
        assert end == 5

    def test_empty_list(self, find_bound):
        """Empty token list → start=0, end=None."""
        start, end = find_bound([], self.BOS, self.EOS)
        assert start == 0
        assert end is None

    def test_custom_token_ids(self, find_bound):
        """Works with arbitrary token IDs (not just defaults)."""
        tokens = [1, 2, 99, 3, 4, 100, 5]
        start, end = find_bound(tokens, tts_bos_id=99, tts_eos_id=100)
        assert start == 3
        assert end == 5


# ---------------------------------------------------------------------------
# thinker2talker
# ---------------------------------------------------------------------------


class TestThinker2Talker:
    """Tests for the thinker→talker stage input processor."""

    def _make_stage_list(self, token_ids, hidden_states,
                         tts_bos_id=151703, tts_eos_id=151704,
                         prompt_len=5):
        """Create a mock stage_list with thinker outputs.

        Args:
            token_ids: Full token sequence (prompt + generated).
            hidden_states: Hidden states tensor [total_tokens, hidden_dim].
            prompt_len: Number of tokens to treat as prompt (rest are generated).
        """
        from unittest.mock import MagicMock
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
            MiniCPMOConfig,
        )

        # Mock thinker output
        output = MagicMock()
        output.token_ids = token_ids[prompt_len:]
        output.multimodal_output = {"thinker_hidden_states": hidden_states}

        thinker_output = MagicMock()
        thinker_output.prompt_token_ids = token_ids[:prompt_len]
        thinker_output.outputs = [output]

        # Mock stage with config
        tts_config = MagicMock()
        tts_config.tts_bos_token_id = tts_bos_id
        tts_config.tts_eos_token_id = tts_eos_id

        config = MagicMock()
        config.tts_config = tts_config

        stage = MagicMock()
        stage.engine_outputs = [thinker_output]
        stage.vllm_config.model_config.hf_config = config

        return [stage]

    def test_basic_slicing(self):
        """Extracts tokens between tts_bos and tts_eos."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker,
        )

        BOS, EOS = 151703, 151704
        # 5 prompt tokens + 5 generated tokens
        token_ids = [1, 2, 3, 4, 5, BOS, 10, 20, 30, EOS]
        hidden = torch.randn(10, 4096)  # 10 tokens total

        stage_list = self._make_stage_list(token_ids, hidden)
        result = thinker2talker(stage_list, [0])

        assert len(result) == 1
        prompt = result[0]
        info = prompt.additional_information

        # TTS content: tokens [10, 20, 30] (indices 6-8, after BOS at 5)
        assert info["thinker_token_ids"].shape[0] == 3
        assert info["thinker_hidden_states"].shape[0] == 3
        # +2 for text_eos and audio_bos boundary tokens
        assert len(prompt.prompt_token_ids) == 3 + 2

    def test_reads_token_ids_from_config(self):
        """Token IDs come from config, not hardcoded."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker,
        )

        CUSTOM_BOS, CUSTOM_EOS = 99999, 99998
        token_ids = [1, 2, 3, 4, 5, CUSTOM_BOS, 10, 20, CUSTOM_EOS, 30]
        hidden = torch.randn(10, 4096)

        stage_list = self._make_stage_list(
            token_ids, hidden,
            tts_bos_id=CUSTOM_BOS, tts_eos_id=CUSTOM_EOS,
        )
        result = thinker2talker(stage_list, [0])

        info = result[0].additional_information
        # Content between custom BOS and EOS: [10, 20] (indices 6-7)
        assert info["thinker_token_ids"].shape[0] == 2

    def test_no_markers_uses_all(self):
        """No TTS markers → uses all tokens as fallback."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker,
        )

        token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        hidden = torch.randn(10, 4096)

        stage_list = self._make_stage_list(token_ids, hidden)
        result = thinker2talker(stage_list, [0])

        info = result[0].additional_information
        assert info["thinker_token_ids"].shape[0] == 10

    def test_hidden_shorter_than_tokens_clamps(self):
        """When hidden_states has fewer entries than tokens, clamp to hidden len."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker,
        )

        BOS, EOS = 151703, 151704
        token_ids = [1, 2, 3, 4, 5, BOS, 10, 20, 30, EOS]
        # Only 7 hidden states for 10 tokens (simulates length mismatch)
        hidden = torch.randn(7, 4096)

        stage_list = self._make_stage_list(token_ids, hidden)
        result = thinker2talker(stage_list, [0])

        info = result[0].additional_information
        # Should not crash; lengths should be clamped to min of hidden/token
        assert info["thinker_token_ids"].shape[0] == info["thinker_hidden_states"].shape[0]
        assert info["thinker_token_ids"].shape[0] <= 7

    def test_hidden_state_dtype_preserved(self):
        """Hidden states should preserve native dtype (no fp32 conversion)."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker,
        )

        BOS, EOS = 151703, 151704
        token_ids = [1, 2, 3, 4, 5, BOS, 10, EOS]
        hidden = torch.randn(8, 4096, dtype=torch.bfloat16)

        stage_list = self._make_stage_list(token_ids, hidden)
        result = thinker2talker(stage_list, [0])

        info = result[0].additional_information
        assert info["thinker_hidden_states"].dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# talker_preprocess (mock-based, no GPU)
# ---------------------------------------------------------------------------


class TestTalkerPreprocess:
    """Tests for the talker preprocess conditioning logic."""

    @pytest.fixture
    def preprocess_env(self):
        """Create a minimal mock environment for talker_preprocess."""
        from unittest.mock import MagicMock
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
            MiniCPMOConfig,
            MiniCPMTTSConfig,
        )

        config = MiniCPMOConfig()

        # Mock talker with real embeddings (small dims for CPU)
        talker = MagicMock()
        hidden_size = 16  # talker hidden size (real: 768)
        llm_dim = 32      # thinker hidden size (real: 4096)

        # emb_text: returns [seq, hidden_size] from token IDs
        def mock_emb_text(ids):
            return torch.randn(ids.shape[0], hidden_size)
        talker.emb_text = mock_emb_text

        # build_conditioning: returns [seq, hidden_size]
        def mock_build_conditioning(t_ids, t_hid):
            return torch.randn(t_ids.shape[0], hidden_size)
        talker.build_conditioning = mock_build_conditioning

        # embed_input_ids: codec embedding
        def mock_embed_input_ids(ids):
            return torch.randn(ids.shape[0], hidden_size)
        talker.embed_input_ids = mock_embed_input_ids

        # Mock parameters for _module_device
        talker.parameters = MagicMock(
            return_value=iter([torch.zeros(1)])  # CPU
        )

        # Build the unified model shell (just enough for preprocess)
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o import (
            MiniCPMOForConditionalGeneration,
        )
        model = MiniCPMOForConditionalGeneration.__new__(
            MiniCPMOForConditionalGeneration
        )
        model.config = config
        model.talker = talker

        return model, hidden_size, llm_dim

    def test_prefill_builds_conditioning(self, preprocess_env):
        """Prefill (span_len > 1) builds conditioning from thinker outputs."""
        model, hidden_size, llm_dim = preprocess_env
        seq_len = 5

        input_ids = torch.zeros(seq_len, dtype=torch.long)
        info_dict = {
            "thinker_token_ids": torch.zeros(3, dtype=torch.long),
            "thinker_hidden_states": torch.randn(3, llm_dim),
        }

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds is not None
        assert embeds.shape[0] == seq_len
        assert embeds.shape[1] == hidden_size

    def test_prefill_no_trailing_when_exact_fit(self, preprocess_env):
        """When span_len == conditioning length, no trailing is stored."""
        model, hidden_size, llm_dim = preprocess_env
        # 3 thinker tokens + 2 boundary = 5 conditioning positions
        # Request exactly 5 → no trailing
        seq_len = 5

        input_ids = torch.zeros(seq_len, dtype=torch.long)
        info_dict = {
            "thinker_token_ids": torch.zeros(3, dtype=torch.long),
            "thinker_hidden_states": torch.randn(3, llm_dim),
        }

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds.shape[0] == seq_len
        assert "trailing_conditioning" not in update

    def test_prefill_stores_trailing(self, preprocess_env):
        """When conditioning is longer than span, trailing is stored."""
        model, hidden_size, llm_dim = preprocess_env
        # 3 thinker tokens + 2 boundary = 5 conditioning positions
        # But only request 3 positions → 2 trailing
        span_len = 3

        input_ids = torch.zeros(span_len, dtype=torch.long)
        info_dict = {
            "thinker_token_ids": torch.zeros(3, dtype=torch.long),
            "thinker_hidden_states": torch.randn(3, llm_dim),
        }

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds.shape[0] == span_len
        assert "trailing_conditioning" in update
        assert update["trailing_conditioning"].shape[0] == 2  # remaining
        assert update["num_processed_tokens"] == span_len

    def test_prefill_second_chunk_continues(self, preprocess_env):
        """Chunked prefill: 2nd chunk starts from num_processed_tokens."""
        model, hidden_size, llm_dim = preprocess_env
        # 3 thinker tokens + 2 boundary = 5 conditioning positions
        # 1st chunk consumed 2, 2nd chunk requests 2 more → 1 trailing
        span_len = 2

        input_ids = torch.zeros(span_len, dtype=torch.long)
        info_dict = {
            "thinker_token_ids": torch.zeros(3, dtype=torch.long),
            "thinker_hidden_states": torch.randn(3, llm_dim),
            "num_processed_tokens": 2,  # 1st chunk already consumed 2
        }

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds.shape[0] == span_len
        assert update["num_processed_tokens"] == 4  # 2 + 2
        # 5 total - 4 consumed = 1 trailing
        assert "trailing_conditioning" in update
        assert update["trailing_conditioning"].shape[0] == 1

    def test_decode_consumes_trailing(self, preprocess_env):
        """Decode (span_len=1) consumes trailing conditioning."""
        model, hidden_size, _ = preprocess_env

        trailing = torch.randn(2, hidden_size)
        input_ids = torch.zeros(1, dtype=torch.long)
        info_dict = {"trailing_conditioning": trailing}

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds is not None
        assert embeds.shape == (1, hidden_size)
        # One trailing consumed, one remaining
        assert "trailing_conditioning" in update
        assert update["trailing_conditioning"].shape[0] == 1

    def test_decode_pure_ar_after_trailing_exhausted(self, preprocess_env):
        """After trailing is exhausted, uses pure codec embedding."""
        model, hidden_size, _ = preprocess_env

        input_ids = torch.zeros(1, dtype=torch.long)
        # No trailing conditioning
        info_dict = {}

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds is not None
        assert embeds.shape == (1, hidden_size)
        # No trailing update
        assert "trailing_conditioning" not in update

    def test_decode_empty_trailing_is_pure_ar(self, preprocess_env):
        """Empty trailing tensor (shape[0]=0) → pure AR mode."""
        model, hidden_size, _ = preprocess_env

        empty_trailing = torch.zeros(0, hidden_size)
        input_ids = torch.zeros(1, dtype=torch.long)
        info_dict = {"trailing_conditioning": empty_trailing}

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds.shape == (1, hidden_size)
        assert "trailing_conditioning" not in update

    def test_prefill_no_conditioning_fallback(self, preprocess_env):
        """Prefill without thinker data falls back to codec embedding."""
        model, hidden_size, _ = preprocess_env

        input_ids = torch.zeros(5, dtype=torch.long)
        # No thinker_token_ids or thinker_hidden_states
        info_dict = {}

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds is not None
        assert embeds.shape == (5, hidden_size)


# ---------------------------------------------------------------------------
# Config: new tts_bos/eos token ID fields
# ---------------------------------------------------------------------------


class TestTTSConfigTokenIds:
    """Verify the new tts_bos/eos token ID config fields."""

    def test_default_tts_bos_token_id(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
            MiniCPMTTSConfig,
        )
        cfg = MiniCPMTTSConfig()
        assert cfg.tts_bos_token_id == 151703

    def test_default_tts_eos_token_id(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
            MiniCPMTTSConfig,
        )
        cfg = MiniCPMTTSConfig()
        assert cfg.tts_eos_token_id == 151704

    def test_custom_tts_bos_eos(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
            MiniCPMTTSConfig,
        )
        cfg = MiniCPMTTSConfig(tts_bos_token_id=9999, tts_eos_token_id=9998)
        assert cfg.tts_bos_token_id == 9999
        assert cfg.tts_eos_token_id == 9998

    def test_accessible_via_parent_config(self):
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import (
            MiniCPMOConfig,
        )
        cfg = MiniCPMOConfig()
        assert cfg.tts_config.tts_bos_token_id == 151703
        assert cfg.tts_config.tts_eos_token_id == 151704
