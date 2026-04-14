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
        """Only tts_eos without preceding tts_bos u2192 EOS is ignored (start=0, end=None)."""
        tokens = [10, 20, 30, self.EOS, 50]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        assert start == 0  # no bos found u2192 fallback to 0
        # EOS is only recorded after BOS is seen u2014 absent BOS means end=None
        assert end is None

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

    def test_im_end_fallback(self, find_bound):
        """When tts_eos absent, <|im_end|> after BOS is used as fallback boundary."""
        IM_END = 151645
        tokens = [10, self.BOS, 30, 40, IM_END, 50]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        # BOS at index 1 -> start = 2
        # No tts_eos present; im_end at index 4 -> end = 4
        assert start == 2
        assert end == 4

    def test_eos_before_bos(self, find_bound):
        """EOS before BOS is ignored (EOS only recorded after BOS)."""
        tokens = [self.EOS, 10, self.BOS, 20]
        start, end = find_bound(tokens, self.BOS, self.EOS)
        # BOS at index 2 -> start = 3
        # EOS at index 0 preceded BOS -- not recorded -> end = None
        assert start == 3
        assert end is None


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
        info = prompt["additional_information"]

        # TTS content: tokens [10, 20, 30] (indices 6-8, after BOS at 5)
        assert info["thinker_token_ids"].shape[0] == 3
        assert info["thinker_hidden_states"].shape[0] == 3
        assert info["thinker_token_ids"].tolist() == [10, 20, 30]
        # +2 for text_eos and audio_bos boundary tokens
        assert len(prompt["prompt_token_ids"]) == 3 + 2

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

        info = result[0]["additional_information"]
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

        info = result[0]["additional_information"]
        assert info["thinker_token_ids"].shape[0] == 10
        # +2 for boundary tokens
        assert len(result[0]["prompt_token_ids"]) == 10 + 2

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

        info = result[0]["additional_information"]
        # Should not crash; lengths should be clamped to min of hidden/token
        assert info["thinker_token_ids"].shape[0] == info["thinker_hidden_states"].shape[0]
        assert info["thinker_token_ids"].shape[0] <= 7

    def test_tts_boundary_beyond_hidden_fallback(self):
        """When tts_start >= hidden_len, fallback to empty slice u2192 gen-portion path."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker,
        )

        BOS, EOS = 151703, 151704
        # BOS at index 7 with only 7 hidden states: tts_start=8 >= hidden_len=7
        token_ids = [1, 2, 3, 4, 5, 6, 7, BOS, 10, EOS]
        hidden = torch.randn(7, 4096)

        stage_list = self._make_stage_list(token_ids, hidden)
        # Should not crash; falls back gracefully
        result = thinker2talker(stage_list, [0])

        info = result[0]["additional_information"]
        # Lengths must always be equal and non-negative
        assert info["thinker_token_ids"].shape[0] == info["thinker_hidden_states"].shape[0]
        assert info["thinker_token_ids"].shape[0] >= 0

    def test_hidden_state_cast_to_float32(self):
        """Hidden states are cast to float32 for serialization compatibility."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker,
        )

        BOS, EOS = 151703, 151704
        token_ids = [1, 2, 3, 4, 5, BOS, 10, EOS]
        # Input in bfloat16 u2014 code explicitly casts to float32 for serialization
        # (numpy doesn't support bfloat16 and shared-memory transport requires f32)
        hidden = torch.randn(8, 4096, dtype=torch.bfloat16)

        stage_list = self._make_stage_list(token_ids, hidden)
        result = thinker2talker(stage_list, [0])

        info = result[0]["additional_information"]
        assert info["thinker_hidden_states"].dtype == torch.float32


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

        # Mock parameters for _module_device — use side_effect so each call
        # returns a fresh iterator (return_value shares one iterator across calls).
        talker.parameters = MagicMock(
            side_effect=lambda: iter([torch.zeros(1)])  # CPU
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

    def test_prefill_exact_fit(self, preprocess_env):
        """Prefill with span_len == conditioning length: builds embeds, no trailing."""
        model, hidden_size, llm_dim = preprocess_env
        # 3 thinker tokens + 2 boundary = 5 conditioning positions
        # Request exactly 5 → all consumed, no trailing
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
        assert embeds.shape == (seq_len, hidden_size)
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
        """Decode (span_len=1) consumes trailing conditioning alone (no codec add)."""
        model, hidden_size, _ = preprocess_env

        trailing = torch.randn(2, hidden_size)
        input_ids = torch.zeros(1, dtype=torch.long)
        info_dict = {"trailing_conditioning": trailing}

        _, embeds, update = model.talker_preprocess(
            input_ids, None, **info_dict
        )

        assert embeds is not None
        assert embeds.shape == (1, hidden_size)
        # Conditioning used alone — should match first trailing position exactly.
        # Both tensors are float32 in this test; dtype is preserved through
        # the .to(dtype=codec_embeds.dtype) call in talker_preprocess.
        assert torch.allclose(embeds[0], trailing[0])
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


# ---------------------------------------------------------------------------
# talker2code2wav
# ---------------------------------------------------------------------------


class TestTalker2Code2Wav:
    """Tests for the talker→code2wav stage input processor."""

    def _make_talker_stage(self, token_ids):
        """Create a mock stage_list with talker outputs."""
        from unittest.mock import MagicMock

        output = MagicMock()
        output.token_ids = token_ids

        talker_output = MagicMock()
        talker_output.outputs = [output]

        stage = MagicMock()
        stage.engine_outputs = [talker_output]

        return [stage]

    def test_strips_trailing_eos(self):
        """Stop token (EOS) at end is stripped before passing to Code2Wav."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            talker2code2wav,
        )

        # Simulates vllm including stop_token_id 6561 in output
        token_ids = [100, 200, 300, 6561]
        stage_list = self._make_talker_stage(token_ids)
        result = talker2code2wav(stage_list, [0])

        assert len(result) == 1
        # Last token (6561) should be stripped
        assert result[0]["prompt_token_ids"] == [100, 200, 300]

    def test_no_stop_token_preserves_all(self):
        """Tokens without 6561 are all preserved (value-based filter, not [:-1] trim)."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            talker2code2wav,
        )

        token_ids = [10, 20, 30, 40, 50]
        stage_list = self._make_talker_stage(token_ids)
        result = talker2code2wav(stage_list, [0])

        # All 5 tokens preserved u2014 none is the stop token 6561
        assert result[0]["prompt_token_ids"] == [10, 20, 30, 40, 50]

    def test_empty_token_ids(self):
        """Empty token_ids produces empty codec_codes."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            talker2code2wav,
        )

        stage_list = self._make_talker_stage([])
        result = talker2code2wav(stage_list, [0])

        assert result[0]["prompt_token_ids"] == []

    def test_single_token_produces_empty(self):
        """Single token (just EOS) → empty codec after strip."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            talker2code2wav,
        )

        stage_list = self._make_talker_stage([6561])
        result = talker2code2wav(stage_list, [0])

        assert result[0]["prompt_token_ids"] == []


# ---------------------------------------------------------------------------
# thinker2talker_async_chunk
# ---------------------------------------------------------------------------


class TestThinker2TalkerAsyncChunk:
    """Smoke tests for the async_chunk thinker->talker accumulator."""

    def _make_transfer_manager(self):
        """Minimal mock for ChunkTransferAdapter."""
        from unittest.mock import MagicMock
        tm = MagicMock()
        tm.request_payload = {}
        return tm

    def _make_request(self, req_id="req0", prompt_ids=None, output_ids=None):
        """Minimal mock for OmniEngineCoreRequest."""
        from unittest.mock import MagicMock
        req = MagicMock()
        req.external_req_id = req_id
        req.prompt_token_ids = prompt_ids or []
        req.output_token_ids = output_ids or []
        return req

    def test_returns_none_while_accumulating(self):
        """Returns None until is_finished=True."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker_async_chunk,
        )
        tm = self._make_transfer_manager()
        req = self._make_request(output_ids=[1, 2, 3])
        pooling = {"thinker_hidden_states": torch.randn(3, 4096)}

        result = thinker2talker_async_chunk(tm, pooling, req, is_finished=False)
        assert result is None
        # Hidden states accumulated in transfer_manager
        assert "req0" in tm.request_payload

    def test_emits_payload_when_finished(self):
        """Returns dict with thinker_token_ids and thinker_hidden_states on finish."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker_async_chunk,
        )
        BOS, EOS = 151703, 151704
        tm = self._make_transfer_manager()
        req = self._make_request(
            prompt_ids=[1, 2, BOS],
            output_ids=[10, 20, EOS],
        )
        pooling = {"thinker_hidden_states": torch.randn(6, 4096)}

        result = thinker2talker_async_chunk(tm, pooling, req, is_finished=True)
        assert result is not None
        assert "thinker_token_ids" in result
        assert "thinker_hidden_states" in result
        assert isinstance(result["thinker_token_ids"], torch.Tensor)
        # TTS slice: tokens between BOS and EOS (index 3 to 5 in all_token_ids)
        assert result["thinker_token_ids"].shape[0] == result["thinker_hidden_states"].shape[0]

    def test_tts_boundary_beyond_hidden_returns_valid_payload(self):
        """When tts_start >= hidden_len, fallback path produces valid payload."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            thinker2talker_async_chunk,
        )
        BOS = 151703
        tm = self._make_transfer_manager()
        # BOS at index 7, but only 5 hidden states
        req = self._make_request(
            prompt_ids=[1, 2, 3, 4, 5, 6, 7, BOS],
            output_ids=[10],
        )
        pooling = {"thinker_hidden_states": torch.randn(5, 4096)}

        result = thinker2talker_async_chunk(tm, pooling, req, is_finished=True)
        # When is_finished=True and hidden states are available, the fallback path
        # always returns a dict (uses all available states, not None).
        assert result is not None, "is_finished=True with hidden states must return a payload dict"
        assert result["thinker_token_ids"].shape[0] == result["thinker_hidden_states"].shape[0]


# ---------------------------------------------------------------------------
# talker2code2wav_async_chunk
# ---------------------------------------------------------------------------


class TestTalker2Code2WavAsyncChunk:
    """Smoke tests for the async_chunk talker->code2wav token streamer."""

    def _make_transfer_manager(self):
        from collections import defaultdict
        from unittest.mock import MagicMock
        tm = MagicMock()
        tm.code_prompt_token_ids = defaultdict(list)
        tm.connector = None
        # Explicitly set real dicts so hasattr() guards in talker2code2wav_async_chunk
        # are not bypassed by MagicMock's auto-attribute creation.
        tm._talker_token_cursor = {}
        tm._talker_emitted_len = {}
        return tm

    def _make_request(self, req_id="req0", output_ids=None):
        from unittest.mock import MagicMock
        req = MagicMock()
        req.external_req_id = req_id
        req.output_token_ids = output_ids or []
        return req

    def test_returns_none_below_chunk_threshold(self):
        """No emission when pending tokens < chunk_size (default 25)."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            talker2code2wav_async_chunk,
        )
        tm = self._make_transfer_manager()
        req = self._make_request(output_ids=[1, 2, 3])

        result = talker2code2wav_async_chunk(tm, None, req, is_finished=False)
        assert result is None

    def test_emits_on_finish(self):
        """On is_finished=True, emits all accumulated tokens."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            talker2code2wav_async_chunk,
        )
        tm = self._make_transfer_manager()
        tokens = list(range(10, 20))
        req = self._make_request(output_ids=tokens)

        result = talker2code2wav_async_chunk(tm, None, req, is_finished=True)
        assert result is not None
        assert "code_predictor_codes" in result
        assert result["finished"].item() is True
        # All 10 tokens emitted (none is stop token 6561)
        assert result["code_predictor_codes"] == tokens

    def test_filters_stop_token_6561(self):
        """Stop token 6561 is filtered from codec codes."""
        from vllm_omni.model_executor.stage_input_processors.minicpm_o import (
            talker2code2wav_async_chunk,
        )
        tm = self._make_transfer_manager()
        tokens = [100, 200, 6561, 300]
        req = self._make_request(output_ids=tokens)

        result = talker2code2wav_async_chunk(tm, None, req, is_finished=True)
        assert result is not None
        assert 6561 not in result["code_predictor_codes"]
        assert result["code_predictor_codes"] == [100, 200, 300]
