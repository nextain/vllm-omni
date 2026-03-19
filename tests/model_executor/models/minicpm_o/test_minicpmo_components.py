# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 component tests for MiniCPM-o 4.5 model building blocks.

All tests run on CPU with random weights.  No GPU required.

Covers:
- MiniCPMOResampler: 2D perceiver cross-attention (shape correctness)
- MiniCPMOTalkerResizeMLP: thinker→talker dimension projection
- _build_minicpmo_cosyvoice_config: CosyVoice3Config for MiniCPM-o
- WeightsMapper prefix ordering (thinker + talker)
- Registry entries for all four MiniCPM-o architectures
"""

import pytest
import torch

pytestmark = [pytest.mark.cpu]


# ---------------------------------------------------------------------------
# MiniCPMOResampler
# ---------------------------------------------------------------------------


class TestMiniCPMOResampler:
    """Tests for the 2D perceiver-resampler used in the vision encoder."""

    @pytest.fixture
    def resampler(self):
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_thinker import MiniCPMOResampler

        return MiniCPMOResampler(
            num_queries=8,   # small for CPU speed
            embed_dim=64,
            num_heads=4,
            kv_dim=32,
            max_size=(16, 16),
        )

    def test_output_shape(self, resampler):
        """Output should be [batch, num_queries, embed_dim]."""
        batch = 2
        patch_h, patch_w = 4, 4
        num_patches = patch_h * patch_w

        x = torch.randn(batch, num_patches, 32)  # kv_dim=32
        tgt_sizes = torch.tensor([[patch_h, patch_w]] * batch)

        out = resampler(x, tgt_sizes)

        assert out.shape == (batch, 8, 64)  # (B, num_queries, embed_dim)

    def test_pos_cache_expands_for_larger_images(self, resampler):
        """Resampler should automatically expand pos_cache for larger images."""
        batch = 1
        patch_h, patch_w = 20, 20  # > max_size=(16,16)
        x = torch.randn(batch, patch_h * patch_w, 32)
        tgt_sizes = torch.tensor([[patch_h, patch_w]])

        out = resampler(x, tgt_sizes)

        assert out.shape == (batch, 8, 64)
        assert resampler.max_size[0] >= 20
        assert resampler.max_size[1] >= 20

    def test_identity_kv_proj_when_dims_equal(self):
        """When kv_dim == embed_dim, kv_proj should be nn.Identity."""
        import torch.nn as nn
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_thinker import MiniCPMOResampler

        r = MiniCPMOResampler(num_queries=4, embed_dim=32, num_heads=2, kv_dim=32)
        assert isinstance(r.kv_proj, nn.Identity)

    def test_kv_proj_is_linear_when_dims_differ(self, resampler):
        """kv_dim != embed_dim → kv_proj should be nn.Linear."""
        import torch.nn as nn

        assert isinstance(resampler.kv_proj, nn.Linear)
        assert resampler.kv_proj.in_features == 32    # kv_dim
        assert resampler.kv_proj.out_features == 64   # embed_dim


# ---------------------------------------------------------------------------
# MiniCPMOTalkerResizeMLP
# ---------------------------------------------------------------------------


class TestMiniCPMOTalkerResizeMLP:
    """Tests for the 2-layer MLP that projects thinker → talker dimension."""

    @pytest.fixture
    def mlp(self):
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_talker import MiniCPMOTalkerResizeMLP

        return MiniCPMOTalkerResizeMLP(
            llm_dim=16,
            intermediate_size=32,
            hidden_size=8,
        )

    def test_output_shape(self, mlp):
        x = torch.randn(5, 16)  # [seq, llm_dim]
        out = mlp(x)
        assert out.shape == (5, 8)  # [seq, hidden_size]

    def test_batched_output_shape(self, mlp):
        x = torch.randn(3, 7, 16)  # [batch, seq, llm_dim]
        out = mlp(x)
        assert out.shape == (3, 7, 8)

    def test_activation_is_silu(self, mlp):
        import torch.nn as nn

        assert isinstance(mlp.act_fn, nn.SiLU)

    def test_linear_dimensions(self, mlp):
        assert mlp.linear_fc1.in_features == 16
        assert mlp.linear_fc1.out_features == 32
        assert mlp.linear_fc2.in_features == 32
        assert mlp.linear_fc2.out_features == 8


# ---------------------------------------------------------------------------
# _build_minicpmo_cosyvoice_config
# ---------------------------------------------------------------------------


class TestBuildMiniCPMOCosyVoiceConfig:
    """Tests for the CosyVoice3Config factory used in Code2Wav."""

    @pytest.fixture
    def cfg(self):
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_code2wav import (
            _build_minicpmo_cosyvoice_config,
        )

        return _build_minicpmo_cosyvoice_config()

    def test_flow_vocab_size(self, cfg):
        """s3tokenizer has vocab_size≈6562, one more than CosyVoice3's 6561."""
        assert cfg.flow["vocab_size"] == 6562

    def test_llm_speech_token_size(self, cfg):
        assert cfg.llm["speech_token_size"] == 6562

    def test_llm_eos_token_id(self, cfg):
        assert cfg.llm["eos_token_id"] == 6563

    def test_sample_rate(self, cfg):
        """CosyVoice default sample_rate=24000."""
        assert cfg.sample_rate == 24000

    def test_token_frame_rate(self, cfg):
        """25 codec frames per second → 960 samples/token at 24kHz."""
        assert cfg.token_frame_rate == 25

    def test_token_mel_ratio(self, cfg):
        assert cfg.token_mel_ratio == 2


# ---------------------------------------------------------------------------
# WeightsMapper prefix ordering
# ---------------------------------------------------------------------------


class TestThinkerWeightsMapper:
    """Verify WeightsMapper prefix dict has no shadowing issues.

    The mapper must translate HF key prefixes correctly.  More-specific
    prefixes (e.g. ``llm.lm_head.``) must appear before less-specific ones
    (e.g. ``llm.``).  Python dicts preserve insertion order (3.7+), so we
    check the iteration order matches expected specificity.
    """

    @pytest.fixture
    def mapper(self):
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_thinker import (
            MiniCPMOThinkerForConditionalGeneration,
        )

        return MiniCPMOThinkerForConditionalGeneration.hf_to_vllm_mapper

    def test_llm_lm_head_before_llm_model(self, mapper):
        """llm.lm_head. must appear before llm.model. in the mapping."""
        keys = list(mapper.orig_to_new_prefix.keys())
        assert "llm.lm_head." in keys
        assert "llm.model." in keys
        assert keys.index("llm.lm_head.") < keys.index("llm.model.")

    def test_llm_model_before_llm(self, mapper):
        """llm.model. must appear before the catch-all llm. prefix."""
        keys = list(mapper.orig_to_new_prefix.keys())
        assert keys.index("llm.model.") < keys.index("llm.")

    def test_required_prefixes_present(self, mapper):
        keys = list(mapper.orig_to_new_prefix.keys())
        for prefix in ("vpm.", "resampler.", "apm.", "audio_projection_layer."):
            assert prefix in keys, f"Missing prefix: {prefix!r}"

    def test_vpm_maps_to_visual_encoder(self, mapper):
        assert mapper.orig_to_new_prefix["vpm."] == "visual.encoder."

    def test_resampler_maps_to_visual_resampler(self, mapper):
        assert mapper.orig_to_new_prefix["resampler."] == "visual.resampler."

    def test_apm_maps_to_audio_encoder(self, mapper):
        assert mapper.orig_to_new_prefix["apm."] == "audio_encoder.encoder."


class TestTalkerWeightsMapper:
    """Verify the Talker WeightsMapper prefix ordering."""

    @pytest.fixture
    def mapper(self):
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_talker import (
            MiniCPMOTalkerForConditionalGeneration,
        )

        return MiniCPMOTalkerForConditionalGeneration.hf_to_vllm_mapper

    def test_required_prefixes_present(self, mapper):
        keys = list(mapper.orig_to_new_prefix.keys())
        for prefix in (
            "tts.model.",
            "tts.emb_code.0.",
            "tts.head_code.0.",
            "tts.projector_semantic.",
            "tts.projector_spk.",
            "tts.",
        ):
            assert prefix in keys, f"Missing prefix: {prefix!r}"

    def test_specific_before_catchall(self, mapper):
        """All specific prefixes must precede the catch-all tts."""
        keys = list(mapper.orig_to_new_prefix.keys())
        catchall_idx = keys.index("tts.")
        for specific in ("tts.model.", "tts.emb_code.0.", "tts.head_code.0."):
            assert keys.index(specific) < catchall_idx, (
                f"{specific!r} must appear before 'tts.' (catch-all)"
            )

    def test_tts_emb_code_maps_to_codec_embedding(self, mapper):
        """tts.emb_code.0. must map to codec_embedding (CosyVoice3 codec vocab)."""
        assert mapper.orig_to_new_prefix["tts.emb_code.0."] == "codec_embedding."

    def test_tts_head_code_maps_to_codec_head(self, mapper):
        assert mapper.orig_to_new_prefix["tts.head_code.0."] == "codec_head."

    def test_tts_projector_semantic_maps_to_text_projection(self, mapper):
        assert mapper.orig_to_new_prefix["tts.projector_semantic."] == "text_projection."

    def test_tts_projector_spk_maps_to_hidden_projection(self, mapper):
        assert mapper.orig_to_new_prefix["tts.projector_spk."] == "hidden_projection."


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestMiniCPMORegistry:
    """Verify all four MiniCPM-o architectures are registered in the registry."""

    @pytest.fixture
    def omni_models(self):
        from vllm_omni.model_executor.models.registry import _OMNI_MODELS

        return _OMNI_MODELS

    def test_main_entry_registered(self, omni_models):
        assert "MiniCPMOForConditionalGeneration" in omni_models

    def test_thinker_registered(self, omni_models):
        assert "MiniCPMOThinkerModel" in omni_models

    def test_talker_registered(self, omni_models):
        assert "MiniCPMOTalkerModel" in omni_models

    def test_code2wav_registered(self, omni_models):
        assert "MiniCPMOCode2Wav" in omni_models

    def test_main_entry_module_path(self, omni_models):
        mod_folder, mod_relname, cls_name = omni_models["MiniCPMOForConditionalGeneration"]
        assert mod_folder == "minicpm_o"
        assert mod_relname == "minicpm_o"
        assert cls_name == "MiniCPMOForConditionalGeneration"

    def test_code2wav_module_path(self, omni_models):
        mod_folder, mod_relname, cls_name = omni_models["MiniCPMOCode2Wav"]
        assert mod_folder == "minicpm_o"
        assert mod_relname == "minicpm_o_code2wav"
        assert cls_name == "MiniCPMOCode2Wav"

    def test_thinker_module_path(self, omni_models):
        mod_folder, mod_relname, cls_name = omni_models["MiniCPMOThinkerModel"]
        assert cls_name == "MiniCPMOThinkerForConditionalGeneration"

    def test_talker_module_path(self, omni_models):
        mod_folder, mod_relname, cls_name = omni_models["MiniCPMOTalkerModel"]
        assert cls_name == "MiniCPMOTalkerForConditionalGeneration"


# ---------------------------------------------------------------------------
# MiniCPMOCode2Wav instantiation (no vllm_config needed)
# ---------------------------------------------------------------------------


class TestMiniCPMOCode2WavInstantiation:
    """Verify Code2Wav can be instantiated without a VllmConfig."""

    def test_instantiate_without_vllm_config(self):
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_code2wav import MiniCPMOCode2Wav

        model = MiniCPMOCode2Wav(vllm_config=None)
        assert model._model_dir is None

    def test_has_flow_model(self):
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_code2wav import MiniCPMOCode2Wav

        model = MiniCPMOCode2Wav(vllm_config=None)
        assert hasattr(model, "flow_model")

    def test_total_upsample_calculation(self):
        """total_upsample = sample_rate / input_frame_rate = 24000/25 = 960."""
        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_code2wav import MiniCPMOCode2Wav

        model = MiniCPMOCode2Wav(vllm_config=None)
        expected = int(
            model.flow_model.config.sample_rate / model.flow_model.input_frame_rate
        )
        assert expected == 960


# ---------------------------------------------------------------------------
# Phase 6: Thinker forward returns (hidden_states, inputs_embeds) tuple
# ---------------------------------------------------------------------------


class TestThinkerForwardTuple:
    """Verify that the thinker forward returns a 2-tuple for talker conditioning."""

    @pytest.fixture
    def thinker(self):
        from unittest.mock import MagicMock, patch
        import torch.nn as nn

        from vllm_omni.model_executor.models.minicpm_o.minicpm_o_thinker import (
            MiniCPMOThinkerForConditionalGeneration,
        )
        from vllm_omni.model_executor.models.minicpm_o.configuration_minicpmo import MiniCPMOConfig

        # Build a tiny thinker with a stubbed language_model
        cfg = MiniCPMOConfig()
        vllm_config = MagicMock()
        vllm_config.model_config.hf_config = cfg
        vllm_config.model_config.dtype = torch.float32
        vllm_config.quant_config = None

        thinker = MiniCPMOThinkerForConditionalGeneration.__new__(
            MiniCPMOThinkerForConditionalGeneration
        )
        # Stub embed_input_ids and language_model
        seq = 4
        thinker_hidden = 8
        thinker.language_model = MagicMock(
            return_value=torch.randn(seq, thinker_hidden)
        )
        thinker.embed_input_ids = MagicMock(
            return_value=torch.randn(seq, thinker_hidden)
        )
        return thinker, seq, thinker_hidden

    def test_forward_returns_tuple(self, thinker):
        """forward() must return a 2-tuple (hidden_states, inputs_embeds)."""
        thinker_obj, seq, hidden = thinker
        input_ids = torch.zeros(seq, dtype=torch.long)
        positions = torch.arange(seq)

        out = thinker_obj.forward(input_ids=input_ids, positions=positions)

        assert isinstance(out, tuple), "forward() should return a tuple"
        assert len(out) == 2, "tuple must have exactly 2 elements"

    def test_forward_tuple_shapes(self, thinker):
        """Both tuple elements must have shape [seq, hidden_dim]."""
        thinker_obj, seq, hidden = thinker
        input_ids = torch.zeros(seq, dtype=torch.long)
        positions = torch.arange(seq)

        hidden_states, inputs_embeds = thinker_obj.forward(
            input_ids=input_ids, positions=positions
        )

        assert hidden_states.shape == (seq, hidden)
        assert inputs_embeds.shape == (seq, hidden)

    def test_forward_builds_embeds_when_none(self, thinker):
        """When inputs_embeds=None, thinker builds and returns them."""
        thinker_obj, seq, hidden = thinker
        input_ids = torch.zeros(seq, dtype=torch.long)
        positions = torch.arange(seq)

        _, returned_embeds = thinker_obj.forward(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=None,
        )

        # embed_input_ids should have been called once
        thinker_obj.embed_input_ids.assert_called_once()
        assert returned_embeds is not None

    def test_forward_passes_provided_embeds(self, thinker):
        """When inputs_embeds is provided, it is passed through and returned."""
        thinker_obj, seq, hidden = thinker
        input_ids = torch.zeros(seq, dtype=torch.long)
        positions = torch.arange(seq)
        provided_embeds = torch.randn(seq, hidden)

        _, returned_embeds = thinker_obj.forward(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=provided_embeds,
        )

        # embed_input_ids should NOT have been called (embeds already provided)
        thinker_obj.embed_input_ids.assert_not_called()
        assert returned_embeds is provided_embeds
