# [Model][MiniCPM-o 4.5] Fix talker conditioning architecture to match original model

## Summary

Align the MiniCPM-o 4.5 3-stage pipeline (Thinker → Talker → Code2Wav) with the original openbmb/MiniCPM-o-4_5 `modeling_minicpmo.py` inference implementation.

The previous implementation had fundamental architecture mismatches in the Talker conditioning that produced audio output but with degraded quality. This PR fixes 12 issues discovered through systematic cross-referencing against the HF checkpoint (`config.json`, `tokenizer_config.json`, `model.safetensors.index.json`) and original source code.

## Changes

### Architecture Fixes (Talker Conditioning)

| Fix | Before | After | File |
|-----|--------|-------|------|
| **A1** | `emb_text` weights skipped (`skip_prefixes=["emb_text."]`) | `emb_text` loaded — primary text conditioning | `minicpm_o_talker.py` |
| **A2** | `projector_semantic` fed thinker input embeddings | `semantic_projection` fed thinker hidden states | `minicpm_o_talker.py`, `minicpm_o.py` |
| **A7** | `nn.SiLU()` activation | `nn.ReLU()` — matches original `MultiModalProjector` | `minicpm_o_talker.py` |
| **A8** | `projector_spk` used as hidden state projector | Loaded but unused (matches original: defined but never called) | `minicpm_o_talker.py` |
| **R4** | Audio projection MLP used `nn.SiLU()` | `nn.ReLU()` — same `MultiModalProjector` class | `minicpm_o_thinker.py` |

**Before (incorrect):**
```
thinker_text_embeds → projector_semantic(SiLU) → [768]
                                                   +
thinker_hidden_states → projector_spk(SiLU) → [768]
                                                   +
codec_embedding(input_ids) → [768]
```

**After (matches original):**
```
thinker_token_ids → emb_text(152064, 768) → [768]
                                               +
thinker_hidden_states → semantic_projection(ReLU) → L2_normalize → [768]

Then: cat([conditioning, text_eos_embed, audio_bos_embed])
→ Talker AR decode (EOS=6561)
```

### Pipeline Fixes

| Fix | Before | After | File |
|-----|--------|-------|------|
| **A3** | No L2 normalization | `F.normalize(p=2, dim=-1)` when `normalize_projected_hidden=True` | `minicpm_o_talker.py` |
| **A4** | No boundary tokens | Append `text_eos + audio_bos` via `emb_text` | `minicpm_o.py` |
| **A9** | EOS=6563 (wrong, out of vocab) or missing | EOS=6561 (`num_audio_tokens - 1`) | `pipeline.yaml`, `minicpm_o_ci.yaml` |
| **A10** | Entire sequence passed to talker | `_find_tts_bound()` filters `<|tts_bos|>`/`<|tts_eos|>` | `stage_input_processors/minicpm_o.py` |
| **A11** | First codec token skipped (`token_ids[1:]`) | No skip — audio_bos is in conditioning, not generated | `stage_input_processors/minicpm_o.py` |
| **tensor fix** | `token_ids` and `hidden_states` length mismatch (28 vs 27) | Align via `min()` with bounds clamping | `stage_input_processors/minicpm_o.py` |

### Config Fixes

| Fix | Before | After | File |
|-----|--------|-------|------|
| **A5** | 8 defaults wrong | All aligned with HF `config.json` | `configuration_minicpmo.py` |
| **A6** | Placeholder token IDs (151859/151860) | Correct IDs from `tokenizer_config.json` | `minicpm_o.py` |

### Cleanup

- Remove 7 debug `logger.warning("[DBG ...")` lines from `gpu_model_runner.py` and `gpu_ar_model_runner.py`
- Fix all CosyVoice3 → CosyVoice2 references
- Remove "Phase N" internal project references
- Update stale docstrings

## Test Plan

- [x] L1 config tests pass (defaults match HF config.json)
- [x] L1 component tests pass (ReLU activation, WeightsMapper prefix ordering)
- [x] E2E: RunPod A40 46GB — text + audio response confirmed
- [x] E2E: naia-os Tauri app connected to vllm-omni server, conversation working
- [ ] L1: Add `thinker2talker` / `_find_tts_bound` unit tests
- [ ] L2: Weight loading verification with real checkpoint

## Verification

Cross-referenced against:
- `openbmb/MiniCPM-o-4_5/config.json` — all config defaults
- `openbmb/MiniCPM-o-4_5/tokenizer_config.json` — special token IDs
- `openbmb/MiniCPM-o-4_5/model.safetensors.index.json` — all 1220 weight keys
- `openbmb/MiniCPM-o-4_5/modeling_minicpmo.py` — `_generate_speech_non_streaming()`, `MiniCPMTTS.__init__()`, `MultiModalProjector`, `create_projector()`

---

Generated with [Claude Code](https://claude.com/claude-code)
