# Memory — vllm-omni

## 2026-04-14
- **Issue #8 (review(minicpm-o))**: Completed Pass 8 & 9 (Comprehensive Recheck).
- **Key fixes**:
  - `serving_chat.py`: empty `audio_data` guard (IndexError).
  - `omni_connection.py`: sliding window (max 20 turns) + spec compliance.
  - `minicpm_o.py`: `left_context_size` propagation to Code2Wav stage.
  - `audio_utils_mixin.py`: `PCM_16` WAV subtype (fixes double-speed playback).
  - `async_omni_engine.py`: stage handshake lock (fixes port race).
- **Lessons**:
  - Code2Wav stage must receive `left_context_size` from `runtime_additional_information`.
  - OpenAI Realtime clients often expect `PCM_16` WAV; `float32` causes double-speed artifacts.
  - Concurrent stage launches in `AsyncOmniEngine` need locking around port binding.
