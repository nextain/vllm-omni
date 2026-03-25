# [Model][MiniCPM-o 4.5] Talker conditioning 아키텍처를 원본 모델에 맞게 수정

## 요약

MiniCPM-o 4.5 3-stage 파이프라인 (Thinker → Talker → Code2Wav)을 원본 openbmb/MiniCPM-o-4_5 `modeling_minicpmo.py`의 추론 구현에 맞게 정렬합니다.

이전 구현은 Talker conditioning에서 근본적인 아키텍처 불일치가 있어 오디오는 출력되지만 품질이 저하되었습니다. 이 PR은 HF 체크포인트 (`config.json`, `tokenizer_config.json`, `model.safetensors.index.json`)와 원본 소스 코드를 체계적으로 대조하여 발견한 12개 이슈를 수정합니다.

## 변경 사항

### 아키텍처 수정 (Talker Conditioning)

| 수정 | 이전 | 이후 | 파일 |
|------|------|------|------|
| **A1** | `emb_text` 가중치 스킵 | `emb_text` 로드 — 기본 텍스트 conditioning | `minicpm_o_talker.py` |
| **A2** | `projector_semantic`에 입력 임베딩 전달 | `semantic_projection`에 hidden states 전달 | `minicpm_o_talker.py`, `minicpm_o.py` |
| **A7** | `nn.SiLU()` 활성함수 | `nn.ReLU()` — 원본 `MultiModalProjector`와 동일 | `minicpm_o_talker.py` |
| **A8** | `projector_spk`를 hidden state projector로 사용 | 로드만 하고 미사용 (원본: 정의만 있고 호출 안 됨) | `minicpm_o_talker.py` |
| **R4** | 오디오 projection MLP도 `nn.SiLU()` | `nn.ReLU()` — 동일 `MultiModalProjector` 클래스 | `minicpm_o_thinker.py` |

**수정 전 (잘못된 구조):**
```
thinker 입력 임베딩 → projector_semantic(SiLU) → [768]
                                                   +
thinker hidden states → projector_spk(SiLU) → [768]
                                                   +
codec_embedding(input_ids) → [768]
```

**수정 후 (원본과 동일):**
```
thinker 토큰 ID → emb_text(152064, 768) → [768]
                                             +
thinker hidden states → semantic_projection(ReLU) → L2 정규화 → [768]

이후: cat([conditioning, text_eos_embed, audio_bos_embed])
→ Talker AR 디코딩 (EOS=6561)
```

### 파이프라인 수정

| 수정 | 이전 | 이후 | 파일 |
|------|------|------|------|
| **A3** | L2 정규화 없음 | `F.normalize(p=2, dim=-1)` (`normalize_projected_hidden=True`일 때) | `minicpm_o_talker.py` |
| **A4** | 경계 토큰 없음 | `text_eos + audio_bos`를 `emb_text`로 추가 | `minicpm_o.py` |
| **A9** | EOS=6563 (잘못됨, vocab 범위 밖) 또는 미설정 | EOS=6561 (`num_audio_tokens - 1`) | `pipeline.yaml`, `minicpm_o_ci.yaml` |
| **A10** | 전체 시퀀스를 talker에 전달 | `_find_tts_bound()`로 `<\|tts_bos\|>`/`<\|tts_eos\|>` 필터링 | `stage_input_processors/minicpm_o.py` |
| **A11** | 첫 codec 토큰 스킵 (`token_ids[1:]`) | 스킵 안 함 — audio_bos는 conditioning에 있음 | `stage_input_processors/minicpm_o.py` |
| **텐서 수정** | `token_ids`와 `hidden_states` 길이 불일치 (28 vs 27) | `min()`으로 정렬 + bounds clamping | `stage_input_processors/minicpm_o.py` |

### Config 수정

| 수정 | 이전 | 이후 | 파일 |
|------|------|------|------|
| **A5** | 8개 기본값 잘못됨 | 전부 HF `config.json`에 맞춤 | `configuration_minicpmo.py` |
| **A6** | placeholder 토큰 ID (151859/151860) | `tokenizer_config.json`에서 확인한 정확한 ID | `minicpm_o.py` |

### 정리

- `gpu_model_runner.py`, `gpu_ar_model_runner.py`에서 디버그 로그 7곳 제거
- CosyVoice3 → CosyVoice2 참조 전부 수정
- "Phase N" 내부 프로젝트 참조 제거
- 오래된 docstring 업데이트

## 테스트

- [x] L1 config 테스트 통과 (기본값 HF config.json 일치)
- [x] L1 컴포넌트 테스트 통과 (ReLU 활성함수, WeightsMapper prefix 순서)
- [x] E2E: RunPod A40 46GB — 텍스트 + 오디오 응답 확인
- [x] E2E: naia-os Tauri 앱에서 vllm-omni 서버 연결, 대화 동작
- [ ] L1: `thinker2talker` / `_find_tts_bound` 단위 테스트 추가
- [ ] L2: 실제 체크포인트로 가중치 로딩 검증

## 검증 방법

다음 원본 파일들과 대조 검증:
- `openbmb/MiniCPM-o-4_5/config.json` — 모든 config 기본값
- `openbmb/MiniCPM-o-4_5/tokenizer_config.json` — 특수 토큰 ID
- `openbmb/MiniCPM-o-4_5/model.safetensors.index.json` — 1220개 전체 가중치 키
- `openbmb/MiniCPM-o-4_5/modeling_minicpmo.py` — `_generate_speech_non_streaming()`, `MiniCPMTTS.__init__()`, `MultiModalProjector`, `create_projector()`
