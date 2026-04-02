# 우리가 수정한 것 — 현재 상태

## 브랜치: feat/minicpm-o (upstream/main 기반)

### 추가한 파일
| 파일 | 줄수 | 목적 |
|------|------|------|
| models/minicpm_o/__init__.py | 13 | 모듈 export |
| models/minicpm_o/configuration_minicpmo.py | 294 | Config 클래스 (커스텀, transformers에 없음) |
| models/minicpm_o/minicpm_o.py | ~550 | 통합 엔트리 + talker_preprocess |
| models/minicpm_o/minicpm_o_thinker.py | ~620 | Thinker (Idefics2 vision + Whisper audio + Qwen3 LLM) |
| models/minicpm_o/minicpm_o_talker.py | 304 | Talker (MiniCPMTTS Llama AR) |
| models/minicpm_o/minicpm_o_code2wav.py | ~260 | Code2Wav (CosyVoice2 + HiFi-GAN) |
| stage_input_processors/minicpm_o.py | ~190 | thinker2talker + talker2code2wav |
| stage_configs/minicpmo.yaml | 범용 2-GPU config |
| stage_configs/minicpmo_24gb.yaml | 24GB 단일 GPU config |

### 수정한 파일
| 파일 | 변경 |
|------|------|
| models/registry.py | MiniCPM-o 5개 엔트리 추가 |

### 신뢰도 상태

| 항목 | 상태 | 비고 |
|------|:----:|------|
| 모델 코드 (thinker/talker/code2wav) | ⚠️ | 24GB curl 테스트만 통과, 2-GPU 미검증 |
| stage_configs YAML | ❌ | 기본 사용법 미숙지 상태에서 작성, 반복 수정 |
| registry | ✅ | 단순 추가 |
| audio input (MiniCPMO processor) | ⚠️ | import만 변경, 2-GPU에서 미검증 |
| embed_multimodal | ⚠️ | vllm MiniCPMO4_5 패턴 따랐으나 profile_run OOM |
| Code2Wav CPU load + lazy GPU | ⚠️ | upstream에 없는 패턴 (hotfix) |

### 주요 교훈
1. **upstream 기본 사용법 먼저** — stage_configs, process:true, multi-GPU 동작을 모른 채 코드 작성
2. **24GB에서 E2E 했다고 "됐다"고 판단** — 실제 사용 환경 (2-GPU, Naia Shell)에서 안 됨
3. **NCCL_P2P_DISABLE=1** — RTX 3090 TP=2에 필수, upstream issue #308에서 2024년부터 알려짐
4. **process: true** — Qwen2.5-Omni 공식 config에 기본 포함, 우리 config에는 누락
