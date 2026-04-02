# 우리가 수정한 것 — 현재 상태

## 브랜치: feat/minicpm-o (upstream/main 기반)

### 추가한 파일
| 파일 (vllm_omni/model_executor/ 기준) | 목적 |
|------|------|
| models/minicpm_o/__init__.py | 모듈 export |
| models/minicpm_o/configuration_minicpmo.py | Config 클래스 (커스텀, transformers에 없음) |
| models/minicpm_o/minicpm_o.py | 통합 엔트리 + talker_preprocess |
| models/minicpm_o/minicpm_o_thinker.py | Thinker (Idefics2 vision + Whisper audio + Qwen3 LLM) |
| models/minicpm_o/minicpm_o_talker.py | Talker (MiniCPMTTS Llama AR) |
| models/minicpm_o/minicpm_o_code2wav.py | Code2Wav (CosyVoice2 + HiFi-GAN) |
| stage_input_processors/minicpm_o.py | thinker2talker + talker2code2wav |
| stage_configs/minicpmo.yaml | 2-GPU config (현재 미작동 — #1387) |
| stage_configs/minicpmo_24gb.yaml | 24GB 단일 GPU config (max_model_len 제한) |

### 수정한 파일
| 파일 | 변경 |
|------|------|
| models/registry.py | MiniCPM-o 6개 엔트리 추가 |

### 신뢰도 상태

| 항목 | 상태 | 비고 |
|------|:----:|------|
| 모델 코드 (thinker/talker/code2wav) | ✅ | 2x RTX 3090 E2E 통과 (텍스트+오디오 출력) |
| stage_configs YAML (minicpmo.yaml) | ✅ | 2x RTX 3090 검증 완료 |
| stage_configs YAML (minicpmo_24gb.yaml) | ⚠️ | 24GB 단일 GPU, curl 테스트만 |
| registry | ✅ | 단순 추가 |
| audio input (MiniCPMO processor) | ⚠️ | 2-GPU에서 미검증 (text→audio만 확인) |
| embed_multimodal | ✅ | max_num_batched_tokens 조정으로 profile_run 통과 |
| Code2Wav CPU load + lazy GPU | ✅ | 2-GPU에서 정상 작동 확인 |
| 디바이스 격리 (CUDA_VISIBLE_DEVICES) | ✅ | 진단 스크립트 + 실제 서버 모두 정상 |

### 주요 교훈
1. **upstream 기본 사용법 먼저** — stage_configs, process:true, multi-GPU 동작을 모른 채 코드 작성
2. **24GB에서 E2E 했다고 "됐다"고 판단** — 실제 사용 환경 (2-GPU, Naia Shell)에서 안 됨
3. **NCCL_P2P_DISABLE=1** — RTX 3090 TP=2에 필수 (vllm-project/vllm#308, NVLink 없는 GPU 간 P2P 통신 비활성)
4. **process: true** — Qwen2.5-Omni 공식 config에 기본 포함, 우리 config에는 누락
5. **디바이스 격리는 정상** — 이전 "업스트림 버그" 판정은 잘못됨. 실제 원인은 gpu_memory_utilization/max_num_batched_tokens 설정
6. **CosyVoice2 flow model ~15GB** — Code2Wav가 예상보다 훨씬 큰 GPU 메모리 필요
7. **2x RTX 3090 메모리 분배**: Thinker(0.88) + Talker(0.1) + Code2Wav(0.02). Talker KV를 최소화해야 Code2Wav가 들어감
8. **max_num_batched_tokens** — vision profile 크기에 직결. 24GB에서 2048이 한계
