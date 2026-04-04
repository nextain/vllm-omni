# nextain/vllm-omni — MiniCPM-o 4.5 기여 Fork

**Language / 언어**: [English](README.md) | 한국어

> **[vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) Fork**  
> Upstream README: [README.upstream.md](README.upstream.md) · Upstream PR 대상: [vllm-omni#1182](https://github.com/vllm-project/vllm-omni/issues/1182)

---

## 이 Fork가 추가하는 것

이 Fork는 **MiniCPM-o 4.5** omni 모델 지원을 vllm-omni에 추가하고 upstream에 기여합니다.

**하드웨어 타깃**: 2× RTX 3090 (총 48 GB, NVLink 없음) — [Naia OS](https://github.com/nextain/naia-os)의 로컬 AI 추론을 위한 소비자급 멀티 GPU 환경.

**방법론**: 업스트림 패턴 분석, 구현, 헤드리스 서브에이전트를 이용한 적대적 코드 리뷰까지 전 과정을 AI 에이전트가 수행했습니다. **AI-native 오픈소스 기여** 방법론 실험입니다.

---

## 아키텍처

MiniCPM-o 4.5는 완전 분리된 3단계 파이프라인을 사용합니다:

```
입력 (텍스트 + 이미지 / 오디오)
  → Stage 0: Thinker  (GPU 0) — Idefics2 비전 + Whisper 오디오 + Qwen3 LLM 백본
  → Stage 1: Talker   (GPU 1) — MiniCPMTTS Llama AR 코덱 생성기
  → Stage 2: Code2Wav (GPU 1) — CosyVoice2 flow 모델 + HiFi-GAN 보코더
출력: PCM 오디오 스트림
```

Qwen3-Omni와의 주요 차이: 단일 RVQ 레이어 (num_vq=1), 인라인 TTS 토큰 경계 감지 필요, 1D 코덱 토큰 시퀀스.

---

## 빠른 시작

### 오프라인 추론

```bash
cd examples/offline_inference/minicpm_o

# 단일 프롬프트 (24 GB 단일 GPU)
bash run_single_prompt.sh

# 단일 프롬프트 async_chunk 스트리밍 (2× RTX 3090)
bash run_single_prompt_async_chunk.sh

# 다중 프롬프트
bash run_multiple_prompts.sh
```

### 온라인 서빙

```bash
# 2× RTX 3090 — async_chunk 스트리밍 (권장)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000

# 단일 24 GB GPU
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --trust-remote-code --port 8000
```

> RTX 3090 (NVLink 없음) 환경에서는 `NCCL_P2P_DISABLE=1`이 필수입니다.

클라이언트 스크립트 및 평가: [`examples/online_serving/minicpm_o/`](examples/online_serving/minicpm_o/)  
오프라인 추론: [`examples/offline_inference/minicpm_o/`](examples/offline_inference/minicpm_o/)

---

## Stage Config 옵션

| Config | 사용 환경 | GPU |
|--------|----------|-----|
| `minicpmo.yaml` | 단일 24 GB GPU | RTX 3090 × 1 |
| `minicpmo_48gb_2gpu.yaml` | 2× 24 GB GPU, 동기 모드 | RTX 3090 × 2 |
| `minicpmo_async_chunk.yaml` | 2× 24 GB GPU, 스트리밍 (TTFP ~0.07s) | RTX 3090 × 2 |

---

## 벤치마크 결과

**2× RTX 3090** (48 GB, NVLink 없음), `minicpmo_async_chunk.yaml` 기준.

### VoiceBench (영어, 텍스트 전용)

| 카테고리 | Pass Rate |
|---------|----------:|
| 지식 (20) | 100.0% |
| 지시 (20) | 95.0% |
| 견고성 (20) | 100.0% |
| 안전 (15) | 100.0% |
| **종합 (75)** | **98.7%** |

### 음성 대화 품질

| 언어 | 정확도 | 지연 | 상태 |
|------|:------|:-----|:----:|
| 영어 | 단어 정확도 92% | 평균 2.3s | ✅ 프로덕션 준비 완료 |
| 중국어 | CER 76.1% / 의미 유사도 95% | 평균 1.5s | ✅ 정상 작동 |
| 한국어 | 텍스트 OK / TTS 왜곡 | — | ⚠️ CosyVoice2 한국어 미지원 |

**TTFP** (첫 오디오 패킷 도달 시간, async_chunk): **~0.07s**

전체 벤치마크 보고서: [`examples/online_serving/minicpm_o/BENCHMARK.ko.md`](examples/online_serving/minicpm_o/BENCHMARK.ko.md)

---

## 추가된 파일

| 경로 | 목적 |
|-----|------|
| `vllm_omni/model_executor/models/minicpm_o/` | 모델 코드 (Thinker / Talker / Code2Wav + config) |
| `vllm_omni/model_executor/stage_input_processors/minicpm_o.py` | 스테이지 간 데이터 전송 (sync + async_chunk) |
| `vllm_omni/model_executor/stage_configs/minicpmo*.yaml` | stage config 3종 (단일 GPU / 2-GPU / async_chunk) |
| `examples/offline_inference/minicpm_o/` | 오프라인 추론 스크립트 |
| `examples/online_serving/minicpm_o/` | 온라인 서빙 스크립트 + 벤치마크 모음 |

수정: `models/registry.py` (모델 엔트리 6개), `pyproject.toml` (선택적 의존성 추가).

---

## 알려진 한계

| 문제 | 영향 | 완화책 |
|------|------|-------|
| Stop token 6561 미생성 | 오디오 ~20s 고정 | `_trim_silence()` 후처리 |
| 한국어 TTS 실패 | 오디오 왜곡 | CosyVoice2 한국어 미학습 |

---

## 구현 참고사항

- `NCCL_P2P_DISABLE=1` — RTX 3090 필수 (NVLink 없음)
- `max_inflight: 1` — 동시 스테이지 메모리 OOM 방지 (upstream [#1387](https://github.com/vllm-project/vllm-omni/issues/1387))
- `_find_tts_bound()` — MiniCPM-o는 Thinker 출력에 TTS 토큰을 인라인 삽입; 경계 감지 필요
- `_ensure_list()` — vLLM `ConstantList` 타입은 `list()`로 올바르게 이터레이션되지 않음

전체 구현 히스토리: [`.agents/context/contribution-journey.md`](.agents/context/contribution-journey.md)

---

## 라이선스

Apache License 2.0 — upstream vllm-omni와 동일. [LICENSE](./LICENSE) 참조.
