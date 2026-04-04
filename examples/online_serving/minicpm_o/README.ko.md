# MiniCPM-o 4.5 — 온라인 서빙 예제

**Language / 언어**: [English](README.md) | 한국어

---

## 한국어

이 디렉토리는 vllm-omni의 OpenAI 호환 API로 **MiniCPM-o 4.5**를 서빙하는 스크립트와 벤치마크를 담고 있습니다.
MiniCPM-o 4.5는 3단계 omni 모델입니다:
**Thinker** (멀티모달 LLM) → **Talker** (TTS 코덱 생성기) → **Code2Wav** (CosyVoice2 보코더).

### 빠른 시작

```bash
# 1. 서버 시작 (2× RTX 3090)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path ../../vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --trust-remote-code --host 0.0.0.0 --port 8000

# 2. 대화 벤치마크 실행
cd examples/online_serving/minicpm_o/
python conversation_benchmark.py --omni

# 3. VoiceBench 평가 실행
python voicebench_runner.py
```

### 스크립트 목록

| 스크립트 | 목적 |
|--------|------|
| `conversation_benchmark.py` | 6개 영어 시나리오 멀티턴 대화 벤치마크 |
| `language_test.py` | EN / ZH / KO 비교 (CER + 의미 유사도) |
| `voicebench_runner.py` | VoiceBench (지식 / 지시 / 견고성 / 안전) |
| `e2e_conversation_test.py` | 핵심 Speaker / Monitor 프레임워크 (위 스크립트에서 import) |
| `metrics/cjk_metrics.py` | CER, 의미 유사도 (sentence-transformers) |
| `metrics/conversation_quality.py` | 관련성, 일관성, 지식 유지 |

### Stage Config 옵션

| Config | 사용 환경 | GPU |
|--------|----------|-----|
| `minicpmo.yaml` | 단일 24 GB GPU | RTX 3090 × 1 |
| `minicpmo_48gb_2gpu.yaml` | 2× 24 GB GPU, 동기 모드 | RTX 3090 × 2 |
| `minicpmo_async_chunk.yaml` | 2× 24 GB GPU, 스트리밍 (TTFP ~0.07s) | RTX 3090 × 2 |

### 벤치마크 결과

전체 결과는 [BENCHMARK.ko.md](BENCHMARK.ko.md) 참조.

**요약 (2× RTX 3090, async_chunk 모드):**

| 언어 | STT 정확도 | 지연 | 상태 |
|------|:----------|:-----|:----:|
| 영어 | 92% (단어) | 평균 2.3s | ✅ 프로덕션 준비 완료 |
| 중국어 | CER 76.1% / 의미 95% | 평균 1.5s | ✅ 정상 작동 |
| 한국어 | 텍스트 OK / TTS 실패 | — | ⚠️ TTS 미지원 |

**VoiceBench (영어, 텍스트 전용):** 전체 98.0% · pass rate 98.7%

### 알려진 한계

- **Stop token 6561** Talker가 거의 생성하지 않음 → 오디오 ~20s 고정
  (별도 이슈로 추적; 완화: `_trim_silence()` 후처리)
- **한국어 TTS**: CosyVoice2가 한국어 미학습 → 오디오 왜곡
- **async_chunk TTFP**: 첫 오디오 지연 ~0.07s. 참고: 벤치마크 클라이언트의
  `modality=="audio"` 감지가 스트리밍 청크를 놓칠 수 있음 — `e2e_conversation_test.py` 참조.
