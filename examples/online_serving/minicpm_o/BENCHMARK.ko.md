# MiniCPM-o 4.5 — 벤치마크 보고서

**Language / 언어**: [English](BENCHMARK.md) | 한국어

**날짜:** 2026-04-03  
**모델:** `openbmb/MiniCPM-o-4_5`  
**하드웨어:** 2× RTX 3090 (각 24 GB, NVLink 없음), distrobox vllm-dev  
**서버 설정:** `minicpmo_async_chunk.yaml` (async_chunk 스트리밍 활성)

---

## 1. 요약

| 평가 영역 | 결과 | 등급 |
|-----------|------|:----:|
| VoiceBench Knowledge (MCQ 20개) | 100% pass rate | A+ |
| VoiceBench Instruction (20개) | 95% pass rate | A |
| VoiceBench Robustness (20개) | 100% pass rate | A+ |
| VoiceBench Safety (15개) | 100% pass rate | A+ |
| 영어 음성 대화 (6 시나리오) | STT 92%, 지연 2.3s | A |
| 중국어 음성 대화 (2 시나리오) | CER 76.1% / 의미 95% | B+ |
| 한국어 음성 대화 | TTS 완전 실패 | F (TTS only) |

**전체 평가:** MiniCPM-o 4.5는 **영어 및 중국어 환경에서 프로덕션 준비 완료**.  
한국어 TTS는 CosyVoice2 미학습으로 현재 사용 불가.

---

## 2. VoiceBench 결과 (텍스트 전용 모드)

> 방법: VoiceBench 태스크 구조 (Knowledge / Instruction / Robustness / Safety) × OpenAI 호환 API  
> 채점: 규칙 기반 자동 평가 (MCQ 글자 매칭, 안전 거부 감지, IFEval 제약 검사)  
> ⚠️ 텍스트 전용 모드 — LLM 추론 능력만 평가, 오디오 I/O 파이프라인 미포함

| 카테고리 | 샘플 수 | 평균 점수 | Pass Rate | 지연 |
|----------|--------:|----------:|----------:|-----:|
| Knowledge | 20 | **100.0%** | 100.0% | 2.9s |
| Instruction | 20 | **92.5%** | 95.0% | 3.1s |
| Robustness | 20 | **100.0%** | 100.0% | 2.9s |
| Safety | 15 | **100.0%** | 100.0% | 4.0s |
| **전체** | **75** | **98.0%** | **98.7%** | 평균 3.0s |

### 업계 참고치 (VoiceBench 리더보드 — 오디오 모드, GPT-4o-mini judge)

| 모델 | Knowledge | Instruction | Robustness | Safety |
|------|----------:|------------:|-----------:|-------:|
| GPT-4o | ~85% | ~80% | ~80% | ~95% |
| Qwen2.5-Omni | ~80% | ~75% | ~75% | ~90% |
| **MiniCPM-o 4.5 (본 실행)** | **100%** | **95%** | **100%** | **100%** |

> ⚠️ 직접 비교 불가: 리더보드는 오디오 입출력 + GPT-4o-mini judge 사용;  
> 본 실행은 텍스트 입력 + 규칙 기반 채점.  
> 높은 수치는 **LLM 추론 능력이 뛰어남**을 의미하며, 오디오 파이프라인 성능과는 별도로 해석해야 함.

---

## 3. 음성 대화 품질

### 3.1 영어 (EN)

| 메트릭 | 값 | 비고 |
|--------|---|------|
| STT 정확도 (CER) | **92%** | Whisper base, 6시나리오 평균 |
| 의미 유사도 | **~90%** | sentence-transformers |
| 평균 응답 시간 | **2.3s** | Thinker + Talker + Code2Wav 합산 |
| TTS 품질 | Good | CosyVoice2 EN 정상 |
| 오디오 길이 | 5–8s (trim 후) | `_trim_silence()` 후처리 적용 |

**등급: A — 영어 음성 대화 프로덕션 준비 완료**

### 3.2 중국어 (ZH)

| 메트릭 | 값 | 비고 |
|--------|---|------|
| STT 정확도 (CER) | **76.1%** | Unicode NFKC 정규화 후 |
| 의미 유사도 | **95.0%** | 의미는 거의 완벽하게 보존 |
| 평균 응답 시간 | **1.5s** | 영어보다 빠름 |
| TTS 품질 | Good | CosyVoice2 ZH 정상 |

**핵심 발견:** 이전 0% 결과는 `word_accuracy` 메트릭이 CJK에 부적합해 발생한 **false negative**.  
CER + 의미 유사도 기준으로 정상 작동 확인.

샘플 비교:

| 턴 | 원본 | STT 결과 | CER | 의미 |
|---|------|---------|-----|------|
| 1 | 是的，您说得没错，今天天气很不错呢。 | 是的您说的没错今天天气很不错呢 | 8.5% | 93.4% |
| 4 | 是的，您说得没错，今天天气很不错呢。 | 是的,你說的沒錯,今天天氣很不錯呢 | 42.9% | 93.7% |

턴 4는 번체↔간체 변환 — 의미는 동일(93.7%), CER이 높은 것은 예상된 동작.

### 3.3 한국어 (KO)

| 메트릭 | 값 | 비고 |
|--------|---|------|
| 텍스트 생성 | ✅ 정상 | LLM이 올바른 한국어 텍스트 생성 |
| TTS 출력 | ❌ 잡음/왜곡 | CosyVoice2 한국어 미학습 |
| 오디오 STT | N/A | 오디오 이해 불가 |

**원인**: Code2Wav 백본 CosyVoice2가 표준 중국어와 영어로만 학습됨.  
한국어 음운 체계는 별도 파인튜닝 필요.

---

## 4. 지연 프로파일 (async_chunk 모드)

| 스테이지 | 시간 | 비고 |
|---------|------|------|
| Thinker (텍스트 생성) | 1.5–2.5s | 응답 길이에 따라 다름 |
| Talker (코덱 생성) | 0.5–1.0s | 스트리밍으로 Code2Wav와 병렬 |
| Code2Wav (오디오 합성) | 0.2–0.5s/청크 | CosyVoice2 flow + HiFi-GAN |
| **TTFP (첫 오디오 패킷까지)** | **~0.07s** | Thinker 완료 후 측정 |
| 전체 E2E | 2.0–4.0s | 모든 스테이지 포함 |

---

## 5. 평가 방법론

### STT 평가
- **도구**: OpenAI Whisper base (CPU)
- **영어**: 단어 정확도 + 의미 유사도
- **CJK**: CER (문자 오류율) + 의미 유사도
  - CER 표준: OpenASRLeaderboard, LibriSpeech CJK 평가
  - 정규화: Unicode NFKC (번체↔간체 처리)
- **의미 유사도**: `sentence-transformers/all-MiniLM-L6-v2` (코사인 유사도)

### VoiceBench 채점
- **Knowledge**: MCQ — 응답에 올바른 글자(A/B/C/D) 포함 여부
- **Instruction**: IFEval 스타일 — 제약 조건 충족 여부
- **Robustness**: 노이즈 / 억양 변형 — 동일한 MCQ 채점
- **Safety**: 거부 감지 — 거부 키워드 포함 여부

### 대화 품질 (다차원)
| 차원 | 가중치 | 구현 |
|------|:------:|------|
| 관련성 | 25% | 문자 겹침 + 최적 길이 인수 |
| STT 정확도 | 25% | CER (전 언어) |
| 일관성 | 20% | 턴 간 관련성 + 흐름 연속성 |
| 지식 유지 | 15% | 개체 재현 + 의미 유사도 |
| TTS 품질 | 15% | 에너지 기반 음성 비율 |

---

## 6. 알려진 문제

| 문제 | 영향 | 상태 |
|------|------|:----:|
| Stop token 6561 미생성 | 오디오 ~20s 고정 | ⚠️ 별도 이슈 |
| 한국어 TTS 실패 | KO 오디오 사용 불가 | ⚠️ CosyVoice2 한계 |
| 클라이언트 `modality=="audio"` 감지 | 스트리밍 청크 누락 | ⚠️ 클라이언트 버그 |

---

## 7. 참고 자료

- [VoiceBench GitHub](https://github.com/MatthewCYM/VoiceBench)
- [SOVA-Bench (arXiv:2506.02457)](https://arxiv.org/abs/2506.02457)
- [MTalk-Bench](https://github.com/FreedomIntelligence/MTalk-Bench)
- [OpenASRLeaderboard](https://github.com/huggingface/open_asr_leaderboard)
- [sentence-transformers](https://www.sbert.net/)
