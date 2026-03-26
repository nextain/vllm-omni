# MiniCPM-o 4.5 벤치마크 가이드

## 개요

MiniCPM-o 4.5 (openbmb/MiniCPM-o-4_5)의 text-to-speech 성능을 측정합니다.

| 벤치마크 | 설명 | 스크립트 |
|----------|------|----------|
| HF Transformers | 오프라인 단일 프로세스 baseline | `transformers/eval_minicpm_o_transformers.sh` |
| vLLM-Omni | 3-stage 파이프라인 E2E | `vllm_omni/eval_minicpm_o.sh` |

## 측정 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| **RTF** (Real-Time Factor) | E2E 지연 / 오디오 길이 | < 1.0 (실시간 이상) |
| **E2E Latency** | 요청→완료 총 시간 | P95 기준 리포트 |
| **Audio Throughput** | 벽시계 1초당 생성 오디오 길이 | > 1.0x (실시간 이상) |
| **Per-stage Latency** | Thinker/Talker/Code2Wav 각 단계 소요 시간 | 병목 식별용 |

## 1) 데이터셋 준비 (SeedTTS top100)

```bash
cd benchmarks/build_dataset
pip install gdown

# SeedTTS 테스트셋 다운로드
gdown --id 1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP
tar -xf seedtts_testset.tar

# top-100 프롬프트 추출
cp seedtts_testset/en/meta.lst meta.lst
python extract_prompts.py -i meta.lst -o top100.txt -n 100

# 정리
rm -rf seedtts_testset seedtts_testset.tar meta.lst
```

결과물: `benchmarks/build_dataset/top100.txt` (100개 텍스트 프롬프트)

## 2) HF Transformers 벤치마크 (baseline)

```bash
# 빠른 테스트 (10개 프롬프트)
bash benchmarks/minicpm-o/transformers/eval_minicpm_o_transformers.sh 10

# 전체 벤치마크 (100개)
bash benchmarks/minicpm-o/transformers/eval_minicpm_o_transformers.sh 100
```

출력:
- `benchmark_results/perf_stats.json` — RTF, latency, throughput (전체 + prompt별)
- `benchmark_results/results.json` — prompt별 텍스트 출력 + 오디오 경로
- `benchmark_results/audio/` — 생성된 wav 파일

확인사항:
- `rtf_avg` < 1.0인지 (실시간 합성 가능 여부)
- `total_time_s_p95` 이상치 없는지
- 오디오 파일 수 = 프롬프트 수

## 3) vLLM-Omni 파이프라인 벤치마크

```bash
# 빠른 테스트
bash benchmarks/minicpm-o/vllm_omni/eval_minicpm_o.sh 10

# 전체 벤치마크
bash benchmarks/minicpm-o/vllm_omni/eval_minicpm_o.sh 100
```

출력:
- `vllm_omni/logs/*.orchestrator.stats.jsonl` — stage별 latency
- `vllm_omni/logs/*.overall.stats.jsonl` — E2E latency/TPS
- `vllm_omni/logs/*.stage{0,1,2}.log` — stage별 상세 로그
- `vllm_omni/outputs/` — 생성된 텍스트 + wav 파일

확인사항:
- E2E latency가 HF 대비 개선되었는지
- stage별 latency가 안정적인지 (long tail 없는지)
- stage 로그에 에러 없는지

## 4) 결과 비교

HF vs vLLM-Omni 비교 차트 생성 예정.

## 하드웨어 요구사항

| 구성 | VRAM | 비고 |
|------|------|------|
| A40 (46GB) | ~27GB | 검증 완료 |
| RTX 3090 (24GB) | OOM 예상 | Thinker만 16GB+ |
| 2x RTX 3090 (48GB) | ~27GB | 미검증, TP 또는 stage 분산 필요 |
| RTX 5090 (32GB) | ~27GB | 미검증, tight fit |

## 디렉토리 구조

```
benchmarks/minicpm-o/
├── README.md                  ← 이 파일
├── transformers/
│   ├── eval_minicpm_o_transformers.sh
│   ├── minicpm_o_transformers.py
│   └── benchmark_results/     (실행 후 생성)
└── vllm_omni/
    ├── eval_minicpm_o.sh
    ├── logs/                  (실행 후 생성)
    └── outputs/               (실행 후 생성)
```
