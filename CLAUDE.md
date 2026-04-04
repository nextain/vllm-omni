# vllm-omni — MiniCPM-o Contribution Fork

This repository is a fork of [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni),
adding MiniCPM-o 4.5 omni model support as an upstream contribution.

## Context

**Practical goal**: Local omni model inference on consumer hardware (2× RTX 3090, no NVLink),
enabling real-time voice conversation in AI-native desktop applications.

**Open-source goal**: Contribute MiniCPM-o 4.5 support to upstream vllm-omni so the broader
community can serve omni models efficiently.

**Methodology**: The entire contribution workflow — upstream pattern analysis, implementation,
and headless adversarial code review — is driven by an AI coding agent following an
issue-driven development process. This is an experiment in **AI-native open-source contribution**:
can an AI agent understand a complex framework, write consistent upstream-quality code,
and self-review it to a production standard?

---

## Mandatory Reads

At the start of every session, read these files to understand the current state:

1. `.agents/context/upstream-basics.md` — vllm-omni architecture, stage config, multi-GPU
2. `.agents/context/our-changes.md` — what we implemented, trust state, benchmark results
3. `.agents/context/contribution-journey.md` — failures, pivots, and key decisions (the *why*)
4. `docs/configuration/stage_configs.md` — official upstream stage config docs
5. `docs/configuration/gpu_memory_utilization.md` — GPU memory planning guide

Human-readable Korean mirrors are in `.users/context/`.

---

## Core Principles

### 1. Upstream first
- Before modifying any code, search upstream docs / issues / PRs
- **Read similar model implementation files directly** and compare signatures, param values,
  and error handling patterns (qwen3_omni, qwen2_5_omni, mimo_audio)
- Follow the upstream standard where possible; if the standard is wrong, fix our code
  AND propose an upstream issue/PR in parallel
- Upstream/OSS code is read-only — modifications only in our files
- Use validated patterns only; do not invent new abstractions

### 2. Know the basics before coding
- Understand `process: true`, multi-GPU device assignment, and CLI options before editing
- E2E test on actual target hardware (2× RTX 3090) before declaring something works
- Never judge correctness from single-GPU behavior alone

### 3. Development process
- Issue-driven development: Understand → Plan → Build → Review → E2E → Post-test Review
  (`.agents/context/contribution-journey.md` tracks the actual implementation phases as "Phase 1–6" —
  these are project history phases, not the same as the development process steps above)
- Code review uses headless adversarial subagents (adversarial frame, per-pass lenses)
- Session handoff via `.agents/context/` files — no conversation memory assumed

---

## Upstream References

| Resource | Location |
|----------|---------|
| Official docs | https://vllm-omni.readthedocs.io/en/latest/ |
| Stage Configs guide | `docs/configuration/stage_configs.md` |
| GPU Memory guide | `docs/configuration/gpu_memory_utilization.md` |
| Adding Omni Model | `docs/contributing/model/adding_omni_model.md` |
| Qwen3-Omni example | `examples/online_serving/qwen3_omni/` |
| Upstream issues | https://github.com/vllm-project/vllm-omni/issues |

---

## Quick Start

```bash
# Basic (stage_configs auto-loaded)
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --port 8000

# Custom stage_configs
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo.yaml \
  --trust-remote-code --port 8000

# 2× RTX 3090 (NCCL P2P disable required — no NVLink)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_48gb_2gpu.yaml \
  --trust-remote-code --port 8000

# async_chunk streaming (TTFP optimization)
NCCL_P2P_DISABLE=1 vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
  --trust-remote-code --port 8000
```

---

## Related Issues

| Issue | Topic | Status |
|-------|-------|:------:|
| vllm-omni#1182 | MiniCPM-o upstream (parallel work by Allyyi) | OPEN |
| vllm-omni#1387 | Multi-GPU OOM (known bug) | OPEN |
