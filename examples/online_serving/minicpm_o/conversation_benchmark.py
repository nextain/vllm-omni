#!/usr/bin/env python3
"""MiniCPM-o English Conversation Ability Benchmark.

Measures the omni model's conversation capabilities across multiple dimensions:
  1. Response quality   — relevance, coherence, grammar
  2. Audio quality      — STT accuracy, duration, clarity
  3. Latency            — time to first token, total response time
  4. Conversation flow  — turn-taking, topic coherence, context retention

Each scenario produces:
  - WAV audio files (one per turn)
  - benchmark_report.json (full metrics)
  - benchmark_summary.txt (human-readable)

Usage (from distrobox vllm-dev):
    # Standard mode (edge-tts TTS, sync server)
    cd examples/online_serving/minicpm_o/
    python conversation_benchmark.py
    python conversation_benchmark.py --output-dir /tmp/bench_run1

    # Omni native audio mode — requires async_chunk server:
    vllm serve openbmb/MiniCPM-o-4_5 --omni \
      --stage-configs-path vllm_omni/model_executor/stage_configs/minicpmo_async_chunk.yaml \
      --trust-remote-code --port 8000
    python conversation_benchmark.py --omni
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Reuse from e2e_conversation_test
from e2e_conversation_test import (
    Monitor,
    OmniSpeaker,
    Speaker,
    _normalize,
    get_openai_client,
    get_whisper_model,
)

# CJK-aware metrics for speech evaluation
from metrics.cjk_metrics import (
    calculate_cjk_metrics,
)
from metrics.conversation_quality import (
    ConversationEvaluator,
)


# ---------------------------------------------------------------------------
# Benchmark Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "id": "greeting",
        "name": "Greeting Exchange",
        "description": "Basic greeting and introduction",
        "opener": "Hi! Nice to meet you. I'm Bob. What's your name?",
        "n_turns": 4,
        "eval_criteria": ["response mentions name", "friendly tone", "asks follow-up"],
    },
    {
        "id": "qa",
        "name": "Question-Answering",
        "description": "Factual questions and answers",
        "opener": "Can you tell me what the capital of France is?",
        "n_turns": 6,
        "eval_criteria": ["correct answer", "elaboration", "follow-up question"],
    },
    {
        "id": "storytelling",
        "name": "Collaborative Storytelling",
        "description": "Build a story together turn by turn",
        "opener": "Let's tell a story together. I'll start: Once upon a time, there was a robot who lived in a small village. Your turn!",
        "n_turns": 6,
        "eval_criteria": ["continues story", "creative", "maintains narrative"],
    },
    {
        "id": "opinion",
        "name": "Opinion Discussion",
        "description": "Express and discuss opinions",
        "opener": "Do you think artificial intelligence will replace most jobs in the next 20 years?",
        "n_turns": 6,
        "eval_criteria": ["states opinion", "provides reasoning", "acknowledges other views"],
    },
    {
        "id": "topic_switch",
        "name": "Topic Switching",
        "description": "Handle abrupt topic changes gracefully",
        "opener": "So, what's your favorite season of the year?",
        "switch_prompts": [
            "Interesting! Now, completely unrelated — what do you think about space travel?",
            "Back to seasons — do you prefer summer or winter activities?",
        ],
        "n_turns": 6,
        "eval_criteria": ["adapts to topic change", "maintains context", "smooth transition"],
    },
    {
        "id": "long_context",
        "name": "Long Context Retention",
        "description": "Remember details from earlier in conversation",
        "opener": "I have two cats named Luna and Stella. Luna is orange and Stella is gray. Remember those details!",
        "n_turns": 8,
        "recall_prompt": "Quick question — what are the names and colors of my cats?",
        "eval_criteria": ["recalls details", "correct names", "correct colors"],
    },
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class TurnMetrics:
    turn: int
    speaker: str
    text_original: str
    audio_path: str
    audio_duration_s: float
    stt_transcript: str
    stt_cer: float  # CER (for all languages)
    stt_semantic_similarity: float  # Embedding-based similarity
    stt_avg_logprob: float
    response_time_s: float
    ttfp_s: float  # Time To First (text) Packet — streaming latency
    wall_time_s: float
    text_length_chars: int
    text_length_words: int


@dataclass
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    turns: list[dict]
    metrics: dict  # Aggregated metrics
    audio_dir: str


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_relevance(response: str, prompt: str) -> float:
    """Heuristic: overlap of content words between prompt and response."""
    prompt_words = set(_normalize(prompt))
    resp_words = set(_normalize(response))
    # Remove common stopwords
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "about", "its", "it", "and", "but", "or",
        "not", "no", "so", "if", "that", "this", "these", "those",
        "i", "you", "he", "she", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "our", "their",
    }
    prompt_content = prompt_words - stopwords
    resp_content = resp_words - stopwords
    if not prompt_content:
        return 1.0
    overlap = len(prompt_content & resp_content) / len(prompt_content)
    return min(overlap * 2, 1.0)  # Scale up: 50% overlap = full score


def evaluate_grammar(text: str) -> float:
    """Heuristic grammar check: sentence completeness, no obvious errors."""
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    if not sentences:
        return 0.0
    scores = []
    for s in sentences:
        score = 1.0
        words = s.split()
        if len(words) < 2:
            score -= 0.3  # Very short fragment
        # Starts with capital (after stripping)
        if s and s[0].isupper():
            pass  # Good
        else:
            score -= 0.1
        # No double spaces
        if "  " in s:
            score -= 0.1
        scores.append(max(score, 0.0))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------
class ConversationBenchmark:
    def __init__(
        self,
        test_speaker: Speaker | OmniSpeaker,
        partner: Speaker,
        monitor: Monitor,
        output_dir: str = "/tmp/minicpmo_benchmark",
    ):
        self.test_speaker = test_speaker
        self.partner = partner
        self.monitor = monitor
        self.output_dir = Path(output_dir)

    async def run_scenario(self, scenario: dict) -> ScenarioResult:
        """Run one benchmark scenario."""
        scenario_dir = self.output_dir / scenario["id"]
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Reset histories
        self.test_speaker.history = []
        self.partner.history = []

        turns: list[TurnMetrics] = []
        t_start = time.perf_counter()
        current_text = scenario["opener"]
        switch_idx = 0

        print(f"\n{'='*50}")
        print(f"  Scenario: {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'='*50}")

        for i in range(scenario["n_turns"]):
            # Check for topic switch prompts
            if "switch_prompts" in scenario and switch_idx < len(scenario["switch_prompts"]):
                if i == 2 or i == 4:  # Inject switch at turns 3 and 5
                    current_text = scenario["switch_prompts"][switch_idx]
                    switch_idx += 1
                    print(f"  [TOPIC SWITCH]")

            # Check for recall prompt
            if "recall_prompt" in scenario and i == scenario["n_turns"] - 2:
                current_text = scenario["recall_prompt"]
                print(f"  [RECALL TEST]")

            # Alternate: partner (even) → test speaker (odd)
            if i % 2 == 0:
                speaker = self.partner
                role = "Partner"
            else:
                speaker = self.test_speaker
                role = "Test"

            label = f"  Turn {i+1} [{role}]"
            print(f"{label}: thinking...", end=" ", flush=True)
            text, resp_time = speaker.generate_text(current_text)
            _last_ttfp = getattr(speaker, "_last_ttfp", None)
            ttfp = _last_ttfp if _last_ttfp is not None else resp_time
            print(f"({resp_time:.1f}s, TTFP={ttfp:.2f}s)")

            # Synthesize
            audio_path = str(scenario_dir / f"turn_{i+1:03d}_{role}.wav")
            audio_dur = await speaker.synthesize(text, audio_path)

            # STT with CJK-aware metrics
            stt_text, stt_logprob = self.monitor.transcribe(audio_path)
            cjk_metrics = calculate_cjk_metrics(text, stt_text, "en")
            stt_acc = cjk_metrics["cer"]  # CER for all languages
            semantic_sim = cjk_metrics["semantic_similarity"]

            print(f"    Text: {text[:80]}{'...' if len(text) > 80 else ''}")
            print(f"    STT:  {stt_text[:80]}{'...' if len(stt_text) > 80 else ''}")
            print(f"    CER: {stt_acc:.0%} | Semantic: {semantic_sim:.0%} | {cjk_metrics['notes']}")

            turn = TurnMetrics(
                turn=i + 1,
                speaker=role,
                text_original=text,
                audio_path=audio_path,
                audio_duration_s=audio_dur,
                stt_transcript=stt_text,
                stt_cer=stt_acc,
                stt_semantic_similarity=semantic_sim,
                stt_avg_logprob=stt_logprob,
                response_time_s=resp_time,
                ttfp_s=ttfp,
                wall_time_s=time.perf_counter() - t_start,
                text_length_chars=len(text),
                text_length_words=len(text.split()),
            )
            turns.append(turn)
            current_text = text

        # Compute aggregated metrics
        test_turns = [t for t in turns if t.speaker == "Test"]
        metrics = {
            "n_turns": len(turns),
            "n_test_turns": len(test_turns),
            "avg_response_time_s": float(np.mean([t.response_time_s for t in test_turns])) if test_turns else 0,
            "avg_ttfp_s": float(np.mean([t.ttfp_s for t in test_turns])) if test_turns else 0,
            "avg_stt_cer": float(np.mean([t.stt_cer for t in test_turns])) if test_turns else 0,
            "avg_stt_semantic_similarity": float(np.mean([t.stt_semantic_similarity for t in test_turns])) if test_turns else 0,
            "avg_text_length_words": float(np.mean([t.text_length_words for t in test_turns])) if test_turns else 0,
            "avg_audio_duration_s": float(np.mean([t.audio_duration_s for t in test_turns])) if test_turns else 0,
            "total_wall_time_s": turns[-1].wall_time_s if turns else 0,
        }

        result = ScenarioResult(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
            turns=[asdict(t) for t in turns],
            metrics=metrics,
            audio_dir=str(scenario_dir),
        )
        print(f"\n  Scenario metrics:")
        print(f"    Avg response time:           {metrics['avg_response_time_s']:.1f}s")
        print(f"    Avg STT accuracy (CER):     {metrics['avg_stt_cer']:.0%}")
        print(f"    Avg STT semantic similarity: {metrics['avg_stt_semantic_similarity']:.0%}")
        print(f"    Avg text length:             {metrics['avg_text_length_words']:.1f} words")

        return result

    async def run_all(self, scenarios: list[dict]) -> dict:
        """Run all scenarios and produce final report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        overall_metrics = {
            "scenarios_run": 0,
            "total_test_turns": 0,
            "avg_response_time_s": 0,
            "avg_ttfp_s": 0,
            "avg_stt_cer": 0,
            "avg_stt_semantic_similarity": 0,
            "avg_text_length_words": 0,
            "scenario_summaries": [],
        }

        for scenario in scenarios:
            result = await self.run_scenario(scenario)
            results.append(result)
            overall_metrics["scenarios_run"] += 1
            overall_metrics["total_test_turns"] += result.metrics["n_test_turns"]
            overall_metrics["scenario_summaries"].append({
                "id": result.scenario_id,
                "name": result.scenario_name,
                "stt_cer": result.metrics["avg_stt_cer"],
                "semantic_similarity": result.metrics["avg_stt_semantic_similarity"],
                "response_time": result.metrics["avg_response_time_s"],
                "ttfp_s": result.metrics["avg_ttfp_s"],
                "text_length": result.metrics["avg_text_length_words"],
                "audio_duration": result.metrics["avg_audio_duration_s"],
            })

        # Overall averages
        test_counts = [r.metrics["n_test_turns"] for r in results]
        total_test = sum(test_counts)
        if total_test > 0:
            overall_metrics["avg_response_time_s"] = float(np.average(
                [r.metrics["avg_response_time_s"] for r in results],
                weights=test_counts,
            ))
            overall_metrics["avg_ttfp_s"] = float(np.average(
                [r.metrics["avg_ttfp_s"] for r in results],
                weights=test_counts,
            ))
            overall_metrics["avg_stt_cer"] = float(np.average(
                [r.metrics["avg_stt_cer"] for r in results],
                weights=test_counts,
            ))
            overall_metrics["avg_stt_semantic_similarity"] = float(np.average(
                [r.metrics["avg_stt_semantic_similarity"] for r in results],
                weights=test_counts,
            ))
            overall_metrics["avg_text_length_words"] = float(np.average(
                [r.metrics["avg_text_length_words"] for r in results],
                weights=test_counts,
            ))

        report = {
            "model": self.test_speaker.model,
            "speaker_type": type(self.test_speaker).__name__,
            "overall": overall_metrics,
            "scenarios": [asdict(r) for r in results],
        }

        # Save report
        report_path = self.output_dir / "benchmark_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved: {report_path}")

        # Print summary
        self._print_summary(report)

        return report

    def _print_summary(self, report: dict):
        overall = report["overall"]
        print(f"\n{'='*60}")
        print(f"  BENCHMARK SUMMARY — {report['model']}")
        print(f"  Speaker: {report['speaker_type']}")
        print(f"{'='*60}")
        print(f"  Scenarios:           {overall['scenarios_run']}")
        print(f"  Test turns:          {overall['total_test_turns']}")
        print(f"  Avg TTFP:            {overall['avg_ttfp_s']:.2f}s")
        print(f"  Avg resp time:       {overall['avg_response_time_s']:.1f}s")
        print(f"  Avg STT CER:         {overall['avg_stt_cer']:.0%}")
        print(f"  Avg STT semantic sim: {overall['avg_stt_semantic_similarity']:.0%}")
        print(f"  Avg text length:       {overall['avg_text_length_words']:.1f} words")
        print()
        print(f"  {'Scenario':<25s} {'CER':>4s} {'Sem':>4s} {'TTFP':>6s} {'Resp':>6s} {'Words':>6s} {'Audio':>6s}")
        print(f"  {'-'*57}")
        for s in overall["scenario_summaries"]:
            print(f"  {s['name']:<25s} {s['stt_cer']:>4.0%} {s['semantic_similarity']:>4.0%} {s['ttfp_s']:>5.2f}s {s['response_time']:>5.1f}s {s['text_length']:>5.1f}w {s['audio_duration']:>5.1f}s")

        # Save summary text
        summary_path = self.output_dir / "benchmark_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"MiniCPM-o Conversation Benchmark — {report['model']}\n")
            f.write(f"Speaker: {report['speaker_type']}\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"{'='*60}\n\n")
            for sr in report["scenarios"]:
                f.write(f"Scenario: {sr['scenario_name']}\n")
                for t in sr["turns"]:
                    f.write(f"  [{t['speaker']}] {t['text_original']}\n")
                    f.write(f"         STT: {t['stt_transcript']} ({t['stt_cer']:.0%})\n")
                f.write(f"\n  Metrics: resp={sr['metrics']['avg_response_time_s']:.1f}s "
                        f"stt={sr['metrics']['avg_stt_cer']:.0%} "
                        f"words={sr['metrics']['avg_text_length_words']:.1f}\n\n")
        print(f"\nSummary saved: {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o Conversation Benchmark")
    parser.add_argument(
        "--output-dir", default="/tmp/minicpmo_benchmark", help="Output directory"
    )
    parser.add_argument(
        "--api-base", default="http://localhost:8000/v1", help="LLM API base URL"
    )
    parser.add_argument(
        "--whisper-model", default="base", help="Whisper model size"
    )
    parser.add_argument(
        "--omni", action="store_true",
        help="Test speaker uses MiniCPM-o native audio",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Run specific scenarios only (by id: greeting qa storytelling opinion topic_switch long_context)",
    )
    args = parser.parse_args()

    # Create speakers
    if args.omni:
        test_speaker = OmniSpeaker(
            name="MiniCPM-o",
            system_prompt=(
                "You are a helpful AI assistant having a natural conversation. "
                "Keep responses to 1-3 sentences. Be engaging and relevant."
            ),
            api_base=args.api_base,
        )
    else:
        test_speaker = Speaker(
            name="MiniCPM-o",
            voice="en-US-AriaNeural",
            system_prompt=(
                "You are a helpful AI assistant having a natural conversation. "
                "Keep responses to 1-3 sentences. Be engaging and relevant."
            ),
            api_base=args.api_base,
        )

    partner = Speaker(
        name="Partner",
        voice="en-US-GuyNeural",
        system_prompt=(
            "You are a conversation partner. Ask interesting questions and "
            "share your thoughts. Keep responses to 1-3 sentences."
        ),
        api_base=args.api_base,
    )

    monitor = Monitor(whisper_size=args.whisper_model)

    # Select scenarios
    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s["id"] in args.scenarios]

    bench = ConversationBenchmark(
        test_speaker=test_speaker,
        partner=partner,
        monitor=monitor,
        output_dir=args.output_dir,
    )
    await bench.run_all(scenarios)


if __name__ == "__main__":
    asyncio.run(main())
