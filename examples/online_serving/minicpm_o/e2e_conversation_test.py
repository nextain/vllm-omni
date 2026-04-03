#!/usr/bin/env python3
"""E2E Conversation Test Framework (#205).

Two AI speakers converse for N turns using external STT/TTS.
A 3rd-party STT monitor records timestamps and content for verification.

Phase 1: External STT (whisper) + TTS (edge-tts) + LLM (vllm-omni text API)
Phase 2: MiniCPM-o omni native audio I/O (replaces edge-tts + whisper for one speaker)

Usage (from distrobox vllm-dev):
    python scripts/e2e_conversation_test.py
    python scripts/e2e_conversation_test.py --turns 5 --output-dir /tmp/conv_test
    python scripts/e2e_conversation_test.py --list-voices
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Lazy imports (heavy, loaded on first use)
# ---------------------------------------------------------------------------
_whisper_model = None
_openai_client = None


def get_whisper_model(model_size: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        import warnings

        import torch
        import whisper

        print(f"[STT] Loading whisper {model_size} (CPU)...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _whisper_model = whisper.load_model(model_size, device="cpu")
    return _whisper_model


def get_openai_client(api_base: str) -> "OpenAI":
    global _openai_client
    if _openai_client is None or _openai_client._base_url != api_base:
        from openai import OpenAI

        _openai_client = OpenAI(base_url=api_base, api_key="unused")
    return _openai_client


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    """One conversation turn."""

    turn: int
    speaker: str
    text_original: str  # LLM-generated text
    audio_path: str  # Saved audio file
    audio_duration_s: float
    stt_transcript: str  # What monitor STT heard
    stt_avg_logprob: float  # STT confidence (higher = better)
    wall_time_s: float  # Elapsed since conversation start
    response_time_s: float  # Time to generate this turn


@dataclass
class ConversationReport:
    """Final conversation quality report."""

    meta: dict  # Config info
    turns: list[dict]  # All turns
    total_duration_s: float
    avg_response_time_s: float
    stt_accuracy: float  # 0-1, fraction of words matched
    n_turns: int
    transcript: str  # Full formatted transcript


# ---------------------------------------------------------------------------
# Speaker — one AI participant
# ---------------------------------------------------------------------------
class Speaker:
    def __init__(
        self,
        name: str,
        voice: str,
        system_prompt: str,
        api_base: str = "http://localhost:8000/v1",
        model: str = "openbmb/MiniCPM-o-4_5",
    ):
        self.name = name
        self.voice = voice
        self.system_prompt = system_prompt
        self.api_base = api_base
        self.model = model
        self.history: list[dict] = []

    def generate_text(self, user_message: str) -> tuple[str, float]:
        """Generate text response via LLM. Returns (text, elapsed_seconds)."""
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        client = get_openai_client(self.api_base)
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - t0
        text = resp.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": text})
        return text, elapsed

    async def synthesize(self, text: str, output_path: str) -> float:
        """TTS via edge-tts. Returns audio duration in seconds."""
        import edge_tts

        comm = edge_tts.Communicate(text, self.voice)
        await comm.save(output_path)
        data, sr = sf.read(output_path)
        return len(data) / sr


# ---------------------------------------------------------------------------
# Monitor — 3rd-party STT observer
# ---------------------------------------------------------------------------
class Monitor:
    """3rd-party STT that records every utterance for verification."""

    def __init__(self, whisper_size: str = "base"):
        self.whisper_size = whisper_size

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        """Returns (transcript, avg_logprob)."""
        model = get_whisper_model(self.whisper_size)
        result = model.transcribe(audio_path, fp16=False)
        text = result["text"].strip()
        segments = result.get("segments", [])
        if segments:
            avg_logprob = float(np.mean([s.get("avg_logprob", 0) for s in segments]))
        else:
            avg_logprob = 0.0
        return text, avg_logprob


# ---------------------------------------------------------------------------
# Word-level accuracy metric
# ---------------------------------------------------------------------------
def word_accuracy(original: str, transcript: str) -> float:
    """Fraction of original words found in transcript (case-insensitive)."""
    orig_words = set(original.lower().split())
    trans_words = set(transcript.lower().split())
    if not orig_words:
        return 1.0
    return len(orig_words & trans_words) / len(orig_words)


# ---------------------------------------------------------------------------
# Conversation Test Runner
# ---------------------------------------------------------------------------
class ConversationTest:
    def __init__(
        self,
        speaker_a: Speaker,
        speaker_b: Speaker,
        monitor: Monitor,
        output_dir: str = "/tmp/conv_test",
        n_turns: int = 10,
        max_duration_s: float = 300,
    ):
        self.speakers = [speaker_a, speaker_b]
        self.monitor = monitor
        self.output_dir = Path(output_dir)
        self.n_turns = n_turns
        self.max_duration_s = max_duration_s
        self.turns: list[Turn] = []

    async def run(self) -> ConversationReport:
        """Run the conversation and return the report."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        turns: list[Turn] = []
        t_start = time.perf_counter()

        # Speaker A opens the conversation
        opener = "Hi there! Let's have a casual chat. What's on your mind today?"

        current_text = opener
        print(f"\n{'='*60}")
        print(f"  E2E Conversation Test  —  {self.n_turns} turns")
        print(f"{'='*60}\n")

        for i in range(self.n_turns):
            elapsed = time.perf_counter() - t_start
            if elapsed > self.max_duration_s:
                print(f"\n[!] Max duration ({self.max_duration_s}s) reached, stopping.")
                break

            speaker_idx = i % 2
            speaker = self.speakers[speaker_idx]
            label = f"[Turn {i+1}] {speaker.name}"

            # 1. LLM generates response
            print(f"{label}: thinking...", end=" ", flush=True)
            text, resp_time = speaker.generate_text(current_text)
            print(f"({resp_time:.1f}s)")

            # 2. TTS
            audio_path = str(self.output_dir / f"turn_{i+1:03d}_{speaker.name}.wav")
            audio_dur = await speaker.synthesize(text, audio_path)
            print(f"  Text:  {text}")
            print(f"  Audio: {audio_dur:.1f}s → {audio_path}")

            # 3. Monitor STT
            stt_text, stt_logprob = self.monitor.transcribe(audio_path)
            acc = word_accuracy(text, stt_text)
            print(f"  STT:   {stt_text}")
            print(f"  Match: {acc:.0%}  (logprob={stt_logprob:.2f})")

            turn = Turn(
                turn=i + 1,
                speaker=speaker.name,
                text_original=text,
                audio_path=audio_path,
                audio_duration_s=audio_dur,
                stt_transcript=stt_text,
                stt_avg_logprob=stt_logprob,
                wall_time_s=time.perf_counter() - t_start,
                response_time_s=resp_time,
            )
            turns.append(turn)

            # Next speaker gets this text as input
            current_text = text
            print()

        return self._build_report(turns, time.perf_counter() - t_start)

    def _build_report(self, turns: list[Turn], total_s: float) -> ConversationReport:
        accuracies = [
            word_accuracy(t.text_original, t.stt_transcript) for t in turns
        ]
        avg_resp = np.mean([t.response_time_s for t in turns]) if turns else 0

        # Build readable transcript
        lines = []
        for t in turns:
            ts = f"{t.wall_time_s:6.1f}s"
            lines.append(f"[{ts}] {t.speaker}: {t.text_original}")
            lines.append(f"{'':10s} STT: {t.stt_transcript}")
        transcript = "\n".join(lines)

        meta = {
            "n_turns_requested": self.n_turns,
            "max_duration_s": self.max_duration_s,
            "speaker_a": {
                "name": self.speakers[0].name,
                "voice": self.speakers[0].voice,
                "model": self.speakers[0].model,
            },
            "speaker_b": {
                "name": self.speakers[1].name,
                "voice": self.speakers[1].voice,
                "model": self.speakers[1].model,
            },
            "monitor": f"whisper {self.monitor.whisper_size}",
            "output_dir": str(self.output_dir),
        }

        report = ConversationReport(
            meta=meta,
            turns=[asdict(t) for t in turns],
            total_duration_s=total_s,
            avg_response_time_s=float(avg_resp),
            stt_accuracy=float(np.mean(accuracies)) if accuracies else 0,
            n_turns=len(turns),
            transcript=transcript,
        )

        # Save report
        report_path = self.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        print(f"\nReport saved: {report_path}")

        # Print transcript
        print(f"\n{'='*60}")
        print("  CONVERSATION TRANSCRIPT")
        print(f"{'='*60}")
        print(transcript)
        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  Turns:          {report.n_turns}")
        print(f"  Total time:     {report.total_duration_s:.1f}s")
        print(f"  Avg resp time:  {report.avg_response_time_s:.1f}s")
        print(f"  STT accuracy:   {report.stt_accuracy:.0%}")

        return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
async def list_voices():
    """List available edge-tts voices."""
    import edge_tts

    voices = await edge_tts.list_voices()
    en_voices = [v for v in voices if v["Locale"].startswith("en-")]
    for v in en_voices:
        print(f"  {v['ShortName']:30s}  {v['Gender']:6s}  {v['Locale']}")


async def main():
    parser = argparse.ArgumentParser(description="E2E Conversation Test (#205)")
    parser.add_argument("--turns", type=int, default=10, help="Number of turns")
    parser.add_argument(
        "--max-duration", type=float, default=300, help="Max duration (seconds)"
    )
    parser.add_argument(
        "--output-dir", default="/tmp/conv_test", help="Output directory"
    )
    parser.add_argument(
        "--api-base", default="http://localhost:8000/v1", help="LLM API base URL"
    )
    parser.add_argument(
        "--whisper-model", default="base", help="Whisper model size"
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List edge-tts voices and exit"
    )
    args = parser.parse_args()

    if args.list_voices:
        await list_voices()
        return

    # Default speakers
    speaker_a = Speaker(
        name="Alice",
        voice="en-US-AriaNeural",
        system_prompt=(
            "You are Alice, a friendly and curious person. "
            "Keep your responses to 1-2 short sentences. "
            "Have a natural casual conversation. "
            "Do not use filler words like 'um' or 'uh'."
        ),
        api_base=args.api_base,
    )
    speaker_b = Speaker(
        name="Bob",
        voice="en-US-GuyNeural",
        system_prompt=(
            "You are Bob, a thoughtful and witty person. "
            "Keep your responses to 1-2 short sentences. "
            "Have a natural casual conversation. "
            "Do not use filler words like 'um' or 'uh'."
        ),
        api_base=args.api_base,
    )

    monitor = Monitor(whisper_size=args.whisper_model)

    test = ConversationTest(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        monitor=monitor,
        output_dir=args.output_dir,
        n_turns=args.turns,
        max_duration_s=args.max_duration,
    )
    await test.run()


if __name__ == "__main__":
    asyncio.run(main())
