#!/usr/bin/env python3
"""Language-specific conversation test for MiniCPM-o.

Tests Chinese and Korean conversation ability:
- Chinese: CosyVoice2 supports it (trained on Chinese)
- Korean: CosyVoice2 does NOT support it (expected to fail for audio)

Usage:
    python examples/online_serving/minicpm_o/language_test.py --lang zh
    python examples/online_serving/minicpm_o/language_test.py --lang ko
    python examples/online_serving/minicpm_o/language_test.py --lang all
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from e2e_conversation_test import (
    Monitor,
    OmniSpeaker,
    Speaker,
)

# CJK-aware metrics for speech evaluation
from metrics.cjk_metrics import (
    calculate_cjk_metrics,
)
from metrics.conversation_quality import (
    ConversationEvaluator,
    format_conversation_report,
)

LANG_CONFIG = {
    "en": {
        "name": "English",
        "voice_a": "en-US-AriaNeural",
        "voice_b": "en-US-GuyNeural",
        "system_prompt": "You are a helpful AI assistant. Keep responses to 1-3 sentences.",
        "scenarios": [
            {"opener": "Hi! How are you today?", "n_turns": 4},
            {"opener": "What do you think about the future of AI?", "n_turns": 4},
        ],
    },
    "zh": {
        "name": "Chinese (中文)",
        "voice_a": "zh-CN-XiaoxiaoNeural",
        "voice_b": "zh-CN-YunxiNeural",
        "system_prompt": "你是一个有用的AI助手。用中文回答，保持1-3句话。",
        "scenarios": [
            {"opener": "你好！今天天气怎么样？", "n_turns": 4},
            {"opener": "你觉得人工智能的未来会怎样？", "n_turns": 4},
        ],
    },
    "ko": {
        "name": "Korean (한국어)",
        "voice_a": "ko-KR-SunHiNeural",
        "voice_b": "ko-KR-InJoonNeural",
        "system_prompt": "당신은 유용한 AI 어시스턴트입니다. 한국어로 답변하고, 1-3문장으로 유지하세요.",
        "scenarios": [
            {"opener": "안녕하세요! 오늘 날씨가 어떤가요?", "n_turns": 4},
            {"opener": "인공지능의 미래에 대해 어떻게 생각하세요?", "n_turns": 4},
        ],
    },
}


async def test_language(
    lang: str,
    config: dict,
    output_dir: Path,
    monitor: Monitor,
    api_base: str,
):
    """Run conversation test for one language."""
    lang_dir = output_dir / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Language Test: {config['name']} ({lang})")
    print(f"{'='*60}")

    # Create speakers
    omni = OmniSpeaker(
        name=f"MiniCPM-o",
        system_prompt=config["system_prompt"],
        api_base=api_base,
    )
    partner = Speaker(
        name="Partner",
        voice=config["voice_b"],
        system_prompt=config["system_prompt"],
        api_base=api_base,
    )

    results = []
    for si, scenario in enumerate(config["scenarios"]):
        omni.history = []
        partner.history = []
        current_text = scenario["opener"]

        print(f"\n  Scenario {si+1}: {scenario['opener'][:50]}...")

        for i in range(scenario["n_turns"]):
            if i % 2 == 0:
                speaker = partner
                role = "Partner"
            else:
                speaker = omni
                role = "Omni"

            text, resp_time = speaker.generate_text(current_text)
            audio_path = str(lang_dir / f"s{si+1}_turn_{i+1:02d}_{role}.wav")
            audio_dur = await speaker.synthesize(text, audio_path)

            # STT verification with CJK-aware metrics
            stt_text, stt_logprob = monitor.transcribe(audio_path)
            cjk_metrics = calculate_cjk_metrics(text, stt_text, lang)
            stt_acc = cjk_metrics["cer"] if lang in ["zh", "ko"] else cjk_metrics["word_accuracy"]
            semantic_sim = cjk_metrics["semantic_similarity"]

            print(f"    [{role}] {text[:60]}{'...' if len(text) > 60 else ''}")
            print(f"     STT: {stt_text[:60]}{'...' if len(stt_text) > 60 else ''}")
            print(f"     CER: {stt_acc:.0%} | Semantic: {semantic_sim:.0%} | {cjk_metrics['notes']}")
            print(f"     Audio: {audio_dur:.1f}s, Resp: {resp_time:.1f}s")

            results.append({
                "scenario": si + 1,
                "turn": i + 1,
                "role": role,
                "text": text,
                "stt_text": stt_text,
                "stt_accuracy": stt_acc,
                "stt_semantic_similarity": semantic_sim,
                "stt_cer": cjk_metrics["cer"],
                "stt_notes": cjk_metrics["notes"],
                "stt_logprob": stt_logprob,
                "audio_duration_s": audio_dur,
                "response_time_s": resp_time,
                "audio_path": audio_path,
            })

            current_text = text

    # Summary for this language
    omni_results = [r for r in results if r["role"] == "Omni"]
    partner_results = [r for r in results if r["role"] == "Partner"]

    # Conversation quality evaluation
    evaluator = ConversationEvaluator()
    conversation_metrics = evaluator.evaluate_conversation(
        [{"role": r["role"], "text": r["text"]} for r in results]
    )

    # Calculate summary with new metrics
    summary = {
        "language": lang,
        "language_name": config["name"],
        "omni_turns": len(omni_results),
        "partner_turns": len(partner_results),
        "omni_avg_stt_accuracy": float(np.mean([r["stt_accuracy"] for r in omni_results])) if omni_results else 0,
        "partner_avg_stt_accuracy": float(np.mean([r["stt_accuracy"] for r in partner_results])) if partner_results else 0,
        "omni_avg_stt_semantic_similarity": float(np.mean([r["stt_semantic_similarity"] for r in omni_results])) if omni_results else 0,
        "partner_avg_stt_semantic_similarity": float(np.mean([r["stt_semantic_similarity"] for r in partner_results])) if partner_results else 0,
        "omni_avg_response_time": float(np.mean([r["response_time_s"] for r in omni_results])) if omni_results else 0,
        "omni_avg_audio_duration": float(np.mean([r["audio_duration_s"] for r in omni_results])) if omni_results else 0,
        "conversation_quality": {
            "relevance": conversation_metrics.relevance,
            "coherence": conversation_metrics.coherence,
            "knowledge_retention": conversation_metrics.knowledge_retention,
            "overall_quality": conversation_metrics.overall_quality,
        },
        "results": results,
    }

    print(f"\n  --- {config['name']} Summary ---")
    print(f"  Omni STT accuracy:          {summary['omni_avg_stt_accuracy']:.0%}")
    print(f"  Omni STT semantic similarity: {summary['omni_avg_stt_semantic_similarity']:.0%}")
    print(f"  Partner STT accuracy:         {summary['partner_avg_stt_accuracy']:.0%}")
    print(f"  Omni avg resp time:           {summary['omni_avg_response_time']:.1f}s")
    print(f"  Omni avg audio dur:           {summary['omni_avg_audio_duration']:.1f}s")
    print(f"\n  --- Conversation Quality ---")
    print(f"  Relevance:                  {summary['conversation_quality']['relevance']:.0%}")
    print(f"  Coherence:                  {summary['conversation_quality']['coherence']:.0%}")
    print(f"  Knowledge Retention:          {summary['conversation_quality']['knowledge_retention']:.0%}")
    print(f"  Overall Quality:             {summary['conversation_quality']['overall_quality']:.0%}")

    # Save
    report_path = lang_dir / "language_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Report: {report_path}")

    return summary


async def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o Language Test")
    parser.add_argument("--lang", default="all", choices=["en", "zh", "ko", "all"])
    parser.add_argument("--output-dir", default="/tmp/lang_test")
    parser.add_argument("--api-base", default="http://localhost:8000/v1")
    parser.add_argument("--whisper-model", default="base")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monitor = Monitor(whisper_size=args.whisper_model)

    langs = list(LANG_CONFIG.keys()) if args.lang == "all" else [args.lang]
    all_results = []

    for lang in langs:
        result = await test_language(
            lang, LANG_CONFIG[lang], output_dir, monitor, args.api_base
        )
        all_results.append(result)

    # Final comparison
    print(f"\n{'='*60}")
    print(f"  LANGUAGE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Language':<15s} {'Omni STT%':>10s} {'Partner STT%':>13s} {'Audio':>8s}")
    print(f"  {'-'*50}")
    for r in all_results:
        print(f"  {r['language_name']:<15s} {r['omni_avg_stt_accuracy']:>9.0%} "
              f"{r['partner_avg_stt_accuracy']:>12.0%} {r['omni_avg_audio_duration']:>6.1f}s")

    # Save comparison
    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Comparison: {comparison_path}")


if __name__ == "__main__":
    asyncio.run(main())
