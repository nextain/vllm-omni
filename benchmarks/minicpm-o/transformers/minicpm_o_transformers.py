"""HF Transformers baseline benchmark for MiniCPM-o 4.5.

Runs text-to-speech inference using openbmb's MiniCPM-o pipeline and
collects per-stage latency + throughput metrics.

Usage:
    python minicpm_o_transformers.py \
        --prompts-file ../../build_dataset/top100.txt \
        --num-prompts 10
"""

import argparse
import json
import os
import time

import soundfile as sf
import torch
from tqdm import tqdm

MODEL_PATH = "openbmb/MiniCPM-o-4_5"
SAMPLE_RATE = 24000  # CosyVoice2 output sample rate


def load_prompts(prompts_file: str) -> list[str]:
    prompts = []
    with open(prompts_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def run_benchmark(
    model,
    tokenizer,
    prompts: list[str],
    output_dir: str = "benchmark_results",
):
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    all_stats = []
    results = []

    for idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        # Thinker stage
        t0 = time.perf_counter()
        response = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
            omni_output="audio",
        )
        t_total = time.perf_counter() - t0

        # Extract outputs
        text_output = response.text if hasattr(response, "text") else str(response)
        audio_data = None
        audio_path = None

        if hasattr(response, "audio") and response.audio is not None:
            audio_tensor = response.audio
            if isinstance(audio_tensor, torch.Tensor):
                audio_data = audio_tensor.reshape(-1).detach().cpu().numpy()
            else:
                audio_data = audio_tensor

            audio_path = os.path.join(audio_dir, f"output_{idx:04d}.wav")
            sf.write(audio_path, audio_data, samplerate=SAMPLE_RATE)

        # Collect stats
        audio_duration = len(audio_data) / SAMPLE_RATE if audio_data is not None else 0
        rtf = t_total / audio_duration if audio_duration > 0 else float("inf")

        stats = {
            "prompt_idx": idx,
            "prompt": prompt,
            "total_time_s": t_total,
            "audio_duration_s": audio_duration,
            "rtf": rtf,
        }
        all_stats.append(stats)

        results.append({
            "idx": idx,
            "prompt": prompt,
            "output": text_output,
            "audio_path": audio_path,
            "perf_stats": stats,
        })

    # Aggregate
    aggregated = aggregate_stats(all_stats)

    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "perf_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"aggregated": aggregated, "per_prompt": all_stats}, f, ensure_ascii=False, indent=2)

    num_audio = sum(1 for s in all_stats if s["audio_duration_s"] > 0)
    print(f"\nSaved {num_audio} audio files to {audio_dir}/")

    return aggregated, results


def aggregate_stats(all_stats: list[dict]) -> dict:
    if not all_stats:
        return {}

    agg = {"num_samples": len(all_stats)}

    for key in ["total_time_s", "audio_duration_s", "rtf"]:
        values = [s[key] for s in all_stats if s[key] != float("inf")]
        if values:
            agg[f"{key}_sum"] = sum(values)
            agg[f"{key}_avg"] = sum(values) / len(values)
            agg[f"{key}_min"] = min(values)
            agg[f"{key}_max"] = max(values)
            # Percentiles
            sorted_v = sorted(values)
            n = len(sorted_v)
            agg[f"{key}_p50"] = sorted_v[n // 2]
            agg[f"{key}_p90"] = sorted_v[int(n * 0.9)]
            agg[f"{key}_p95"] = sorted_v[int(n * 0.95)]
            agg[f"{key}_p99"] = sorted_v[min(int(n * 0.99), n - 1)]

    total_audio = agg.get("audio_duration_s_sum", 0)
    total_time = agg.get("total_time_s_sum", 0)
    if total_time > 0:
        agg["overall_rtf"] = total_time / total_audio if total_audio > 0 else float("inf")
        agg["audio_throughput_ratio"] = total_audio / total_time

    return agg


def print_stats(stats: dict):
    print("\n" + "=" * 60)
    print("MiniCPM-o 4.5 HF Transformers Benchmark Results")
    print("=" * 60)
    print(f"\nSamples: {stats.get('num_samples', 0)}")

    print("\n--- Latency ---")
    print(f"  Avg E2E:    {stats.get('total_time_s_avg', 0):.2f}s")
    print(f"  P50:        {stats.get('total_time_s_p50', 0):.2f}s")
    print(f"  P90:        {stats.get('total_time_s_p90', 0):.2f}s")
    print(f"  P95:        {stats.get('total_time_s_p95', 0):.2f}s")

    print("\n--- Real-Time Factor (RTF) ---")
    print(f"  Avg RTF:    {stats.get('rtf_avg', 0):.3f}")
    print(f"  Overall:    {stats.get('overall_rtf', 0):.3f}")
    print(f"  (RTF < 1.0 = faster than realtime)")

    print("\n--- Audio ---")
    print(f"  Total audio: {stats.get('audio_duration_s_sum', 0):.1f}s")
    print(f"  Total time:  {stats.get('total_time_s_sum', 0):.1f}s")
    print(f"  Throughput:  {stats.get('audio_throughput_ratio', 0):.2f}x realtime")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="MiniCPM-o 4.5 HF Benchmark")
    parser.add_argument("--prompts-file", type=str,
                        default="../../build_dataset/top100.txt")
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--num-prompts", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        init_audio=True,
        init_tts=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True,
    )

    prompts = load_prompts(args.prompts_file)
    if args.num_prompts:
        prompts = prompts[:args.num_prompts]

    print(f"Running benchmark on {len(prompts)} prompts...")
    aggregated, results = run_benchmark(model, tokenizer, prompts, args.output_dir)
    print_stats(aggregated)


if __name__ == "__main__":
    main()
