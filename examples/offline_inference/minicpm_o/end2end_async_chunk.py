# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline inference with async_chunk enabled via AsyncOmni for MiniCPM-o 4.5.

This script uses AsyncOmni (the async orchestrator) to run offline inference
with async_chunk semantics: Talker and Code2Wav stages start *before* Thinker
finishes, consuming hidden state chunks as they arrive via the in-worker
OmniChunkTransferAdapter / SharedMemoryConnector.

Compared to the synchronous ``end2end.py`` (which uses ``Omni``), this
entry point achieves true stage-level concurrency — TTFP drops from ~6.5s
to ~0.07s on 2× RTX 3090.

Usage
-----
    python end2end_async_chunk.py --query-type use_audio \\
        --stage-configs-path <path-to-minicpmo_async_chunk.yaml>

See ``--help`` for all options.
"""

import asyncio
import os
import time
import uuid
from typing import NamedTuple

import librosa
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.async_omni import AsyncOmni

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SEED = 42

default_system = (
    "You are a helpful multimodal AI assistant capable of understanding "
    "text, images, and audio, and generating spoken responses."
)


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


def _build_prompt(user_content: str) -> str:
    return (
        f"<|im_start|>system\n{default_system}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_text_query(question: str = None) -> QueryResult:
    if question is None:
        question = "Describe the key features of a multimodal AI assistant in 15 words."
    return QueryResult(inputs={"prompt": _build_prompt(question)}, limit_mm_per_prompt={})


def get_image_query(question: str = None, image_path: str | None = None) -> QueryResult:
    if question is None:
        question = "What is the content of this image?"
    prompt = _build_prompt(f"(<image>./</image>)\n{question}")
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = Image.open(image_path).convert("RGB")
    else:
        image_data = ImageAsset("cherry_blossom").pil_image.convert("RGB")
    return QueryResult(
        inputs={"prompt": prompt, "multi_modal_data": {"image": image_data}},
        limit_mm_per_prompt={"image": 1},
    )


def get_audio_query(
    question: str = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    if question is None:
        question = "What is being said in this audio?"
    prompt = _build_prompt(f"(<audio>./</audio>)\n{question}")
    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate
    return QueryResult(
        inputs={"prompt": prompt, "multi_modal_data": {"audio": audio_data}},
        limit_mm_per_prompt={"audio": 1},
    )


query_map = {
    "text": get_text_query,
    "use_image": get_image_query,
    "use_audio": get_audio_query,
}


def _default_async_chunk_stage_configs_path() -> str | None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    candidate = os.path.join(
        repo_root, "vllm_omni", "model_executor", "stage_configs", "minicpmo_async_chunk.yaml"
    )
    return candidate if os.path.exists(candidate) else None


def clone_prompt_for_request(template: dict) -> dict:
    cloned = dict(template)
    for key in ("multi_modal_data", "mm_processor_kwargs"):
        value = template.get(key)
        if isinstance(value, dict):
            cloned[key] = dict(value)
        elif isinstance(value, list):
            cloned[key] = list(value)
    return cloned


async def run_single_request(
    async_omni: AsyncOmni,
    prompt: dict,
    request_id: str,
    sampling_params_list,
    output_dir: str,
    output_modalities: list[str] | None = None,
    stream_audio_to_disk: bool = False,
) -> dict:
    t_start = time.perf_counter()
    text_parts: list[str] = []
    audio_chunks: list[torch.Tensor] = []
    audio_sr: int | None = None
    first_audio_ts: float | None = None
    audio_list_consumed: int = 0
    samplerate = 24000

    wav_file = os.path.join(output_dir, f"output_{request_id}.wav")
    sf_writer = None
    audio_samples_written = 0

    try:
        async for omni_output in async_omni.generate(
            prompt=prompt,
            request_id=request_id,
            sampling_params_list=sampling_params_list,
            output_modalities=output_modalities,
        ):
            output = omni_output.request_output
            if omni_output.final_output_type == "text":
                if output.finished:
                    text_parts.append(output.outputs[0].text)
            elif omni_output.final_output_type == "audio":
                mm_out = output.outputs[0].multimodal_output
                if mm_out and "audio" in mm_out:
                    if first_audio_ts is None:
                        first_audio_ts = time.perf_counter()
                    if audio_sr is None and "sr" in mm_out:
                        sr_val = mm_out["sr"]
                        audio_sr = sr_val.item() if hasattr(sr_val, "item") else int(sr_val)
                        samplerate = audio_sr
                    audio_data = mm_out["audio"]
                    if isinstance(audio_data, list):
                        new_chunks = audio_data[audio_list_consumed:]
                        audio_list_consumed = len(audio_data)
                    elif isinstance(audio_data, torch.Tensor):
                        new_chunks = [audio_data]
                    else:
                        new_chunks = []

                    if stream_audio_to_disk and new_chunks:
                        if sf_writer is None:
                            sf_writer = sf.SoundFile(
                                wav_file, mode="w", samplerate=samplerate,
                                channels=1, subtype="FLOAT",
                            )
                        for chunk in new_chunks:
                            chunk_np = chunk.float().detach().cpu().numpy().flatten()
                            sf_writer.write(chunk_np)
                            audio_samples_written += len(chunk_np)
                    else:
                        audio_chunks.extend(new_chunks)
    finally:
        if sf_writer is not None:
            sf_writer.close()

    t_end = time.perf_counter()
    result = {"request_id": request_id, "e2e_latency_s": t_end - t_start, "saved_files": []}

    if text_parts:
        text_file = os.path.join(output_dir, f"{request_id}.txt")
        with open(text_file, "w", encoding="utf-8") as f:
            f.write("\n".join(text_parts))
        result["saved_files"].append(text_file)
        print(f"[{request_id}] Text saved to {text_file}")

    ttfa = (first_audio_ts - t_start) if first_audio_ts else None
    ttfa_str = f"{ttfa:.3f}s" if ttfa is not None else "N/A"

    if stream_audio_to_disk and audio_samples_written > 0:
        result["saved_files"].append(wav_file)
        result["audio_duration_s"] = audio_samples_written / samplerate
        print(
            f"[{request_id}] Audio streamed to {wav_file} "
            f"(duration={result['audio_duration_s']:.2f}s, TTFA={ttfa_str}, e2e={t_end - t_start:.3f}s)"
        )
    elif audio_chunks:
        if len(audio_chunks) > 1:
            audio_tensor = torch.cat(audio_chunks, dim=-1)
        else:
            audio_tensor = audio_chunks[0]
        audio_numpy = audio_tensor.float().detach().cpu().numpy()
        if audio_numpy.ndim > 1:
            audio_numpy = audio_numpy.flatten()
        sf.write(wav_file, audio_numpy, samplerate=samplerate, format="WAV")
        result["saved_files"].append(wav_file)
        result["audio_duration_s"] = len(audio_numpy) / samplerate
        print(
            f"[{request_id}] Audio saved to {wav_file} "
            f"(duration={result['audio_duration_s']:.2f}s, TTFA={ttfa_str}, e2e={t_end - t_start:.3f}s)"
        )

    result["time_to_first_audio_s"] = ttfa
    return result


async def run_all(args):
    query_func = query_map[args.query_type]
    if args.query_type == "use_image":
        query_result = query_func(image_path=getattr(args, "image_path", None))
    elif args.query_type == "use_audio":
        query_result = query_func(
            audio_path=getattr(args, "audio_path", None),
            sampling_rate=getattr(args, "sampling_rate", 16000),
        )
    else:
        query_result = query_func()

    if args.txt_prompts is not None:
        assert args.query_type == "text", "--txt-prompts is only supported for text query type"
        with open(args.txt_prompts, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        prompts = [get_text_query(ln).inputs for ln in lines]
        print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")
    else:
        prompts = [clone_prompt_for_request(query_result.inputs) for _ in range(args.num_prompts)]

    output_modalities = None
    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for prompt in prompts:
            prompt["modalities"] = output_modalities

    async_omni = None
    try:
        async_omni = AsyncOmni(
            model=args.model,
            stage_configs_path=args.stage_configs_path,
            log_stats=args.log_stats,
            stage_init_timeout=args.stage_init_timeout,
        )

        # Use sampling params from stage config YAML (pre-configured for MiniCPM-o)
        sampling_params_list = None

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        semaphore = asyncio.Semaphore(args.max_in_flight)
        stream_audio = getattr(args, "stream_audio_to_disk", False)
        request_timeout = getattr(args, "request_timeout_s", None)

        async def _run_one(idx: int, prompt: dict):
            async with semaphore:
                request_id = f"req_{idx}_{uuid.uuid4().hex[:8]}"
                coro = run_single_request(
                    async_omni=async_omni,
                    prompt=prompt,
                    request_id=request_id,
                    sampling_params_list=sampling_params_list,
                    output_dir=output_dir,
                    output_modalities=output_modalities,
                    stream_audio_to_disk=stream_audio,
                )
                if request_timeout and request_timeout > 0:
                    return await asyncio.wait_for(coro, timeout=request_timeout)
                return await coro

        wall_start = time.perf_counter()
        tasks = [_run_one(i, p) for i, p in enumerate(prompts)]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_end = time.perf_counter()

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        success_count = 0
        total_audio_dur = 0.0
        for r in all_results:
            if isinstance(r, Exception):
                print(f"  [ERROR] {type(r).__name__}: {r}")
            else:
                success_count += 1
                total_audio_dur += r.get("audio_duration_s", 0.0)
                ttfa = r.get("time_to_first_audio_s")
                ttfa_str = f"{ttfa:.3f}s" if ttfa is not None else "N/A"
                print(
                    f"  [{r['request_id']}] e2e={r['e2e_latency_s']:.3f}s "
                    f"TTFA={ttfa_str}  files={r['saved_files']}"
                )
        wall_time = wall_end - wall_start
        print(f"\nTotal: {success_count}/{len(prompts)} succeeded")
        print(f"Wall time: {wall_time:.3f}s")
        if total_audio_dur > 0:
            print(f"Total audio duration: {total_audio_dur:.2f}s")
            print(f"Real-time factor: {total_audio_dur / wall_time:.2f}x")
        print("=" * 60)
    finally:
        if async_omni is not None:
            async_omni.shutdown()


def parse_args():
    parser = FlexibleArgumentParser(
        description=(
            "MiniCPM-o 4.5 offline inference with async_chunk via AsyncOmni. "
            "Talker and Code2Wav start before Thinker finishes for low TTFP (~0.07s)."
        )
    )
    parser.add_argument("--model", type=str, default="openbmb/MiniCPM-o-4_5")
    parser.add_argument(
        "--query-type", "-q", type=str, default="use_audio",
        choices=query_map.keys(),
    )
    parser.add_argument(
        "--stage-configs-path", type=str,
        default=_default_async_chunk_stage_configs_path(),
        help="Path to minicpmo_async_chunk.yaml (must have async_chunk: true).",
    )
    parser.add_argument("--log-stats", action="store_true", default=False)
    parser.add_argument("--stage-init-timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default="output_audio_async_chunk")
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument("--txt-prompts", type=str, default=None)
    parser.add_argument("--max-in-flight", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=None)
    parser.add_argument("--batch-timeout-s", type=float, default=None)
    parser.add_argument("--stream-audio-to-disk", action="store_true", default=False)
    parser.add_argument(
        "--modalities", type=str, default=None,
        help="Comma-separated output modalities: text, audio, or text,audio.",
    )
    parser.add_argument("--audio-path", "-a", type=str, default=None)
    parser.add_argument("--image-path", "-i", type=str, default=None)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    async def _main():
        batch_timeout = getattr(args, "batch_timeout_s", None)
        if batch_timeout and batch_timeout > 0:
            await asyncio.wait_for(run_all(args), timeout=batch_timeout)
        else:
            await run_all(args)

    try:
        asyncio.run(_main())
    except asyncio.TimeoutError:
        print(f"\n[TIMEOUT] Batch exceeded --batch-timeout-s={args.batch_timeout_s}s.")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Keyboard interrupt received.")
