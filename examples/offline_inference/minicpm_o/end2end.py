# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This example shows how to use vLLM for running offline inference
with the correct prompt format on MiniCPM-o 4.5.

MiniCPM-o 4.5 is a 3-stage omni model:
  Stage 0 (Thinker):  text + image + audio → LLM hidden states
  Stage 1 (Talker):   TTS conditioning → codec tokens
  Stage 2 (Code2Wav): codec tokens → PCM audio

Supported input modalities: text, image, audio, video
Output: PCM audio (WAV)
"""

import os
import time
from typing import NamedTuple

import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from vllm import SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42

# NOTE: MiniCPM-o uses (<image>./</image>), (<audio>./</audio>),
# (<video>./</video>) as multimodal placeholders (not Qwen-style tokens).
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
    return QueryResult(
        inputs={"prompt": _build_prompt(question)},
        limit_mm_per_prompt={},
    )


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
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"image": image_data},
        },
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
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"audio": audio_data},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_image_audio_query(
    image_path: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    question = "Describe what you see in the image and what you hear in the audio."
    prompt = _build_prompt(f"(<image>./</image>)\n(<audio>./</audio>)\n{question}")

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image_data = Image.open(image_path).convert("RGB")
    else:
        image_data = ImageAsset("cherry_blossom").pil_image.convert("RGB")

    if audio_path:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"image": image_data, "audio": audio_data},
        },
        limit_mm_per_prompt={"image": 1, "audio": 1},
    )


query_map = {
    "text": get_text_query,
    "use_image": get_image_query,
    "use_audio": get_audio_query,
    "use_image_audio": get_image_audio_query,
}


def main(args):
    model_name = args.model

    audio_path = getattr(args, "audio_path", None)
    image_path = getattr(args, "image_path", None)

    query_func = query_map[args.query_type]
    if args.query_type == "use_image":
        query_result = query_func(image_path=image_path)
    elif args.query_type == "use_audio":
        query_result = query_func(
            audio_path=audio_path,
            sampling_rate=getattr(args, "sampling_rate", 16000),
        )
    elif args.query_type == "use_image_audio":
        query_result = query_func(
            image_path=image_path,
            audio_path=audio_path,
            sampling_rate=getattr(args, "sampling_rate", 16000),
        )
    else:
        query_result = query_func()

    omni = Omni(
        model=model_name,
        dtype=args.dtype,
        stage_configs_path=args.stage_configs_path,
        log_stats=args.log_stats,
        stage_init_timeout=args.stage_init_timeout,
        init_timeout=args.init_timeout,
    )

    # Sampling parameters — aligned with minicpmo.yaml defaults
    thinker_sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        top_k=20,
        max_tokens=2048,
        seed=SEED,
        repetition_penalty=1.05,
    )

    talker_sampling_params = SamplingParams(
        temperature=0.9,
        top_k=50,
        max_tokens=4096,
        seed=SEED,
        detokenize=False,
        stop_token_ids=[6561],  # Talker codec EOS
    )

    code2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        max_tokens=65536,
        seed=SEED,
        detokenize=True,
    )

    num_stages = omni.num_stages
    all_sampling_params = [
        thinker_sampling_params,
        talker_sampling_params,
        code2wav_sampling_params,
    ]
    sampling_params_list = all_sampling_params[:num_stages]

    if args.txt_prompts is None:
        prompts = [query_result.inputs for _ in range(args.num_prompts)]
    else:
        assert args.query_type == "text", "--txt-prompts is only supported for text query type"
        with open(args.txt_prompts, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        prompts = [get_text_query(ln).inputs for ln in lines]
        print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for prompt in prompts:
            prompt["modalities"] = output_modalities

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"query type: {args.query_type}")
    omni_generator = omni.generate(prompts, sampling_params_list, py_generator=args.py_generator)

    for stage_outputs in omni_generator:
        output = stage_outputs.request_output
        request_id = output.request_id

        if stage_outputs.final_output_type == "text":
            text_output = output.outputs[0].text
            out_txt = os.path.join(output_dir, f"{request_id}.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(f"Prompt:\n{output.prompt}\n\nResponse:\n{text_output.strip()}\n")
            print(f"[{request_id}] Text saved to {out_txt}")

        elif stage_outputs.final_output_type == "audio":
            audio_tensor = output.outputs[0].multimodal_output["audio"]
            output_wav = os.path.join(output_dir, f"output_{request_id}.wav")
            audio_numpy = audio_tensor.float().detach().cpu().numpy()
            if audio_numpy.ndim > 1:
                audio_numpy = audio_numpy.flatten()
            sf.write(output_wav, audio_numpy, samplerate=24000, format="WAV")
            print(f"[{request_id}] Audio saved to {output_wav}")

    omni.close()


def parse_args():
    parser = FlexibleArgumentParser(
        description="MiniCPM-o 4.5 offline inference via vllm-omni Omni class."
    )
    parser.add_argument(
        "--model", type=str, default="openbmb/MiniCPM-o-4_5",
        help="Model name or path.",
    )
    parser.add_argument(
        "--query-type", "-q", type=str, default="use_audio",
        choices=query_map.keys(),
        help="Query type: text, use_image, use_audio, use_image_audio.",
    )
    parser.add_argument(
        "--stage-configs-path", type=str, default=None,
        help="Path to a stage configs YAML file.",
    )
    parser.add_argument(
        "--log-stats", action="store_true", default=False,
    )
    parser.add_argument(
        "--stage-init-timeout", type=int, default=300,
    )
    parser.add_argument(
        "--init-timeout", type=int, default=300,
    )
    parser.add_argument(
        "--output-dir", type=str, default="output_audio",
        help="Directory to save output WAV and text files.",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--txt-prompts", type=str, default=None,
        help="Path to a .txt file with one prompt per line (text query only).",
    )
    parser.add_argument(
        "--py-generator", action="store_true", default=False,
        help="Use Python generator mode (recommended for large batches).",
    )
    parser.add_argument(
        "--modalities", type=str, default=None,
        help="Comma-separated output modalities: text, audio, or text,audio.",
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
    )
    parser.add_argument(
        "--audio-path", "-a", type=str, default=None,
        help="Path to local audio file. Uses default asset if not provided.",
    )
    parser.add_argument(
        "--image-path", "-i", type=str, default=None,
        help="Path to local image file. Uses default asset if not provided.",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=16000,
        help="Sampling rate for audio loading (default: 16000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
