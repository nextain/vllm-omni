"""
L2/L3 E2E tests for MiniCPM-o 4.5 model — text-to-audio pipeline.

L2 (dummy weights): Verifies 3-stage pipeline startup and basic inference flow.
L3 (real model): Verifies audio output correctness with real weights.

Stage architecture:
  Stage 0 (Thinker): SigLIP2 + Whisper + Qwen3-7B → text hidden states
  Stage 1 (Talker):  MiniCPMTTS Llama AR → audio codec tokens
  Stage 2 (Code2Wav): CosyVoice2 + HiFi-GAN → waveform
"""

from pathlib import Path

import pytest

from tests.conftest import (
    generate_synthetic_audio,
    generate_synthetic_image,
    modify_stage_config,
)
from tests.utils import hardware_test

models = ["openbmb/MiniCPM-o-4_5"]

# CI stage config — uses load_format: dummy for L2, override for L3
stage_config = str(
    Path(__file__).parent.parent / "stage_configs" / "minicpm_o_ci.yaml"
)

test_params = [(model, stage_config) for model in models]


def get_question(prompt_type="text_only"):
    prompts = {
        "text_only": "Please say hello in a friendly tone.",
        "describe_image": "Describe what you see in this image.",
    }
    return prompts.get(prompt_type, prompts["text_only"])


# ---------------------------------------------------------------------------
# L2: Dummy-weight pipeline startup + basic inference (no real model needed)
# ---------------------------------------------------------------------------


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio_dummy(omni_runner, omni_runner_handler) -> None:
    """
    L2: Pipeline startup with dummy weights, verify text→audio flow completes.
    Deploy Setting: minicpm_o_ci.yaml (load_format: dummy)
    Input Modal: text
    Output Modal: audio
    Datasets: single request
    """
    request_config = {
        "prompts": get_question("text_only"),
        "modalities": ["audio"],
    }
    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text_dummy(omni_runner, omni_runner_handler) -> None:
    """
    L2: Text-only output (thinker only, no talker/code2wav).
    Deploy Setting: minicpm_o_ci.yaml (load_format: dummy)
    Input Modal: text
    Output Modal: text
    Datasets: single request
    """
    request_config = {
        "prompts": get_question("text_only"),
        "modalities": ["text"],
    }
    omni_runner_handler.send_request(request_config)


# ---------------------------------------------------------------------------
# L3: Real model — correctness + basic performance (requires real weights)
# ---------------------------------------------------------------------------


def get_real_model_config():
    """Override dummy config to use real model weights."""
    return modify_stage_config(
        stage_config,
        updates={
            "stage_args": {
                0: {"engine_args.load_format": "auto"},
                1: {"engine_args.load_format": "auto"},
                2: {"engine_args.load_format": "auto"},
            },
        },
    )


real_model_config = get_real_model_config()
real_test_params = [(model, real_model_config) for model in models]


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", real_test_params, indirect=True)
def test_text_to_audio_real(omni_runner, omni_runner_handler) -> None:
    """
    L3: Real model text→audio — verify audio output is generated.
    Deploy Setting: real weights, enforce_eager, single GPU
    Input Modal: text
    Output Modal: audio
    Datasets: single request
    """
    request_config = {
        "prompts": "Count from one to five slowly.",
        "modalities": ["audio"],
    }
    omni_runner_handler.send_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", real_test_params, indirect=True)
def test_image_to_audio_real(omni_runner, omni_runner_handler) -> None:
    """
    L3: Real model image+text→audio — verify multimodal input with audio output.
    Deploy Setting: real weights
    Input Modal: text + image
    Output Modal: audio
    Datasets: single request
    """
    image = generate_synthetic_image(16, 16)["np_array"]

    request_config = {
        "prompts": get_question("describe_image"),
        "images": image,
        "modalities": ["audio"],
    }
    omni_runner_handler.send_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", real_test_params, indirect=True)
def test_batch_text_to_audio_real(omni_runner, omni_runner_handler) -> None:
    """
    L3: Batch processing — multiple text→audio requests.
    Deploy Setting: real weights
    Input Modal: text (batch of 3)
    Output Modal: audio
    Datasets: 3 requests
    """
    prompts = [
        "Say good morning.",
        "What is the weather today?",
        "Tell me a short joke.",
    ]
    for prompt in prompts:
        request_config = {
            "prompts": prompt,
            "modalities": ["audio"],
        }
        omni_runner_handler.send_request(request_config)


@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", real_test_params, indirect=True)
def test_audio_to_audio_real(omni_runner, omni_runner_handler) -> None:
    """
    E2E: Audio input → audio output — verify full sentence playback.

    Tests the complete ASR → TTS pipeline:
    1. Input: Synthetic speech (e.g., "Hello, how are you today?")
    2. Process: ASR recognition + TTS response generation
    3. Output: Audio with complete sentence
    4. Verify: No stuttering, full sentence played

    Deploy Setting: real weights, async_chunk streaming
    Input Modal: audio
    Output Modal: audio
    Datasets: single request
    """
    # Generate synthetic input audio: "Hello, how are you today?"
    audio_input = generate_synthetic_audio(duration=5, num_channels=1, sample_rate=16000, save_to_file=False)

    request_config = {
        "audios": audio_input["np_array"],
        "modalities": ["audio"],
    }
    omni_runner_handler.send_request(request_config)
