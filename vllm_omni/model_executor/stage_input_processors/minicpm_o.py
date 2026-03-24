# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The OpenBMB Team. All rights reserved.
"""Stage input processor for MiniCPM-o 4.5: Thinker → Talker → Code2Wav."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")
    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")
    return stage.engine_outputs


# =========================
# Thinker → Talker
# =========================


def _find_tts_bound(
    token_ids: list[int],
    tts_bos_id: int = 151703,
    tts_eos_id: int = 151704,
) -> tuple[int, int | None]:
    """Find the TTS content boundary in the thinker token sequence.

    Returns (start, end) where start is the position AFTER <|tts_bos|>
    and end is the position OF <|tts_eos|> (or None if not found).
    This matches the original MiniCPM-o slicing: [tts_bos_idx+1 : tts_eos_idx].
    """
    tts_bos_idx = -1
    tts_eos_idx = None
    for i, tok in enumerate(token_ids):
        if tok == tts_bos_id:
            tts_bos_idx = i + 1  # +1 to skip the marker itself
        elif tok == tts_eos_id:
            tts_eos_idx = i
    if tts_bos_idx < 0:
        # No TTS boundary found — use entire sequence as fallback
        tts_bos_idx = 0
    return tts_bos_idx, tts_eos_idx


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: "OmniTokensPrompt | TextPrompt | None" = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build talker inputs from thinker outputs.

    Extracts ``thinker_hidden_states`` and thinker token IDs from the
    thinker's output, applies tts_bound filtering (only the content between
    <|tts_bos|> and <|tts_eos|>), and packages them as
    ``additional_information`` for the talker stage.

    The talker's ``talker_preprocess`` uses these to build
    hidden_text_merge conditioning (matching original MiniCPM-o):
        tts_embeds = emb_text(token_ids) + normalize(semantic_projection(hidden_states))
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]

        thinker_hidden_states = output.multimodal_output.get("thinker_hidden_states")
        if thinker_hidden_states is None:
            raise RuntimeError(
                "thinker stage did not emit 'thinker_hidden_states' in "
                "multimodal_output. Check that model_stage='thinker' is "
                "running with the MiniCPMOForConditionalGeneration model."
            )

        # Reconstruct full token sequence (prompt + generated)
        full_token_ids = list(thinker_output.prompt_token_ids) + list(output.token_ids)

        # Find TTS content boundary: [tts_bos+1 : tts_eos]
        # Original only passes the speech-relevant portion to the talker.
        tts_start, tts_end = _find_tts_bound(full_token_ids)

        # Slice hidden states and token IDs to TTS-relevant portion
        tts_token_ids = full_token_ids[tts_start:tts_end]
        tts_hidden = thinker_hidden_states[tts_start:tts_end]

        num_tts_tokens = len(tts_token_ids)
        if num_tts_tokens == 0:
            # Fallback: no TTS markers found, use all tokens
            tts_token_ids = full_token_ids
            tts_hidden = thinker_hidden_states
            num_tts_tokens = len(tts_token_ids)

        info: dict[str, Any] = {
            "thinker_token_ids": torch.tensor(
                tts_token_ids, dtype=torch.long, device=device,
            ),
            "thinker_hidden_states": (
                tts_hidden.detach().to(device=device, dtype=torch.float)
            ),
        }

        # +2 for text_eos and audio_bos boundary tokens appended in talker_preprocess
        talker_prompt_len = num_tts_tokens + 2

        talker_inputs.append(
            OmniTokensPrompt(
                # Placeholder IDs — talker_preprocess will replace them with
                # conditioning embeddings before the Llama backbone runs.
                prompt_token_ids=[0] * talker_prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: "OmniEngineCoreRequest",
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Streaming (async-chunk) variant of thinker2talker.

    Called once per thinker decode step to pass freshly generated token
    embeddings to the talker stage without waiting for the full thinker run.

    NOTE: This function is not registered in pipeline.yaml (async_chunk=false)
    and needs to be updated to use thinker_token_ids + build_conditioning()
    when async_chunk mode is enabled.  Currently kept for future use.
    """
    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk.get(request_id, 0)

    thinker_text_embeds = pooling_output.get("thinker_text_embeds")
    thinker_hidden_states = pooling_output.get("thinker_hidden_states")

    if thinker_text_embeds is None and thinker_hidden_states is None:
        return None

    if chunk_id == 0:
        all_token_ids = list(getattr(request, "all_token_ids", []))
        prompt_token_ids = list(getattr(request, "prompt_token_ids", []))
        talker_info: dict[str, Any] = {
            "finished": torch.tensor(is_finished, dtype=torch.bool),
        }
        if thinker_text_embeds is not None:
            talker_info["thinker_text_embeds"] = thinker_text_embeds.detach().cpu()
        if thinker_hidden_states is not None:
            talker_info["thinker_hidden_states"] = thinker_hidden_states.detach().cpu()
        talker_info["thinker_sequences"] = all_token_ids
        talker_info["thinker_input_ids"] = prompt_token_ids
    else:
        talker_info = {
            "finished": torch.tensor(is_finished, dtype=torch.bool),
        }
        output_token_ids = list(getattr(request, "output_token_ids", []))
        if output_token_ids:
            talker_info["override_keys"] = [
                "thinker_decode_text_embeds",
                "thinker_decode_hidden_states",
                "thinker_output_token_ids",
            ]
            if thinker_text_embeds is not None:
                talker_info["thinker_decode_text_embeds"] = thinker_text_embeds.detach().cpu()
            if thinker_hidden_states is not None:
                talker_info["thinker_decode_hidden_states"] = thinker_hidden_states.detach().cpu()
            talker_info["thinker_output_token_ids"] = output_token_ids

    return talker_info


# =========================
# Talker → Code2Wav
# =========================


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: "OmniTokensPrompt | TextPrompt | None" = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build code2wav inputs from talker codec-token outputs.

    MiniCPM-o uses num_vq=1 (single RVQ layer), so the codec tokens are
    flat (1D).  Code2Wav expects them flattened as the prompt_token_ids.
    """
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []

    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        # All generated tokens are valid codec tokens (audio_bos is in the
        # conditioning, not in the generated output). EOS token (6561) is
        # excluded by the stop_token_ids mechanism.
        codec_codes = list(output.token_ids)
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
