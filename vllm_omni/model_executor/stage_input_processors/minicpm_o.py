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


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: "OmniTokensPrompt | TextPrompt | None" = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Build talker inputs from thinker outputs.

    Extracts ``thinker_text_embeds`` and ``thinker_hidden_states`` from the
    thinker's multimodal_output dict and packages them as ``additional_information``
    for the talker stage.  The talker's ``talker_preprocess`` uses these to build
    hidden_text_merge conditioning:

        talker_input[t] = text_projection(thinker_text_embeds[t])
                        + hidden_projection(thinker_hidden_states[t])
                        + codec_embedding(input_ids[t])

    The talker's prompt length equals the total number of thinker tokens
    (prompt + generated), because there is a 1:1 correspondence between
    thinker positions and talker codec-generating positions.
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]

        # multimodal_output is already sliced per-request and moved to CPU
        # by gpu_ar_model_runner.  Shape: [num_thinker_tokens, thinker_hidden].
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[DBG thinker2talker] output type=%s, hasattr(output,'multimodal_output')=%s, "
            "multimodal_output=%s",
            type(output).__name__,
            hasattr(output, "multimodal_output"),
            getattr(output, "multimodal_output", "MISSING"),
        )
        thinker_text_embeds = output.multimodal_output.get("thinker_text_embeds")
        thinker_hidden_states = output.multimodal_output.get("thinker_hidden_states")

        if thinker_text_embeds is None and thinker_hidden_states is None:
            raise RuntimeError(
                "thinker stage did not emit 'thinker_text_embeds' or "
                "'thinker_hidden_states' in multimodal_output.  "
                "Check that model_stage='thinker' is running with the "
                "MiniCPMOForConditionalGeneration model."
            )

        # Total thinker tokens = prompt tokens + generated tokens
        total_thinker_tokens = len(thinker_output.prompt_token_ids) + len(output.token_ids)

        info: dict[str, Any] = {}
        if thinker_text_embeds is not None:
            info["thinker_text_embeds"] = (
                thinker_text_embeds.detach().to(device=device, dtype=torch.float)
            )
        if thinker_hidden_states is not None:
            info["thinker_hidden_states"] = (
                thinker_hidden_states.detach().to(device=device, dtype=torch.float)
            )

        talker_inputs.append(
            OmniTokensPrompt(
                # Placeholder IDs — talker_preprocess will replace them with
                # projected thinker embeddings before the Llama backbone runs.
                prompt_token_ids=[0] * total_thinker_tokens,
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
        # Skip BOS token (first generated token is the codec BOS sentinel)
        codec_codes = list(output.token_ids[1:])
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
