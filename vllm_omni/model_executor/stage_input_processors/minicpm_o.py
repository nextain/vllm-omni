# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 The OpenBMB Team. All rights reserved.
"""Stage input processor for MiniCPM-o 4.5: Thinker → Talker → Code2Wav."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

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
    tts_bos_id: int,
    tts_eos_id: int,
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

    # Read TTS boundary token IDs from model config (MiniCPMTTSConfig)
    # Fall back to MiniCPM-o 4.5 defaults if config doesn't have the fields
    # (e.g. when HF config.json is loaded via --trust-remote-code from upstream)
    thinker_stage = stage_list[engine_input_source[0]]
    tts_config = thinker_stage.vllm_config.model_config.hf_config.tts_config
    tts_bos_id = getattr(tts_config, "tts_bos_token_id", 151703)
    tts_eos_id = getattr(tts_config, "tts_eos_token_id", 151704)

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
        tts_start, tts_end = _find_tts_bound(full_token_ids, tts_bos_id, tts_eos_id)

        # Slice hidden states and token IDs to TTS-relevant portion.
        # Use hidden_states.shape[0] as authoritative length — token_ids
        # may differ by 1 due to EOS inclusion/exclusion in output.token_ids.
        hidden_len = thinker_hidden_states.shape[0]
        token_len = len(full_token_ids)

        # Clamp tts_start/tts_end to hidden_states bounds
        if tts_start >= hidden_len:
            tts_start = 0
            tts_end = None
        if tts_end is not None and tts_end > hidden_len:
            tts_end = hidden_len

        tts_hidden = thinker_hidden_states[tts_start:tts_end]
        tts_token_ids = full_token_ids[tts_start:tts_end]

        # Align lengths: trim the longer one to match the shorter
        num_tts_tokens = min(len(tts_token_ids), tts_hidden.shape[0])
        tts_token_ids = tts_token_ids[:num_tts_tokens]
        tts_hidden = tts_hidden[:num_tts_tokens]

        if num_tts_tokens == 0:
            # Fallback: no TTS markers found — use only the GENERATED tokens
            # (not the prompt), since those are the actual response.
            # thinker_hidden_states covers the full sequence (prompt + generated).
            prompt_len = len(thinker_output.prompt_token_ids)
            gen_len = len(output.token_ids)
            # Slice hidden states to the generated portion only
            gen_start = min(prompt_len, hidden_len)
            gen_end = min(prompt_len + gen_len, hidden_len)
            if gen_end > gen_start:
                tts_token_ids = list(output.token_ids[: gen_end - gen_start])
                tts_hidden = thinker_hidden_states[gen_start:gen_end]
                num_tts_tokens = len(tts_token_ids)
            else:
                # Last resort: use all tokens
                num_tts_tokens = min(token_len, hidden_len)
                tts_token_ids = full_token_ids[:num_tts_tokens]
                tts_hidden = thinker_hidden_states[:num_tts_tokens]

        info: dict[str, Any] = {
            "thinker_token_ids": torch.tensor(
                tts_token_ids, dtype=torch.long, device=device,
            ),
            # Cast to float32 for serialization (numpy doesn't support bfloat16)
            "thinker_hidden_states": tts_hidden.detach().float().to(device=device),
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
        # vllm includes stop_token_ids in output.token_ids, so strip the
        # trailing stop token before passing to Code2Wav.  This matches
        # the qwen3_omni pattern: seq_len = len(output.token_ids) - 1.
        codec_codes = list(output.token_ids[:-1]) if output.token_ids else []
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
