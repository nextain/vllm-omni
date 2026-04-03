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
    im_end_id: int = 151645,
) -> tuple[int, int | None]:
    """Find the TTS content boundary in the thinker token sequence.

    Returns (start, end) where start is the position AFTER <|tts_bos|>
    and end is the position of the content boundary token.

    Priority (searched AFTER tts_bos only):
      1. <|tts_eos|> — ideal TTS boundary (original MiniCPM-o pattern)
      2. <|im_end|>  — fallback when model skips <|tts_eos|> and generates
                       <|im_end|> directly (common with standard vLLM decoding)
      3. None        — no boundary found at all

    Note: When <|tts_eos|> is not found, we use <|im_end|> as the
    fallback boundary. This handles the case where the model generates <im_end>
    instead of <tts_eos>.
    """
    tts_bos_idx = -1
    tts_eos_idx = None
    im_end_idx = None
    for i, tok in enumerate(token_ids):
        if tok == tts_bos_id:
            tts_bos_idx = i + 1  # +1 to skip the marker itself
        elif tts_bos_idx >= 0:
            # Only search for end markers AFTER tts_bos
            if tok == tts_eos_id:
                tts_eos_idx = i
                break  # tts_eos found — no need to continue
            elif tok == im_end_id and im_end_idx is None:
                im_end_idx = i
    if tts_bos_idx < 0:
        # No TTS boundary found — use entire sequence as fallback
        tts_bos_idx = 0
    # Prefer tts_eos, fall back to im_end (only after tts_bos)
    # Important: When tts_eos is None (not found), use im_end as fallback
    # This handles the case where the model generates <im_end> instead of <tts_eos>
    if tts_eos_idx is None:
        tts_eos_idx = im_end_idx
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
    hf_config = thinker_stage.vllm_config.model_config.hf_config
    tts_config = getattr(hf_config, "tts_config", None)
    tts_bos_id = getattr(tts_config, "tts_bos_token_id", 151703) if tts_config else 151703
    tts_eos_id = getattr(tts_config, "tts_eos_token_id", 151704) if tts_config else 151704

    for thinker_output in thinker_outputs:
        if not thinker_output.outputs:
            raise RuntimeError(
                "thinker stage produced empty outputs. Check that max_tokens > 0 "
                "and the model generated at least one token."
            )
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

        # Keys match MiniCPM-o's talker_preprocess expectations:
        #   thinker_token_ids  → emb_text(ids) for conditioning
        #   thinker_hidden_states → semantic_projection input
        # Note: differs from qwen3 (thinker_sequences/thinker_prefill_embeddings)
        # and qwen2.5 (thinker_result/prompt_embeds) — model-specific keys.
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
        if not talker_output.outputs:
            raise RuntimeError(
                "talker stage produced empty outputs. Check that max_tokens > 0 "
                "and the model generated at least one codec token."
            )
        output = talker_output.outputs[0]
        # MiniCPM-o uses num_vq=1 (single RVQ layer), so codec tokens are
        # directly in output.token_ids (flat 1D).  This differs from
        # qwen3_omni which extracts from multimodal_output["code_predictor_codes"]
        # because qwen3 uses multi-layer RVQ requiring transpose+reshape.
        # Strip trailing stop token (vllm includes stop_token_ids in output).
        codec_codes = list(output.token_ids[:-1]) if output.token_ids else []

        # Debug: Log codec tokens generated by Talker
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async chunk: stream codec tokens to Code2Wav as they're generated.

    MiniCPM-o uses num_vq=1 (single RVQ layer), so codec tokens are
    flat (1D). Each generated token is a single codec frame.

    Args:
        transfer_manager: Manages per-request token accumulation
        pooling_output: Current Talker output with codec tokens
        request: vLLM request object
        is_finished: Whether generation is complete

    Returns:
        dict with chunk info for Code2Wav, or None if not ready yet
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    # Initialize per-request token storage if needed
    if not hasattr(transfer_manager, "code_prompt_token_ids"):
        transfer_manager.code_prompt_token_ids = {}

    if request_id not in transfer_manager.code_prompt_token_ids:
        transfer_manager.code_prompt_token_ids[request_id] = []

    # Extract codec token from Talker output
    if isinstance(pooling_output, dict):
        token_ids = pooling_output.get("token_ids")
        if token_ids is not None:
            if isinstance(token_ids, torch.Tensor):
                # MiniCPM-o: single codec token per position (num_vq=1)
                # Skip stop token (6561)
                tokens = token_ids.cpu().tolist()
                tokens = [t for t in tokens if t != 6561]
                transfer_manager.code_prompt_token_ids[request_id].extend(tokens)
            elif isinstance(token_ids, list):
                tokens = [t for t in token_ids if t != 6561]
                transfer_manager.code_prompt_token_ids[request_id].extend(tokens)

    # Get chunk config from stage config
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 50))  # 50 codec frames per chunk
    left_context_size = int(cfg.get("codec_left_context_frames", 10))

    current_length = len(transfer_manager.code_prompt_token_ids[request_id])

    if current_length <= 0:
        if finished:
            return {
                "code_predictor_codes": [],
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return None

    # Emit chunks when we reach chunk_size or when finished
    # But ensure we have at least some tokens to work with
    if not finished and current_length % chunk_size != 0:
        return None

    # Determine chunk length with left context
    if finished:
        chunk_length = current_length
    else:
        chunk_length = min(chunk_size, current_length)

    # Calculate left context (frames before current chunk)
    start_index = max(0, current_length - left_context_size - chunk_length)
    chunk_start = current_length - chunk_length
    left_context_count = chunk_start - start_index

    # Extract window: left_context + current_chunk
    window_frames = transfer_manager.code_prompt_token_ids[request_id][
        start_index:chunk_start + chunk_length
    ]

    # For MiniCPM-o (num_vq=1), code_predictor_codes is just the flat list
    # Qwen3-TTS interleaves quantizers: [f0[q0], f0[q1], f1[q0], f1[q1], ...
    code_predictor_codes = window_frames

    info: dict[str, Any] = {
        "code_predictor_codes": code_predictor_codes,
        "left_context_size": left_context_count,
        "finished": torch.tensor(finished, dtype=torch.bool),
    }

    return info
