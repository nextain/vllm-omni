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


def _ensure_list(x: Any) -> list:
    """Convert ConstantList / tensor-like to a plain Python list.

    vLLM may wrap token ID sequences in a ConstantList (with a ``_x`` attribute).
    Based on qwen3_omni._ensure_list; differs in that we always convert to a true
    Python list (``list(x)``) rather than returning non-list types as-is.
    """
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return x
    return list(x)


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


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async chunk: accumulate Thinker hidden states until generation is done.

    Each forward step, vLLM's model runner slices ``pooling_output["thinker_hidden_states"]``
    to only the tokens processed in that step (1 token per decode step).  We must
    concatenate these slices across all steps to reconstruct the full-sequence hidden
    states that the Talker's ``talker_preprocess`` expects.

    MiniCPM-o differs from Qwen3-Omni in two ways:
    1. Hidden states are under a single ``thinker_hidden_states`` key
       (Qwen3 splits into prefill layer "0" and decode layer "24").
    2. Only the TTS-relevant portion (between <|tts_bos|> and <|tts_eos|>)
       should be passed to the Talker — ``_find_tts_bound()`` handles this.

    Accumulation strategy (matches Qwen3's ``request_payload`` pattern):
      - Each step: append this step's hidden states into ``request_payload[request_id]``.
      - Returns None until ``is_finished=True`` (Talker needs the full slice).
      - When finished: concatenate all accumulated slices, apply tts_bound, emit.

    Known limitation: chunked prefill (where ``is_finished=False`` but
    ``chunk_id==0`` spans multiple forward passes) will accumulate prefill
    hidden states alongside decode hidden states without distinction.
    This is safe when ``enable_prefix_caching=false`` and chunked prefill is
    not active (current configuration).  If chunked prefill is ever enabled,
    a ``put_req_chunk[request_id]`` check (as in Qwen3) would be needed.

    Args:
        transfer_manager: Per-request state manager (holds ``request_payload`` dict).
        pooling_output: Current-step Thinker output (``thinker_hidden_states`` = this step only).
        request: vLLM request object (provides token IDs).
        is_finished: Whether Thinker generation is complete.

    Returns:
        dict payload for Talker when finished, or None while accumulating.
    """
    request_id = request.external_req_id

    # Accumulate this step's hidden states.
    # Each step provides only the tokens processed in that step (typically 1 for decode).
    # We concatenate across all steps to reconstruct the full-sequence hidden states.
    new_hidden = pooling_output.get("thinker_hidden_states")
    if new_hidden is not None:
        new_hidden_cpu = new_hidden.detach().float().cpu()
        if request_id not in transfer_manager.request_payload:
            transfer_manager.request_payload[request_id] = new_hidden_cpu
        else:
            transfer_manager.request_payload[request_id] = torch.cat(
                [transfer_manager.request_payload[request_id], new_hidden_cpu], dim=0
            )

    if not is_finished:
        return None

    # Thinker finished — retrieve accumulated hidden states.
    # If new_hidden was not None above, it was stored in request_payload, so pop always
    # returns a value.  The None branch is only reachable if new_hidden was also None
    # (e.g. model emitted no thinker_hidden_states on the final step), in which case
    # we have no data to send and must skip this request.
    thinker_hidden_states = transfer_manager.request_payload.pop(request_id, None)
    if thinker_hidden_states is None:
        return None

    # Use MiniCPM-o 4.5 default TTS boundary token IDs.
    # In async_chunk mode the request object does not expose stage config,
    # so we fall back to hardcoded defaults (same values as thinker2talker).
    tts_bos_id = 151703
    tts_eos_id = 151704

    # Reconstruct full token sequence (prompt + generated).
    # Use _ensure_list() to handle vLLM's ConstantList wrapper correctly.
    all_token_ids: list[int] = []
    try:
        prompt_ids = _ensure_list(request.prompt_token_ids)
        output_ids = _ensure_list(request.output_token_ids)
        all_token_ids = prompt_ids + output_ids
    except AttributeError:
        # Fallback: use whatever is available on the request object.
        try:
            all_token_ids = _ensure_list(request.all_token_ids)
        except AttributeError:
            pass

    # Find TTS content boundary.
    tts_start, tts_end = _find_tts_bound(all_token_ids, tts_bos_id, tts_eos_id)

    hidden_len = thinker_hidden_states.shape[0]
    if tts_start >= hidden_len:
        # tts boundary is beyond available hidden states — produce an empty
        # slice so that num_tts_tokens == 0 fallback fires gracefully.
        # (resetting to 0 would feed the entire unfiltered sequence to Talker)
        tts_start = hidden_len
        tts_end = hidden_len
    if tts_end is not None and tts_end > hidden_len:
        tts_end = hidden_len

    tts_hidden = thinker_hidden_states[tts_start:tts_end]
    tts_token_ids = all_token_ids[tts_start:tts_end] if all_token_ids else []

    # Align lengths.
    num_tts_tokens = min(len(tts_token_ids), tts_hidden.shape[0])
    if num_tts_tokens == 0:
        # Fallback: use generated portion (prompt excluded).
        try:
            prompt_len = len(_ensure_list(request.prompt_token_ids))
            gen_ids = _ensure_list(request.output_token_ids)
            gen_start = min(prompt_len, hidden_len)
            gen_end = min(prompt_len + len(gen_ids), hidden_len)
            if gen_end > gen_start:
                tts_hidden = thinker_hidden_states[gen_start:gen_end]
                tts_token_ids = gen_ids[: gen_end - gen_start]
                num_tts_tokens = len(tts_token_ids)
            else:
                # Last resort: use all accumulated hidden states
                num_tts_tokens = min(len(all_token_ids), hidden_len) if all_token_ids else hidden_len
                tts_token_ids = all_token_ids[:num_tts_tokens]
                tts_hidden = thinker_hidden_states[:num_tts_tokens]
        except AttributeError:
            num_tts_tokens = hidden_len
            tts_hidden = thinker_hidden_states
            tts_token_ids = all_token_ids[:hidden_len] if all_token_ids else []

    tts_token_ids = tts_token_ids[:num_tts_tokens]
    tts_hidden = tts_hidden[:num_tts_tokens]

    return {
        # Keys match MiniCPM-o talker_preprocess expectations (same as sync mode).
        "thinker_token_ids": torch.tensor(tts_token_ids, dtype=torch.long),
        "thinker_hidden_states": tts_hidden,
        "finished": torch.tensor(True, dtype=torch.bool),
    }


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
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
            # tts boundary beyond available states u2014 produce empty slice so
            # num_tts_tokens == 0 fallback fires gracefully.
            # (resetting to 0 would feed the entire unfiltered sequence to Talker)
            tts_start = hidden_len
            tts_end = hidden_len
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
    prompt: OmniTokensPrompt | TextPrompt | None = None,
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
        # Filter stop token by value (6561) rather than unconditional [:-1] trim,
        # so that codec frames are preserved when max_tokens is reached without EOS.
        codec_codes = [t for t in output.token_ids if t != 6561] if output.token_ids else []

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
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Async chunk: stream codec tokens to Code2Wav as they're generated.

    MiniCPM-o uses num_vq=1 (single RVQ layer), so codec tokens are
    flat (1D). Each generated token is a single codec frame.

    Why ``request.output_token_ids`` instead of ``pooling_output["code_predictor_codes"]``
    (the pattern used by Qwen3-Omni and MIMO-Audio):
        Qwen3/MIMO have a *separate code predictor head* whose multi-layer RVQ
        predictions are stored in ``pooling_output["code_predictor_codes"]``.
        MiniCPM-o Talker has no such head — it performs plain auto-regressive
        decoding where each sampled token ID *is* the codec frame (num_vq=1).
        ``pooling_output`` therefore contains only raw hidden states (``"hidden"``
        key) and no codec codes.  Reading the sampled IDs via ``request.output_token_ids``
        with a cursor-based delta is the only correct approach for this model.

    Args:
        transfer_manager: Manages per-request token accumulation
        pooling_output: Current Talker output (hidden states only — no codec codes)
        request: vLLM request object (codec tokens read from output_token_ids)
        is_finished: Whether generation is complete

    Returns:
        dict with chunk info for Code2Wav, or None if not ready yet
    """
    request_id = request.external_req_id
    # Use is_finished parameter authoritatively; avoid racing against request state.
    finished = bool(is_finished)

    # code_prompt_token_ids is initialized as defaultdict(list) by the framework
    # (ChunkTransferAdapter.__init__), so no hasattr check needed.
    # _talker_token_cursor and _talker_emitted_len are MiniCPM-o-specific state;
    # initialize them lazily if not present.
    if not hasattr(transfer_manager, "_talker_token_cursor"):
        transfer_manager._talker_token_cursor = {}
    if not hasattr(transfer_manager, "_talker_emitted_len"):
        transfer_manager._talker_emitted_len = {}

    if request_id not in transfer_manager._talker_token_cursor:
        transfer_manager._talker_token_cursor[request_id] = 0
        transfer_manager._talker_emitted_len[request_id] = 0

    # Extract newly generated codec tokens from request.output_token_ids.
    # pooling_output["hidden"] contains raw hidden states, NOT sampled token ids.
    # Cumulative output is in request.output_token_ids; read only the delta via cursor.
    try:
        all_output_ids = _ensure_list(request.output_token_ids)
    except AttributeError:
        all_output_ids = []

    cursor = transfer_manager._talker_token_cursor[request_id]
    new_tokens = all_output_ids[cursor:]
    # MiniCPM-o Talker stop token is 6561 — filter before advancing cursor
    new_tokens = [t for t in new_tokens if t != 6561]
    # Advance cursor by original delta length (including any filtered stop tokens)
    transfer_manager._talker_token_cursor[request_id] = len(all_output_ids)
    transfer_manager.code_prompt_token_ids[request_id].extend(new_tokens)

    # Get chunk config from stage config
    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size = int(cfg.get("codec_left_context_frames", 25))

    current_length = len(transfer_manager.code_prompt_token_ids[request_id])
    emitted_len = transfer_manager._talker_emitted_len[request_id]
    pending = current_length - emitted_len  # tokens not yet sent to Code2Wav

    if pending <= 0:
        if finished:
            # Clean up our custom-state keys (framework handles code_prompt_token_ids
            # via cleanup_sender; we own _talker_token_cursor and _talker_emitted_len)
            transfer_manager._talker_token_cursor.pop(request_id, None)
            transfer_manager._talker_emitted_len.pop(request_id, None)
            # Return an empty finished payload so the framework proceeds to
            # cleanup_sender (line 244 in chunk_transfer_adapter.py).
            # Returning None would cause early return at line 230 and skip cleanup,
            # leaking put_req_chunk / request_payload / code_prompt_token_ids.
            # MiniCPMOCode2Wav.forward() handles empty codes gracefully (returns zeros).
            # Matches MIMO-Audio's _make_finished_sentinel() pattern.
            return {
                "code_predictor_codes": [],
                "left_context_size": 0,
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        # No pending tokens and not finished — nothing to emit yet
        return None

    # Emit when we have at least chunk_size pending tokens, or when finished.
    # Use >= threshold (not % modulo) so tokens are never permanently blocked
    # even when multiple tokens arrive in one call and skip the exact boundary.
    if not finished and pending < chunk_size:
        return None

    # Emit only the pending (not-yet-sent) tokens plus left context.
    chunk_length = pending  # all pending tokens form the new chunk
    chunk_start = emitted_len  # start of new tokens in the accumulated list
    left_start = max(0, chunk_start - left_context_size)
    left_context_count = chunk_start - left_start

    window_frames = transfer_manager.code_prompt_token_ids[request_id][
        left_start : chunk_start + chunk_length
    ]

    # Advance emitted pointer
    transfer_manager._talker_emitted_len[request_id] = current_length

    if finished:
        # Clean up our custom-state keys (framework handles code_prompt_token_ids)
        transfer_manager._talker_token_cursor.pop(request_id, None)
        transfer_manager._talker_emitted_len.pop(request_id, None)

    # For MiniCPM-o (num_vq=1), code_predictor_codes is just the flat list.
    # Qwen3-TTS interleaves quantizers: [f0[q0], f0[q1], f1[q0], f1[q1], ...]
    info: dict[str, Any] = {
        "code_predictor_codes": window_frames,
        "left_context_size": left_context_count,
        "finished": torch.tensor(finished, dtype=torch.bool),
    }

    return info
