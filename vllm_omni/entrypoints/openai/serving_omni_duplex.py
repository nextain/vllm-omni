"""Full-duplex omni WebSocket handler.

New endpoint: POST /v1/omni

Protocol:
    Client -> Server:
        {"type": "session.config", "model": "...", "system": "..."}
        <binary: PCM16 16kHz mono audio>   (one or more binary frames)
        {"type": "input.done"}             (end of audio input)
        OR {"type": "input.text", "text": "..."}  (text-only turn)
        ... (repeat per turn; multi-turn conversation supported)

    Server -> Client:
        {"type": "turn.start"}
        {"type": "transcript.delta", "text": "..."}  (zero or more, LLM text tokens)
        {"type": "audio.start", "format": "wav_chunk", "sample_rate": 24000}
            Each subsequent binary frame is a self-contained WAV file (RIFF header + PCM16 data).
            Clients must decode each frame independently, NOT as a concatenated stream.
        <binary: WAV chunk>  (one or more binary frames)
        {"type": "audio.done", "total_bytes": N}
        {"type": "turn.done"}
        {"type": "error", "message": "..."}  (non-fatal; session continues)

Audio pipeline:
    PCM16 16kHz (client) -> WAV -> ChatCompletionRequest (audio_url) ->
    OmniOpenAIServingChat (stream=True) -> SSE -> WAV chunks -> binary frames (client)

Why not _add_streaming_input_request?
    MiniCPM-o's thinker2talker_async_chunk requires the full Thinker output to
    find TTS boundary (_find_tts_bound). Incremental audio input gives no TTFP benefit;
    buffered VAD-gated input is equally fast and simpler.
"""

import asyncio
import base64
import io
import json
import struct
from contextlib import aclosing

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ValidationError
from vllm.logger import init_logger
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

logger = init_logger(__name__)

_DEFAULT_IDLE_TIMEOUT = 60.0
_DEFAULT_CONFIG_TIMEOUT = 10.0
_PCM_SAMPLE_RATE = 16000       # client sends 16kHz PCM16 mono
_OUTPUT_SAMPLE_RATE = 24000    # MiniCPM-o Code2Wav outputs 24kHz
_MAX_CONFIG_SIZE = 4096        # bytes
_MAX_AUDIO_FRAME_SIZE = 64 * 1024  # 64 KB per binary frame
_MIN_PCM_BYTES = 16000         # 0.5s at 16kHz 16-bit mono


class OmniDuplexSessionConfig(BaseModel):
    """Parameters sent in the initial session.config message."""

    model: str | None = None
    system: str | None = None
    max_tokens: int = Field(default=1024, ge=1, le=4096)


class OmniFullDuplexHandler:
    """WebSocket handler for /v1/omni full-duplex omni conversation.

    Reuses OmniOpenAIServingChat.create_chat_completion() (the proven REST path).
    Each turn: buffer audio -> submit request -> stream WAV output chunks.

    Args:
        chat_service: The existing chat serving instance.
        idle_timeout: Max seconds to wait for a client message.
        config_timeout: Max seconds to wait for the initial session.config.
    """

    def __init__(
        self,
        chat_service: OmniOpenAIServingChat,
        idle_timeout: float = _DEFAULT_IDLE_TIMEOUT,
        config_timeout: float = _DEFAULT_CONFIG_TIMEOUT,
    ) -> None:
        self._chat_service = chat_service
        self._idle_timeout = idle_timeout
        self._config_timeout = config_timeout

    async def handle_session(self, websocket: WebSocket) -> None:
        """Main session loop for a single WebSocket connection."""
        await websocket.accept()
        try:
            config = await self._receive_config(websocket)
            if config is None:
                return  # Error already sent; _receive_config closes on error

            # Look up the served model name if client did not specify one
            model_name = config.model or self._chat_service.models.model_name(None)

            messages: list[dict] = []
            if config.system:
                messages.append({"role": "system", "content": config.system})

            # Multi-turn loop: each iteration handles one user turn
            while True:
                pcm_bytes, text_input = await self._receive_turn_input(websocket)
                if pcm_bytes is None and text_input is None:
                    return  # Session ended (closed by _receive_turn_input on error)

                if pcm_bytes is not None:
                    wav = _pcm16_to_wav(pcm_bytes, _PCM_SAMPLE_RATE)
                    b64 = base64.b64encode(wav).decode()
                    user_content: list | str = [
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}
                    ]
                else:
                    user_content = text_input or ""

                messages.append({"role": "user", "content": user_content})

                await websocket.send_json({"type": "turn.start"})
                transcript = await self._run_turn(websocket, model_name, config, messages)
                # Store assistant reply as plain text (audio not preserved in history).
                # Known limitation: multi-turn audio context not maintained.
                messages.append({"role": "assistant", "content": transcript})

        except WebSocketDisconnect:
            logger.info("omni-duplex: client disconnected")
        except asyncio.TimeoutError:
            try:
                await self._send_error(websocket, "Idle timeout")
            except Exception:
                pass
        except Exception as e:
            logger.exception("omni-duplex session error: %s", e)
            try:
                await self._send_error(websocket, f"Internal error: {e}")
            except Exception:
                pass

    async def _receive_config(
        self, websocket: WebSocket
    ) -> OmniDuplexSessionConfig | None:
        """Wait for and validate the session.config message."""
        try:
            raw = await asyncio.wait_for(
                websocket.receive_text(), timeout=self._config_timeout
            )
        except asyncio.TimeoutError:
            await self._send_error(websocket, "Timeout waiting for session.config")
            await websocket.close()
            return None
        except RuntimeError as e:
            # receive_text() raises RuntimeError if client sends a binary frame first
            await self._send_error(websocket, f"Expected text session.config message: {e}")
            await websocket.close()
            return None

        if len(raw) > _MAX_CONFIG_SIZE:
            await self._send_error(websocket, "session.config too large")
            await websocket.close()
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON in session.config")
            await websocket.close()
            return None

        if not isinstance(msg, dict) or msg.get("type") != "session.config":
            got = msg.get("type") if isinstance(msg, dict) else type(msg).__name__
            await self._send_error(websocket, f"Expected session.config, got: {got}")
            await websocket.close()
            return None

        try:
            config = OmniDuplexSessionConfig(
                **{k: v for k, v in msg.items() if k != "type"}
            )
        except ValidationError as e:
            await self._send_error(websocket, f"Invalid session config: {e}")
            await websocket.close()
            return None

        return config

    async def _receive_turn_input(
        self, websocket: WebSocket
    ) -> tuple[bytes | None, str | None]:
        """Collect one turn of input (binary audio or text).

        Returns:
            (pcm_bytes, None) when audio input received via binary frames + input.done
            (None, text_str) when text input received via input.text
            (None, None) when session should close (error path)
        """
        pcm_chunks: list[bytes] = []

        while True:
            try:
                msg = await asyncio.wait_for(
                    websocket.receive(), timeout=self._idle_timeout
                )
            except asyncio.TimeoutError:
                await self._send_error(websocket, "Idle timeout waiting for input")
                await websocket.close()
                return None, None

            if msg["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(code=msg.get("code", 1000))

            bytes_data: bytes | None = msg.get("bytes")
            text_data: str | None = msg.get("text")

            if bytes_data is not None:
                # Zero-length binary frame: valid but skip
                if len(bytes_data) > _MAX_AUDIO_FRAME_SIZE:
                    await self._send_error(websocket, "Audio frame too large")
                    continue
                pcm_chunks.append(bytes_data)

            elif text_data is not None:
                try:
                    parsed = json.loads(text_data)
                except json.JSONDecodeError:
                    await self._send_error(websocket, "Invalid JSON")
                    continue

                if not isinstance(parsed, dict):
                    await self._send_error(websocket, "Message must be a JSON object")
                    continue

                msg_type = parsed.get("type")

                if msg_type == "input.done":
                    total = b"".join(pcm_chunks)
                    if len(total) < _MIN_PCM_BYTES:
                        # Audio too short to be meaningful — close session
                        await self._send_error(
                            websocket,
                            f"Audio too short ({len(total)} bytes < {_MIN_PCM_BYTES} minimum), closing session",
                        )
                        await websocket.close()
                        return None, None
                    return total, None

                elif msg_type == "input.text":
                    text = parsed.get("text", "")
                    if not isinstance(text, str) or not text.strip():
                        await self._send_error(websocket, "input.text requires a non-empty string")
                        continue
                    return None, text

                elif msg_type == "session.config":
                    # Re-configuration not supported mid-session
                    await self._send_error(websocket, "Cannot reconfigure session after start")
                    await websocket.close()
                    return None, None

                else:
                    await self._send_error(websocket, f"Unknown message type: {msg_type}")

    async def _run_turn(
        self,
        websocket: WebSocket,
        model_name: str,
        config: OmniDuplexSessionConfig,
        messages: list[dict],
    ) -> str:
        """Submit one chat turn to the LLM and stream audio/text output to client.

        Returns the transcript (text output) for conversation history.
        """
        # Pre-generate request_id so we can abort before the first SSE chunk arrives.
        # _base_request_id(raw_request=None, default=pre_uuid) -> pre_uuid
        # engine registers the request as f"chatcmpl-{pre_uuid}"
        pre_uuid = random_uuid()
        abort_id = f"chatcmpl-{pre_uuid}"

        request = ChatCompletionRequest(
            model=model_name,
            messages=messages,  # type: ignore[arg-type]
            modalities=["audio"],
            stream=True,
            chat_template_kwargs={"use_tts_template": True},
            max_tokens=config.max_tokens,
            request_id=pre_uuid,
        )

        try:
            result = await self._chat_service.create_chat_completion(request)
        except Exception as e:
            # Synchronous exception from create_chat_completion (e.g., AssertionError)
            logger.error("omni-duplex: create_chat_completion raised: %s", e)
            await self._send_error(websocket, f"Request error: {e}")
            await websocket.send_json({"type": "turn.done"})
            return ""

        # create_chat_completion returns ErrorResponse on model/validation errors
        if isinstance(result, ErrorResponse):
            await self._send_error(websocket, f"Model error: {result.message}")
            await websocket.send_json({"type": "turn.done"})
            return ""

        # result is AsyncGenerator[str, None] — SSE event stream
        generator = result
        audio_started = False
        total_bytes = 0
        transcript = ""

        try:
            async with aclosing(generator) as stream:
                async for line in stream:
                    if not isinstance(line, str):
                        line = line.decode("utf-8", errors="replace")
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    modality = chunk.get("modality")
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    if modality == "audio":
                        b64 = delta.get("content", "")
                        if b64:
                            try:
                                audio_bytes = base64.b64decode(b64)
                            except Exception:
                                continue
                            if not audio_started:
                                await websocket.send_json(
                                    {
                                        "type": "audio.start",
                                        "format": "wav_chunk",
                                        "sample_rate": _OUTPUT_SAMPLE_RATE,
                                    }
                                )
                                audio_started = True
                            await websocket.send_bytes(audio_bytes)
                            total_bytes += len(audio_bytes)

                    elif modality == "text":
                        text = delta.get("content", "")
                        if text:
                            transcript += text
                            await websocket.send_json(
                                {"type": "transcript.delta", "text": text}
                            )

        except WebSocketDisconnect:
            # Client disconnected mid-stream. aclosing() will close the generator
            # (GeneratorExit -> engine abort internally). The explicit abort below
            # is a safety net in case the generator teardown is async-delayed.
            try:
                await self._chat_service.engine_client.abort(abort_id)
            except Exception:
                logger.debug("omni-duplex: abort %s failed (may already be done)", abort_id, exc_info=True)
            raise

        if audio_started:
            await websocket.send_json({"type": "audio.done", "total_bytes": total_bytes})
        await websocket.send_json({"type": "turn.done"})
        return transcript

    @staticmethod
    async def _send_error(websocket: WebSocket, message: str) -> None:
        """Send a non-fatal error message to the client."""
        try:
            await websocket.send_json({"type": "error", "message": message})
        except Exception:
            pass  # Connection may already be closing; safe to ignore


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Convert raw PCM16 mono bytes to WAV (RIFF) format."""
    buf = io.BytesIO()
    data_len = len(pcm_bytes)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_len))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))          # fmt chunk size
    buf.write(struct.pack("<H", 1))           # PCM
    buf.write(struct.pack("<H", 1))           # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))           # block align
    buf.write(struct.pack("<H", 16))          # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    buf.write(pcm_bytes)
    return buf.getvalue()
