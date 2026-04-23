import asyncio
import base64
import io
import json
import wave as wave_mod
from contextlib import aclosing

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.utils import random_uuid

from vllm_omni.entrypoints.openai.realtime.connection import RealtimeConnection
from vllm_omni.entrypoints.openai.realtime.protocol import (
    ResponseCreate, ResponseCancel,
    ResponseCreated, ResponseAudioDelta, ResponseAudioDone,
    ResponseAudioTranscriptDelta, ResponseAudioTranscriptDone,
    ResponseDone, ResponseCancelled,
)
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat


class OmniRealtimeConnection(RealtimeConnection):
    """
    vllm-omni extension: full audio output via /v1/realtime.
    Subclasses RealtimeConnection to add omni-specific event routing.
    
    ASR (Transcription only): use upstream RealtimeConnection
    Audio output (Omni): use OmniRealtimeConnection
    """
    
    def __init__(self, websocket, serving, chat_service: OmniOpenAIServingChat):
        super().__init__(websocket, serving)
        self.chat_service = chat_service
        self._audio_buffer: list[bytes] = []  # Accumulated PCM16 audio chunks
        self._response_task: asyncio.Task | None = None
        self._system_prompt: str | None = None
        self._conversation_history: list[dict] = []  # Multi-turn context; bounded to _MAX_HISTORY_TURNS
        self._MAX_HISTORY_TURNS = 20  # Keep last 20 user/assistant exchange pairs (40 entries)
        self._session_config: dict = {}  # Store other session parameters
    
    async def handle_event(self, event: dict) -> None:
        """Extend upstream handle_event() with response.create/cancel."""
        event_type = event.get("type", "")
        
        if event_type == "response.create":
            await self._start_response()
        elif event_type == "response.cancel":
            await self._cancel_response()
        elif event_type == "session.update":
            # Handle locally only — upstream expects top-level "model" key
            # which OpenAI-spec clients put inside "session" dict.
            # Omni mode doesn't use upstream's model validation.
            session_config = event.get("session", {})
            self._system_prompt = session_config.get("instructions", self._system_prompt)
            self._session_config.update(session_config)
        elif event_type == "session.config":
            # Legacy alias — handle locally only
            session_config = event.get("session", {})
            self._system_prompt = session_config.get("instructions", self._system_prompt)
            self._session_config.update(session_config)
        elif event_type == "input_audio_buffer.append":
            # Accumulate audio buffer locally for omni response
            await self._buffer_audio(event)
            # Do NOT pass to upstream — omni mode uses its own pipeline,
            # upstream's audio_queue would grow without bound since
            # input_audio_buffer.commit routes to _start_response() instead
            # of upstream's start_generation().
        elif event_type == "input_audio_buffer.commit":
            # Route based on presence of chat_service
            if self.chat_service is not None:
                await self._start_response()
            else:
                await super().handle_event(event)
        else:
            await super().handle_event(event)
    
    async def _buffer_audio(self, event: dict) -> None:
        """Decode base64 PCM16 audio and accumulate in buffer."""
        audio_b64 = event.get("audio", "")
        if not audio_b64:
            return
        audio_bytes = base64.b64decode(audio_b64)
        # Store raw PCM16 bytes — will convert to WAV when committing
        self._audio_buffer.append(audio_bytes)
    
    async def _start_response(self) -> None:
        """Triggered by input_audio_buffer.commit or response.create."""
        if self._response_task and not self._response_task.done():
            return  # Ignore if response already in progress
        if not self._audio_buffer:
            await self.send_error("No audio input provided")
            await self.send(ResponseDone())
            return

        pcm_chunks = self._audio_buffer
        self._audio_buffer = []

        self._response_task = asyncio.create_task(
            self._run_omni_response(pcm_chunks)
        )
    
    async def _run_omni_response(self, pcm_chunks: list[bytes]) -> None:
        """Execute full Omni pipeline and stream to WebSocket.

        Uses the same proven pattern as serving_omni_duplex.py:
        PCM16 -> WAV -> audio_url data URI -> ChatCompletionRequest (stream=True)
        -> SSE chunks -> WebSocket events.
        """
        await self.send(ResponseCreated())

        # PCM16 chunks -> single WAV (same pattern as serving_omni_duplex.py)
        pcm_bytes = b"".join(pcm_chunks)
        buf = io.BytesIO()
        with wave_mod.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(16000)
            wf.writeframes(pcm_bytes)
        wav_b64 = base64.b64encode(buf.getvalue()).decode()

        user_message = {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{wav_b64}"}}
            ]
        }

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._conversation_history)
        messages.append(user_message)

        full_text = ""

        pre_uuid = random_uuid()
        abort_id = f"chatcmpl-{pre_uuid}"

        request = ChatCompletionRequest(
            model=self.serving.model_config.model,
            messages=messages,
            stream=True,
            modalities=["audio", "text"],
            temperature=self._session_config.get("temperature", None),
            max_tokens=self._session_config.get("max_response_output_tokens", None),
            chat_template_kwargs={"use_tts_template": True},
            request_id=pre_uuid,
        )

        try:
            result = await self.chat_service.create_chat_completion(request)
        except Exception as e:
            await self.send_error(str(e))
            await self.send(ResponseDone())
            return

        if isinstance(result, ErrorResponse):
            await self.send_error(result.error.message)
            await self.send(ResponseDone())
            return

        # result is AsyncGenerator[str, None] — SSE event stream
        generator = result

        try:
            async with aclosing(generator):
                async for line in generator:
                    if not isinstance(line, str):
                        line = line.decode("utf-8", errors="replace")
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break

                    try:
                        chunk_dict = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    modality = chunk_dict.get("modality")
                    choices = chunk_dict.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    if not content:
                        continue

                    if modality == "text":
                        full_text += content
                        await self.send(ResponseAudioTranscriptDelta(delta=content))
                    elif modality == "audio":
                        await self.send(ResponseAudioDelta(delta=content))

        except asyncio.CancelledError:
            # Abort engine request to prevent zombie GPU work
            try:
                await self.chat_service.engine_client.abort(abort_id)
            except Exception:
                pass
            await self.send(ResponseCancelled())
            await self.send(ResponseDone())
            raise
        except Exception as e:
            await self.send_error(str(e))
            await self.send(ResponseDone())
            return

        # NOTE: Multi-turn history accumulation disabled.
        # Storing "[audio input]" text placeholders for audio turns confuses the model
        # on subsequent turns — it sees the literal string "[audio input]" as prior user
        # speech, causing incorrect responses and degraded TTS audio quality.
        # Proper multi-turn support requires transcribing user audio turns (ASR) before
        # storing them in history. Until then, each turn is processed independently.
        await self.send(ResponseAudioTranscriptDone(transcript=full_text))
        await self.send(ResponseAudioDone())
        await self.send(ResponseDone())
    
    async def _cancel_response(self) -> None:
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()

    async def cleanup(self) -> None:
        """Override to also cancel the omni response task on disconnect."""
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
        await super().cleanup()

    async def send(self, event) -> None:
        """Extend send() to support vllm-omni specific event types."""
        omni_event_types = (
            ResponseCreated, ResponseAudioDelta, ResponseAudioDone,
            ResponseAudioTranscriptDelta, ResponseAudioTranscriptDone,
            ResponseDone, ResponseCancelled,
        )
        if isinstance(event, omni_event_types):
            data = event.model_dump_json()
            await self.websocket.send_text(data)
        else:
            await super().send(event)
