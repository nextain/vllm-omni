import asyncio
import base64
import json
import numpy as np
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
        self._audio_buffer: list[np.ndarray] = []  # Accumulated input audio
        self._response_task: asyncio.Task | None = None
        self._system_prompt: str | None = None
        self._conversation_history: list[dict] = []  # Multi-turn context
        self._session_config: dict = {}  # Store other session parameters
    
    async def handle_event(self, event: dict) -> None:
        """Extend upstream handle_event() with response.create/cancel."""
        event_type = event.get("type", "")
        
        if event_type == "response.create":
            await self._start_response()
        elif event_type == "response.cancel":
            await self._cancel_response()
        elif event_type == "session.update" or event_type == "session.config":
            # Add support for updating session parameters
            # OpenAI spec uses session.update, some older ones use session.config
            session_config = event.get("session", {})
            self._system_prompt = session_config.get("instructions", self._system_prompt)
            # Update internal session config
            self._session_config.update(session_config)
            await super().handle_event(event)
        elif event_type == "input_audio_buffer.append":
            # Accumulate audio buffer locally for omni response
            await self._buffer_audio(event)
            # Also pass to upstream for ASR transcription path
            await super().handle_event(event)
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
        # PCM16 -> float32 (same normalization as upstream connection.py)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer.append(audio_np)
    
    async def _start_response(self) -> None:
        """Triggered by input_audio_buffer.commit or response.create."""
        if self._response_task and not self._response_task.done():
            return  # Ignore if response already in progress
        if not self._audio_buffer:
            return
        
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []
        
        self._response_task = asyncio.create_task(
            self._run_omni_response(audio)
        )
    
    async def _run_omni_response(self, audio: np.ndarray) -> None:
        """Execute full Omni pipeline and stream to WebSocket.
        
        1. Audio -> ChatCompletionRequest
        2. serving_chat.create_chat_completion(stream=True)
        3. SSE chunks -> WebSocket events
        """
        await self.send(ResponseCreated())
        
        # Convert audio to base64 encoded PCM16 bytes for ChatCompletionRequest
        # We convert float32 back to int16 to match the expected format in multimodal message.
        audio_pcm16 = (audio * 32767.0).astype(np.int16)
        audio_b64 = base64.b64encode(audio_pcm16.tobytes()).decode()
        
        user_message = {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "pcm16"}}
            ]
        }
        
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._conversation_history)
        messages.append(user_message)
        
        # Add to history for future turns
        self._conversation_history.append(user_message)
        
        full_text = ""
        
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        # Map OpenAI Realtime session config to ChatCompletionRequest fields
        request = ChatCompletionRequest(
            model=self.serving.model_config.model if hasattr(self.serving, 'model_config') else None,
            messages=messages,
            stream=True,
            modalities=["audio", "text"],
            temperature=self._session_config.get("temperature", None),
            max_tokens=self._session_config.get("max_response_output_tokens", None),
        )
        
        try:
            async for sse_chunk in self.chat_service.create_chat_completion(
                request=request,
                raw_request=None,
            ):
                if not isinstance(sse_chunk, str):
                    continue
                if not sse_chunk.startswith("data: "):
                    continue
                json_str = sse_chunk[6:].strip()
                if json_str == "[DONE]":
                    break
                
                try:
                    chunk_dict = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                
                modality = chunk_dict.get("modality", "text")
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
                    # Stream base64 PCM16 chunks to WebSocket
                    await self.send(ResponseAudioDelta(delta=content))
                    
        except asyncio.CancelledError:
            await self.send(ResponseCancelled())
            await self.send(ResponseDone())
            raise
        except Exception as e:
            await self.send({"type": "error", "error": {"message": str(e)}})
            await self.send(ResponseDone())
            return
        
        # Finish turn
        self._conversation_history.append({"role": "assistant", "content": full_text})
        await self.send(ResponseAudioTranscriptDone(transcript=full_text))
        await self.send(ResponseAudioDone())
        await self.send(ResponseDone())
    
    async def _cancel_response(self) -> None:
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
    
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
