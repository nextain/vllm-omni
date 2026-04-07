# Plan: Issue #6 u2014 OmniRealtime uc624ub514uc624 ucd9cub825 ud30cuc774ud504ub77cuc778 (/v1/realtime)

## ubaa9uc801
`/v1/realtime` WebSocketuc5d0uc11c MiniCPM-o 3ub2e8uacc4 ud30cuc774ud504ub77cuc778uc744 ud1b5ud55c uc624ub514uc624 ucd9cub825 uc2a4ud2b8ub9acubc0d uad6cud604.
upstream vllm `RealtimeConnection`uc744 uc11cube0cud074ub798uc2f1ud558uc5ec vllm-omniuc5d0ub9cc ud655uc7a5.

## ud575uc2ec uc124uacc4 uacb0uc815

### uc655 ASR uacbdub85c vs uc624ub514uc624 uc751ub2f5 uacbdub85c

**ASR uacbdub85c (upstream uae30uc874):**
```
audio_stream u2192 buffer_realtime_audio() u2192 TokensPrompt(max_tokens=1)
u2192 engine.generate() u2192 text delta u2192 TranscriptionDelta
```

**uc624ub514uc624 uc751ub2f5 uacbdub85c (uc2e0uaddc, ud574ub2f9 uc774uc288):**
```
audio_buffer u2192 ChatCompletionRequest(audio_message)
u2192 serving_chat.create_chat_completion() u2192 audio+text uc2a4ud2b8ub9d8
u2192 response.audio.delta + response.audio_transcript.delta
```

**uc774uc720**: upstream `_run_generation()`uc740 `output.outputs[0].text`ub9cc ucd94ucd9c. uc624ub514uc624ub294 `multimodal_outputs`ub97c ud1b5ud574 serving_chatuc774 ucc98ub9ac. ub530ub77cuc11c uc624ub514uc624 ucd9cub825uc740 uc11cube0cud074ub798uc2f1uc73cub85c ud655uc7a5ub418uc9c0 uc54auace0, uae30uc874 chat completion ud30cuc774ud504ub77cuc778uc744 uc7acuc0acuc6a9ud558uc5ec WebSocket uc774ubca4ud2b8ub85c uc0acuc6a9ud55cub2e4.

### ud540 uc2a4ud2b8ub9bc uc544ud0a4ud14ducc98

`input_audio_buffer.commit` ub370uc774ud130uc758 ud750ub984:
```
1. ub204uc801ub41c uc624ub514uc624 ubc84ud37c u2192 float32 np.ndarray ubcd1ud569
2. ChatCompletionRequest uc0dduc131 (uc624ub514uc624 uba40ud2f0ubaa8ub2ec uba54uc2dcuc9c0)
3. serving_chat.create_chat_completion() ud638ucd9c (stream=True)
4. SSE uccadud06c uc21cud68c:
   a. ud14duc2a4ud2b8 delta u2192 ResponseAudioTranscriptDelta uc804uc1a1
   b. uc624ub514uc624 uccadud06c u2192 ResponseAudioDelta uc804uc1a1
5. uc644ub8cc u2192 ResponseAudioTranscriptDone + ResponseAudioDone + ResponseDone uc804uc1a1
```

### uc778ud130ub7fdud2b8 ud540 uc2a4ud2b8ub9bc
```
ResponseCancel uc218uc2e0 u2192 uc9c4ud589 uc911 generation_task.cancel() u2192 ResponseCancelled uc804uc1a1
```

## uc791uc5c5 uacc4ud68d

### uc0dduc131ud560 ud30cuc77c

**1. `vllm_omni/entrypoints/openai/realtime/omni_connection.py`**

```python
import asyncio
import base64
import json
import numpy as np
from vllm.entrypoints.openai.realtime.connection import RealtimeConnection
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
    
    ASR (uc804uc0acub9cc): upstream RealtimeConnection uc0acuc6a9
    uc624ub514uc624 ucd9cub825: OmniRealtimeConnection uc0acuc6a9
    """
    
    def __init__(self, websocket, serving, chat_service: OmniOpenAIServingChat):
        super().__init__(websocket, serving)
        self.chat_service = chat_service
        self._audio_buffer: list[np.ndarray] = []  # ub208uc801ub41c uc785ub825 uc624ub514uc624
        self._response_task: asyncio.Task | None = None
        self._system_prompt: str | None = None
        self._conversation_history: list[dict] = []  # uba40ud2f0ud134 ucee8ud14duc2a4ud2b8
    
    async def handle_event(self, event: dict) -> None:
        """upstream handle_event() ud655uc7a5: response.create, response.cancel ucd94uac00."""
        event_type = event.get("type", "")
        
        if event_type == "response.create":
            await self._start_response()
        elif event_type == "response.cancel":
            await self._cancel_response()
        elif event_type == "session.update":
            # system_prompt ud56dubaa9 ucd94uac00 uc9c0uc6d0
            self._system_prompt = event.get("system_prompt", None)
            await super().handle_event(event)
        elif event_type == "input_audio_buffer.append":
            # uc624ub514uc624 ubc84ud37cub9cc ub204uc801 (uc804uc1a1 x)
            await self._buffer_audio(event)
            # upstream ASR ubc84ud37cub3c4 ub3d9uc2dc ub204uc801 (ASR uc804uc6a9 uacbdub85cub3c4 uc720uc9c0)
            await super().handle_event(event)
        elif event_type == "input_audio_buffer.commit":
            # uacbdub85c ubd84uae30: chat_service uc788uc73cba74 omni ud30cuc774ud504ub77cuc778, uc5c6uc73cba74 ASR uacbdub85c
            if self.chat_service is not None:
                await self._start_response()
            else:
                await super().handle_event(event)
        else:
            await super().handle_event(event)
    
    async def _buffer_audio(self, event: dict) -> None:
        """base64 PCM16 uc624ub514uc624ub97c float32ub85c ub514ucf54ub529ud558uc5ec ubc84ud37cuc5d0 ub208uc801."""
        audio_b64 = event.get("audio", "")
        if not audio_b64:
            return
        audio_bytes = base64.b64decode(audio_b64)
        # PCM16 u2192 float32 (upstream connection.pyuc640 ub3d9uc77c uc815uaddcud654)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._audio_buffer.append(audio_np)
    
    async def _start_response(self) -> None:
        """input_audio_buffer.commit ub610ub294 response.create uc2dc ud638ucd9c."""
        if self._response_task and not self._response_task.done():
            return  # uc9c4ud589 uc911uc778 uc751ub2f5 uc788uc73cuba74 ubb34uc2dc
        if not self._audio_buffer:
            return
        
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []
        
        self._response_task = asyncio.create_task(
            self._run_omni_response(audio)
        )
    
    async def _run_omni_response(self, audio: np.ndarray) -> None:
        """MiniCPM-o uc804uccb4 ud30cuc774ud504ub77cuc778 uc2e4ud589 ubc0f uc2a4ud2b8ub9acubc0d.
        
        1. uc624ub514uc624 u2192 ChatCompletionRequest
        2. serving_chat.create_chat_completion() ud638ucd9c (stream=True)
        3. SSE uccadud06c u2192 WebSocket uc774ubca4ud2b8 ubcc0ud658
        """
        await self.send(ResponseCreated())
        
        # uc624ub514uc624 float32 u2192 base64 (serving_chatuc774 uc218uc6a9ud558ub294 ud615uc2dd)
        audio_b64 = base64.b64encode(audio.tobytes()).decode()
        
        # ChatCompletionRequest uc0dduc131
        # serving_chatuc758 uae30uc874 uba40ud2f0ubaa8ub2ec uc624ub514uc624 uba54uc2dcuc9c0 ud615uc2dd ub530ub984
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(self._conversation_history)
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "pcm16"}}
            ]
        })
        
        # serving_chat.create_chat_completion() ud638ucd9c (stream=True)
        # ubc18ud658: AsyncGenerator[str, None] u2014 SSE ud615uc2dd ubb38uc790uc5f4 "data: {...}\n\n"
        # uc624ub514uc624 uccadud06c: choices[0].delta.content uc5d0 base64 PCM16 (modality="audio", serving_chat uc218uc815 ud6c4)
        # ud14duc2a4ud2b8 uccadud06c: choices[0].delta.content uc5d0 ud14duc2a4ud2b8 (modality="text")
        full_text = ""
        
        # ChatCompletionRequest uc0dduc131 (from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest)
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        request = ChatCompletionRequest(
            model=self.serving._model_name() if hasattr(self.serving, '_model_name') else None,
            messages=messages,
            stream=True,
        )
        
        try:
            async for sse_chunk in self.chat_service.create_chat_completion(
                request=request,
                raw_request=None,
            ):
                # SSE uc2a4ud2b8ub9acubc0d ud30cuc2f1: "data: {...}\n\n" u2192 dict
                if not sse_chunk.startswith("data: "):
                    continue
                json_str = sse_chunk[6:].strip()
                if json_str == "[DONE]":
                    break
                chunk_dict = json.loads(json_str)
                
                modality = chunk_dict.get("modality", "text")
                choices = chunk_dict.get("choices", [])
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if not content:
                    continue
                
                if modality == "text":
                    full_text += content
                    await self.send(ResponseAudioTranscriptDelta(delta=content))
                elif modality == "audio":
                    # base64 PCM16 uccadud06c u2192 WebSocket uc804uc1a1 (serving_chat pcm ud3ecub9f7 ud6c4)
                    await self.send(ResponseAudioDelta(delta=content))
        except asyncio.CancelledError:
            await self.send(ResponseCancelled())
            return
        
        # uc644ub8cc
        self._conversation_history.append({"role": "assistant", "content": full_text})
        await self.send(ResponseAudioTranscriptDone(transcript=full_text))
        await self.send(ResponseAudioDone())
        await self.send(ResponseDone())
    
    async def _cancel_response(self) -> None:
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            # ResponseCancelledub294 _run_omni_responseuc5d0uc11c CancelledError uc2dc uc804uc1a1
    
    async def send(self, event) -> None:
        """upstream send() ud655uc7a5: uc2e0uaddc uc774ubca4ud2b8 ud0c0uc785 uc9c0uc6d0."""
        # OmniRealtimeConnection uc804uc6a9 uc774ubca4ud2b8ub294 uc9c1uc811 uc2dcub9acuc5bcub77cuc774uc988
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
```

**2. `serving_chat.py` `_create_audio_choice()` PCM ud3ecub9f7 uc218uc815**

```python
# _create_audio_choice() ub0b4 uc2a4ud2b8ub9acubc0d uc2dcuc758 response_format uc218uc815:
# stream=True uc2dc: response_format="pcm" (raw PCM16, WAV ud5e4ub354 uc5c6uc74c) u2192 ud504ub85cud1a0ucf5c uc77cuce58
# stream=False uc2dc: response_format="wav" uc720uc9c0 (uae30uc874 REST API ud638ud658uc131)
audio_obj = CreateAudio(
    audio_tensor=audio_tensor,
    sample_rate=24000,
    response_format="pcm" if stream else "wav",  # uc218uc815: stream uc2dc PCM16
    speed=1.0,
    stream_format="audio",
    base64_encode=True,
)
```

**3. `api_server.py` `/v1/realtime` ub77cuc6b0ud2b8 uc218uc815**

```python
@router.websocket("/v1/realtime")
async def realtime_websocket(websocket: WebSocket):
    serving = getattr(websocket.app.state, "openai_serving_realtime", None)
    chat_service = getattr(websocket.app.state, "openai_serving_chat", None)
    
    if serving is None:
        ...
        return
    
    # chat_serviceuac00 uc788uc73cuba74 OmniRealtimeConnection (uc624ub514uc624 ucd9cub825 uc9c0uc6d0)
    # uc5c6uc73cuba74 upstream RealtimeConnection (uae30ubcf8 ASRub9cc)
    if chat_service is not None:
        from vllm_omni.entrypoints.openai.realtime.omni_connection import OmniRealtimeConnection
        connection = OmniRealtimeConnection(websocket, serving, chat_service)
    else:
        connection = RealtimeConnection(websocket, serving)
    
    await connection.handle_connection()
```

## uc870uc0ac uc644ub8cc uc0acud56d (2026-04-07 uc0ddud3ec uc2e4uc81c ucf54ub4dc ubd84uc11d)

### 1. serving_chat.create_chat_completion() uc2a4ud2b8ub9acubc0d ud615uc2dd
- **ud655uc778 uc644ub8cc**: `AsyncGenerator[str, None]` u2014 `"data: {...}\n\n"` SSE ubb38uc790uc5f4
- uc624ub514uc624: `modality="audio"`, `choices[0].delta.content` = base64 PCM16 (serving_chat uc218uc815: `response_format="pcm"` stream=True uc2dc)
- ud14duc2a4ud2b8: `modality="text"`, `choices[0].delta.content` = ud14duc2a4ud2b8
- `_run_omni_response()`uc5d0 SSE ud30cuc2f1 ub85cuc9c1 ucd94uac00ub428

### 2. ChatCompletionRequest raw_request ud30cub77cubbf8ud130
- **ud655uc778 uc644ub8cc**: `raw_request=None` uc815uc0c1 ub3d9uc791 (Optional ud30cub77cubbf8ud130)

### 3. uba40ud2f0ubaa8ub2ec uc624ub514uc624 uba54uc2dcuc9c0 ud615uc2dd
- **ud655uc778 uc644ub8cc**: `input_audio` ud0c0uc785 + base64 float32 PCM (uc11cubc84uc5d0uc11c PCM16u2192float32 ubcc0ud658 ud544uc694)
- uc11cubc84uac00 PCM16 uc785ub825uc744 uc720uc9c0ud560 uc218 uc788ub294uc9c0ub294 `_buffer_audio()`uc5d0uc11c ub2f9uc7a5 ubcc0ud658: `int16 / 32768.0 u2192 float32`

## uac80uc99d ubc29ubc95
1. uc11cubc84 uae30ub3d9 + `/v1/realtime` ASR uc815uc0c1 uc720uc9c0
2. MiniCPM-o + uc624ub514uc624 uc785ub825 u2192 `response.audio.delta` uc218uc2e0
   - base64 ub370ucf54ub529 ud6c4 PCM16 uc720ud6a8uc131 ud655uc778 (waveform ud3ecub9f7 + uc0d8ud50cub808uc774ud2b8 24kHz)
3. uc778ud130ub7fdud2b8 (`response.cancel`) u2192 `response.cancelled` uc218uc2e0
4. uba40ud2f0ud134: 2ubc88uc9f8 uc624ub514uc624 uc785ub825 u2192 ub300ud654 ub9e5ub77d uc720uc9c0ub418ub294uc9c0 ud655uc778
5. uc9c1uc811 `/v1/chat/completions` uc624ub514uc624 E2Euc640 ub3d9uc77cud55c uc751ub2f5 ud488uc9c8
6. ASR ub3c4ub294 uc801uc0c1 ub3d9uc791 ud655uc778 (`chat_service=None` ub610ub294 commit ubc1cuc0dd uc2dc upstream uc6d0 ASR uacbdub85c)

## ud30cuc77c ucd5cuc885 uc0c1ud0dc
```
ucd94uac00: vllm_omni/entrypoints/openai/realtime/omni_connection.py
uc218uc815: vllm_omni/entrypoints/openai/api_server.py (/v1/realtime ub77cuc6b0ud2b8)
uc218uc815: vllm_omni/entrypoints/openai/serving_chat.py (_create_audio_choice stream=True PCM ud3ecub9f7)
```

## uc120ud589 uc870uc0ac uac00ud0c4 uc0acud56d (Build uc804 uc218ud589)
1. `vllm_omni/entrypoints/openai/serving_chat.py` uc2a4ud2b8ub9acubc0d uc624ub514uc624 ucf54ub4dc ubd84uc11d
2. uc2a4ud2b8ub9acubc0d uc2dc uc624ub514uc624 ucf54ub4dcuac00 uc5b4ub5bbuac8c exposed ub418ub294uc9c0 uc2e4uc81c ud655uc778
3. `raw_request=None` uc2e4uc81c ub3d9uc791 ud655uc778
