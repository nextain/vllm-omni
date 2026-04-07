# Plan: Issue #5 u2014 OpenAI Realtime API ud480 uc774ubca4ud2b8 ud504ub85cud1a0ucf5c ud655uc7a5

## ubaa9uc801
vllm-omniuc5d0 OpenAI Realtime API ud45cuc900uc758 ud480 ub300ud654 uc774ubca4ud2b8ub97c ucd94uac00.
upstream vllm `protocol.py`ub294 ASRuc804uc6a9 uc774ubca4ud2b8ub9cc uc788uc74c u2014 uc9c1uc811 uc218uc815 ubd88uac00.

## ubd84uc11d uc694uc57d

### upstream vllm protocol.py ud604uc7ac uc774ubca4ud2b8 (uc218uc815 ubd88uac00)

**Client u2192 Server:**
- `InputAudioBufferAppend` (input_audio_buffer.append)
- `InputAudioBufferCommit` (input_audio_buffer.commit)
- `SessionUpdate` (session.update)

**Server u2192 Client:**
- `SessionCreated` (session.created)
- `TranscriptionDelta` (transcription.delta)
- `TranscriptionDone` (transcription.done)
- `ErrorEvent` (error)

### ucd94uac00ud560 uc774ubca4ud2b8 (OpenAI Realtime API ud45cuc900)

**Client u2192 Server (uc2e0uaddc):**
- `response.create` u2014 LLM+TTS uc751ub2f5 uc0dduc131 uc694uccad
- `response.cancel` u2014 uc778ud130ub7fdud2b8 (uc751ub2f5 ucde8uc18c)

**Server u2192 Client (uc2e0uaddc):**
- `response.created` u2014 uc751ub2f5 uc2dcuc791 uc54cub9bc
- `response.audio.delta` u2014 base64 PCM16 uc624ub514uc624 uccadud06c (24kHz)
- `response.audio.done` u2014 uc624ub514uc624 uc2a4ud2b8ub9bc uc885ub8cc
- `response.audio_transcript.delta` u2014 ubaa8ub378 ubc1cud654 ud14duc2a4ud2b8 (uc2a4ud2b8ub9acubc0d)
- `response.audio_transcript.done` u2014 ubc1cud654 ud14duc2a4ud2b8 uc644ub8cc
- `response.done` u2014 uc804uccb4 uc751ub2f5 uc644ub8cc
- `response.cancelled` u2014 uc778ud130ub7fdud2b8ub85c ucde8uc18cub428
- `input_audio_buffer.speech_started` u2014 VAD ubc1cud654 uc2dcuc791 uac10uc9c0
- `input_audio_buffer.speech_stopped` u2014 VAD ubc1cud654 uc885ub8cc uac10uc9c0

## uc791uc5c5 uacc4ud68d

### uc0dduc131ud560 ud30cuc77c
`vllm_omni/entrypoints/openai/realtime/protocol.py`

### uae30uc900: upstream vllm protocol.py ud328ud134 ub3d9uc77cud558uac8c ub530ub978ub2e4

```python
# vllm_omni/entrypoints/openai/realtime/protocol.py
"""
vllm-omni extension of vllm realtime protocol.
Adds full-conversation events per OpenAI Realtime API spec.

Reference: https://platform.openai.com/docs/api-reference/realtime-client-events

Base events (from vllm):
  InputAudioBufferAppend, InputAudioBufferCommit, SessionUpdate,
  SessionCreated, TranscriptionDelta, TranscriptionDone, ErrorEvent

Added here:
  Client: ResponseCreate, ResponseCancel
  Server: ResponseCreated, ResponseAudioDelta, ResponseAudioDone,
          ResponseAudioTranscriptDelta, ResponseAudioTranscriptDone,
          ResponseDone, ResponseCancelled,
          InputAudioBufferSpeechStarted, InputAudioBufferSpeechStopped
"""
from typing import Literal, Optional
from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel
from pydantic import Field
import time
from vllm.utils import random_uuid


# ============================================================
# Client u2192 Server (uc2e0uaddc)
# ============================================================

class ResponseCreate(OpenAIBaseModel):
    """Client uc694uccad: uc751ub2f5 uc0dduc131 uc2dcuc791.
    
    uc0acuc6a9uc790uac00 uc9c1uc811 uc694uccadud558ub294 uacbduc6b0 ub610ub294 VAD ubc1cud654 uc885ub8cc ud6c4
    ud074ub77cuc774uc5b8ud2b8uac00 uc790ub3d9 uc804uc1a1.
    """
    type: Literal["response.create"] = "response.create"


class ResponseCancel(OpenAIBaseModel):
    """Client uc694uccad: uc9c4ud589 uc911uc778 uc751ub2f5 ucde8uc18c (uc778ud130ub7fdud2b8)."""
    type: Literal["response.cancel"] = "response.cancel"


# ============================================================
# Server u2192 Client (uc2e0uaddc)
# ============================================================

class ResponseCreated(OpenAIBaseModel):
    """Server: uc751ub2f5 uc2dcuc791 uc54cub9bc."""
    type: Literal["response.created"] = "response.created"
    response_id: str = Field(
        default_factory=lambda: f"resp-{random_uuid()}"
    )


class ResponseAudioDelta(OpenAIBaseModel):
    """Server: PCM16 uc624ub514uc624 uccadud06c uc2a4ud2b8ub9acubc0d.
    
    audio: base64uc73cub85c uc778ucf54ub529ub41c PCM16 (24kHz, mono)
    """
    type: Literal["response.audio.delta"] = "response.audio.delta"
    delta: str  # base64 encoded PCM16 24kHz


class ResponseAudioDone(OpenAIBaseModel):
    """Server: uc624ub514uc624 uc2a4ud2b8ub9bc uc885ub8cc."""
    type: Literal["response.audio.done"] = "response.audio.done"


class ResponseAudioTranscriptDelta(OpenAIBaseModel):
    """Server: ubaa8ub378 ubc1cud654 ud14duc2a4ud2b8 uc2a4ud2b8ub9acubc0d."""
    type: Literal["response.audio_transcript.delta"] = "response.audio_transcript.delta"
    delta: str


class ResponseAudioTranscriptDone(OpenAIBaseModel):
    """Server: ubc1cud654 ud14duc2a4ud2b8 uc644ub8cc."""
    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    transcript: str  # uc804uccb4 ubc1cud654 ud14duc2a4ud2b8


class ResponseDone(OpenAIBaseModel):
    """Server: uc804uccb4 uc751ub2f5 uc644ub8cc."""
    type: Literal["response.done"] = "response.done"


class ResponseCancelled(OpenAIBaseModel):
    """Server: uc778ud130ub7fdud2b8ub85c uc751ub2f5 ucde8uc18cub428."""
    type: Literal["response.cancelled"] = "response.cancelled"


class InputAudioBufferSpeechStarted(OpenAIBaseModel):
    """Server: VADuac00 ubc1cud654 uc2dcuc791 uac10uc9c0. Serveru2192Client uc774ubca4ud2b8.
    
    uc11cubc84 VADuac00 ubc1cud654 uc2dcuc791uc744 uac10uc9c0ud588uc744 ub54c uc11cubc84uac00 ud074ub77cuc774uc5b8ud2b8uc5d0 uc804uc1a1.
    Naiaub294 ud074ub77cuc774uc5b8ud2b8 uce21 Silero VADub97c uc0acuc6a9ud558uc9c0ub9cc, OpenAI Realtime API ud45cuc900uc5d0 ub530ub77c uc815uc758.
    """
    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"


class InputAudioBufferSpeechStopped(OpenAIBaseModel):
    """Server: VADuac00 ubc1cud654 uc885ub8cc uac10uc9c0. Serveru2192Client uc774ubca4ud2b8.
    
    uc11cubc84 VADuac00 ubc1cud654 uc885ub8ccub97c uac10uc9c0ud588uc744 ub54c uc11cubc84uac00 ud074ub77cuc774uc5b8ud2b8uc5d0 uc804uc1a1.
    """
    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
```

### uac80uc99d ubc29ubc95
1. import uc624ub958 uc5c6uc5b4uc57c ud568:
   ```bash
   python -c "from vllm_omni.entrypoints.openai.realtime.protocol import ResponseAudioDelta, ResponseDone, ResponseCreated; print('ok')"
   ```
2. ud544ub4dc uc815ud655uc131 uac80uc99d (type ud544ub4dcuac12 + delta ud544ub4dcuba85 ud655uc778):
   ```bash
   python -c "from vllm_omni.entrypoints.openai.realtime.protocol import ResponseAudioDelta; d = ResponseAudioDelta(delta='test'); j = d.model_dump_json(); import json; obj = json.loads(j); assert obj['type'] == 'response.audio.delta'; assert 'delta' in obj; print('fields OK')"
   ```
3. ub610ud55c ub4f1ub85d uc774ubca4ud2b8 uc77cucfb45 ud655uc778:
   ```bash
   python -c "from vllm_omni.entrypoints.openai.realtime.protocol import ResponseCreated, ResponseAudioTranscriptDelta, ResponseDone, ResponseCancelled; print('all events OK')"
   ```

## uc8fcuc758uc0acud56d
- upstream `OpenAIBaseModel` import uacbdub85c: `from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel` (uc2e4uc81c upstream protocol.py uc5d0uc11c ud655uc778ub428)
- `random_uuid`: `from vllm.utils import random_uuid`
- ud544ub4dcuba85uc740 OpenAI Realtime API uc2a4ud399 ud3ecub9f7uacfc uc815ud655ud788 uc77cuce58
- `ResponseAudioDelta.delta` ud544ub4dcuba85: OpenAI uc2a4ud399 ud655uc778 uc644ub8cc u2014 `delta` (audio buffer uccadud06cuc5d0 `audio`uac00 uc544ub2cc `delta` uc0acuc6a9)

## ud30cuc77c ucd5cuc885 uc0c1ud0dc
```
ucd94uac00: vllm_omni/entrypoints/openai/realtime/protocol.py (uc2e0uaddc)
uc218uc815: uc5c6uc74c (uae30uc874 ub77cuc6b0ud2b8 uac74ub4dcub9acuc9c0 uc54auc74c)
```
