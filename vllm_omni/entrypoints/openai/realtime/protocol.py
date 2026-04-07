"""
vllm-omni extension of vllm realtime protocol.
Adds full-conversation events per OpenAI Realtime API spec.

Reference: https://platform.openai.com/docs/api-reference/realtime-client-events
"""
from typing import Literal, Optional
from vllm.entrypoints.openai.protocol import OpenAIBaseModel
from pydantic import Field
from vllm.utils import random_uuid


# ============================================================
# Client -> Server
# ============================================================

class ResponseCreate(OpenAIBaseModel):
    """Client request: start generating a response."""
    type: Literal["response.create"] = "response.create"


class ResponseCancel(OpenAIBaseModel):
    """Client request: cancel the ongoing response."""
    type: Literal["response.cancel"] = "response.cancel"


# ============================================================
# Server -> Client
# ============================================================

class ResponseCreated(OpenAIBaseModel):
    """Server: response created notification."""
    type: Literal["response.created"] = "response.created"
    response_id: str = Field(
        default_factory=lambda: f"resp-{random_uuid()}"
    )


class ResponseAudioDelta(OpenAIBaseModel):
    """Server: PCM16 audio chunk streaming."""
    type: Literal["response.audio.delta"] = "response.audio.delta"
    delta: str  # base64 encoded PCM16 24kHz


class ResponseAudioDone(OpenAIBaseModel):
    """Server: audio stream done."""
    type: Literal["response.audio.done"] = "response.audio.done"


class ResponseAudioTranscriptDelta(OpenAIBaseModel):
    """Server: model transcript streaming."""
    type: Literal["response.audio_transcript.delta"] = "response.audio_transcript.delta"
    delta: str


class ResponseAudioTranscriptDone(OpenAIBaseModel):
    """Server: model transcript done."""
    type: Literal["response.audio_transcript.done"] = "response.audio_transcript.done"
    transcript: str


class ResponseDone(OpenAIBaseModel):
    """Server: full response done."""
    type: Literal["response.done"] = "response.done"


class ResponseCancelled(OpenAIBaseModel):
    """Server: response cancelled."""
    type: Literal["response.cancelled"] = "response.cancelled"


class InputAudioBufferSpeechStarted(OpenAIBaseModel):
    """Server: VAD detected speech start."""
    type: Literal["input_audio_buffer.speech_started"] = "input_audio_buffer.speech_started"


class InputAudioBufferSpeechStopped(OpenAIBaseModel):
    """Server: VAD detected speech stop."""
    type: Literal["input_audio_buffer.speech_stopped"] = "input_audio_buffer.speech_stopped"
