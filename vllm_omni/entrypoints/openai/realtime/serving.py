from vllm.entrypoints.openai.realtime.serving import (
    OpenAIServingRealtime as UpstreamServingRealtime,
)


class OpenAIServingRealtime(UpstreamServingRealtime):
    """Extended OpenAI Realtime serving for omni models (audio in -> audio out).

    Inherits from upstream's OpenAIServingRealtime which handles standard
    ASR-only realtime. This subclass adds omni model awareness for the
    full audio conversation loop.
    """

    pass
