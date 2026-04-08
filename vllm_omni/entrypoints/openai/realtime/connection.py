from vllm.entrypoints.openai.realtime.connection import (
    RealtimeConnection as UpstreamRealtimeConnection,
)

# Re-export upstream RealtimeConnection so that omni_connection.py
# can import from here without knowing whether it's local or upstream.
RealtimeConnection = UpstreamRealtimeConnection

__all__ = ["RealtimeConnection"]
