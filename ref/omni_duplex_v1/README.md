# /v1/omni Duplex V1 (Archived)

This directory contains the original `/v1/omni` WebSocket implementation that has been
replaced by the OpenAI Realtime API-compatible `/v1/realtime` endpoint.

## Why it was removed

- `/v1/omni` used a proprietary protocol (PCM16 binary frames + custom JSON events)
- `/v1/realtime` uses the standard OpenAI Realtime API protocol, enabling compatibility
  with existing clients and SDKs
- Maintaining two WebSocket endpoints for the same functionality was unnecessary

## Replacement

Use `/v1/realtime` with `OmniRealtimeConnection`. See:
- `vllm_omni/entrypoints/openai/realtime/omni_connection.py`
- `examples/online_serving/minicpm_o/realtime_e2e_test.py`

## Archive date

2026-04-08
