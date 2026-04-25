# MiniCPM-o Realtime — TypeScript reference client

> **Fork-only.** Sibling of [`realtime_e2e_test.py`](../realtime_e2e_test.py). Not
> proposed upstream — `vllm-project/vllm-omni` examples are Python-only by
> convention. Kept here as a reference for downstream TS/JS consumers (e.g.
> the naia-os Tauri shell at
> [`shell/src/lib/voice/minicpm-o.ts`](https://github.com/nextain/naia-os/blob/main/shell/src/lib/voice/minicpm-o.ts))
> and as wire-protocol regression coverage for the fork.

## What this is

A Node.js + TypeScript client that:

1. Connects to `vllm-omni` `/v1/realtime` (OpenAI Realtime API extended for
   omni models).
2. Sends a 16 kHz mono PCM16 WAV file as `input_audio_buffer.append` chunks,
   commits explicitly, and collects the response stream.
3. Asserts that both `response.audio.delta` (TTS audio) and
   `response.audio_transcript.delta` (text) are emitted, then prints a
   timing summary identical in spirit to `realtime_e2e_test.py`.

Two files:

| File | Role |
|---|---|
| `src/realtime-client.ts` | Reusable client — `runRealtimeTurn(cfg, pcm16Chunks)` returns a structured `ResponseSummary`. |
| `src/e2e-test.ts` | CLI — parses args, loads WAV, calls the client, exits 0/1. |

## Server-side constraints honored

These mirror the lessons recorded in
`.agents/context/contribution-journey.md` (Phase 11 of the MiniCPM-o
contribution journey):

- `server_vad` is declared but unimplemented in `OmniRealtimeConnection`.
  This client always drives responses with **explicit
  `input_audio_buffer.commit`**.
- Multi-turn history is **disabled** server-side (commit `fd273bdc`).
  `runRealtimeTurn` is therefore a single-turn-per-WebSocket function;
  state does not survive between calls.
- The `session.update` envelope carries `model` at the top level, matching
  the Python sibling and the legacy fork extension. The omni handler
  currently reads only the `session.*` subfields.

## Quick start

```bash
# In this directory
npm install

# Generate or supply a 16kHz mono PCM16 WAV (4 seconds is plenty)
espeak-ng -v en -s 145 -w /tmp/in.wav "Hello, please introduce yourself in one short sentence." \
  && ffmpeg -y -loglevel error -i /tmp/in.wav -ar 16000 -ac 1 -sample_fmt s16 /tmp/test_input.wav

# Start vllm-omni in another terminal (see ../README.md)
#   distrobox enter vllm-dev -- bash scripts/serve_async_chunk.sh

# Run the e2e
npm run e2e -- --audio /tmp/test_input.wav --host localhost --port 8000
```

Expected output ends with:

```
RESULT: SUCCESS (~7s wall)
```

## CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--audio <path>` | _required_ | 16 kHz mono PCM16 WAV file. Other formats throw. |
| `--host <host>` | `localhost` | vllm-omni server hostname. |
| `--port <port>` | `8000` | vllm-omni server port. |
| `--model <id>` | `openbmb/MiniCPM-o-4_5` | Model id passed via `session.update`. |

## Differences from the Python sibling

| | Python (`realtime_e2e_test.py`) | TypeScript (this client) |
|---|---|---|
| Sample-rate enforcement | Soft (warns, continues) | Hard (throws) |
| Connection timeout | none (default `wait_for` 30s response only) | 15 s connect + 30 s response |
| URL input | `host` + `port` only | `serverUrl` accepts http(s)://, ws(s):// (with scheme allowlist) |
| Multi-listener safety | n/a | single message handler with explicit `stage` state machine |
| Cancellation | n/a (script-level Ctrl-C) | not yet exposed (future work: `AbortSignal`) |

These deltas are intentional. The TS client doubles as a quality bar for
downstream Tauri/browser clients (which need stricter validation), so it
errors on what the Python sibling lets through.

## Wire-protocol coverage

The `ResponseSummary` returned by `runRealtimeTurn` records every observed
event, so the same harness can be folded into a CI smoke later. Events
asserted by the e2e test:

- `session.created` → `session.update` (handshake)
- `input_audio_buffer.append` × N (1 second each) → `input_audio_buffer.commit`
- `response.created` → `response.audio_transcript.delta` × N → `response.audio.delta` × N → `response.audio_transcript.done` → `response.audio.done` → `response.done`

Server-emitted `error` events terminate the response promise with the
extracted message. Pre-handshake `error` rejects the connect promise with
the server's text — useful when the model is mis-configured.

## License

Apache-2.0, same as the rest of the vllm-omni fork.
