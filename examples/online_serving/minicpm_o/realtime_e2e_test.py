import asyncio
import base64
import json
import argparse
import sys
import time
import numpy as np
import websockets
import soundfile as sf
from pathlib import Path

async def test_realtime_audio_to_audio(audio_path: str, host: str, port: int, model: str):
    uri = f"ws://{host}:{port}/v1/realtime"
    
    print(f"\nConnecting to {uri}...")
    async with websockets.connect(uri) as ws:
        # 1. Wait for session.created
        msg = await ws.recv()
        response = json.loads(msg)
        print(f"Server -> Client: {response['type']}")
        if response["type"] != "session.created":
            print(f"FAILED: Expected session.created, got {response['type']}")
            return

        # 2. Update session (required: upstream session.update needs 'model' field)
        await ws.send(json.dumps({
            "type": "session.update",
            "model": model,
            "session": {
                "instructions": "You are a helpful assistant. Keep your responses very short.",
                "temperature": 0.6
            }
        }))
        print("Client -> Server: session.update")

        # 3. Load and send audio
        print(f"Loading audio: {audio_path}")
        audio_data, sr = sf.read(audio_path)
        if sr != 16000:
            # Simple resample if needed, but we assume 16k for test
            print(f"WARNING: Audio SR is {sr}, expected 16000. Test might fail.")
        
        # PCM16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_b64 = base64.b64encode(audio_int16.tobytes()).decode("utf-8")
        
        # Send in 1s chunks
        chunk_size_samples = 16000
        total_samples = len(audio_int16)
        print(f"Sending {total_samples} samples in chunks...")
        
        for i in range(0, total_samples, chunk_size_samples):
            chunk = audio_int16[i : i + chunk_size_samples]
            payload = base64.b64encode(chunk.tobytes()).decode("utf-8")
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": payload
            }))
        
        # 4. Commit and trigger response
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        print("Client -> Server: input_audio_buffer.commit")

        # 5. Receive response stream
        print("\nReceiving response events:")
        has_audio = False
        has_transcript = False
        transcript = ""
        audio_chunks = []

        t0 = time.perf_counter()
        while True:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                event = json.loads(msg)
                etype = event["type"]
                print(f"  [+{time.perf_counter()-t0:.2f}s] {etype}")

                if etype == "response.audio_transcript.delta":
                    has_transcript = True
                    transcript += event["delta"]
                elif etype == "response.audio.delta":
                    has_audio = True
                    audio_chunks.append(base64.b64decode(event["delta"]))
                elif etype == "response.done":
                    break
                elif etype == "error":
                    print(f"ERROR: {event}")
                    break
            except asyncio.TimeoutError:
                print("TIMEOUT: No response from server for 30s")
                break

        print("\nTest Summary:")
        print(f"  Transcript received: {has_transcript}")
        if has_transcript:
            print(f"  Final Transcript: {transcript}")
        print(f"  Audio received:      {has_audio} ({len(audio_chunks)} chunks)")
        
        if has_audio and has_transcript:
            print("\nRESULT: SUCCESS")
        else:
            print("\nRESULT: FAILED (Missing audio or transcript)")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to input PCM16 16kHz audio")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="openbmb/MiniCPM-o-4_5")
    args = parser.parse_args()
    
    asyncio.run(test_realtime_audio_to_audio(args.audio, args.host, args.port, args.model))
