#!/usr/bin/env python3
'''
diag_codec_tokens.py — Talker codec token 진단

vllm-omni 서버 로그에서 codec 토큰을 포착해서 Talker의 출력이
올바른지 확인합니다.

Run:
  # 서버 실행 중 다른 터미널에서:
  python3 scripts/diag_codec_tokens.py

Expected healthy codec tokens:
  - Range: mostly 0-6560 (6561 is stop token, should appear at the end)
  - Not all zeros, not all same value
  - Length: 50-500 tokens for a short sentence
  - Stop token 6561 should appear near the end
'''

import os
from pathlib import Path

SERVER_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
MODEL = os.environ.get("VLLM_MODEL", "openbmb/MiniCPM-o-4_5")


def check_server_health():
    """Check if server is running.

    NOTE: Always called from capture_codec_tokens() after its dependency
    import block — requests is guaranteed importable by that point.
    The bare except here covers network errors only, not ImportError.
    """
    try:
        import requests
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def capture_codec_tokens():
    """
    Make a request and analyze the codec tokens in the response.
    Uses the internal debug endpoint if available, otherwise analyzes audio.
    """
    try:
        import base64
        import numpy as np
        import requests
        import wave
        import io
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return

    print("=== Codec Token Diagnosis ===")
    print(f"Server: {SERVER_URL}")
    print(f"Model: {MODEL}")

    if not check_server_health():
        print("\nERROR: Server not running. Start with:")
        print("  bash scripts/serve_async_chunk.sh  # or see README for setup options")
        return

    # Request with WAV output (non-streaming) to analyze audio quality
    test_text = "Hello, how are you doing today?"
    print(f"\nPrompt: {test_text!r}")

    print("\n[1] Non-streaming WAV test...")
    try:
        resp = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": test_text}],
                "modalities": ["audio", "text"],
                "audio": {"format": "wav"},
                "stream": False,
                "max_tokens": 100,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            return
        data = resp.json()
        if "error" in data:
            print(f"  API error: {data['error']}")
            return

        # vllm-omni returns text and audio in separate choices
        choices = data.get("choices")
        if not choices:
            print(f"  ERROR: No choices in response: {str(data)[:200]}")
            return

        text = None
        audio_b64 = None
        for choice in choices:
            msg = choice.get("message", {})
            if msg.get("content"):
                text = msg["content"]
            if msg.get("audio") and msg["audio"].get("data"):
                audio_b64 = msg["audio"]["data"]

        print(f"  Text: {text!r}")

        if not audio_b64:
            print("  WARNING: No audio in response")
            print(f"  Choices: {len(choices)}, keys: {[list(c.get('message', {}).keys()) for c in choices]}")
            return

        wav_bytes = base64.b64decode(audio_b64)
        print(f"  WAV bytes: {len(wav_bytes)}")

        # Parse WAV
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        print(f"  WAV: {sr}Hz, {channels}ch, {sampwidth*8}bit, {n_frames} frames = {n_frames/sr:.2f}s")

        # Analyze audio quality
        if sampwidth == 2:
            audio_int16 = np.frombuffer(raw, dtype=np.int16)
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
        elif sampwidth == 4:
            # Assumes float32 PCM. Standard 32-bit WAV is int32, but vllm-omni
            # uses float32 internally u2014 reinterpret if values look out of range.
            # If RMS/Peak appear unreasonably large (>1e4), the WAV may be int32
            # not float32; rerun with dtype=np.int32 and divide by 2**31.
            audio_f32 = np.frombuffer(raw, dtype=np.float32)
        else:
            print(f"  WARNING: Unexpected sample width {sampwidth}")
            return

        rms = np.sqrt(np.mean(audio_f32 ** 2))
        peak = np.abs(audio_f32).max()
        print(f"  Audio: RMS={rms:.4f}, Peak={peak:.4f}")

        # Check for common gibberish patterns
        if rms < 0.001:
            print("  DIAGNOSIS: Near-silent audio — possible Code2Wav or flow not generating")
        elif rms > 0.3:
            print("  DIAGNOSIS: Very loud audio — possible clipping or wrong format")
        else:
            print("  Audio amplitude looks reasonable")

        # Spectral analysis to detect gibberish (noise) vs speech.
        # No window applied u2014 spectral leakage may inflate high-freq noise slightly,
        # making the SNR estimate conservative. Threshold 2.0 accounts for this.
        # For typical max_tokens=100 responses (<5 s at 24kHz ~120k samples) the
        # FFT is fast; longer inputs would still complete but take a few seconds.
        if len(audio_f32) > 1024:
            freqs = np.fft.rfftfreq(len(audio_f32), d=1/sr)
            magnitudes = np.abs(np.fft.rfft(audio_f32))
            # Speech energy is concentrated in 80-8000 Hz
            speech_band = (freqs >= 80) & (freqs <= 8000)
            noise_band = freqs > 8000
            speech_energy = magnitudes[speech_band].mean()
            noise_energy = magnitudes[noise_band].mean()
            snr_estimate = speech_energy / (noise_energy + 1e-9)
            print(f"  Speech/noise energy ratio: {snr_estimate:.2f}")
            if snr_estimate < 2.0:
                print("  DIAGNOSIS: Low speech/noise ratio — likely gibberish (noise dominates)")
            else:
                print("  DIAGNOSIS: Higher speech-band energy — likely intelligible speech")

        # Save for listening (intentionally in /tmp — diagnostic tool, not persistent storage)
        out_path = Path("/tmp/diag_audio/diag_server_output.wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_bytes)
        print(f"\n  Saved: {out_path}")
        print(f"  Play: ffplay {out_path}")

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    capture_codec_tokens()
