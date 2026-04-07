#!/usr/bin/env python3
"""Phase 2: KsponSpeech 2h 서브셋 전처리

입력: /var/home/luke/data/aihub/10.한국어음성/KsponSpeech_01.zip
출력: data/ko_2h/train.list, data/ko_2h/dev.list, data/ko_2h/wavs/

실행: python3 tools/ko_finetune/preprocess_kspon.py
"""
import os, zipfile, re, argparse, random
import soundfile as sf
import numpy as np
from pathlib import Path

ZIP_PATH = '/var/home/luke/data/aihub/10.한국어음성/KsponSpeech_01.zip'
OUT_DIR   = 'data/ko_2h'
WAV_DIR   = f'{OUT_DIR}/wavs'
N_SAMPLES = 1240          # 앞 1,240개 (약 2h)
MIN_SEC   = 2.0
MAX_SEC   = 15.0
DEV_RATIO = 0.2           # 20% dev set

# KsponSpeech 특수토큰 제거 패턴
SPECIAL_TOKEN_RE = re.compile(r'[bn]?/|\+|[\(\)]|\*')

def clean_text(raw: str) -> str:
    """cp949 텍스트 특수토큰 제거"""
    text = SPECIAL_TOKEN_RE.sub('', raw)
    return re.sub(r'\s+', ' ', text).strip()

def pcm_to_wav(pcm_bytes: bytes, wav_path: str, sr: int = 16000) -> float:
    """16kHz 16bit mono PCM -> WAV, 길이(초) 반환"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(wav_path, samples, sr, subtype='PCM_16')
    return len(samples) / sr

def main():
    os.makedirs(WAV_DIR, exist_ok=True)

    entries = []  # (utt_id, wav_path, text)
    filtered_short = filtered_long = 0

    print(f'ZIP 열기: {ZIP_PATH}')
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        # PCM 파일 목록 수집 (알파벳 정렬)
        pcm_names = sorted(
            n for n in zf.namelist()
            if n.lower().endswith('.pcm')
        )[:N_SAMPLES]

        print(f'대상 PCM 파일: {len(pcm_names)}개')

        for pcm_name in pcm_names:
            # 대응 TXT 경로
            txt_name = pcm_name.replace('.pcm', '.txt').replace('.PCM', '.txt')
            if txt_name not in zf.namelist():
                continue

            utt_id = Path(pcm_name).stem
            wav_path = os.path.join(WAV_DIR, f'{utt_id}.wav')

            # PCM -> WAV
            pcm_bytes = zf.read(pcm_name)
            duration = pcm_to_wav(pcm_bytes, wav_path)

            # 오디오 품질 필터
            if duration < MIN_SEC:
                filtered_short += 1
                os.remove(wav_path)
                continue
            if duration > MAX_SEC:
                filtered_long += 1
                os.remove(wav_path)
                continue

            # TXT 디코딩 + 특수토큰 제거
            raw = zf.read(txt_name).decode('cp949', errors='ignore')
            text = clean_text(raw)
            if not text:
                continue

            entries.append((utt_id, os.path.abspath(wav_path), text))

    print(f'
필터링 결과:')
    print(f'  유효 샘플: {len(entries)}개')
    print(f'  제외 (< {MIN_SEC}s): {filtered_short}개')
    print(f'  제외 (> {MAX_SEC}s): {filtered_long}개')

    # 80/20 분할
    random.seed(42)
    random.shuffle(entries)
    split = int(len(entries) * (1 - DEV_RATIO))
    train_entries = entries[:split]
    dev_entries   = entries[split:]

    # list 파일 작성 (TSV: utt_id	path	text)
    def write_list(path, items):
        with open(path, 'w', encoding='utf-8') as f:
            for utt_id, wav_path, text in items:
                f.write(f'{utt_id}	{wav_path}	{text}
')

    write_list(f'{OUT_DIR}/train.list', train_entries)
    write_list(f'{OUT_DIR}/dev.list',   dev_entries)

    print(f'
✅ train.list: {len(train_entries)}개')
    print(f'✅ dev.list:   {len(dev_entries)}개')
    print(f'✅ WAV 디렉터리: {WAV_DIR}')

if __name__ == '__main__':
    main()
