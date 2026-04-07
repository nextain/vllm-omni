#!/usr/bin/env python3
"""Phase 2: KsponSpeech 2h 서브셋 전처리

입력: /var/home/luke/data/aihub/10.한국어음성/KsponSpeech_01.zip
출력: data/ko_2h/{train,dev}/{wav.scp,text,utt2spk}
         (CosyVoice make_parquet_list.py와 호환하는 Kaldi 포맷)

실행: python3 tools/ko_finetune/preprocess_kspon.py
"""
import os, sys, zipfile, re, random
import soundfile as sf
import numpy as np
from pathlib import Path
from g2pk import G2p

ZIP_PATH = '/var/home/luke/data/aihub/10.\ud55c\uad6d\uc5b4\uc74c\uc131/KsponSpeech_01.zip'
OUT_DIR   = 'data/ko_2h'
WAV_DIR   = f'{OUT_DIR}/wavs'
N_SAMPLES = 1240
MIN_SEC   = 2.0
MAX_SEC   = 15.0
DEV_RATIO = 0.2

SPK_ID = 'KsponSpeech'  # 관레 스피커 ID (utt2spk 사용)

# KsponSpeech 특수토큰 제거 패턴 (b/ n/ l/ o/ / + ~ * 및 괄호 내 잡음 표기)
SPECIAL_TOKEN_RE = re.compile(r'\([^)]*\)|[bnlo]/|[/+~*]')

_g2p = G2p()

def clean_text(raw: str) -> str:
    """cp949 텍스트 특수토큰 제거"""
    text = SPECIAL_TOKEN_RE.sub('', raw)
    return re.sub(r'\s+', ' ', text).strip()

def pcm_to_wav(pcm_bytes: bytes, wav_path: str, sr: int = 16000) -> float:
    """16kHz 16bit mono PCM -> WAV, 길이(초) 반환"""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write(wav_path, samples, sr, subtype='PCM_16')
    return len(samples) / sr

def write_kaldi_files(split_dir: str, items: list):
    """Kaldi 포맷 파일 작성 (CosyVoice make_parquet_list.py 입력 포맷)"""
    os.makedirs(split_dir, exist_ok=True)
    with open(f'{split_dir}/wav.scp', 'w', encoding='utf-8') as f:
        for utt_id, wav_path, _ in items:
            f.write(f'{utt_id}\t{wav_path}\n')
    with open(f'{split_dir}/text', 'w', encoding='utf-8') as f:
        for utt_id, _, phoneme_text in items:
            f.write(f'{utt_id}\t{phoneme_text}\n')
    with open(f'{split_dir}/utt2spk', 'w', encoding='utf-8') as f:
        for utt_id, _, _ in items:
            f.write(f'{utt_id}\t{SPK_ID}\n')

def main():
    # CWD 자기 강제화: tools/ko_finetune/ 존재 = repo 루트 확인
    if not os.path.isdir('tools/ko_finetune'):
        sys.exit(f'❌ repo 루트에서 실행할 것 (현재: {os.getcwd()})')

    os.makedirs(WAV_DIR, exist_ok=True)

    entries = []  # (utt_id, wav_path_abs, phoneme_text)
    filtered_short = filtered_long = 0

    print(f'ZIP \uc5f4\uae30: {ZIP_PATH}')
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        name_set = set(zf.namelist())
        pcm_names = sorted(
            n for n in name_set
            if n.lower().endswith('.pcm')
        )[:N_SAMPLES]

        print(f'\ub300\uc0c1 PCM \ud30c\uc77c: {len(pcm_names)}\uac1c')

        for pcm_name in pcm_names:
            stem = pcm_name[:-4]
            txt_name = stem + '.txt'
            if txt_name not in name_set:
                txt_name = stem + '.TXT'
            if txt_name not in name_set:
                continue

            utt_id = Path(pcm_name).stem
            wav_path_abs = os.path.abspath(os.path.join(WAV_DIR, f'{utt_id}.wav'))

            pcm_bytes = zf.read(pcm_name)
            duration = pcm_to_wav(pcm_bytes, wav_path_abs)

            if duration < MIN_SEC:
                filtered_short += 1; os.remove(wav_path_abs); continue
            if duration > MAX_SEC:
                filtered_long += 1; os.remove(wav_path_abs); continue

            raw = zf.read(txt_name).decode('cp949', errors='ignore')
            text = clean_text(raw)
            if not text:
                continue

            phoneme_text = _g2p(text)
            # Kaldi wav.scp requires absolute paths to be safe across CWD changes
            entries.append((utt_id, wav_path_abs, phoneme_text))

    print(f'\n\ud544\ud130\ub9c1 \uacb0\uacfc:')
    print(f'  \uc720\ud6a8 \uc0d8\ud50c: {len(entries)}\uac1c')
    print(f'  \uc81c\uc678 (< {MIN_SEC}s): {filtered_short}\uac1c')
    print(f'  \uc81c\uc678 (> {MAX_SEC}s): {filtered_long}\uac1c')

    random.seed(42)
    random.shuffle(entries)
    split = int(len(entries) * (1 - DEV_RATIO))

    write_kaldi_files(f'{OUT_DIR}/train', entries[:split])
    write_kaldi_files(f'{OUT_DIR}/dev',   entries[split:])

    print(f'\n\u2705 train/: {split}\uac1c')
    print(f'\u2705 dev/:   {len(entries) - split}\uac1c')
    print(f'\u2705 \ub2e4\uc74c \ub2e8\uacc4: train_cosyvoice2_ko.sh \uc2e4\ud589 (\ud30c\ucf08 \ubcc0\ud658 \ud3ec\ud568)')

if __name__ == '__main__':
    main()
