# audio_loader.py
# -*- coding: utf-8 -*-
"""
カレントディレクトリの音源ファイル(MP3/WAV)を一覧化して、
選択 → BPM推定 → 拍位置(beat_times)を返すミニユーティリティ
"""

import os

import librosa
import numpy as np

# 対応ファイル
AUDIO_EXT = (".mp3", ".wav", ".m4a")

def list_audio_files():
    files = [f for f in os.listdir(".") if f.lower().endswith(AUDIO_EXT)]
    return files

def load_audio(path):
    # librosaで読み込む（モノラル, sr=44.1k固定）
    y, sr = librosa.load(path, sr=44100, mono=True)
    return y, sr

def analyze_bpm_and_beats(y, sr):
    # 1) onset envelope（音の打点）
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    # 2) BPM推定
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        units='frames'
    )

    # 3) beatフレーム→秒に変換
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return tempo, beat_times

def pick_audio_interactively():
    files = list_audio_files()

    if not files:
        print("音源ファイル(mp3/wav/m4a)が見つからんかった…")
        return None

    print("=== 見つかった音源一覧 ===")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")

    idx = int(input("使う曲の番号を入力してくれ: "))
    return files[idx]

# 実行テスト
if __name__ == "__main__":
    fname = pick_audio_interactively()
    if fname is None:
        exit()

    print(f"\n>>> '{fname}' を解析中…")

    y, sr = load_audio(fname)
    bpm, beats = analyze_bpm_and_beats(y, sr)

    print(f"\nBPM推定値：{bpm:.1f}")
    print(f"最初の10拍(秒)：{beats[:10]}")
    print(f"総拍数：{len(beats)}")
