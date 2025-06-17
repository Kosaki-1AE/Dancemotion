import os

import numpy as np
from scipy.io.wavfile import write

# 各音階（C4～B4）の周波数（Hz）
notes_freq = {
    'C': 261.63,
    'i_F1': 270.00,
    'C#': 277.18,
    'D': 293.66,
    'u_F1': 300.00,
    'D#': 311.13,
    'E': 329.63,
    'F': 349.23,
    'F#': 369.99,
    'G': 392.00,
    'G#': 415.30,
    'A': 440.00,
    'A#': 466.16,
    'B': 493.88,
    'e_F1': 530.00,
    'o_F1': 570.00,
    'a_F1': 730.00,
    'o_F2': 840.00,
    'u_F2': 870.00,
    'a_F2': 1090.00,
    'e_F2': 1840.00,
    'u_F3': 2240.00,
    'i_F2': 2290.00,
    'o_F3': 2410.00,
    'a_F3': 2440.00,
    'e_F3': 2480.00,
    'i_F3': 3010.00
}

# パラメータ
duration = 1.0  # 1秒
samplerate = 44100

# 出力フォルダ
os.makedirs("notes", exist_ok=True)

for note, freq in notes_freq.items():
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    wave = np.int16(wave * 32767)
    write(f"scales/{note}.wav", samplerate, wave)

print("✅ ドレミ音階のWAV音源を生成しました！notesフォルダをチェック！")
