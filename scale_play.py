import os

import numpy as np
from scipy.io.wavfile import write

# 各音階（C4～B4）の周波数（Hz）
notes_freq = {
    'C': 261.63,
    'C#': 277.18,
    'D': 293.66,
    'D#': 311.13,
    'E': 329.63,
    'F': 349.23,
    'F#': 369.99,
    'G': 392.00,
    'G#': 415.30,
    'A': 440.00,
    'A#': 466.16,
    'B': 493.88
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
