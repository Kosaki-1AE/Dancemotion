# -*- coding: utf-8 -*-
"""
拍量子仮説の最小実装
- リズム = 拍量子の系列
- Stillness = 発音しない確率
- Responsibility Arrow = 位相バイアス（早め/遅め）

出力:
  straight.wav (均等拍クリック)
"""

import numpy as np
import wave

# ====== 拍量子モデル ======
class BeatQuantum:
    def __init__(self, duration, velocity=1.0, phase=0.0):
        self.duration = duration  # 秒
        self.velocity = velocity  # 音量(0=休符)
        self.phase = phase        # 位相(秒)

# ====== パラメータ ======
tempo_bpm = 120
bars = 4
beats_per_bar = 4
stillness_prob = 0.2      # 20%は休符
responsibility_phase_ms = 0.0  # 責任の矢 (正なら遅れる)
sr = 44100
click_freq = 880
click_ms = 30
# ====== 関数 ======
def beat_duration(bpm):
    return 60.0 / bpm

def sine_click(freq, dur_ms, sr):
    t = np.linspace(0, dur_ms/1000, int(sr*dur_ms/1000), endpoint=False)
    env = np.linspace(1, 0, len(t))  # フェードアウト
    return np.sin(2*np.pi*freq*t) * env

def render(quanta, sr):
    total = sum(q.duration for q in quanta) + 1.0
    audio = np.zeros(int(sr*total))
    click = sine_click(click_freq, click_ms, sr)

    t_cursor = 0.0
    for q in quanta:
        if q.velocity > 0:
            idx = int(sr*(t_cursor+q.phase))
            audio[idx:idx+len(click)] += q.velocity * click[:len(audio)-idx]
        t_cursor += q.duration
    return audio / np.max(np.abs(audio))

def save_wav(path, audio, sr):
    audio = (audio*32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())

# ====== 実行 ======
beat_sec = beat_duration(tempo_bpm)
quanta = []
rng = np.random.default_rng()

for _ in range(bars*beats_per_bar):
    # Stillness (確率で休符)
    if rng.random() < stillness_prob:
        quanta.append(BeatQuantum(beat_sec, velocity=0.0))
    else:
        phase = responsibility_phase_ms/1000.0
        quanta.append(BeatQuantum(beat_sec, velocity=1.0, phase=phase))

audio = render(quanta, sr)
save_wav("straight.wav", audio, sr)
print("[ok] straight.wav saved.")
