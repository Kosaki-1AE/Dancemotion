# emotion_player.py
import json

import numpy as np
import sounddevice as sd

EMOTION_FILE = "emotion_trial.json"

def load_emotions():
    with open(EMOTION_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def play_emotion_from_score(score):
    emotions = load_emotions()
    closest = min(emotions.items(), key=lambda item: abs(score - item[1][0]))
    freq, vol, dur = closest[1]
    print(f"🎵 感情音再生: {closest[0]} (freq={freq}, vol={vol}, dur={dur})")

    fs = 44100
    t = np.linspace(0, dur, int(fs * dur), False)
    tone = np.sin(2 * np.pi * freq * t) * vol
    sd.play(tone, fs)
    sd.wait()
