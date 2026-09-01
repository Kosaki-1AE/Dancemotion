import json
import random

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
    name = closest[0]
    print(f"🎵 感情音再生: {name} (freq={freq}, vol={vol}, dur={dur})")
    fs = 44100
    t = np.linspace(0, dur, int(fs * dur), False)
    tone = np.sin(2 * np.pi * freq * t) * vol
    sd.play(tone, fs)

def play_emotion_from_score(score):
    emotions = load_emotions()
    closest = min(emotions.items(), key=lambda item: abs(score - item[1][0]))
    freq, vol, dur = closest[1]
    name = closest[0]
    print(f"🎵 感情音再生: {name} (freq={freq}, vol={vol}, dur={dur})")
    fs = 44100
    t = np.linspace(0, dur, int(fs * dur), False)
    tone = np.sin(2 * np.pi * freq * t) * vol
    sd.play(tone, fs)
    sd.wait()

def detect_sound_level(duration=1.0, fs=44100):
    print("🎙 録音中…（声を出してみて！）")
    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        rms = np.sqrt(np.mean(audio**2))
        print(f"🔊 音量 (RMS): {round(rms, 5)}")
        return rms
    except Exception as e:
        print("⚠️ 録音失敗:", e)
        return 0.0

maze = [
    [0, 0, 0, 1],
    [0, -1, 0, -1],
    [0, 0, 0, 0]
]
goal = (0, 3)
start = (2, 0)
actions = ['up', 'down', 'left', 'right']
q_table = np.zeros((3, 4, len(actions)))

alpha = 0.1
gamma = 0.9
epsilon = 0.1

choice_log = []
identity_score = 0
emotion_log = []
identity_log = []

def move(pos, action):
    x, y = pos
    if action == 'up': x -= 1
    elif action == 'down': x += 1
    elif action == 'left': y -= 1
    elif action == 'right': y += 1
    if 0 <= x < 3 and 0 <= y < 4 and maze[x][y] != -1:
        return (x, y)
    return pos

for episode in range(3):
    pos = start
    print(f"\n🌱 Episode {episode+1}")
    for step in range(50):
        # 自己制御：感情が高ければ積極的、低ければ慎重に
        adaptive_epsilon = epsilon + (0.3 if identity_score < 0 else -0.05)
        if random.random() < adaptive_epsilon:
            a = random.randint(0, 3)
        else:
            a = np.argmax(q_table[pos[0], pos[1]])

        action_name = actions[a]
        next_pos = move(pos, action_name)
        reward = 1 if next_pos == goal else 0

        expected = np.max(q_table[pos[0], pos[1]])
        actual = reward + gamma * np.max(q_table[next_pos[0], next_pos[1]])
        q_table[pos[0], pos[1], a] += alpha * (actual - q_table[pos[0], pos[1], a])

        sound_level = detect_sound_level()
        emotion_score = actual * 1000 + sound_level * 1000
        emotion_log.append(emotion_score)

        play_emotion_from_score(emotion_score)

        choice_log.append((pos, action_name, round(emotion_score, 2)))
        if choice_log.count((pos, action_name, round(emotion_score, 2))) > 1:
            identity_score += 0.5
        else:
            identity_score -= 0.2
        identity_log.append(identity_score)

        print(f"🧠 自我スコア: {round(identity_score, 2)}")
        print(f"📜 選択ログ: 位置{pos} → 行動'{action_name}' → 感情スコア {round(emotion_score, 2)}")

        pos = next_pos
        if pos == goal:
            print("🎯 ゴール達成！")
            break
