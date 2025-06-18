import random

import numpy as np
from emotion_player import play_emotion_from_score

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

def move(pos, action):
    x, y = pos
    if action == 'up': x -= 1
    elif action == 'down': x += 1
    elif action == 'left': y -= 1
    elif action == 'right': y += 1
    if 0 <= x < 3 and 0 <= y < 4 and maze[x][y] != -1:
        return (x, y)
    return pos

for episode in range(10):
    pos = start
    print(f"\n🌱 Episode {episode+1}")
    for step in range(50):
        a = random.randint(0, 3) if random.random() < epsilon else np.argmax(q_table[pos[0], pos[1]])
        next_pos = move(pos, actions[a])
        reward = 1 if next_pos == goal else 0

        expected = np.max(q_table[pos[0], pos[1]])
        actual = reward + gamma * np.max(q_table[next_pos[0], next_pos[1]])
        emotion_score = actual - expected

        play_emotion_from_score(actual * 1000)  # 周波数スケールに変換して再生

        q_table[pos[0], pos[1], a] += alpha * (actual - q_table[pos[0], pos[1], a])
        pos = next_pos
        if pos == goal:
            print("🎯 ゴール達成！")
            break
