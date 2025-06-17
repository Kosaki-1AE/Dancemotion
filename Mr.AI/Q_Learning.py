import numpy as np
import random

# 環境定義（3×4のグリッド）
maze = [
    [0, 0, 0, 1],   # ゴールは(0,3)
    [0, -1, 0, -1], # -1は壁
    [0, 0, 0, 0]    # スタートは(2,0)
]
goal = (0, 3)
start = (2, 0)
actions = ['up', 'down', 'left', 'right']

# Qテーブル初期化（状態は3x4、各状態で4つの行動）
q_table = np.zeros((3, 4, len(actions)))

# 次の位置を計算する関数
def move(pos, action):
    x, y = pos
    if action == 'up': x -= 1
    elif action == 'down': x += 1
    elif action == 'left': y -= 1
    elif action == 'right': y += 1
    if 0 <= x < 3 and 0 <= y < 4 and maze[x][y] != -1:
        return (x, y)
    return pos  # 壁 or 外に出たら元の位置

# 学習パラメータ
alpha = 0.1    # 学習率
gamma = 0.9    # 割引率
epsilon = 0.2  # 探索率（ε-greedy）

# 学習ループ
for episode in range(500):
    pos = start
    for _ in range(100):
        # ε-greedyで行動を選ぶ
        if random.random() < epsilon:
            a = random.randint(0, 3)
        else:
            a = np.argmax(q_table[pos[0], pos[1]])
        next_pos = move(pos, actions[a])
        reward = 1 if next_pos == goal else 0
        q_table[pos[0], pos[1], a] += alpha * (
            reward + gamma * np.max(q_table[next_pos[0], next_pos[1]]) - q_table[pos[0], pos[1], a]
        )
        pos = next_pos
        if pos == goal:
            break

# 結果のQテーブルを出力
print("学習後のQテーブル：")
print(np.round(q_table, 2))
