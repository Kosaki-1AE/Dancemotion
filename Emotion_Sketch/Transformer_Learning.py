import random

import numpy as np

# 環境定義（3×4のグリッド）状況確認
maze = [
    [0, 0, 0, 1],   # ゴールは(0,3)
    [0, -1, 0, -1], # -1は壁
    [0, 0, 0, 0]    # スタートは(2,0)
]
goal = (0, 3)
start = (2, 0)
actions = ['up', 'down', 'left', 'right']

# ソフトマックス関数 迷う確率を決める
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# 迷った結果の状態を表す行列
x = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])

# Qテーブル初期化（状態は3x4、各状態で4つの行動）迷った結果を活かすために一旦初期化
q_table = np.zeros((3, 4, len(actions)))

# 重み初期化 最後の調整はいりまぁす
W_q = np.random.randn(4, 4)
W_k = np.random.randn(4, 4)
W_v = np.random.randn(4, 4)

# Q, K, V 計算
Q = x @ W_q
K = x @ W_k
V = x @ W_v

# Attentionスコア計算
scores = Q @ K.T / np.sqrt(Q.shape[-1])  # スケーリング付き内積
weights = softmax(scores)                # softmaxで重み
output = weights @ V                    # 加重平均で出力

# 次の位置を計算する関数 行動中
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

# 結果のQテーブルを出力 学習後のQテーブル
print("学習後の最適行動ポリシー（矢印だけ）：")
for i in range(3):  # 行（y座標）
    row = ""
    for j in range(4):  # 列（x座標）
        if maze[i][j] == -1:
            row += "■ "  # 壁
        elif maze[i][j] == 1:
            row += "G "  # ゴール
        else:
            best_action = np.argmax(q_table[i][j])
            arrow = ["↑", "↓", "←", "→"][best_action]
            row += f"{arrow} "
    print(row)
