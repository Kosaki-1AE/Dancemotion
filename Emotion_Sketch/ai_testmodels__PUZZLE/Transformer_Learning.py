import random

import numpy as np

# 人間でいうところの「世界の状況（環境）」を定義する
maze = [
    [0, 0, 0, 1],     # G: 目標（目標達成の報酬が得られる場所）
    [0, -1, 0, -1],   # ■: 壁（進めない制約）
    [0, 0, 0, 0]      # S: スタート地点（ここから思考スタート）
]
goal = (0, 3)
start = (2, 0)
actions = ['up', 'down', 'left', 'right']

# 人間でいうところの「迷い・判断のふわっとさ（選択傾向）」を計算する関数
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# 人間でいうところの「過去経験からくる印象や直感のパターン」
x = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])

# 行動価値の記憶（経験からの重みづけ：いわば“価値地図”）
q_table = np.zeros((3, 4, len(actions)))

# 人間でいうところの「注目の向け方（Attention）」に使う重みたち
W_q = np.random.randn(4, 4)
W_k = np.random.randn(4, 4)
W_v = np.random.randn(4, 4)
Q = x @ W_q
K = x @ W_k
V = x @ W_v
scores = Q @ K.T / np.sqrt(Q.shape[-1])
weights = softmax(scores)
output = weights @ V

# 人間でいうところの「実際に動いてみたときの変化（実行部）」
def move(pos, action):
    x, y = pos
    if action == 'up': x -= 1
    elif action == 'down': x += 1
    elif action == 'left': y -= 1
    elif action == 'right': y += 1
    if 0 <= x < 3 and 0 <= y < 4 and maze[x][y] != -1:
        return (x, y)
    return pos  # 壁や外に出たら変化なし（現実世界の制約）

# 人間でいうところの「学習感度（どれくらい反省するか）」と「未来への期待度」
alpha = 0.1  # 学習率：反省の深さ
gamma = 0.9  # 割引率：未来をどれくらい信じるか
epsilon = 0.2  # 探索率：気まぐれに他の行動を試すかどうか

# 感情：期待と現実のギャップ（嬉しさ・悲しさ）/ 自我：自分で選んだ行動の履歴
emotion = 0
choice_log = []

# 人間でいう「経験の積み重ねによる学び（試行錯誤）」
for episode in range(500):
    pos = start
    for _ in range(100):
        # 人間の「気まぐれ or 慣れた選択」
        if random.random() < epsilon:
            a = random.randint(0, 3)
        else:
            a = np.argmax(q_table[pos[0], pos[1]])

        next_pos = move(pos, actions[a])
        reward = 1 if next_pos == goal else 0  # 成功したら嬉しい！

        # 人間の「感情反応」＝期待と現実のギャップ（報酬予測誤差）
        expected = np.max(q_table[pos[0], pos[1]])
        emotion = reward + gamma * np.max(q_table[next_pos[0], next_pos[1]]) - expected

        # 人間の「自我の記録」＝自分で選んだ判断履歴を覚える
        choice_log.append((pos, actions[a]))

        # 行動の価値を更新（記憶の書き換え）
        q_table[pos[0], pos[1], a] += alpha * (
            reward + gamma * np.max(q_table[next_pos[0], next_pos[1]]) - q_table[pos[0], pos[1], a]
        )
        pos = next_pos
        if pos == goal:
            break

# 人間が後から「こう動けばいいんだ！」と理解するような知見のまとめ（矢印表示）
print("学習後の最適行動ポリシー（矢印だけ）：")
for i in range(3):
    row = ""
    for j in range(4):
        if maze[i][j] == -1:
            row += "■ "
        elif maze[i][j] == 1:
            row += "G "
        else:
            best_action = np.argmax(q_table[i][j])
            row += ["↑", "↓", "←", "→"][best_action] + " "
    print(row)

# 人間の「最終的な気持ち」と「これまでの判断の軌跡」
print(f"\n最終的な感情スコア: {round(emotion, 3)}")
print(f"選んだ行動ログ（自我）:")
for log in choice_log[-10:]:
    print(f"位置{log[0]} → 行動: {log[1]}")
