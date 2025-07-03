import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 迷路定義
maze = [
    [0, 0, 0, 1],   # ゴールは(0,3)
    [0, -1, 0, -1],
    [0, 0, 0, 0]    # スタートは(2,0)
]

# 教師データ（状態, 行動）  ← ここが教師あり！
# 行動: 0=up, 1=down, 2=left, 3=right
train_data = [
    ((2, 0), 3),
    ((2, 1), 3),
    ((2, 2), 0),
    ((1, 2), 0),
    ((0, 2), 3),
]

X = np.array([pos for pos, _ in train_data])
y = np.array([label for _, label in train_data])

# モデル学習（RandomForest）
clf = RandomForestClassifier()
clf.fit(X, y)

# 出力：迷路を矢印で表示
print("教師あり学習による迷路ポリシー：")
for i in range(3):
    row = ""
    for j in range(4):
        if maze[i][j] == -1:
            row += "■ "
        elif maze[i][j] == 1:
            row += "G "
        else:
            pred = clf.predict([[i, j]])[0] if [i, j] in X.tolist() else -1
            arrow = ["↑", "↓", "←", "→", "."][pred] if pred != -1 else "."
            row += arrow + " "
    print(row)
