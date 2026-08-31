import numpy as np
from sklearn.cluster import KMeans

# 迷路定義（0:通路, -1:壁, 1:ゴール）
maze = [
    [0, 0, 0, 1],
    [0, -1, 0, -1],
    [0, 0, 0, 0]
]
maze = np.array(maze)

# 迷路の全位置を取り出す
positions = [(i, j) for i in range(maze.shape[0]) for j in range(maze.shape[1])]
features = np.array([maze[i, j] for (i, j) in positions]).reshape(-1, 1)

# クラスタリング（3つのグループに分ける）
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(features)

# 結果を迷路上に表示
print("教師なし学習によるクラスタリング結果：")
label_map = np.array(labels).reshape(maze.shape)
for i in range(maze.shape[0]):
    row = ""
    for j in range(maze.shape[1]):
        if maze[i][j] == -1:
            row += "■ "
        else:
            row += f"{label_map[i][j]} "
    print(row)
