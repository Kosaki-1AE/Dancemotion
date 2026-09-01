# structure_visualizer.py
# 構造知ベクトルの3軸（Robot / Wave / Rhythm）を可視化するユーティリティ

import matplotlib.pyplot as plt
import numpy as np

def plot_motion_vector(vector: np.ndarray, title: str = "Motion Vector"):
    """
    単一の3軸ベクトルをバーグラフで表示
    """
    labels = ["Robot", "Wave", "Rhythm"]
    plt.bar(labels, vector, color="skyblue")
    plt.ylim(0, 1)
    plt.title(title)
    plt.grid(True)
    plt.show()

def plot_motion_sequence(vectors: list[np.ndarray], title: str = "Motion Sequence"):
    """
    時系列でのベクトル推移（3軸）を折れ線グラフで表示
    """
    t = list(range(len(vectors)))
    robot = [v[0] for v in vectors]
    wave = [v[1] for v in vectors]
    rhythm = [v[2] for v in vectors]

    plt.plot(t, robot, label="Robot")
    plt.plot(t, wave, label="Wave")
    plt.plot(t, rhythm, label="Rhythm")
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    vec = np.array([0.5, 0.3, 0.7])
    plot_motion_vector(vec, "Example Vector")

    seq = [
        np.array([0.4, 0.3, 0.2]),
        np.array([0.5, 0.35, 0.25]),
        np.array([0.55, 0.4, 0.3])
    ]
    plot_motion_sequence(seq, "Motion History")
