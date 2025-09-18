# responsibility_allow/linops.py　線形変換だけ
import numpy as np


def linear_transform(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(x, W) + b
