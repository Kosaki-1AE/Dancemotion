# responsibility_allow/linops.py
import numpy as np

def linear_transform(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(x, W) + b
