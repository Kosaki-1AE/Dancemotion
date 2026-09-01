# structure_calculator.py
# 構造知モデルの中核：構成要素を選択的に加味して次の動作ベクトルを算出

import numpy as np

class StructureCalculator:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.history = []

    def add_action(self, vector: np.ndarray):
        self.history.append(vector)

    def compute_derivative(self) -> np.ndarray:
        if len(self.history) < 2:
            return np.zeros(3)
        return self.history[-1] - self.history[-2]

    def predict_custom_action(self, current_vector: np.ndarray, airflow_score: float, responsibility_balance: float, flags: dict) -> np.ndarray:
        dF_dt = self.compute_derivative()
        vec = np.zeros(3)

        if flags.get("use_emotion", True):
            vec += current_vector
        if flags.get("use_derivative", False):
            vec += self.alpha * dF_dt
        if flags.get("use_airflow", False):
            vec += self.beta * airflow_score
        if flags.get("use_responsibility", False):
            vec += self.gamma * responsibility_balance
        if flags.get("use_inertia", False) and self.history:
            vec += self.delta * self.history[-1]

        self.add_action(vec)
        return np.clip(vec, 0.0, 1.0)

if __name__ == "__main__":
    sc = StructureCalculator()
    v = np.array([0.5, 0.3, 0.2])
    sc.add_action(v)
    flags = {"use_emotion": True, "use_derivative": True, "use_airflow": True, "use_responsibility": True, "use_inertia": True}
    print("Next vector:", sc.predict_custom_action(v, 0.4, -0.1, flags))
