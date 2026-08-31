# log_manager.py
# 各タイムステップの感情・構造・スコアをCSV/JSONに記録するユーティリティ

import pandas as pd
import numpy as np
import os
from datetime import datetime

class LogManager:
    def __init__(self, log_dir="logs"):
        self.records = []
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, step: int, emotion: str, intensity: float,
            airflow_score: float, responsibility_balance: float,
            predicted_vector: np.ndarray):

        record = {
            "step": step,
            "emotion": emotion,
            "intensity": intensity,
            "airflow": airflow_score,
            "responsibility": responsibility_balance,
            "robot": predicted_vector[0],
            "wave": predicted_vector[1],
            "rhythm": predicted_vector[2],
            "timestamp": datetime.now().isoformat()
        }
        self.records.append(record)

    def save_csv(self, filename="log.csv"):
        df = pd.DataFrame(self.records)
        path = os.path.join(self.log_dir, filename)
        df.to_csv(path, index=False)
        print(f"Saved log to {path}")

    def save_json(self, filename="log.json"):
        df = pd.DataFrame(self.records)
        path = os.path.join(self.log_dir, filename)
        df.to_json(path, orient="records", lines=True)
        print(f"Saved log to {path}")

if __name__ == "__main__":
    lm = LogManager()
    for i in range(3):
        vec = np.array([0.4+i*0.1, 0.3+i*0.1, 0.2+i*0.1])
        lm.log(i, "joy", 0.8, 0.3, 0.1, vec)
    lm.save_csv()
    lm.save_json()
