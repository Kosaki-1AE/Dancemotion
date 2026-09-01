# airflow_analyzer.py
# 空気感スコア計算モジュール

import numpy as np

class AirflowAnalyzer:
    def __init__(self):
        self.rhythm_log = []
        self.stillness_log = []
        self.tension_log = []

    def log_state(self, rhythm: float, stillness: float, tension: float):
        self.rhythm_log.append(rhythm)
        self.stillness_log.append(stillness)
        self.tension_log.append(tension)

    def compute_airflow_score(self) -> float:
        if not self.rhythm_log or not self.stillness_log or not self.tension_log:
            return 0.0
        r = self.rhythm_log[-1]
        s = self.stillness_log[-1]
        t = self.tension_log[-1]
        airflow = s * r - t
        return np.clip(airflow, -1.0, 1.0)

    def recent_summary(self):
        return {
            "rhythm": self.rhythm_log[-1] if self.rhythm_log else None,
            "stillness": self.stillness_log[-1] if self.stillness_log else None,
            "tension": self.tension_log[-1] if self.tension_log else None,
            "airflow_score": self.compute_airflow_score()
        }

if __name__ == "__main__":
    af = AirflowAnalyzer()
    af.log_state(rhythm=0.6, stillness=0.7, tension=0.2)
    print("Airflow score:", af.compute_airflow_score())
