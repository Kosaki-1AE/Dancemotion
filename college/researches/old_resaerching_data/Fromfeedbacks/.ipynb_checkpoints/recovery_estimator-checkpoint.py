# recovery_estimator.py
# 空気感スコア（airflow）が崩れたあとの回復速度・傾向を推定するモジュール

class RecoveryEstimator:
    def __init__(self, decay_threshold=-0.3, recovery_window=5):
        self.airflow_log = []
        self.decay_threshold = decay_threshold  # 崩壊とみなす閾値
        self.recovery_window = recovery_window  # 何ステップ以内に回復するか
        self.recovery_data = []

    def log_airflow(self, score: float):
        self.airflow_log.append(score)

    def detect_recovery(self) -> dict:
        """
        崩壊→回復があったかを検出。
        崩壊スコア以下に落ちたのち、回復したステップ差を返す。
        """
        for i in range(len(self.airflow_log) - self.recovery_window):
            if self.airflow_log[i] < self.decay_threshold:
                for j in range(1, self.recovery_window + 1):
                    if self.airflow_log[i + j] >= 0.0:
                        recovery_time = j
                        self.recovery_data.append(recovery_time)
                        return {"detected": True, "recovery_time": recovery_time, "index": i}
        return {"detected": False}

    def get_average_recovery(self) -> float:
        if not self.recovery_data:
            return -1.0
        return sum(self.recovery_data) / len(self.recovery_data)

if __name__ == "__main__":
    re = RecoveryEstimator()
    airflow_sequence = [0.3, 0.2, -0.4, -0.5, -0.6, 0.0, 0.2, 0.3]
    for s in airflow_sequence:
        re.log_airflow(s)
    print(re.detect_recovery())
    print("Avg recovery:", re.get_average_recovery())
