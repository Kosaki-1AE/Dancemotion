import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ===============================
# MLPモデル定義
# ===============================
class ResponsibilityMLP(nn.Module):
    def __init__(self, input_dim=51):
        super(ResponsibilityMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# ===============================
# 候補の矢（責任オブジェクト）スキャン関数
# ===============================
def scan_possible_arrows(stillness):
    return stillness.get("candidates", [])

# ===============================
# 責任スコアをMLPで推論する
# ===============================
def responsibility_estimation(arrow, model):
    features = torch.tensor(arrow["features"], dtype=torch.float32)
    if features.ndim == 1:
        features = features.unsqueeze(0)  # shape: (1, input_dim)
    score = model(features)
    return score.item()

# ===============================
# 意思の発火条件チェック（＋CSVログ保存付き）
# ===============================
def is_will_activated(stillness, model, tension_threshold):
    candidate_arrows = scan_possible_arrows(stillness)
    for arrow in candidate_arrows:
        R = responsibility_estimation(arrow, model)
        if R > tension_threshold and arrow.get("source") == "self":
            # ログ出力
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "decision_log.csv")
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "arrow_id": arrow.get("id", "unknown"),
                "responsibility_score": R,
                "source": arrow.get("source")
            }
            df = pd.DataFrame([log_entry])
            if os.path.exists(log_path):
                df.to_csv(log_path, mode="a", index=False, header=False)
            else:
                df.to_csv(log_path, index=False, header=True)

            return arrow  # 意思が発生した！
    return None  # まだ迷ってる
