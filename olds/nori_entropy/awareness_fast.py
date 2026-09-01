# -*- coding: utf-8 -*-
"""
awareness_fast.py
NumPy だけで A(意識:0..1) を推定する軽量エンジン。T は常に 1.0（無意識オフの目安）。
"""
from collections import deque
import numpy as np

class FastAwarenessEngine:
    """
    共通I/F:
        step(feat: np.ndarray) -> (A: float, T: float)
    """
    def __init__(self, feat_dim: int = 7, win: int = 24, ema: float = 0.3):
        self.buf = deque(maxlen=win)
        self.ema = float(ema)
        self.A_ema = None
        self.feat_dim = feat_dim

    @staticmethod
    def _safe_entropy(p, eps: float = 1e-9) -> float:
        p = np.clip(p, eps, 1.0)
        return float(-(p*np.log(p)).sum())

    def step(self, feat: np.ndarray):
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 1 or feat.shape[0] != self.feat_dim:
            raise ValueError(f"feat must be shape [{self.feat_dim}], got {feat.shape}")
        self.buf.append(feat)
        # ウォームアップ
        if len(self.buf) < 8:
            self.A_ema = 0.5 if self.A_ema is None else (self.ema*0.5 + (1-self.ema)*self.A_ema)
            return float(self.A_ema), 1.0

        X = np.stack(self.buf, axis=0)           # [W,D]
        var = X.var(axis=0)                      # 各次元の揺らぎ
        var_n = var / (var.mean() + 1e-6)        # 粗正規化
        var_n = np.clip(var_n, 0, 4) / 4.0       # 0..1

        # 分散を重みと見立ててエントロピー→俯瞰度
        w = var_n + 1e-6
        w = w / w.sum()
        H = self._safe_entropy(w)                          # 0..log(D)
        Hn = H / (np.log(len(w)) + 1e-6)                   # 0..1
        A = 1.0 - Hn                                       # 拡散→A↓, 集中→A↑

        self.A_ema = A if self.A_ema is None else (self.ema*A + (1-self.ema)*self.A_ema)
        return float(self.A_ema), 1.0  # 温度は固定（無意識OFFの指標）