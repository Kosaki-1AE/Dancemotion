# -*- coding: utf-8 -*-
"""
awareness_torch.py
PyTorch ベースの「無意識」ありエンジン。Self-Attentionで A/T を推定。
※ CPU 環境では重いことがあるので、必要なときだけ使ってください。
"""
from typing import Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise RuntimeError("PyTorch が読み込めませんでした。Torch 版を使うには torch をインストールしてください。") from e

class _TinyAttn1h(nn.Module):
    """1ヘッド・因果マスクなし（履歴俯瞰用）の超軽量Self-Attn"""
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, T: float = 1.0):
        # x: [B,L,D]
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores = (Q @ K.transpose(-1,-2)) / (Q.size(-1) ** 0.5)
        attn = F.softmax(scores / T, dim=-1)     # Boltzmann視点
        out = attn @ V
        return self.proj(out), attn

class TorchAwarenessEngine:
    """
    共通I/F:
        step(feat: np.ndarray) -> (A: float, T: float)
    """
    def __init__(self, feat_dim: int = 7, seq_len: int = 32, device: str = None,
                 T_min: float = 0.7, T_max: float = 1.5):
        self.D = feat_dim
        self.L = seq_len
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.buf = torch.zeros(1, self.L, self.D, device=self.dev)
        self.attn = _TinyAttn1h(self.D).to(self.dev)
        self.T_min, self.T_max = float(T_min), float(T_max)
        self._filled = 0

    @staticmethod
    def _H(prob, eps: float = 1e-9):
        p = prob.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)  # [...,L]→[...]

    def step(self, feat: np.ndarray):
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 1 or feat.shape[0] != self.D:
            raise ValueError(f"feat must be shape [{self.D}], got {feat.shape}")
        x = torch.from_numpy(feat).to(self.dev).view(1,1,self.D)
        # リングバッファに追加
        self.buf = torch.roll(self.buf, shifts=-1, dims=1)
        self.buf[:, -1, :] = x
        self._filled = min(self._filled+1, self.L)

        if self._filled < 8:
            return 0.5, 1.0

        _, attn = self.attn(self.buf, T=1.0)
        last = attn[:, -1, :]               # [1,L]
        H = self._H(last)                   # [1]
        Hn = (H - H.min()) / (H.max() - H.min() + 1e-9)   # 0..1
        # T は拡散度に比例、A はその逆
        T = (self.T_min + (self.T_max - self.T_min) * Hn.mean()).detach().cpu().item()
        A = float(1.0 - Hn.mean().detach().cpu().item())
        return A, T