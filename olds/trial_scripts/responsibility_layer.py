# responsibility_layer.py
import torch
import torch.nn as nn


class ResponsibilityLayer(nn.Module):
    """
    入力 x -> z = Wx + b
    pos = PReLU(z)            # 愛（正側）
    neg = PReLU(-z)           # えぐみ（負側の強さ）
    S = pos + neg             # 安全地帯の幅（余白）
    C = α*pos + β*neg         # 統合責任（出力）
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        self.prelu_pos = nn.PReLU(out_dim)
        self.prelu_neg = nn.PReLU(out_dim)
        self.alpha = nn.Parameter(torch.ones(out_dim))
        self.beta  = nn.Parameter(torch.ones(out_dim))

    def forward(self, x):
        z = self.W(x)
        pos = self.prelu_pos(z)
        neg = self.prelu_neg(-z)
        safety_width = pos + neg
        combined = self.alpha * pos + self.beta * neg
        return {
            "z": z,
            "pos": pos,
            "neg": neg,
            "safety_width": safety_width,
            "combined": combined
        }
