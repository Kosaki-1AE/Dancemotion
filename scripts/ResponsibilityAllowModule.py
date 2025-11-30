# easy version
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResponsibilityLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)   # 学習可能な線形
        self.prelu_p = nn.PReLU(out_dim)                 # 正側の学習可能ReLU
        self.prelu_n = nn.PReLU(out_dim)                 # 負側（-zに適用）
        self.alpha = nn.Parameter(torch.ones(out_dim))   # 愛の係数
        self.beta  = nn.Parameter(torch.ones(out_dim))   # えぐみの係数

    def forward(self, x):
        z = self.W(x)
        pos = self.prelu_p(z)          # 愛
        neg = self.prelu_n(-z)         # えぐみ（絶対値化相当）
        combined = self.alpha*pos + self.beta*neg
        safety_width = pos + neg       # “安全地帯の幅”
        return {"z": z, "pos": pos, "neg": neg,
                "combined": combined, "safety_width": safety_width}
