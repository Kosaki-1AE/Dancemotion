import numpy as np
import torch
import torch.nn as nn

from responsibility_layer import ResponsibilityLayer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RespBackbone(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.resp = ResponsibilityLayer(in_dim, hid_dim)
    def forward(self, x):
        return self.resp(x)["combined"]

class FuseConcatModel(nn.Module):
    def __init__(self, in_a=16, hid_a=32, in_p=54, hid_p=32, fuse_dim=32):
        super().__init__()
        self.back_a = RespBackbone(in_a, hid_a)
        self.back_p = RespBackbone(in_p, hid_p)
        self.align_a = nn.Sequential(nn.LayerNorm(hid_a), nn.Linear(hid_a, fuse_dim))
        self.align_p = nn.Sequential(nn.LayerNorm(hid_p), nn.Linear(hid_p, fuse_dim))
        self.gater = nn.Sequential(nn.LayerNorm(fuse_dim*2), nn.Linear(fuse_dim*2,16),
                                   nn.ReLU(), nn.Linear(16,2))
        self.head  = nn.Sequential(nn.LayerNorm(fuse_dim), nn.Linear(fuse_dim,64),
                                   nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,2))
    def forward(self, xa, xp):
        ha = self.align_a(self.back_a(xa))
        hp = self.align_p(self.back_p(xp))
        w  = torch.softmax(self.gater(torch.cat([ha,hp],-1)), dim=-1)
        h  = w[:, :1]*ha + w[:, 1:2]*hp
        return self.head(h)

# ====== ここから実行例 ======
ckpt = "experts/fuse/ckpt_fuse_concat_seed0.pt"  # 置き場所に合わせて
sd = torch.load(ckpt, map_location="cpu")
model = FuseConcatModel().to(DEVICE)
model.load_state_dict(sd, strict=True)
model.eval()

# ダミー入力（実データに置換）：xa: [B,16], xp: [B,54]
B=2
xa = torch.randn(B,16).to(DEVICE)
xp = torch.randn(B,54).to(DEVICE)

with torch.no_grad():
    logits = model(xa, xp)           # [B,2]
    prob   = torch.softmax(logits,1) # 確率
    pred   = prob.argmax(1)          # 0:neg優勢 / 1:pos優勢
print("probs:", prob.cpu().numpy())
print("pred :", pred.cpu().numpy())