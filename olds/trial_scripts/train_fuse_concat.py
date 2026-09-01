# train_fuse_concat.py  ← 精度寄りの改訂版
import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split

from responsibility_layer import ResponsibilityLayer

# ================== 引数 ==================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
# 専門家の構成（ckptに合わせてデフォルトを揃える）
parser.add_argument("--in-a", type=int, default=16)   # audio in_dim
parser.add_argument("--hid-a", type=int, default=32)  # ★ audio hidden_dim=32 に統一
parser.add_argument("--in-p", type=int, default=54)   # pose  in_dim
parser.add_argument("--hid-p", type=int, default=32)  # pose  hidden_dim（32）
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--n", type=int, default=800)     # ダミーのペア件数
parser.add_argument("--ckpt-a", type=str, default="experts/audio/ckpt_seed0.pt")
parser.add_argument("--ckpt-p", type=str, default="experts/pose/ckpt_seed0.pt")
args, _ = parser.parse_known_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== 専門家バックボーン ==================
class RespBackbone(nn.Module):
    """入力x -> ResponsibilityLayer(in_dim->hid_dim) -> combined を返す（fcは無し）"""
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.resp = ResponsibilityLayer(in_dim, hid_dim)
    def forward(self, x):
        return self.resp(x)["combined"]  # [B, hid_dim]

# ================== 融合モデル（整列＋ゲート＋強めヘッド） ==================
class FuseConcatModel(nn.Module):
    def __init__(self, in_a, hid_a, in_p, hid_p, fuse_dim=32):
        super().__init__()
        self.back_a = RespBackbone(in_a, hid_a)
        self.back_p = RespBackbone(in_p, hid_p)

        # 各モダリティの32次元を“整列（LayerNorm+Linear）”して共通次元へ
        self.align_a = nn.Sequential(nn.LayerNorm(hid_a), nn.Linear(hid_a, fuse_dim))
        self.align_p = nn.Sequential(nn.LayerNorm(hid_p), nn.Linear(hid_p, fuse_dim))

        # サンプルごとの寄与率を出すゲーター（audio/poseの重み：softmaxで和=1）
        self.gater = nn.Sequential(
            nn.LayerNorm(fuse_dim * 2),
            nn.Linear(fuse_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        # 強めのヘッド（非線形＋Dropout）
        self.head = nn.Sequential(
            nn.LayerNorm(fuse_dim),
            nn.Linear(fuse_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, xa, xp):
        ha = self.back_a(xa)                # [B, hid_a]
        hp = self.back_p(xp)                # [B, hid_p]
        ha = self.align_a(ha)               # [B, fuse_dim]
        hp = self.align_p(hp)               # [B, fuse_dim]

        hcat = torch.cat([ha, hp], dim=-1)  # [B, 2*fuse_dim]
        wlog = self.gater(hcat)             # [B, 2]
        w = torch.softmax(wlog, dim=-1)
        h = w[:, :1] * ha + w[:, 1:2] * hp  # 学習的重み付き和 → [B, fuse_dim]

        logits = self.head(h)               # [B, 2]
        return logits, (ha, hp, w)

# ================== ダミーペアデータ ==================
class PairDataset(Dataset):
    """(Xa, Xp, y) のセット。実データでは“同じクリップ/時刻”を揃えて入れる。"""
    def __init__(self, Xa, Xp, y):
        self.Xa = torch.tensor(Xa, dtype=torch.float32)
        self.Xp = torch.tensor(Xp, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.long)
    def __len__(self): return self.Xa.shape[0]
    def __getitem__(self, i):
        return {"xa": self.Xa[i], "xp": self.Xp[i], "y": self.y[i]}

def make_dummy_pair(n=800, in_a=16, hid_a=32, in_p=54, hid_p=32, seed=0):
    """
    以前の OR 方式だと不安定だったので、双方で“似た基準”になるように生成。
    実データに差し替えるときはここを自分のローダに置換してOK。
    """
    rng = np.random.default_rng(seed)
    Xa = rng.normal(size=(n, in_a)).astype(np.float32)
    Xp = rng.normal(size=(n, in_p)).astype(np.float32)

    wa = rng.normal(size=(in_a, hid_a)).astype(np.float32)
    wp = rng.normal(size=(in_p, hid_p)).astype(np.float32)
    za = Xa @ wa; zp = Xp @ wp
    posa, nega = np.clip(za, 0, None), np.clip(-za, 0, None)
    posp, negp = np.clip(zp, 0, None), np.clip(-zp, 0, None)

    score = (posa.sum(1) + posp.sum(1)) - (nega.sum(1) + negp.sum(1))
    y = (score > 0).astype(np.int64)
    return Xa, Xp, y

# ================== ユーティリティ ==================
@torch.no_grad()
def load_resp_only(backbone: RespBackbone, ckpt_path: str):
    """専門家ckptから resp.* のみロード（fcは無視）→ 以降は凍結"""
    sd = torch.load(ckpt_path, map_location="cpu")
    sub = {k: v for k, v in sd.items() if k.startswith("resp.")}
    # 形が合わないときはここで即死してくれる（原因が分かりやすい）
    backbone.load_state_dict(sub, strict=True)
    for p in backbone.parameters():
        p.requires_grad_(False)

def build_class_weight_from_subset(ds: PairDataset, subset) -> torch.Tensor:
    """学習に使うラベル分布からクラス重みを作る"""
    idx = subset.indices if hasattr(subset, "indices") else list(range(len(subset)))
    y_all = ds.y[idx]
    counts = torch.bincount(y_all, minlength=2)
    weights = (counts.sum() / (2.0 * counts.clamp(min=1))).float()
    return weights

def train_epoch(model, loader, opt, criterion):
    model.train()
    totL, totC, N = 0.0, 0, 0
    for b in loader:
        xa = b["xa"].to(DEVICE); xp = b["xp"].to(DEVICE); y = b["y"].to(DEVICE)
        opt.zero_grad()
        logits, _ = model(xa, xp)
        loss = criterion(logits, y)
        loss.backward(); opt.step()
        totL += loss.item() * y.size(0)
        totC += (logits.argmax(1)==y).sum().item()
        N += y.size(0)
    return totL/N, totC/N

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    totL, totC, N = 0.0, 0, 0
    ys, ps = [], []
    for b in loader:
        xa = b["xa"].to(DEVICE); xp = b["xp"].to(DEVICE); y = b["y"].to(DEVICE)
        logits, _ = model(xa, xp)
        loss = criterion(logits, y)
        totL += loss.item() * y.size(0)
        pred = logits.argmax(1)
        totC += (pred==y).sum().item()
        N += y.size(0)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return totL/N, totC/N, ys, ps

# ================== メイン ==================
def main():
    os.makedirs("experts/fuse", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # データ（実データに差し替え可）
    Xa, Xp, y = make_dummy_pair(
        n=args.n, in_a=args.in_a, hid_a=args.hid_a, in_p=args.in_p, hid_p=args.hid_p, seed=args.seed
    )
    ds = PairDataset(Xa, Xp, y)
    n_tr = int(len(ds)*0.8); n_va = len(ds)-n_tr
    tr_ds, va_ds = random_split(ds, [n_tr, n_va], generator=torch.Generator().manual_seed(args.seed))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

    # モデル（専門家respはロード→凍結）
    model = FuseConcatModel(args.in_a, args.hid_a, args.in_p, args.hid_p).to(DEVICE)
    load_resp_only(model.back_a, args.ckpt_a)
    load_resp_only(model.back_p, args.ckpt_p)

    # 不均衡対策：クラス重み（学習に使うsubsetから作成）
    weights = build_class_weight_from_subset(ds, tr_ds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # ヘッド/整列/ゲータのみ学習
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

    best_acc, best_sd, best_cm, best_rep = -1.0, None, None, None
    for ep in range(1, args.epochs+1):
        trL, trA = train_epoch(model, tr_loader, opt, criterion)
        vaL, vaA, ys, ps = eval_epoch(model, va_loader, criterion)
        if ep % 5 == 0 or ep == 1:
            print(f"[{ep:02d}] train loss {trL:.4f} acc {trA:.3f} | val loss {vaL:.4f} acc {vaA:.3f}")
        if vaA > best_acc:
            best_acc = vaA
            best_sd = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            best_cm = confusion_matrix(ys, ps)
            best_rep = classification_report(ys, ps, digits=3, zero_division=0)  # warning抑制

    # 出力
    print("Confusion matrix (best):\n", best_cm)
    print(best_rep)
    print(f"Val Accuracy (best): {best_acc:.4f}")

    ckpt = f"experts/fuse/ckpt_fuse_concat_seed{args.seed}.pt"
    torch.save(best_sd, ckpt)
    print("saved ckpt:", ckpt)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results/log.txt", "a", encoding="utf-8") as f:
        f.write(f"=== FUSE CONCAT {ts} seed={args.seed} ===\n")
        f.write(f"Val Accuracy: {best_acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(best_cm) + "\n")
        f.write("Classification Report:\n")
        f.write(best_rep + "\n\n")

if __name__ == "__main__":
    main()
