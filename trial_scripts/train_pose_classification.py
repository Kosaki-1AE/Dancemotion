import argparse
import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split

from dataset_stub import ResponsibilityDataset
from responsibility_layer import ResponsibilityLayer

# ========= 引数 & シード =========
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--in-dim", type=int, default=54)   # 例: 18関節×xyz=54
parser.add_argument("--hidden-dim", type=int, default=32)  # ★ デフォ32で統一
parser.add_argument("--n", type=int, default=900)       # サンプル数（ダミー時）
args, _ = parser.parse_known_args()
np.random.seed(args.seed); torch.manual_seed(args.seed)

# ========= モデル =========
class Classifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.resp = ResponsibilityLayer(in_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)  # 0: neg優勢, 1: pos優勢
    def forward(self, x):
        out = self.resp(x)
        logits = self.fc(out["combined"])  # hidden_dim=32 -> 2クラス
        return logits, out

# ========= ダミーpose特徴 =========
def make_pose_dummy(n=900, in_dim=54, out_dim=32, seed=123):
    """
    1サンプル=1クリップ分の統計特徴（速度/停止/加速度…の混合を模倣）
    実データでは「各クリップの骨格時系列→統計ベクトル」に置き換えてOK。
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    w = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
    z = X @ w
    pos, neg = np.clip(z, 0, None), np.clip(-z, 0, None)
    y_label = (pos.sum(axis=1) > neg.sum(axis=1)).astype(np.int64)
    return X, y_label

# ========= 学習/評価 =========
def train_epoch(model, loader, optim, device):
    model.train()
    totL, totC, n = 0.0, 0, 0
    for b in loader:
        x = b["x"].to(device); y = b["y_label"].to(device)
        optim.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward(); optim.step()
        totL += loss.item() * x.size(0)
        totC += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return totL/n, totC/n

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    totL, totC, n = 0.0, 0, 0
    ys, ps = [], []
    for b in loader:
        x = b["x"].to(device); y = b["y_label"].to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        totL += loss.item() * x.size(0)
        pred = logits.argmax(1)
        totC += (pred == y).sum().item()
        n += x.size(0)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return totL/n, totC/n, ys, ps

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = args.in_dim
    hid = args.hidden_dim   # ★ 32に統一

    # ==== 実データに差し替えるならここ ====
    X, y_label = make_pose_dummy(n=args.n, in_dim=in_dim, out_dim=hid, seed=args.seed)
    ds = ResponsibilityDataset(X, y_label=y_label)

    n_train = int(len(ds)*0.8); n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = Classifier(in_dim, hid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc, best_sd, best_cm, best_rep = -1.0, None, None, None
    for ep in range(1, 21):
        trL, trA = train_epoch(model, train_loader, opt, device)
        vaL, vaA, ys, ps = eval_epoch(model, val_loader, device)
        if ep % 5 == 0 or ep == 1:
            print(f"[{ep:02d}] train loss {trL:.4f} acc {trA:.3f} | val loss {vaL:.4f} acc {vaA:.3f}")
        if vaA > best_acc:
            best_acc = vaA
            best_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_cm = confusion_matrix(ys, ps)
            best_rep = classification_report(ys, ps, digits=3)

    print("Confusion matrix (best):\n", best_cm)
    print(best_rep)
    print(f"Val Accuracy (best): {best_acc:.4f}")

    os.makedirs("experts/pose", exist_ok=True)
    ckpt = f"experts/pose/ckpt_seed{args.seed}.pt"
    torch.save(best_sd, ckpt)
    print(f"saved ckpt: {ckpt} (val_acc={best_acc:.4f})")

    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results/log.txt", "a", encoding="utf-8") as f:
        f.write(f"=== Run {ts} seed={args.seed} [POSE] ===\n")
        f.write(f"Val Accuracy: {best_acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(best_cm) + "\n")
        f.write("Classification Report:\n")
        f.write(best_rep + "\n\n")

if __name__ == "__main__":
    main()