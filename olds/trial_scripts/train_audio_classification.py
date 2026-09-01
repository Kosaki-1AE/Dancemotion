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

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()
np.random.seed(args.seed); torch.manual_seed(args.seed)

class Classifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.resp = ResponsibilityLayer(in_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)  # 0: neg優勢, 1: pos優勢
    def forward(self, x):
        out = self.resp(x)
        logits = self.fc(out["combined"])  # 32次元 → 2クラス
        return logits, out

def make_dummy(n=512, in_dim=16, out_dim=32, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    w = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
    z = X @ w
    pos, neg = np.clip(z, 0, None), np.clip(-z, 0, None)
    y_label = (pos.sum(axis=1) > neg.sum(axis=1)).astype(np.int64)
    return X, y_label

def train_epoch(model, loader, optim, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for batch in loader:
        x = batch["x"].to(device); y = batch["y_label"].to(device)
        optim.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward(); optim.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_count += x.size(0)
    return total_loss/total_count, total_correct/total_count

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    ys, ps = [], []
    for batch in loader:
        x = batch["x"].to(device); y = batch["y_label"].to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_count += x.size(0)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return total_loss/total_count, total_correct/total_count, ys, ps

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim = 16           # audio入力次元（そのまま）
    hidden_dim = 32       # ★ ここを32に統一 ★

    # 実データに差し替えるならここ
    X, y_label = make_dummy(n=800, in_dim=in_dim, out_dim=hidden_dim, seed=args.seed)
    ds = ResponsibilityDataset(X, y_label=y_label)

    n_train = int(len(ds)*0.8)
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)  # 分割もseed固定
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = Classifier(in_dim, hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = -1.0
    best_sd = None
    best_report = None
    best_cm = None

    for epoch in range(1, 21):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, device)
        va_loss, va_acc, ys, ps = eval_epoch(model, val_loader, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{epoch:02d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_cm = confusion_matrix(ys, ps)
            best_report = classification_report(ys, ps, digits=3)

    print("Confusion matrix (best):\n", best_cm)
    print(best_report)
    print(f"Val Accuracy (best): {best_acc:.4f}")

    os.makedirs("experts/audio", exist_ok=True)
    ckpt_path = f"experts/audio/ckpt_seed{args.seed}.pt"
    torch.save(best_sd, ckpt_path)
    print(f"saved ckpt: {ckpt_path} (val_acc={best_acc:.4f})")

    os.makedirs("results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("results/log.txt", "a", encoding="utf-8") as f:
        f.write(f"=== Run {ts} seed={args.seed} ===\n")
        f.write(f"Val Accuracy: {best_acc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(best_cm) + "\n")
        f.write("Classification Report:\n")
        f.write(best_report + "\n\n")

if __name__ == "__main__":
    main()