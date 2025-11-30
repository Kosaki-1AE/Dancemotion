# eval_keep.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset_stub import ResponsibilityDataset
from responsibility_layer import ResponsibilityLayer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.resp = ResponsibilityLayer(in_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        out = self.resp(x)
        logits = self.fc(out["combined"])
        return logits

# ===== ダミー（train_xxx.py と合わせる）=====
def make_dummy(n=800, in_dim=16, out_dim=16, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    w = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
    z = X @ w
    pos = np.clip(z, 0, None)
    neg = np.clip(-z, 0, None)
    y = (pos.sum(axis=1) > neg.sum(axis=1)).astype(np.int64)
    return X, y

def make_pose_dummy(n=900, in_dim=54, out_dim=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    w = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
    z = X @ w
    pos = np.clip(z, 0, None); neg = np.clip(-z, 0, None)
    y = (pos.sum(axis=1) > neg.sum(axis=1)).astype(np.int64)
    return X, y

@torch.no_grad()
def eval_acc(model_sd_path, in_dim, hidden_dim, maker, seed=0):
    # 同じアーキでモデルを作って state_dict をロード
    m = Classifier(in_dim, hidden_dim).to(DEVICE)
    sd = torch.load(model_sd_path, map_location="cpu")
    m.load_state_dict(sd); m.eval()

    X, y = maker(seed=seed, in_dim=in_dim, out_dim=hidden_dim)
    ds = ResponsibilityDataset(X, y_label=y)
    n_train = int(len(ds)*0.8); n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    correct = tot = 0
    for b in val_loader:
        x = b["x"].to(DEVICE); yy = b["y_label"].to(DEVICE)
        logits = m(x)
        pred = logits.argmax(1)
        correct += (pred == yy).sum().item()
        tot += yy.numel()
    return correct / tot

if __name__ == "__main__":
    # ==== audioタスクでの保持 ====
    acc_audio = eval_acc("experts/merged/ckpt_m2n2.pt",
                         in_dim=16, hidden_dim=16, maker=make_dummy, seed=0)
    print(f"[KEEP on AUDIO] acc={acc_audio:.4f}")

    # ==== poseタスクでの保持 ====
    acc_pose  = eval_acc("experts/merged/ckpt_m2n2.pt",
                         in_dim=54, hidden_dim=32, maker=make_pose_dummy, seed=0)
    print(f"[KEEP on POSE ] acc={acc_pose:.4f}")
