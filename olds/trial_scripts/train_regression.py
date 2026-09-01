# train_regression.py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset_stub import ResponsibilityDataset
from responsibility_layer import ResponsibilityLayer


class Regressor(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.resp = ResponsibilityLayer(in_dim, out_dim)

    def forward(self, x):
        return self.resp(x)

def make_dummy(n=512, in_dim=16, out_dim=8, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, in_dim)).astype(np.float32)
    w = rng.normal(size=(in_dim, out_dim)).astype(np.float32)
    z = X @ w
    y_pos = np.clip(z, 0, None).astype(np.float32)
    y_neg = np.clip(-z, 0, None).astype(np.float32)
    return X, y_pos, y_neg

def train_epoch(model, loader, optim, device):
    model.train()
    total = 0; loss_sum = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y_pos = batch["y_pos"].to(device)
        y_neg = batch["y_neg"].to(device)
        optim.zero_grad()
        out = model(x)
        loss = F.mse_loss(out["pos"], y_pos) + F.mse_loss(out["neg"], y_neg)
        loss.backward()
        optim.step()
        loss_sum += loss.item()*x.size(0)
        total += x.size(0)
    return loss_sum/total

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total = 0; loss_sum = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y_pos = batch["y_pos"].to(device)
        y_neg = batch["y_neg"].to(device)
        out = model(x)
        loss = F.mse_loss(out["pos"], y_pos) + F.mse_loss(out["neg"], y_neg)
        loss_sum += loss.item()*x.size(0)
        total += x.size(0)
    return loss_sum/total

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_dim, out_dim = 16, 8
    X, y_pos, y_neg = make_dummy(n=800, in_dim=in_dim, out_dim=out_dim)
    ds = ResponsibilityDataset(X, y_pos=y_pos, y_neg=y_neg)

    n_train = int(len(ds)*0.8)
    n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = Regressor(in_dim, out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 21):
        tr = train_epoch(model, train_loader, opt, device)
        va = eval_epoch(model, val_loader, device)
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{epoch:02d}] train MSE {tr:.4f} | val MSE {va:.4f}")

if __name__ == "__main__":
    main()
