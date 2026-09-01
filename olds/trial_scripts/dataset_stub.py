# dataset_stub.py
import numpy as np
import torch
from torch.utils.data import Dataset


class ResponsibilityDataset(Dataset):
    """
    X: (N, in_dim) 連続特徴（テキスト/音/骨格 由来なんでもOK）
    回帰: y_pos, y_neg (N, out_dim) 実数スコア
    分類: y_label (N,) 0 or 1
    """
    def __init__(self, X: np.ndarray,
                 y_pos: np.ndarray | None = None,
                 y_neg: np.ndarray | None = None,
                 y_label: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_pos = torch.tensor(y_pos, dtype=torch.float32) if y_pos is not None else None
        self.y_neg = torch.tensor(y_neg, dtype=torch.float32) if y_neg is not None else None
        self.y_label = torch.tensor(y_label, dtype=torch.long) if y_label is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = {"x": self.X[idx]}
        if self.y_pos is not None: sample["y_pos"] = self.y_pos[idx]
        if self.y_neg is not None: sample["y_neg"] = self.y_neg[idx]
        if self.y_label is not None: sample["y_label"] = self.y_label[idx]
        return sample
