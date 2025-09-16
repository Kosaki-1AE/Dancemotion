import os
import pickle

import numpy as np
import torch
from train_scripts.improvformer import ImprovFormer  # モデル定義ファイル

# ====== 設定 ======
input_dim = 51
output_dim = 51
seq_len = 30  # 入力シーケンスの長さ
model_path = "models/improvformer_model.pth"
pkl_path = "data/action_features.pkl"

# ====== モデル読み込み ======
model = ImprovFormer(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# ====== 入力データ読み込み（pklから）======
with open(pkl_path, "rb") as f:
    data = pickle.load(f)  # data.shape = (N, 30, 51) の想定

# 1つ目のサンプルを使って推論
input_seq = torch.tensor(data[0], dtype=torch.float32).unsqueeze(0)  # shape: [1, 30, 51]

# ====== 推論実行 ======
with torch.no_grad():
    output = model(input_seq)

# ====== 結果表示 ======
print("Input shape :", input_seq.shape)
print("Output shape:", output.shape)
print("Output tensor (first frame):\n", output[0, 0])
