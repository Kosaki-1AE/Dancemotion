import numpy as np
import torch
from train_scripts.aist_loader import load_dataset_from_folder
from train_scripts.convert_action import convert_to_csv
from train_scripts.decision_engine import is_will_activated
from train_scripts.improvisation_dance import StillnessAI
from train_scripts.replay_buffer import ReplayBuffer

# モデル初期化（例: 17 keypoints × 3D = 51次元）
dim = 51
model = StillnessAI(dim=dim, num_heads=3)
model.eval()

# データ読み込み（AIST++のJSONを入れたフォルダを指定）
dataset = load_dataset_from_folder("./aist_json_sample", max_files=3)

# Replay Buffer 作成
buffer = ReplayBuffer(capacity=100)

# 自己モデル（責任スコア計算用の重み）
rng = np.random.default_rng(seed=42)
self_model = rng.random(10)  # 仮の10次元ベクトル
tension_threshold = 3.0

# 各モーションシーケンスをモデルに流す
for i, motion_seq in enumerate(dataset):
    input_seq = motion_seq.unsqueeze(0)  # [1, seq_len, dim]

    with torch.no_grad():
        output, R = model(input_seq)

    buffer.push(input_seq.squeeze(0), output.squeeze(0), R.squeeze(0))
    print(f"[{i}] Motion processed:")
    print(f"  - Input shape: {input_seq.shape}")
    print(f"  - Output shape: {output.shape}")
    print(f"  - Responsibility shape: {R.shape}")

    # --- 意思決定ロジックの呼び出し ---
    stillness_state = {
        "input": input_seq,
        "output": output,
        "responsibility": R
    }  # 今は空の仮構造（scan_possible_arrows内で処理）
    arrow = is_will_activated(stillness_state, self_model, tension_threshold)
    if arrow:
        print(f"  >>> Action triggered: {arrow['direction']}")
    else:
        print("  ... Stillness maintained")

print(f"✅ Total buffered sequences: {len(buffer)}")