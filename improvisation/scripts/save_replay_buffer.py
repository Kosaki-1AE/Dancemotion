import os
import pickle

import numpy as np
from convert_action import convert_to_csv  # これはconvert_action.pyに実装されている想定
from replay_buffer import ReplayBuffer


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, item):
        self.buffer.append(item)
        
# 事前に Generator を作成（シード値は任意）
rng = np.random.default_rng(seed=42)

# 1. ダミーデータを生成（ここは本番ではAIST++とかのデータで置き換える）
def generate_dummy_motion_sequence(length=30, dims=51):
    return np.random(length, dims)

# 2. ReplayBuffer に追加
buffer = ReplayBuffer()

for _ in range(10):  # 10シーケンス 分
    seq = generate_dummy_motion_sequence()
    buffer.push(seq)

# 3. 保存
with open("data/replay_buffer.pkl", "wb") as f:
    pickle.dump(buffer, f)

print("ReplayBuffer 保存完了 ✅ → data/replay_buffer.pkl")

if __name__ == "__main__":
    print("ReplayBuffer saved to data/replay_buffer.pkl")

