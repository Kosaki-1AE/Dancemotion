import json
import os

import numpy as np
import torch


def load_motion_sequence_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 仮定: 'keypoints3d' キーが含まれている（[フレーム数 x 17 x 3]）
    keypoints = np.array(data['keypoints3d'])  # 例: 3D座標の配列

    # Flattenして [フレーム数, 特徴量次元] 形式に変換
    seq = keypoints.reshape(len(keypoints), -1)
    return torch.tensor(seq, dtype=torch.float32)

def load_dataset_from_folder(folder_path, max_files=None):
    dataset = []
    count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            path = os.path.join(folder_path, filename)
            sequence = load_motion_sequence_from_json(path)
            dataset.append(sequence)
            count += 1
            if max_files and count >= max_files:
                break
    return dataset
