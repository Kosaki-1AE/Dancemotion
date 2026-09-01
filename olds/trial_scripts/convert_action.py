import csv
import os
import pickle

from action_features import extract_features  # 特徴量抽出関数
from replay_buffer import ReplayBuffer


def label_from_features(features):
    # 超簡易ラベル例：Stillness or Moving（あとでジャンプとか追加OK）
    mean_speed = features[..., 0].mean().item()  # 仮にspeedを0番目にしてたとする
    return "Still" if mean_speed < 0.1 else "Move"


def convert_to_csv(pkl_path="data/replay_buffer.pkl"):
    os.makedirs("data/csv_output", exist_ok=True)

    # ReplayBufferか直接リストかを判定
    if "replay_buffer" in pkl_path:
        buffer = ReplayBuffer.load(pkl_path)
    else:
        with open(pkl_path, "rb") as f:
            buffer = pickle.load(f)

    with open("data/csv_output/labeled_actions.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["seq_id", "feature1", "feature2", "...", "label"])

        for idx, sequence in enumerate(buffer):
            features = extract_features(sequence)
            label = label_from_features(features)
            flat = features.flatten().tolist()
            writer.writerow([idx] + flat + [label])


if __name__ == "__main__":
    # デフォルトでReplayBufferを処理
    convert_to_csv()
