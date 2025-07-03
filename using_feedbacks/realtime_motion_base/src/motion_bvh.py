from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from bvh import Bvh

bvh_path = "君の夜をくれ_Sasaki41.bvh"

with open(bvh_path, encoding="utf-8", errors="ignore") as f:
    mocap = Bvh(f.read())

frame_time = float(mocap.frame_time)
num_frames = mocap.nframes
duration = frame_time * num_frames
bins = int(duration / 0.5)

def classify_joint(name):
    name = name.lower()
    if any(k in name for k in ["hip", "spine", "root", "neck"]):
        return "軸"
    elif any(k in name for k in ["shoulder", "arm", "leg", "knee", "elbow"]):
        return "骨"
    else:
        return "筋肉"

joint_category = {joint.name: classify_joint(joint.name) for joint in mocap.get_joints()}
channel_map = mocap.get_joints_names()
channel_counts = [len(mocap.joint_channels(j)) for j in channel_map]
category_activity = defaultdict(lambda: np.zeros(bins))

for frame_idx in range(num_frames):
    frame_data = []
    for joint in channel_map:
        for ch in mocap.joint_channels(joint):
            val = float(mocap.frame_joint_channel(frame_idx, joint, ch))
            frame_data.append(val)

    time_bin = min(int((frame_idx * frame_time) / 0.5), bins - 1)

    idx = 0
    for joint_name, count in zip(channel_map, channel_counts):
        cat = joint_category.get(joint_name, "筋肉")
        joint_values = frame_data[idx:idx+count]
        ch_names = mocap.joint_channels(joint_name)
        rotation_values = [
            val for ch_name, val in zip(ch_names, joint_values)
            if "rotation" in ch_name.lower()
        ]
        movement = sum(abs(v) for v in rotation_values)
        category_activity[cat][time_bin] += movement
        idx += count

# 差分スコア（調和スコア）算出
diff_metric = []
for i in range(bins):
    values = [category_activity["軸"][i], category_activity["骨"][i], category_activity["筋肉"][i]]
    std_dev = np.std(values)
    closeness = 1 / (std_dev + 1e-6)
    diff_metric.append(closeness)

# 微分（変化量）計算
deriv_metric = np.gradient(diff_metric, 0.5)

# 正規化（0〜1にスケーリングして位相合わせ）
diff_norm = (diff_metric - np.min(diff_metric)) / (np.max(diff_metric) - np.min(diff_metric))
deriv_norm = (deriv_metric - np.min(deriv_metric)) / (np.max(deriv_metric) - np.min(deriv_metric))

# 時間軸
times = [i * 0.5 for i in range(bins)]

# グラフ出力
plt.figure(figsize=(10, 5))
plt.plot(times, diff_norm, color="red", linewidth=2, label="score_sin")
plt.plot(times, deriv_norm, color="blue", linestyle="dashed", linewidth=2, label="Derivative_cos")
plt.xlabel("time (0.5min)")
plt.ylabel("score / Derivative (normalized)")
plt.title("Stillness Harmony（Phase-aligned）")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
