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

def classify_joint(name):
    name = name.lower()
    if any(k in name for k in ["hip", "spine", "root", "neck"]):
        return "軸"
    elif any(k in name for k in ["shoulder", "arm", "leg", "knee", "elbow"]):
        return "骨"
    else:
        return "筋肉"

joint_category = {}
for joint in mocap.get_joints():
    joint_category[joint.name] = classify_joint(joint.name)

channel_map = mocap.get_joints_names()
channel_counts = [len(mocap.joint_channels(j)) for j in mocap.get_joints_names()]


bins = int(duration / 0.5)
category_activity = defaultdict(lambda: np.zeros(bins))

for frame_idx in range(num_frames):
    frame_data = []
    for joint in mocap.get_joints_names():
        for ch in mocap.joint_channels(joint):
            val = float(mocap.frame_joint_channel(frame_idx, joint, ch))
            frame_data.append(val)

    time_bin = min(int((frame_idx * frame_time) / 0.5), bins - 1)
    idx = 0
    for joint_name, count in zip(channel_map, channel_counts):
        cat = joint_category.get(joint_name, "筋肉")
        joint_values = frame_data[idx:idx+count]

        # 回転チャンネルのみ抽出（"rotation"を含むチャンネル名だけ）
        ch_names = mocap.joint_channels(joint_name)
        rotation_values = [
            val for ch_name, val in zip(ch_names, joint_values)
            if "rotation" in ch_name.lower()
        ]

        movement = sum(abs(v) for v in rotation_values)
        category_activity[cat][time_bin] += movement
        idx += count

for joint in mocap.get_joints_names():
    for ch in mocap.joint_channels(joint):
        val = float(mocap.frame_joint_channel(frame_idx, joint, ch))
        frame_data.append(val)
    time_bin = min(int((frame_idx * frame_time) / 0.5), bins - 1)
    idx = 0
    for joint_name, count in zip(channel_map, channel_counts):
        cat = joint_category.get(joint_name, "筋肉")
        joint_values = frame_data[idx:idx+count]
        rotation_values = joint_values[-3:] if count >= 3 else []
        movement = sum(abs(v) for v in rotation_values)
        category_activity[cat][time_bin] += movement
        idx += count

diff_metric = []
for i in range(bins):
    values = [category_activity["軸"][i], category_activity["骨"][i], category_activity["筋肉"][i]]
    std_dev = np.std(values)
    closeness = 1 / (std_dev + 1e-6)
    diff_metric.append(closeness)

times = [i * 0.5 for i in range(bins)]

plt.figure(figsize=(10, 4))
plt.plot(times, diff_metric, color="skyblue", linewidth=2)
plt.xlabel("times/0.5min")
plt.ylabel("degree of harmony/1: high, 0: low")
plt.title("Stillness Harmony/Little difference in movement between categories per 0.5min")
plt.grid(True)
plt.tight_layout()
plt.show()
