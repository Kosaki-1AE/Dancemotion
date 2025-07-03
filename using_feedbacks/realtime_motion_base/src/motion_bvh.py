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
channel_counts = mocap.get_joints_channel_count()

bins = int(duration / 0.5)
category_activity = defaultdict(lambda: np.zeros(bins))

for frame_idx in range(num_frames):
    frame_data = list(map(float, mocap.frame_joint_channel(frame_idx)))
    time_bin = int((frame_idx * frame_time) / 0.5)
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

plt.figure(figsize=(6, 8))
plt.barh(times, diff_metric, height=0.4, color="skyblue")
plt.xlabel("3カテゴリ間の“動きの差の少なさ”")
plt.ylabel("時間（秒）")
plt.title("ワイ理論：軸・骨・筋肉の調和度（0.5秒ごと）")
plt.gca().invert_yaxis()
plt.grid(True, axis='x')
plt.tight_layout()
plt.show()
