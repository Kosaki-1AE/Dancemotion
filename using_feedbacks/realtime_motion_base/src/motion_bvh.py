import json
from pathlib import Path

from bvh import Bvh

# === ファイル設定 ===
bvh_path = "君の夜をくれ_Sasaki41.bvh"
output_path = "bvh_pose_vectors.json"

# === BVH読み込み ===
with open(bvh_path, encoding="utf-8") as f:
    mocap = Bvh(f.read())

frame_time = float(mocap.frame_time)
frame_count = mocap.nframes
joints = mocap.get_joints()

output_data = []

for frame_idx in range(frame_count):
    frame_info = {
        "frame": frame_idx,
        "timestamp_sec": round(frame_idx * frame_time, 3),
        "landmarks": {}
    }
    for joint in joints:
        joint_name = joint.name
        try:
            channels = mocap.joint_channels(joint_name)
            if set(["Xposition", "Yposition", "Zposition"]).issubset(set(channels)):
                pos = mocap.frame_joint_channels(frame_idx, joint_name, ["Xposition", "Yposition", "Zposition"])
                frame_info["landmarks"][joint_name] = list(map(float, pos))
        except LookupError:
            continue  # 存在しないJoint名はスキップ


    output_data.append(frame_info)

# === JSON出力 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"✅ 完了！ {frame_count} フレーム分を {output_path} に保存しました")
