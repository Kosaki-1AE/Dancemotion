import json
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

# === 動画ファイルの入力 ===
video_path = input("🎥 処理したい動画ファイルのパスを入力してください：\n> ").strip()
output_json = f"pose_vectors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# === MediaPipe 初期化 ===
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# === 動画読み込み ===
cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"❌ 動画ファイルが開けませんでした: {video_path}")
    exit()
fps = cap.get(cv2.CAP_PROP_FPS)
frame_data = []

frame_idx = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image_height, image_width, _ = frame.shape

    landmarks = {}

    # === Pose (顔含む全身) ===
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            x, y = int(lm.x * image_width), int(lm.y * image_height)
            landmarks[f"pose_{i}"] = [x, y]

    # === 左手 ===
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            x, y = int(lm.x * image_width), int(lm.y * image_height)
            landmarks[f"left_hand_{i}"] = [x, y]

    # === 右手 ===
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            x, y = int(lm.x * image_width), int(lm.y * image_height)
            landmarks[f"right_hand_{i}"] = [x, y]

    # === フレームごとのデータを保存 ===
    frame_info = {
        "frame": frame_idx,
        "timestamp_sec": round(frame_idx / fps, 3),
        "landmarks": landmarks
    }
    frame_data.append(frame_info)

    frame_idx += 1

cap.release()

# === JSON保存 ===
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(frame_data, f, ensure_ascii=False, indent=2)

print(f"✅ 完了！ {len(frame_data)} フレーム分を {output_json} に保存しました")