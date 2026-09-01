from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- カスタムヒートカラーマップ（青→紫→赤→橙→黄→白） ---
cmap = LinearSegmentedColormap.from_list("thermo_heat", ["blue", "purple", "red", "orange", "yellow", "white"])

# MediaPipe設定
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
cap = cv2.VideoCapture(0)

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# スコア用保存
prev_nodes = {}
score_history = {}
max_history = 6

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    unified_nodes = {}

    # --- ノード登録 ---
    pose_face_ids = set(range(0, 11))
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i in pose_face_ids:
                continue
            x, y = int(lm.x * w), int(lm.y * h)
            unified_nodes[f"pose_{i}"] = (x, y)

    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            unified_nodes[f"left_hand_{i}"] = (x, y)

    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            unified_nodes[f"right_hand_{i}"] = (x, y)

    # --- スコア計算 ---
    scores = {}
    for key, (x, y) in unified_nodes.items():
        if key in prev_nodes:
            dx = x - prev_nodes[key][0]
            dy = y - prev_nodes[key][1]
            speed = np.sqrt(dx**2 + dy**2)
            scores[key] = speed
        else:
            scores[key] = 0.0

    # --- 正規化＆スムージング（履歴平均） ---
    max_speed = max(scores.values()) if scores else 0.0
    max_speed = max(max_speed, 1e-6)
    for k in scores:
        norm = scores[k] / max_speed
        if k not in score_history:
            score_history[k] = deque(maxlen=max_history)
        score_history[k].append(norm)
        scores[k] = np.mean(score_history[k])

    # --- 背景ヒートマップ作成 ---
    heatmap = np.zeros((h, w), dtype=np.float32)
    for key, (x, y) in unified_nodes.items():
        intensity = scores[key]
        cv2.circle(heatmap, (x, y), 30, intensity, -1)

    # ブラーで熱の拡散表現
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=25, sigmaY=25)
    heatmap = np.clip(heatmap, 0, 1)

    # マスク化（動きがある部分だけ）
    mask = heatmap > 0.05  # 色を乗せる範囲（少し緩和）

    # カラーマップ変換（マスク範囲のみ）
    heatmap_color = np.zeros_like(image)
    heatmap_rgb = np.uint8(cmap(heatmap)[:, :, :3] * 255)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    heatmap_color[mask] = heatmap_rgb[mask]

    # 重ね描画（ノード部分のみヒート表示）
    blended = cv2.addWeighted(image, 1.0, heatmap_color, 0.6, 0)

    # --- 接続線描画 ---
    for i1, i2 in mp_pose.POSE_CONNECTIONS:
        k1, k2 = f"pose_{i1}", f"pose_{i2}"
        if k1 in unified_nodes and k2 in unified_nodes:
            col = tuple(int(c * 255) for c in cmap((scores[k1] + scores[k2]) / 2)[:3])
            cv2.line(blended, unified_nodes[k1], unified_nodes[k2], col, 3)

    for hand_prefix, connections in [("left_hand", mp_hands.HAND_CONNECTIONS), ("right_hand", mp_hands.HAND_CONNECTIONS)]:
        for i1, i2 in connections:
            k1, k2 = f"{hand_prefix}_{i1}", f"{hand_prefix}_{i2}"
            if k1 in unified_nodes and k2 in unified_nodes:
                col = tuple(int(c * 255) for c in cmap((scores[k1] + scores[k2]) / 2)[:3])
                cv2.line(blended, unified_nodes[k1], unified_nodes[k2], col, 2)

    # --- ノード描画 ---
    for k, (x, y) in unified_nodes.items():
        col = tuple(int(c * 255) for c in cmap(scores[k])[:3])
        cv2.circle(blended, (x, y), 3, col, -1)

    # 更新＆表示
    prev_nodes = unified_nodes.copy()
    cv2.imshow("🔥 Thermographic Dance View", blended)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
