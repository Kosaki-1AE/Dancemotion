# realtime_airfeel.py
# ---------------------------------------------------
# リアルタイム空気感可視化（スコア表示つき）
# - Webカメラ映像入力
# - MediaPipeで骨格推定
# - 静/動のバランスをスコア化
# - 波形表示 + 数値出力
# ---------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# === MediaPipe 初期化 ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# === 波形描画用 ===
window_size = 100  # グラフに表示するフレーム数
scores = deque(maxlen=window_size)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_ylim(-1, 1)
ax.set_xlim(0, window_size)
ax.set_title("Airfeel Score (Stillness vs Motion)")

# === スコア計算関数 ===
def calc_airfeel(landmarks, prev_landmarks):
    if prev_landmarks is None:
        return 0.0

    curr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    prev = np.array([[lm.x, lm.y, lm.z] for lm in prev_landmarks])

    diff = curr - prev
    speed = np.linalg.norm(diff, axis=1).mean()  # 平均速度

    # 静: 動かないと+寄り, 動: 動くと-寄り としてスコア化
    stillness = np.exp(-speed * 50)   # 小さい速度ほど1に近い
    motion = 1 - stillness
    score = stillness - motion        # [-1, +1]

    return float(score)

# === カメラ開始 ===
cap = cv2.VideoCapture(0)
prev_landmarks = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    score = 0.0
    if results.pose_landmarks:
        score = calc_airfeel(results.pose_landmarks.landmark, prev_landmarks)
        prev_landmarks = results.pose_landmarks.landmark

        # 骨格を描画
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # スコア保存
    scores.append(score)

    # === グラフ更新 ===
    line.set_ydata(scores)
    line.set_xdata(range(len(scores)))
    ax.draw_artist(ax.patch)
    ax.draw_artist(line)
    fig.canvas.flush_events()

    # === 数値出力 ===
    print(f"Airfeel Score: {score:.3f}")  # コンソール出力
    cv2.putText(frame, f"Score: {score:.3f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # 映像上に描画

    # カメラ表示
    cv2.imshow('Airfeel Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
