import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
cap = cv2.VideoCapture(0)

# 接続定義
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape

    unified_nodes = {}

    # --- Poseノード登録（顔部分除外） ---
    pose_face_ids = set(range(0, 11))
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            if i in pose_face_ids:
                continue
            x, y = int(lm.x * w), int(lm.y * h)
            unified_nodes[f"pose_{i}"] = (x, y)

    # --- 左手ノード登録 ---
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            unified_nodes[f"left_hand_{i}"] = (x, y)

    # --- 右手ノード登録 ---
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            x, y = int(lm.x * w), int(lm.y * h)
            unified_nodes[f"right_hand_{i}"] = (x, y)

    # --- 接続線（Pose） ---
    for i1, i2 in mp_pose.POSE_CONNECTIONS:
        key1, key2 = f"pose_{i1}", f"pose_{i2}"
        if key1 in unified_nodes and key2 in unified_nodes:
            cv2.line(image, unified_nodes[key1], unified_nodes[key2], (255, 255, 255), 2)

    # --- 接続線（Hands） ---
    for hand_prefix, connections in [("left_hand", mp_hands.HAND_CONNECTIONS), ("right_hand", mp_hands.HAND_CONNECTIONS)]:
        for i1, i2 in connections:
            key1, key2 = f"{hand_prefix}_{i1}", f"{hand_prefix}_{i2}"
            if key1 in unified_nodes and key2 in unified_nodes:
                cv2.line(image, unified_nodes[key1], unified_nodes[key2], (200, 200, 255), 1)

    # --- ノード描画（全ノード） ---
    for _, (x, y) in unified_nodes.items():
        cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

    cv2.imshow("No-Face Holistic View", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
