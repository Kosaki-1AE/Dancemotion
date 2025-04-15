import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Movenet モデル読み込み
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
movenet = model.signatures['serving_default']

def detect_pose(img):
    input_image = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_image = tf.cast(input_image, dtype=tf.int32)
    # 推論
    outputs = movenet(input_image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    return keypoints

def draw_vector(image, kps, i, j):
    h, w, _ = image.shape
    p1 = (int(kps[i][1] * w), int(kps[i][0] * h))
    p2 = (int(kps[j][1] * w), int(kps[j][0] * h))
    cv2.arrowedLine(image, p1, p2, (0, 255, 0), 2)
    vec = np.array(p2) - np.array(p1)
    print(f"ベクトル ({i}->{j}): {vec}")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    kps = detect_pose(frame)

    draw_vector(frame, kps, 5, 7)  # 左肩→左肘
    draw_vector(frame, kps, 6, 8)  # 右肩→右肘

    cv2.imshow('MoveNet Pose', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
