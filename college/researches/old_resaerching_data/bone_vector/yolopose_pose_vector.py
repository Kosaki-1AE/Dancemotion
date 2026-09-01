import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")  # 軽量モデル

cap = cv2.VideoCapture(0)

# Cocoフォーマットでのキーポイント接続ペア（17点）
# [親, 子] のインデックス（0から始まる）
connections = [
    [5, 7],  # 左肩 → 左肘
    [7, 9],  # 左肘 → 左手首
    [6, 8],  # 右肩 → 右肘
    [8,10],  # 右肘 → 右手首
    [5,11],  # 左肩 → 左腰
    [6,12],  # 右肩 → 右腰
    [11,13], # 左腰 → 左膝
    [13,15], # 左膝 → 左足首
    [12,14], # 右腰 → 右膝
    [14,16], # 右膝 → 右足首
    [5,6],   # 左肩 → 右肩
    [11,12], # 左腰 → 右腰
    [0,1],   # 鼻 → 左目
    [0,2],   # 鼻 → 右目
    [1,3],   # 左目 → 左耳
    [2,4],   # 右目 → 右耳
    [0,5],   # 鼻 → 左肩
    [0,6],   # 鼻 → 右肩
]

def draw_vector(image, p1, p2):
    pt1 = tuple(map(int, p1))
    pt2 = tuple(map(int, p2))
    vec = np.array(pt2) - np.array(pt1)
    cv2.arrowedLine(image, pt1, pt2, (0, 255, 0), 2, tipLength=0.2)
    print(f"ベクトル {pt1}→{pt2} : {vec}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        if result.keypoints is not None and result.keypoints.xy.numel() > 0:
            kpts = result.keypoints.xy[0].cpu().numpy()

            for pair in connections:
                i, j = pair
                if i < len(kpts) and j < len(kpts):
                    draw_vector(frame, kpts[i], kpts[j])

        # 骨格の描画（Ultralyticsの内部可視化）
        result.plot()

    cv2.imshow("YOLO-Pose Full Body Vectors", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
