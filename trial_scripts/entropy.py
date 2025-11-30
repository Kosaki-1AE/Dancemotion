import cv2 as cv
import numpy as np
import time
from collections import deque

def shannon_entropy_from_hist(h, eps=1e-12):
    p = h / (np.sum(h) + eps)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))  # nats（自然対数）。log2にしたいなら /np.log(2)

def magnitude_hist(flow, bins=16, mag_max=None):
    fx, fy = flow[...,0], flow[...,1]
    mag = np.sqrt(fx*fx + fy*fy)
    if mag_max is None:
        mag_max = np.percentile(mag, 99) + 1e-6  # 外れ値を切ると安定
    h, _ = np.histogram(np.clip(mag, 0, mag_max), bins=bins, range=(0, mag_max))
    return h

def ema(prev, x, alpha=0.2):
    return alpha*x + (1-alpha)*prev

cap = cv.VideoCapture(0)  # カメラID
assert cap.isOpened(), "カメラが開けないよ :("

prev_gray = None
H_ema = None
hist_win = deque(maxlen=30)  # 条件付きやPEの近似用に短期履歴も保持できる

# ヒートマップ表示用
def draw_meter(frame, value, min_v=0.0, max_v=3.5, x=20, y=20, w=240, h=18, label="H"):
    v = (value - min_v) / (max_v - min_v + 1e-9)
    v = float(np.clip(v, 0, 1))
    cv.rectangle(frame, (x, y), (x+w, y+h), (60,60,60), 1)
    cv.rectangle(frame, (x, y), (x+int(w*v), y+h), (0,200,0), -1)
    cv.putText(frame, f"{label}: {value:.3f}", (x, y-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

# “ファジィ”なStillness/Motion度（会員度）をHから作る
def fuzzy_from_entropy(H, low=0.4, high=1.2):
    # 低エントロピー→Stillness、 高エントロピー→Motion
    # 線形ソフト化（S字にしたければシグモイドでもOK）
    motion = np.clip((H - low) / (high - low + 1e-9), 0, 1)
    still  = 1.0 - motion
    return still, motion

# Optical Flowのパラメータ（Farnebäck）
flow_params = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

bins = 16
H_display_max = 3.5  # メーター上限（適宜調整）

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv.resize(frame, (640, 360))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray.copy()
        cv.putText(frame, "warming up...", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv.imshow("Nori-RT (Shannon baseline)", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
        continue

    # フロー計算
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)

    # 大きさヒスト → Shannon
    hist = magnitude_hist(flow, bins=bins)
    H = shannon_entropy_from_hist(hist)  # nats

    # EMAで安定化（リアタイの揺らぎ抑え）
    if H_ema is None:
        H_ema = H
    else:
        H_ema = ema(H_ema, H, alpha=0.25)

    # ファジィ会員度
    still_m, motion_m = fuzzy_from_entropy(H_ema, low=0.35, high=1.10)

    # （任意）短期履歴を保持：条件付きやPEの近似に使える
    hist_win.append(hist)

    # 表示
    draw_meter(frame, H_ema, 0.0, H_display_max, x=20, y=30,  w=260, h=16, label="H (Shannon, nats)")
    draw_meter(frame, float(motion_m), 0.0, 1.0,        x=20, y=60,  w=260, h=16, label="Motion fuzzy")
    draw_meter(frame, float(still_m),  0.0, 1.0,        x=20, y=90,  w=260, h=16, label="Stillness fuzzy")

    # ざっくりラベル
    state = "MOTION" if motion_m > 0.5 else "STILL"
    color = (0,0,255) if state=="MOTION" else (0,255,0)
    cv.putText(frame, f"STATE: {state}", (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)

    # デバッグで矢印を間引き描画（重いならオフに）
    step = 20
    h, w = gray.shape
    for y in range(0, h, step):
        for x in range(0, w, step):
            dx, dy = flow[y, x].tolist()
            cv.arrowedLine(frame, (x, y), (int(x+dx), int(y+dy)), (200,200,50), 1, tipLength=0.3)

    cv.imshow("Nori-RT (Shannon baseline)", frame)
    prev_gray = gray

    if cv.waitKey(1) & 0xFF == 27:  # ESCで終了
        break

cap.release()
cv.destroyAllWindows()
