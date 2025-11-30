# -*- coding: utf-8 -*-
# Shannon-only Stillness/Gaze/Expression (no mediapipe)
# 1) 肌色マスク(YCrCb) -> 最大輪郭=顔近傍ROI
# 2) ROIを上下3分割: Eyes/Nose/Mouth
# 3) 各ROIでヒスト分布 -> Shannonエントロピー
# 4) ファジィ合成 + ヒステリシス/滞留/ゲートで安定化

import cv2 as cv
import numpy as np
from collections import deque

# ---------- util ----------
def shannon_entropy_from_hist(h, eps=1e-12):
    p = h / (np.sum(h) + eps)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))  # nats

def soft_fuzzy_low(value, low, high):
    # 低いほど"良い"(=Still寄与)を [0,1] に線形マップ
    return float(np.clip((high - value) / (high - low + 1e-9), 0, 1))

def ema(prev, x, a=0.25):
    return x if prev is None else (a*x + (1-a)*prev)

def magnitude_direction(flow):
    fx, fy = flow[...,0], flow[...,1]
    mag = np.sqrt(fx*fx + fy*fy)
    ang = (np.arctan2(fy, fx) + np.pi)  # [0, 2pi)
    return mag, ang

def hist_entropy(values, bins, rng):
    h, _ = np.histogram(np.clip(values, rng[0], rng[1]), bins=bins, range=rng)
    return shannon_entropy_from_hist(h), h

def skin_mask_ycrcb(bgr):
    ycrcb = cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv.split(ycrcb)
    # ゆるめの肌色域（環境で調整）
    mask = cv.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    mask = cv.medianBlur(mask, 5)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    return mask

def largest_bbox(mask, min_area=5000):
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv.contourArea)
    if cv.contourArea(cnt) < min_area: return None
    x,y,w,h = cv.boundingRect(cnt)
    return (x,y,w,h)

# ---------- config ----------
W,H = 640, 360
BINS_MAG = 12
BINS_DIR = 12
BINS_Y   = 32
EMA_A = 0.25

# 閾値（目安）
THR = dict(
    motion_low=0.35, motion_high=1.10,        # H_motion
    eyed_low =0.35, eyed_high =1.10,          # H_dir_eye
    blink_spike=0.20,                         # ΔH_blink がこれ以上なら瞬き＝Motion寄り
    mouth_low=0.25, mouth_high=0.90,          # H_mouth_open-ish
    tongue_low=0.20, tongue_high=0.85,        # H_tongue-ish
)

# 状態安定化
DWELL_FR = 45        # Still判定の滞留
ENTER_HYST = 0.50    # fused<0.5 でStill候補
EXIT_HYST  = 0.60    # fused>0.6 で解除

# 背景差分で入退場ゲート
bg = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
refractory = 0
REFR_FR = 20
prev_fg = None
prev_luma = None

cap = cv.VideoCapture(0)
assert cap.isOpened(), "camera open failed"

prev_gray = None
H_motion_ema = None
H_eye_dir_ema = None
H_blink_prev = None
still_counter = 0
state = "MOTION"

# 履歴
blink_hist = deque(maxlen=15)  # 0.5s程度

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv.resize(frame, (W,H))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray_blur = cv.GaussianBlur(gray, (3,3), 0)

    # 入退場ゲート用
    mask_bg = bg.apply(gray_blur)
    fg_ratio = float(np.count_nonzero(mask_bg==255))/mask_bg.size
    luma = float(np.mean(gray_blur))
    if prev_fg is not None and abs(fg_ratio-prev_fg)>0.08: refractory = REFR_FR
    if prev_luma is not None and abs(luma-prev_luma)>12.0: refractory = REFR_FR
    prev_fg, prev_luma = fg_ratio, luma

    # 肌色で顔っぽい領域
    skin = skin_mask_ycrcb(frame)
    box = largest_bbox(skin, min_area=4000)
    if box is None:
        cv.putText(frame,"No face ROI", (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
        cv.imshow("Shannon-only", frame)
        if cv.waitKey(1)==27: break
        prev_gray = gray_blur
        continue

    x,y,w,h = box
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,200,255),2)

    # 3分割（上:目, 中:鼻, 下:口）
    y1 = y + int(h*0.33)
    y2 = y + int(h*0.66)
    eyeROI  = gray_blur[y:y1, x:x+w]
    noseROI = gray_blur[y1:y2, x:x+w]
    mouthROI= gray_blur[y2:y+h, x:x+w]
    if eyeROI.size==0 or mouthROI.size==0:
        prev_gray = gray_blur
        continue

    # 光フロー（全体＆目ROI）
    if prev_gray is None:
        prev_gray = gray_blur
        cv.putText(frame,"warming up...", (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
        cv.imshow("Shannon-only", frame)
        if cv.waitKey(1)==27: break
        continue

    flow = cv.calcOpticalFlowFarneback(prev_gray, gray_blur, None,
                                       pyr_scale=0.5, levels=3, winsize=15,
                                       iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = magnitude_direction(flow)
    # 大外れ値をクリップ
    mag_max = np.percentile(mag, 99)+1e-6
    mag = np.clip(mag, 0, mag_max)
    mag[mag < 0.03] = 0.0

    # 全体のH_motion
    Hm, _ = hist_entropy(mag.ravel(), BINS_MAG, (0,mag_max))
    H_motion_ema = ema(H_motion_ema, Hm, EMA_A)

    # 目領域の方向エントロピー（目線が一点なら方向が揃って H↓）
    eye_flow = flow[y:y1, x:x+w]
    e_mag, e_ang = magnitude_direction(eye_flow)
    e_ang = e_ang[e_mag>0.01]
    if e_ang.size>50:
        He_dir, _ = hist_entropy(e_ang, BINS_DIR, (0, 2*np.pi))
    else:
        He_dir = 0.0
    H_eye_dir_ema = ema(H_eye_dir_ema, He_dir, 0.35)

    # 目領域の輝度ヒストのエントロピー（瞬きでΔHがドンと動く）
    Hy_eye, _ = hist_entropy(eyeROI.ravel(), BINS_Y, (0,255))
    if H_blink_prev is None: H_blink_prev = Hy_eye
    dH_blink = max(0.0, Hy_eye - H_blink_prev)  # 増えた分を瞬き疑い
    H_blink_prev = ema(H_blink_prev, Hy_eye, 0.4)
    blink_hist.append(dH_blink)
    blink_spike = float(np.mean(blink_hist))  # 平滑化

    # 口：開き＆舌（Shannonのみの代理特徴）
    #   - Cannyで垂直エッジ強度の分布 → 開くと分布が片寄ってH↓
    edges = cv.Canny(mouthROI, 60, 120)
    sobel_y = cv.Sobel(mouthROI, cv.CV_32F, 0, 1, ksize=3)
    vabs = np.abs(sobel_y).ravel()
    H_mouth, _ = hist_entropy(vabs, BINS_MAG, (0, np.percentile(vabs, 99)+1e-6))

    #   - 赤色比率の分布 → 舌が出ると赤が単峰に増えてH↓
    mouth_bgr = frame[y2:y+h, x:x+w]
    hsv = cv.cvtColor(mouth_bgr, cv.COLOR_BGR2HSV)
    Hc,Sc,Vc = cv.split(hsv)
    red = ((Hc < 15) | (Hc > 165)) & (Sc > 40) & (Vc > 40)
    red_ratio = np.mean(red.astype(np.float32))
    # 「比率のヒスト（0..1を10ビン）」のH
    Hr, _ = hist_entropy(np.array([red_ratio]), 1, (0,1))  # 単点だとH=0、変動を見るなら履歴化
    # 簡易に：口Hと赤比率で舌スコア proxy
    tongue_score = float(np.clip(red_ratio*3.0 - 0.2, 0, 1))
    # 舌H proxy（“出たら単峰”として Hを低い方へ倒す用途）：ここでは 1 - tongue_score をH擬似
    H_tongue_proxy = 1.0 - tongue_score

    # -------- ファジィ合成（Still寄与） --------
    f_motion = soft_fuzzy_low(H_motion_ema, THR["motion_low"], THR["motion_high"])
    f_eyedir= soft_fuzzy_low(H_eye_dir_ema, THR["eyed_low"],  THR["eyed_high"])
    f_blink = float(1.0 - np.clip(blink_spike/THR["blink_spike"], 0, 1))  # 瞬き強いとStill下げ
    f_mouth = soft_fuzzy_low(H_mouth, THR["mouth_low"], THR["mouth_high"])
    f_tong  = soft_fuzzy_low(H_tongue_proxy, THR["tongue_low"], THR["tongue_high"])

    # 最終 Stillness（目×体×表情の積）
    still_fused = f_motion * f_eyedir * f_blink
    # 口・舌は補正：口が極端に活動→Still弱め。ただし舌出し中は減点緩める
    mouth_boost = 0.7 + 0.3*(f_mouth*0.7 + f_tong*0.3)
    still_fused *= mouth_boost

    # ヒステリシス＋滞留＋不応
    label = "MOTION"
    if refractory > 0:
        refractory -= 1
        still_counter = 0
        state = "MOTION"
        gate_text = f"GATED {refractory}"
    else:
        gate_text = ""

        if state=="MOTION":
            if still_fused > (1.0-ENTER_HYST):
                still_counter += 1
                if still_counter >= DWELL_FR:
                    state = "STILL"; still_counter = 0
            else:
                still_counter = 0
        else:
            if still_fused < (1.0-EXIT_HYST):
                state = "MOTION"

    label = f"STATE: {state}"
    cv.putText(frame, label, (20, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8,
               (0,255,0) if state=="STILL" else (0,0,255), 2, cv.LINE_AA)
    if gate_text:
        cv.putText(frame, gate_text, (20, 54), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)

    # UI（軽くバー表示）
    def bar(y, txt, val, col):
        cv.rectangle(frame,(20,y),(280,y+12),(60,60,60),1)
        cv.rectangle(frame,(20,y),(20+int(260*val),y+12),col,-1)
        cv.putText(frame, f"{txt}: {val:.2f}", (20, y-2), cv.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1, cv.LINE_AA)

    bar( 90, "f_motion", f_motion, (0,200,0))
    bar(110, "f_eyedir", f_eyedir,(0,180,200))
    bar(130, "f_blink ", f_blink, (200,180,0))
    bar(150, "f_mouth ", f_mouth, (120,180,255))
    bar(170, "f_tong  ", f_tong,  (180,120,255))
    bar(190, "Still*",  still_fused,(0,255,255))

    # 目安として矢印描画（間引き）
    step=24
    for yy in range(y, y+h, step):
        for xx in range(x, x+w, step):
            dx, dy = flow[yy, xx]
            cv.arrowedLine(frame,(xx,yy),(int(xx+dx),int(yy+dy)),(180,180,50),1,tipLength=0.3)

    cv.imshow("Shannon-only", frame)
    prev_gray = gray_blur
    if cv.waitKey(1)==27: break

cap.release()
cv.destroyAllWindows()
