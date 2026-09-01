# -*- coding: utf-8 -*-
# Nori-RT for Animation Dance (plus gates & adaptive thresholds)
# 追加:
# - ENTRY/EXITゲート（前景面積/輝度の急変で不応期間）
# - AE急変ブレーキ（H急降下時はEMA追従を鈍化）
# - 前景最小率（fg_ratio が小さい時は Stillness禁止）
# - 自動適応しきい値（直近履歴から enter_thr を微調整）
# - ROI重み（最大前景にガウシアン重みをかけ、中心を重視）

import cv2 as cv
import numpy as np
from collections import deque

CFG = dict(
    cam_id=0,
    frame_size=(640, 360),
    blur_ksize=(3, 3),
    deadzone=0.03,
    bins=10,
    ema_alpha=0.25,
    clip_percentile=99.0,

    dwell_frames=45,
    enter_thr_H=0.40,   # 初期値（自動適応で軽く動く）
    exit_thr_H=0.65,
    enter_thr_medV=0.12,
    exit_thr_medV=0.25,

    stop_flash_frames=18,

    # 追加：ゲート/適応/ROI
    use_entry_exit_gate=True,
    refr_frames=20,          # 不応フレーム（~0.6-0.8s@30fps）
    fg_ratio_jump=0.08,      # 前景率がこの差分以上で急変扱い
    luma_jump=12.0,          # 平均輝度の急変しきい値
    min_fg_ratio=0.03,       # これ未満の前景率は Stillness禁止
    adapt_enable=True,       # 履歴から適応調整
    adapt_window=60,         # 適応用の履歴長
    adapt_H_quantile=0.35,   # enter_thr_H ≈ q35 あたりに更新
    adapt_V_quantile=0.35,   # enter_thr_medV ≈ q35 あたりに更新
    roi_weight_enable=True,  # ROI重み（最大連結成分中心を重視）
    roi_weight_sigma=0.35,   # ガウシアン径（0~1; ROI包絡の対角で規格化）
)

# ===== utils =====
def shannon_entropy_from_hist(h, eps=1e-12):
    p = h / (np.sum(h) + eps)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))

def ema(prev, x, alpha=0.2):
    return alpha*x + (1-alpha)*prev

def draw_meter(frame, value, min_v, max_v, x, y, w, h, label, color=(0,200,0)):
    v = (value - min_v) / (max_v - min_v + 1e-9)
    v = float(np.clip(v, 0, 1))
    cv.rectangle(frame, (x, y), (x+w, y+h), (60,60,60), 1)
    cv.rectangle(frame, (x, y), (x+int(w*v), y+h), color, -1)
    cv.putText(frame, f"{label}: {value:.3f}", (x, y-6),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

def fuzzy_from_entropy(H, low, high):
    motion = np.clip((H - low) / (high - low + 1e-9), 0, 1)
    still  = 1.0 - motion
    return still, motion

def magnitude_hist_weighted(flow, mask_weight, bins=16, clip_percentile=99.0, deadzone=0.03):
    fx, fy = flow[...,0], flow[...,1]
    mag = np.sqrt(fx*fx + fy*fy)
    # デッドゾーン
    mag = np.where(mag < deadzone, 0.0, mag)
    # ROI重み付け
    if mask_weight is not None:
        mag = mag * mask_weight
    mag_max = np.percentile(mag, clip_percentile) + 1e-6
    mag_clip = np.clip(mag, 0, mag_max)
    h, _ = np.histogram(mag_clip, bins=bins, range=(0, mag_max))
    return h, mag

def gaussian_weight_for_roi(rect, shape, sigma_norm=0.35):
    """
    rect: (x,y,w,h) 最大前景の外接矩形
    shape: (H,W)
    sigma_norm: 0~1, ROI対角に対する相対σ
    -> ROI中心を1.0、周辺を下げるガウシアン重み（背景はそのまま）
    """
    H, W = shape
    (x,y,w,h) = rect
    cx, cy = x + w/2.0, y + h/2.0
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((xx-cx)/max(w,1e-6))**2 + ((yy-cy)/max(h,1e-6))**2)
    sigma = max(1e-3, sigma_norm)  # 正規化空間のσ
    weight = np.exp(-(dist**2)/(2*sigma**2))
    # ROI外でもゼロにはしない（周辺も少しは拾う）
    return np.clip(weight, 0.1, 1.0).astype(np.float32)

# ===== main =====
cap = cv.VideoCapture(CFG["cam_id"])
assert cap.isOpened(), "カメラが開けないよ :("

prev_gray = None
H_ema = None
state = "MOTION"
still_counter = 0
stop_flash = 0

# 背景差分（ENTRY/EXIT, ROI抽出）
bg = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)

# 統計 & 適応
hist_H = deque(maxlen=CFG["adapt_window"])
hist_V = deque(maxlen=CFG["adapt_window"])

# ゲート用
refractory = 0
prev_fg_ratio = None
prev_luma = None
prev_H_raw = None

# フロー
flow_params = dict(pyr_scale=0.5, levels=3, winsize=15,
                   iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

H_display_max = 3.5

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv.resize(frame, CFG["frame_size"])
    gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray  = cv.GaussianBlur(gray, CFG["blur_ksize"], 0)

    if prev_gray is None:
        prev_gray = gray.copy()
        cv.putText(frame, "warming up...", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv.imshow("Nori-RT (Dance Stillness+)", frame)
        if cv.waitKey(1) & 0xFF == 27:
            break
        continue

    # --- 背景差分で前景率/ROI ---
    fgmask = bg.apply(gray)  # 0, 127(影), 255
    # 影は背景扱い
    fg = (fgmask == 255).astype(np.uint8)
    fg_ratio = float(np.count_nonzero(fg)) / fg.size
    luma = float(np.mean(gray))

    # ROI重み（最大連結成分の外接矩形中心にガウシアン）
    weight = None
    if CFG["roi_weight_enable"]:
        cnts, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            cnt = max(cnts, key=cv.contourArea)
            x,y,w,h = cv.boundingRect(cnt)
            if w*h > 100:  # 小さ過ぎる検出は無視
                weight = gaussian_weight_for_roi((x,y,w,h), gray.shape, CFG["roi_weight_sigma"])
                cv.rectangle(frame, (x,y), (x+w, y+h), (50,200,255), 1)

    # --- Optical Flow ---
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)

    # --- ヒスト→H（ROI重み & デッドゾーン適用） ---
    hist, mag = magnitude_hist_weighted(
        flow, mask_weight=weight,
        bins=CFG["bins"],
        clip_percentile=CFG["clip_percentile"],
        deadzone=CFG["deadzone"]
    )
    H_raw = shannon_entropy_from_hist(hist)
    med_v = float(np.median(mag))

    # --- ENTRY/EXIT ゲート & AE急変検出 ---
    if CFG["use_entry_exit_gate"]:
        if prev_fg_ratio is not None:
            if abs(fg_ratio - prev_fg_ratio) > CFG["fg_ratio_jump"]:
                refractory = CFG["refr_frames"]
        if prev_luma is not None:
            if abs(luma - prev_luma) > CFG["luma_jump"]:
                refractory = CFG["refr_frames"]
        prev_fg_ratio = fg_ratio
        prev_luma = luma

    # --- H急降下時はEMA追従を鈍化（入口暴発ブレーキ） ---
    if prev_H_raw is None:
        prev_H_raw = H_raw
    dH = H_raw - prev_H_raw
    prev_H_raw = H_raw
    alpha_now = CFG["ema_alpha"] * (0.35 if dH < -0.5 else 1.0)
    H_ema = H_raw if H_ema is None else ema(H_ema, H_raw, alpha=alpha_now)

    # --- 適応しきい値（ゆっくり更新） ---
    if CFG["adapt_enable"]:
        hist_H.append(H_ema)
        hist_V.append(med_v)
        if len(hist_H) >= int(0.6 * CFG["adapt_window"]):
            # 下位分位側に合わせて「静止入りやすさ」を環境適応
            ent_q = float(np.quantile(hist_H, CFG["adapt_H_quantile"]))
            vel_q = float(np.quantile(hist_V, CFG["adapt_V_quantile"]))
            # 極端に動かし過ぎないように混合
            enter_H = 0.7*CFG["enter_thr_H"] + 0.3*ent_q
            enter_V = 0.7*CFG["enter_thr_medV"] + 0.3*vel_q
        else:
            enter_H, enter_V = CFG["enter_thr_H"], CFG["enter_thr_medV"]
    else:
        enter_H, enter_V = CFG["enter_thr_H"], CFG["enter_thr_medV"]

    exit_H  = CFG["exit_thr_H"]
    exit_V  = CFG["exit_thr_medV"]

    # ファジィ指標（表示用）
    still_m, motion_m = fuzzy_from_entropy(H_ema, low=enter_H, high=exit_H)

    # --- 強制MOTION（不応 & 前景小さすぎ） ---
    forced_motion = False
    if refractory > 0:
        refractory -= 1
        forced_motion = True
    if fg_ratio < CFG["min_fg_ratio"]:
        forced_motion = True

    # --- 状態遷移（ヒステリシス + 滞留） ---
    if forced_motion:
        state = "MOTION"
        still_counter = 0
    else:
        if state == "MOTION":
            if (H_ema < enter_H) and (med_v < enter_V):
                still_counter += 1
                if still_counter >= CFG["dwell_frames"]:
                    state = "STILL"
                    stop_flash = CFG["stop_flash_frames"]
                    still_counter = 0
            else:
                still_counter = 0
        else:
            if (H_ema > exit_H) or (med_v > exit_V):
                state = "MOTION"

    # --- 表示 ---
    draw_meter(frame, H_ema, 0.0, H_display_max, 20, 30, 260, 16, "H (Shannon, nats)")
    draw_meter(frame, float(motion_m), 0.0, 1.0, 20, 60, 260, 16, "Motion fuzzy", (0,170,0))
    draw_meter(frame, float(still_m),  0.0, 1.0, 20, 90, 260, 16, "Stillness fuzzy", (0,150,250))

    label = "STATE: STILL (Stop as Move)" if state=="STILL" else "STATE: MOTION"
    color = (0,255,0) if state=="STILL" else (0,0,255)
    cv.putText(frame, label, (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)

    if refractory > 0:
        cv.putText(frame, f"ENTRY/EXIT GATE ({refractory})",
                   (20, 160), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv.LINE_AA)

    if stop_flash > 0:
        overlay = frame.copy()
        cv.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,255,255), -1)
        frame = cv.addWeighted(overlay, 0.12, frame, 0.88, 0)
        cv.putText(frame, "STOP !", (20, 190), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3, cv.LINE_AA)
        stop_flash -= 1

    # デバッグ矢印（重いならstep↑ or OFF）
    step = 20
    h, w = gray.shape
    for yy in range(0, h, step):
        for xx in range(0, w, step):
            dx, dy = flow[yy, xx].tolist()
            cv.arrowedLine(frame, (xx, yy), (int(xx+dx), int(yy+dy)),
                           (200,200,50), 1, tipLength=0.3)

    cv.imshow("Nori-RT (Dance Stillness+)", frame)
    prev_gray = gray
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
