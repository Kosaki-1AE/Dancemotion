# -*- coding: utf-8 -*-
"""
nori_onepass_unified.py
UIはFace Shannon-onlyそのまま。内部で Face と AV Shannon+ を同時計算し、
信頼度で自動ブレンドして Still* を出力（モード分割なし）。
矢印：枠外=水色 / 枠内=薄い青
依存: opencv-python, numpy, (任意)sounddevice
"""

import cv2 as cv
import numpy as np
import sys, queue
from collections import deque

# --- HUD popup toggle ---
HUD_POPUP = True   # True=バーは別ウィンドウ / False=従来どおり映像に重ねる

def make_hud_canvas(w=300, h=160, bg=(32,32,32)):
    hud = np.zeros((h, w, 3), dtype=np.uint8)
    hud[:] = bg
    return hud

def hud_bar(hud, y, txt, val, col):
    x0, x1, bar_w, bar_h = 20, 280, 260, 12
    cv.rectangle(hud, (x0,y), (x1,y+bar_h), (60,60,60), 1)
    v = np.clip(val, 0, 1)
    cv.rectangle(hud, (x0,y), (x0+int(bar_w*v), y+bar_h), col, -1)
    cv.putText(hud, f"{txt}: {val:.2f}", (x0, y-2),
               cv.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv.LINE_AA)

def show_hud(f_motion, f_eyedir, f_blink, f_mouth, f_tong, still):
    hud = make_hud_canvas()
    hud_bar(hud,  20, "Motion Intensity ", f_motion, (0,200,0))
    hud_bar(hud,  40, "Eye Movement ", f_eyedir,(0,180,200))
    hud_bar(hud,  60, "Blink Rate ", f_blink, (200,180,0))
    hud_bar(hud,  80, "Smile Intensity ", f_mouth, (120,180,255))
    hud_bar(hud, 100, "Tong move ", f_tong,  (180,120,255))
    hud_bar(hud, 120, "Score",  still,    (0,255,255))
    cv.imshow("Shannon-HUD", hud)
    # 固定位置に出したいなら↓のコメントアウトを外して調整
    # cv.moveWindow("Shannon-HUD", 680, 60)

# ====== Audio (optional) ======
USE_AUDIO = True
try:
    import sounddevice as sd
except Exception:
    USE_AUDIO = False
    sd = None
    print("[WARN] sounddevice が読み込めなかったので音声は無効化します。", file=sys.stderr)

audio_q = queue.Queue()
audio_stream = None
AUDIO_OK = False

def audio_callback(indata, frames, time_info, status):
    if indata is None or len(indata)==0: return
    x = indata.copy()
    if x.ndim==2 and x.shape[1]>1:
        x = x.mean(axis=1, keepdims=True)
    audio_q.put(x)

def start_audio(sr=16000, blocksize=512, device=None):
    global audio_stream, AUDIO_OK
    if not USE_AUDIO:
        AUDIO_OK = False; return
    try:
        audio_stream = sd.InputStream(
            channels=1, samplerate=sr, blocksize=blocksize,
            device=device, dtype='float32', callback=audio_callback
        )
        audio_stream.start(); AUDIO_OK = True
    except Exception as e:
        AUDIO_OK = False
        print(f"[WARN] マイク開始に失敗: {e} → 音声は無効化します。", file=sys.stderr)

# ====== Utils ======
def shannon_entropy_from_hist(h, eps=1e-12):
    s = float(np.sum(h)); 
    if s <= eps: return 0.0
    p = np.clip(h/(s+eps), eps, 1.0)
    return float(-np.sum(p*np.log(p)))

def ema(prev, x, a):
    return float(x) if prev is None else float(a*x + (1-a)*prev)

def fuzzy_low(v, low, high):
    return float(np.clip((high - v) / (high - low + 1e-9), 0, 1))

def draw_bar(frame, y, txt, val, col):
    cv.rectangle(frame,(20,y),(280,y+12),(60,60,60),1)
    cv.rectangle(frame,(20,y),(20+int(260*np.clip(val,0,1)),y+12),col,-1)
    cv.putText(frame, f"{txt}: {val:.2f}", (20, y-2),
               cv.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1, cv.LINE_AA)

# ====== Config ======
W,H = 640,360
CFG = dict(
    cam_id=0, frame_size=(W,H),
    # flow
    flow_deadzone=0.03, flow_bins=12, flow_clip_percentile=99.0, flow_ema=0.25,
    # audio
    sr=16000, blocksize=512, audio_bins=32, audio_ema=0.35, audio_gain=1.0, audio_device=None,
    # gate
    gate_fg_delta=0.08, gate_luma_delta=12.0, gate_frames=20,
    # hysteresis
    enter=0.50, exit=0.60, dwell=45,
    # display
    H_display_max=3.5
)

THR = dict(
    motion_low=0.35, motion_high=1.10,
    eyed_low=0.35,   eyed_high=1.10,
    blink_spike=0.20,
    mouth_low=0.25,  mouth_high=0.90,
    tongue_low=0.20, tongue_high=0.85
)

# ====== Face helpers ======
def skin_mask_ycrcb(bgr):
    ycrcb = cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)
    mask = cv.inRange(ycrcb, (0,135,85), (255,180,135))
    mask = cv.medianBlur(mask,5)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3),np.uint8),1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE,np.ones((5,5),np.uint8),2)
    return mask

def largest_bbox(mask, min_area=4000):
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv.contourArea)
    if cv.contourArea(cnt) < min_area: return None
    return cv.boundingRect(cnt)

def audio_shannon(block, bins=32, gain=1.0):
    if block is None or block.size==0: return 0.0
    x = (block[:,0]*gain).astype(np.float32)
    n = len(x); 
    if n<=0: return 0.0
    win = np.hanning(n).astype(np.float32)
    X = np.abs(np.fft.rfft(x*win))**2
    hi = float(np.percentile(X,99))+1e-6
    h,_ = np.histogram(X, bins=bins, range=(0,hi))
    return shannon_entropy_from_hist(h)

FLOW_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15,
                   iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# ====== Main ======
def main():
    cap = cv.VideoCapture(CFG["cam_id"])
    assert cap.isOpened(), "camera open failed"
    start_audio(CFG["sr"], CFG["blocksize"], CFG["audio_device"])

    # 背景ゲート
    bg = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
    refractory = 0; prev_fg=None; prev_luma=None

    prev_gray=None
    # Face state
    H_motion_ema=None; H_eye_dir_ema=None; H_blink_prev=None
    blink_hist = deque(maxlen=15)
    # AV state
    H_video_ema=None; H_audio_ema=None

    state="MOTION"; still_counter=0

    # 矢印色
    COLOR_CYAN      = (255,255,  0)  # ROI外
    COLOR_LIGHTBLUE = (255,210,120)  # ROI内
    YELLOW_BOX      = (0,200,255)

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv.resize(frame, (W,H))
        gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grayb = cv.GaussianBlur(gray, (3,3), 0)

        # ---- Gate ----
        mask_bg = bg.apply(grayb)
        fg_ratio = float(np.count_nonzero(mask_bg==255))/mask_bg.size
        luma = float(np.mean(grayb))
        if prev_fg is not None and abs(fg_ratio-prev_fg)>CFG["gate_fg_delta"]: refractory = CFG["gate_frames"]
        if prev_luma is not None and abs(luma-prev_luma)>CFG["gate_luma_delta"]: refractory = CFG["gate_frames"]
        prev_fg, prev_luma = fg_ratio, luma

        # ---- Face ROI ----
        skin = skin_mask_ycrcb(frame)
        box = largest_bbox(skin, min_area=4000)
        if box is None:
            cv.putText(frame,"No face ROI", (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
            cv.imshow("Shannon-only", frame)
            if cv.waitKey(1)==27: break
            prev_gray = grayb
            continue

        x,y,w,h = box
        cv.rectangle(frame,(x,y),(x+w,y+h),YELLOW_BOX,2)
        y1 = y + int(h*0.33)
        y2 = y + int(h*0.66)
        eyeROI  = grayb[y:y1, x:x+w]
        mouthROI= grayb[y2:y+h, x:x+w]
        if eyeROI.size==0 or mouthROI.size==0:
            prev_gray = grayb
            continue

        # ---- Flow ----
        if prev_gray is None:
            prev_gray = grayb
            cv.putText(frame,"warming up...", (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
            cv.imshow("Shannon-only", frame)
            if cv.waitKey(1)==27: break
            continue

        flow = cv.calcOpticalFlowFarneback(prev_gray, grayb, None, **FLOW_PARAMS)
        fx, fy = flow[...,0], flow[...,1]
        mag = np.sqrt(fx*fx + fy*fy)
        mag_max = np.percentile(mag, CFG["flow_clip_percentile"]) + 1e-6
        mag = np.clip(mag, 0, mag_max); mag[mag < CFG["flow_deadzone"]] = 0.0

        # ---- Face features -> still_face ----
        # 全体H_motion
        h_flow,_ = np.histogram(mag.ravel(), bins=CFG["flow_bins"], range=(0,mag_max))
        Hm = shannon_entropy_from_hist(h_flow)
        H_motion_ema = ema(H_motion_ema, Hm, CFG["flow_ema"])

        # 目方向エントロピー
        eye_flow = flow[y:y1, x:x+w]
        e_mag = np.sqrt(eye_flow[...,0]**2 + eye_flow[...,1]**2)
        e_ang = (np.arctan2(eye_flow[...,1], eye_flow[...,0]) + np.pi)
        e_ang = e_ang[e_mag>0.01]
        if e_ang.size>50:
            h_ed,_ = np.histogram(e_ang, bins=12, range=(0, 2*np.pi))
            He = shannon_entropy_from_hist(h_ed)
        else:
            He = 0.0
        H_eye_dir_ema = ema(H_eye_dir_ema, He, 0.35)

        # 瞬き proxy（輝度ヒスト）
        Hy_eye,_ = np.histogram(eyeROI.ravel(), bins=32, range=(0,255))
        Hy_eye = shannon_entropy_from_hist(Hy_eye)
        if H_blink_prev is None: H_blink_prev = Hy_eye
        dH = max(0.0, Hy_eye - H_blink_prev)
        H_blink_prev = ema(H_blink_prev, Hy_eye, 0.4)
        blink_hist.append(dH)
        blink_spike = float(np.mean(blink_hist))

        # 口/舌 proxy
        sobel_y = cv.Sobel(mouthROI, cv.CV_32F, 0,1, ksize=3)
        vabs = np.abs(sobel_y).ravel()
        H_mouth,_ = np.histogram(vabs, bins=12, range=(0, np.percentile(vabs,99)+1e-6))
        H_mouth = shannon_entropy_from_hist(H_mouth)
        hsv = cv.cvtColor(frame[y2:y+h, x:x+w], cv.COLOR_BGR2HSV)
        Hc,Sc,Vc = cv.split(hsv)
        red = ((Hc < 15) | (Hc > 165)) & (Sc > 40) & (Vc > 40)
        tongue_score = float(np.clip(np.mean(red.astype(np.float32))*3.0 - 0.2, 0, 1))
        H_tong_proxy = 1.0 - tongue_score

        f_motion = fuzzy_low(H_motion_ema, THR["motion_low"], THR["motion_high"])
        f_eyedir= fuzzy_low(H_eye_dir_ema, THR["eyed_low"],  THR["eyed_high"])
        f_blink = float(1.0 - np.clip(blink_spike/THR["blink_spike"], 0, 1))
        f_mouth = fuzzy_low(H_mouth, THR["mouth_low"], THR["mouth_high"])
        f_tong  = fuzzy_low(H_tong_proxy, THR["tongue_low"], THR["tongue_high"])

        still_face = f_motion * f_eyedir * f_blink
        still_face *= 0.7 + 0.3*(f_mouth*0.7 + f_tong*0.3)
        still_face = float(np.clip(still_face, 0, 1))

        # Faceの信頼度（ROI面積ベース）
        c_face = float(np.clip((w*h)/(W*H*0.20), 0.0, 1.0))

        # ---- AV Shannon+ -> still_av ----
        # ※ 映像Hは同じフローから、音声Hはあれば使用
        H_video_ema = ema(H_video_ema, Hm, CFG["flow_ema"])

        H_audio = None
        if AUDIO_OK:
            try:
                blk = audio_q.get_nowait()
                # FFT→ヒスト→H
                XH = audio_shannon(blk, bins=CFG["audio_bins"], gain=CFG["audio_gain"])
                H_audio_ema = ema(H_audio_ema, XH, CFG["audio_ema"])
            except queue.Empty:
                pass

        f_video = fuzzy_low(H_video_ema if H_video_ema is not None else 0.0, 0.35, 1.10)
        f_audio = fuzzy_low(H_audio_ema, 2.0, 5.0) if H_audio_ema is not None else 0.5
        fused_av = float(np.clip(0.6*(1-f_video) + 0.4*(1-f_audio), 0, 1))
        still_av = 1.0 - fused_av
        c_av = 1.0 if H_audio_ema is not None else 0.6  # 音声が取れてれば高信頼

        # ---- 自動ブレンド（UIはFaceのまま）----
        s = c_face + c_av
        if s <= 1e-6:
            still = still_face
            wf, wa = 1.0, 0.0
        else:
            wf, wa = c_face/s, c_av/s
            still = float(np.clip(wf*still_face + wa*still_av, 0, 1))
                # --- Nori: 映像/音声のShannon系特徴から A(意識) と T(温度) を推定 ---
        feat = np.array([
            f_motion, f_eyedir, f_blink, f_mouth, f_tong,
            f_video, (f_audio if H_audio_ema is not None else 0.5)
        ], dtype=np.float32)

        # ---- 滞留＋ヒステリシス（enter/exit/dwell は固定。必要なら自動調整も可）----
        gate_text = ""
        if refractory>0:
            refractory -= 1
            state="MOTION"; still_counter=0; gate_text=f"GATED {refractory}"
        else:
            if state=="MOTION":
                if still > 1.0:
                    still_counter += 1
                    if still_counter >= CFG["dwell"]:
                        state="STILL"; still_counter=0
                else:
                    still_counter=0
            else:
                if still < 1.0:
                    state="MOTION"

        # ===== UI（Face版そのまま）=====
        cv.putText(frame, f"STATE: {state}", (20,28), cv.FONT_HERSHEY_SIMPLEX, 0.8,
                   (0,255,0) if state=="STILL" else (0,0,255), 2, cv.LINE_AA)
        if gate_text:
            cv.putText(frame, gate_text, (20,54), cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)

        if HUD_POPUP:
        # バーは別ウィンドウに表示
            show_hud(f_motion, f_eyedir, f_blink, f_mouth, f_tong, still)
        else:
            # 従来どおりメイン映像に重ねる
            draw_bar(frame,  90, "f_motion", f_motion, (0,200,0))
            draw_bar(frame, 110, "f_eyedir", f_eyedir,(0,180,200))
            draw_bar(frame, 130, "f_blink ", f_blink, (200,180,0))
            draw_bar(frame, 150, "f_mouth ", f_mouth, (120,180,255))
            draw_bar(frame, 170, "f_tong  ", f_tong,  (180,120,255))
            draw_bar(frame, 190, "Still*",  still,    (0,255,255))


        # ===== 矢印（枠内=薄青/外=水色、配置は現状維持）=====
        step=24
        gh, gw = grayb.shape
        x0,y0,x1,y1 = x,y,x+w,y+h
        for yy in range(0, gh, step):
            for xx in range(0, gw, step):
                dx, dy = flow[yy, xx]
                inside = (x0<=xx<x1) and (y0<=yy<y1)
                col = COLOR_LIGHTBLUE if inside else COLOR_CYAN
                cv.arrowedLine(frame,(xx,yy),(int(xx+dx),int(yy+dy)),col,1,tipLength=0.3)

        cv.imshow("Shannon-only", frame)
        prev_gray = grayb
        if cv.waitKey(1) & 0xFF in (27, ord('q')):
            break

    # cleanup
    try: cap.release()
    except: pass
    try:
        if audio_stream is not None:
            audio_stream.stop(); audio_stream.close()
    except Exception: pass
    cv.destroyAllWindows()
    print("[INFO] Bye.")

if __name__ == "__main__":
    main()
