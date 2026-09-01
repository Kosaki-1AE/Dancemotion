# -*- coding: utf-8 -*-
# Video × Audio Shannon Fusion for Real-time Stillness
# しゃきしゃき仕様：
# - 映像：オプティカルフローの大きさヒスト → Shannonエントロピー（揺らぎ）
# - 音声：短時間スペクトル（パワー）ヒスト → Shannonエントロピー（揺らぎ）
# - ファジィ合成 + ヒステリシス + 滞留 + 入退場ゲートで“人間っぽい停止”判定
#
# 依存:
#   pip install opencv-python numpy sounddevice
#
# 実行:
#   python nori_rt_av_shannon.py   # ESCで終了
#
# メモ:
# - カメラ/マイクのデバイスIDは環境によって異なるのでCFGで調整してね
# - まずはShannonだけ。視線/表情の拡張は後から足しやすい骨格にしてます

import cv2 as cv
import numpy as np
import sounddevice as sd
import queue, threading, time
from collections import deque

# =========================
# CONFIG（環境に合わせて調整）
# =========================
CFG = dict(
    # Video
    cam_id=0,
    frame_size=(640, 360),
    flow_deadzone=0.03,         # 微振動カット（0.02〜0.06で調整）
    flow_bins=12,               # ヒストビン
    flow_clip_percentile=99.0,  # 外れ値カット
    flow_ema=0.25,              # HのEMA

    # Audio
    sr=16000,                   # サンプリング周波数
    blocksize=512,              # 1ブロックあたりサンプル数（約32ms@16kHz）
    audio_bins=32,              # スペクトルヒストのビン
    audio_ema=0.35,             # 音声HのEMA
    audio_device=None,          # 使う入力デバイスID（Noneでデフォルト）
    audio_gain=1.0,             # 入力スケール（クリップ対策）

    # Fusion & Decision
    w_video=0.6,                # H融合の重み（映像側）
    w_audio=0.4,                # H融合の重み（音声側）
    dwell_frames=45,            # Still判定に必要な継続フレーム（~1.5秒@30fps）
    enter_thresh=0.50,          # これを超えたらStill候補（0〜1, 高い=厳しめ）
    exit_thresh=0.60,           # これを下回ったら解除（0〜1）

    # Entry/Exit Gate（入る/掃ける瞬間の暴発抑制）
    gate_fg_delta=0.08,         # 前景面積の急変しきい
    gate_luma_delta=12.0,       # 輝度平均の急変しきい
    gate_frames=20,             # 不応フレーム（~0.7秒@30fps）

    # UI
    H_display_max=3.5,
)

# =========================
# Utility
# =========================
def shannon_entropy_from_hist(h, eps=1e-12):
    p = h / (np.sum(h) + eps)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p))  # nats

def ema(prev, x, a):
    return x if prev is None else (a*x + (1-a)*prev)

def fuzzy_low(value, low, high):
    """低いほど“良い”（=Still寄与）を [0,1] に線形マップ"""
    return float(np.clip((high - value) / (high - low + 1e-9), 0, 1))

def draw_meter(frame, value, min_v, max_v, x, y, w, h, label, color=(0,200,0)):
    v = (value - min_v) / (max_v - min_v + 1e-9)
    v = float(np.clip(v, 0, 1))
    cv.rectangle(frame, (x, y), (x+w, y+h), (60,60,60), 1)
    cv.rectangle(frame, (x, y), (x+int(w*v), y+h), color, -1)
    cv.putText(frame, f"{label}: {value:.3f}", (x, y-6),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

# =========================
# Audio Stream（非同期）
# =========================
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        # ドロップ/OVERRUN等はログだけ
        print(status)
    # 1chに揃える
    data = indata.copy()
    if data.ndim == 2 and data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    audio_q.put(data)

def start_audio_stream():
    stream = sd.InputStream(
        channels=1,
        samplerate=CFG["sr"],
        blocksize=CFG["blocksize"],
        device=CFG["audio_device"],
        callback=audio_callback,
        dtype='float32'
    )
    stream.start()
    return stream

def audio_shannon(block, bins=32):
    # block: shape (N,1)
    x = (block[:,0] * CFG["audio_gain"]).astype(np.float32)
    # 窓関数（ハン）
    win = np.hanning(len(x)).astype(np.float32)
    X = np.abs(np.fft.rfft(x * win))**2
    if not np.isfinite(X).any() or np.max(X) <= 0:
        return 0.0
    # 対数スケールにしない代わりに範囲を動的に
    h, _ = np.histogram(X, bins=bins, range=(0, np.percentile(X, 99)+1e-6))
    return shannon_entropy_from_hist(h)

# =========================
# Video準備
# =========================
cap = cv.VideoCapture(CFG["cam_id"])
assert cap.isOpened(), "カメラ開けない🥲"
prev_gray = None

# 入退場ゲート
bg = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
refractory = 0
prev_fg_ratio = None
prev_luma = None

# 状態
H_video_ema = None
H_audio_ema = None
still_counter = 0
state = "MOTION"

# Audio start
stream = start_audio_stream()

# Optical Flow params
flow_params = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# =========================
# Main Loop
# =========================
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv.resize(frame, CFG["frame_size"])
        gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray  = cv.GaussianBlur(gray, (3,3), 0)  # 露出ゆれ・砂ノイズ抑え

        # ---- 入退場ゲート（急変で不応）----
        mask_bg = bg.apply(gray)
        fg_ratio = float(np.count_nonzero(mask_bg==255)) / mask_bg.size
        luma = float(np.mean(gray))
        if prev_fg_ratio is not None and abs(fg_ratio - prev_fg_ratio) > CFG["gate_fg_delta"]:
            refractory = CFG["gate_frames"]
        if prev_luma is not None and abs(luma - prev_luma) > CFG["gate_luma_delta"]:
            refractory = CFG["gate_frames"]
        prev_fg_ratio, prev_luma = fg_ratio, luma

        # ---- 映像H: オプティカルフローの大きさヒスト ----
        if prev_gray is None:
            prev_gray = gray.copy()
            cv.putText(frame, "warming up...", (20, 36),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv.imshow("Nori AV Shannon", frame)
            if cv.waitKey(1) & 0xFF == 27: break
            continue

        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)
        fx, fy = flow[...,0], flow[...,1]
        mag = np.sqrt(fx*fx + fy*fy)
        # クリップ＆デッドゾーン
        mag_max = np.percentile(mag, CFG["flow_clip_percentile"]) + 1e-6
        mag = np.clip(mag, 0, mag_max)
        mag[mag < CFG["flow_deadzone"]] = 0.0

        h_flow, _ = np.histogram(mag, bins=CFG["flow_bins"], range=(0, mag_max))
        H_video = shannon_entropy_from_hist(h_flow)
        H_video_ema = ema(H_video_ema, H_video, CFG["flow_ema"])

        # ---- 音声H: スペクトルパワーのヒスト ----
        H_audio = None
        try:
            audio_block = audio_q.get_nowait()
            H_audio = audio_shannon(audio_block, bins=CFG["audio_bins"])
            H_audio_ema = ema(H_audio_ema, H_audio, CFG["audio_ema"])
        except queue.Empty:
            pass

        # ---- ファジィ化（低HほどStill寄与）----
        # しきいは経験的に：映像H ~ [0.3, 1.2] / 音声H ~ [2, 5]（データで調整してね）
        f_video = fuzzy_low(H_video_ema if H_video_ema is not None else 0.0, 0.35, 1.10)
        if H_audio_ema is not None:
            f_audio = fuzzy_low(H_audio_ema, 2.0, 5.0)
        else:
            f_audio = 0.5  # 音声未取得時は中立

        # ---- 融合（積でも和でもOK。ここは和→正規化）----
        fused = CFG["w_video"]*(1-f_video) + CFG["w_audio"]*(1-f_audio)  # “不安定度”の重み付き和
        fused = float(np.clip(fused, 0, 1))
        # Still度（1に近いほど静）
        still_score = 1.0 - fused

        # ---- 不応ゲート + 滞留 + ヒステリシス ----
        gate_text = ""
        if refractory > 0:
            refractory -= 1
            state = "MOTION"
            still_counter = 0
            gate_text = f"GATED {refractory}"
        else:
            if state == "MOTION":
                if still_score > (1.0 - CFG["enter_thresh"]):
                    still_counter += 1
                    if still_counter >= CFG["dwell_frames"]:
                        state = "STILL"
                        still_counter = 0
                else:
                    still_counter = 0
            else:
                if still_score < (1.0 - CFG["exit_thresh"]):
                    state = "MOTION"

        # ---- UI ----
        label = f"STATE: {state}"
        color = (0,255,0) if state=="STILL" else (0,0,255)
        cv.putText(frame, label, (20, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
        if gate_text:
            cv.putText(frame, gate_text, (20, 52), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv.LINE_AA)

        draw_meter(frame, H_video_ema if H_video_ema is not None else 0.0,
                   0.0, CFG["H_display_max"], 20, 80, 260, 16, "H_video (nats)")
        if H_audio_ema is not None:
            draw_meter(frame, H_audio_ema, 0.0, 6.0, 20, 105, 260, 16, "H_audio (nats)", (0,180,200))
        draw_meter(frame, still_score, 0.0, 1.0, 20, 130, 260, 16, "Stillness score", (0,255,255))

        # 矢印デバッグ（重いならOFF）
        step = 24
        h, w = gray.shape
        for yy in range(0, h, step):
            for xx in range(0, w, step):
                dx, dy = flow[yy, xx]
                cv.arrowedLine(frame,(xx,yy),(int(xx+dx),int(yy+dy)),(180,180,50),1,tipLength=0.3)

        cv.imshow("Nori AV Shannon", frame)
        prev_gray = gray

        if cv.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    stream.stop()
    cv.destroyAllWindows()
