# -*- coding: utf-8 -*-
# Video × Audio Shannon Fusion + Reward/Prediction + CLI
# しゃきしゃき仕様：意思っぽさ付与・堅牢化版

import queue
import sys
import time
from collections import deque

import cv2 as cv
import numpy as np

# =============== 可変: 音声を使うか =================
USE_AUDIO = True
try:
    import sounddevice as sd
except Exception:
    USE_AUDIO = False
    sd = None
    print("[WARN] sounddevice が読み込めなかったので音声は無効化します。")

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
    blocksize=512,              # 1ブロック ≈32ms@16k
    audio_bins=32,              # スペクトルヒストのビン
    audio_ema=0.35,             # 音声HのEMA
    audio_device=None,          # None=デフォルト
    audio_gain=1.0,             # 入力スケール（クリップ対策）

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
    s = float(np.sum(h))
    if s <= eps:
        return 0.0
    p = h / (s + eps)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))  # nats

def ema(prev, x, a):
    return float(x) if prev is None else float(a*x + (1-a)*prev)

def fuzzy_low(value, low, high):
    """低いほどStillに寄与 → [0,1]"""
    return float(np.clip((high - value) / (high - low + 1e-9), 0, 1))

def draw_meter(frame, value, min_v, max_v, x, y, w, h, label, color=(0,200,0)):
    v = (value - min_v) / (max_v - min_v + 1e-9)
    v = float(np.clip(v, 0, 1))
    cv.rectangle(frame, (x, y), (x+w, y+h), (60,60,60), 1)
    cv.rectangle(frame, (x, y), (x+int(w*v), y+h), color, -1)
    cv.putText(frame, f"{label}: {value:.3f}", (x, y-6),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)

# =========================
# Predictor & RewardSystem
# =========================
class Predictor:
    """
    極小コストのオンライン予測器（RLS, AR(1)+b）。
    update_and_predict(name, x) → (x_pred_next, |err|)
    """
    def __init__(self, lam=0.98, eps=1e-6):
        self.lam, self.eps = lam, eps
        self.params = {}
    def _init(self, name, x0):
        self.params[name] = {"a":0.9,"b":0.0,"P":np.eye(2)*10.0,"last":float(x0)}
    def update_and_predict(self, name, x):
        x = float(x)
        if name not in self.params:
            self._init(name, x); return x, 0.0
        p=self.params[name]; x_prev=p["last"]
        phi=np.array([x_prev,1.0], dtype=np.float64)
        theta=np.array([p["a"],p["b"]], dtype=np.float64)
        P=p["P"]/self.lam
        denom=float(phi @ P @ phi) + self.eps
        K=(P @ phi)/denom
        err = x - float(phi @ theta)
        theta = theta + K*err
        P = (np.eye(2) - np.outer(K, phi)) @ P
        p["a"], p["b"], p["P"], p["last"] = float(theta[0]), float(theta[1]), P, x
        x_pred = float(theta[0]*x + theta[1])
        return x_pred, abs(float(err))

class RewardSystem:
    """
    上位調整回路：
      入力: H_video, H_audio, still_score, 予測誤差 e_*
      出力: enter/exit/dwell と w_video/w_audio を自動調整
    """
    def __init__(self, enter=0.50, exit=0.60, dwell=45, wv=0.6, wa=0.4):
        self.enter, self.exit, self.dwell = float(enter), float(exit), int(dwell)
        self.wv, self.wa = float(wv), float(wa)
        self.alpha = 0.2
        self.hist = deque(maxlen=60)
    def _smooth(self, attr, val, clamp):
        cur = getattr(self, attr)
        new = (1-self.alpha)*cur + self.alpha*val
        lo, hi = clamp
        setattr(self, attr, float(np.clip(new, lo, hi)) if attr != "dwell" else int(np.clip(new, lo, hi)))
    def step(self, Hv, Ha, S, ev, ea, es):
        # 正規化
        Hvn = np.clip((Hv-0.35)/(1.10-0.35+1e-9), 0, 1)
        Han = np.clip((Ha-2.0)/(5.0-2.0+1e-9),   0, 1) if Ha is not None else 0.5
        evn, ean, esn = np.clip(ev/0.35,0,1), np.clip((ea or 0.0)/1.0,0,1), np.clip(es/0.20,0,1)
        good = (1-Hvn)*0.35 + (1-Han)*0.15 + (1-evn)*0.20 + (1-ean)*0.10 + (1-esn)*0.20
        self.hist.append(float(good))
        # 驚きが大 → 慎重に
        shock = 0.4*evn + 0.3*ean + 0.3*esn
        self._smooth("enter", self.enter + 0.15*shock, (0.35, 0.75))
        self._smooth("exit",  self.exit  + 0.10*shock, (0.45, 0.85))
        self._smooth("dwell", self.dwell + 30*shock,   (15, 120))
        # 落ち着き → 攻め
        calm = (1-Hvn)*0.6 + (1-evn)*0.4
        self._smooth("enter", self.enter - 0.10*calm,  (0.35, 0.75))
        self._smooth("dwell", self.dwell - 20*calm,    (15, 120))
        # 音がうるさい → 音の重み下げ
        target_wa = np.clip(0.5 - 0.3*Han, 0.1, 0.6)
        target_wv = 1.0 - target_wa
        self._smooth("wa", target_wa, (0.1, 0.8))
        self._smooth("wv", target_wv, (0.2, 0.9))
        return dict(enter=self.enter, exit=self.exit, dwell=int(self.dwell),
                    wv=self.wv, wa=self.wa,
                    reward=float(good), reward_avg=float(np.mean(self.hist)))

# =========================
# Audio Stream（非同期・安全起動）
# =========================
audio_q = queue.Queue()
audio_stream = None
AUDIO_OK = False

def audio_callback(indata, frames, time_info, status):
    if status:
        # オーバーラン等はログだけ
        sys.stderr.write(f"[AUDIO] {status}\n")
    if indata is None or len(indata) == 0:
        return
    data = indata.copy()
    if data.ndim == 2 and data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    audio_q.put(data)

def start_audio():
    global audio_stream, AUDIO_OK
    if not USE_AUDIO:
        AUDIO_OK = False
        return
    try:
        audio_stream = sd.InputStream(
            channels=1,
            samplerate=CFG["sr"],
            blocksize=CFG["blocksize"],
            device=CFG["audio_device"],
            callback=audio_callback,
            dtype='float32'
        )
        audio_stream.start()
        AUDIO_OK = True
    except Exception as e:
        AUDIO_OK = False
        print(f"[WARN] マイク開始に失敗: {e}  → 音声は無効化します。")

def audio_shannon(block, bins=32):
    if block is None or block.size == 0:
        return 0.0
    x = (block[:,0] * CFG["audio_gain"]).astype(np.float32)
    n = len(x)
    if n <= 0:
        return 0.0
    win = np.hanning(n).astype(np.float32)
    X = np.abs(np.fft.rfft(x * win))**2
    xmax = float(np.max(X)) if X.size > 0 else 0.0
    if not np.isfinite(xmax) or xmax <= 0:
        return 0.0
    hi = np.percentile(X, 99) + 1e-6
    h, _ = np.histogram(X, bins=bins, range=(0, hi))
    return shannon_entropy_from_hist(h)

# =========================
# Video 準備
# =========================
cap = cv.VideoCapture(CFG["cam_id"])
assert cap.isOpened(), "カメラが開けませんでした。cam_id を確認してね。"
prev_gray = None

# 入退場ゲート
bg = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
refractory = 0
prev_fg_ratio = None
prev_luma = None

# 状態変数（Noneで初期化 → 逐次EMA）
H_video_ema = None
H_audio_ema = None
state = "MOTION"
still_counter = 0

# 予測・報酬
pred = Predictor()
rew  = RewardSystem()  # enter/exit/dwell/wV/wA の現在値を保持
ctrl = dict(enter=rew.enter, exit=rew.exit, dwell=rew.dwell, wv=rew.wv, wa=rew.wa,
            reward=0.0, reward_avg=0.0)

# UI/CLI
log_enable = False

# 音声開始
start_audio()

# Optical Flow params
flow_params = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERR] フレーム取得に失敗。")
            break
        frame = cv.resize(frame, CFG["frame_size"])
        gray  = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray  = cv.GaussianBlur(gray, (3,3), 0)

        # ---- 入退場ゲート（急変で不応）----
        mask_bg = bg.apply(gray)
        fg_ratio = float(np.count_nonzero(mask_bg==255)) / mask_bg.size
        luma = float(np.mean(gray))
        if prev_fg_ratio is not None and abs(fg_ratio - prev_fg_ratio) > CFG["gate_fg_delta"]:
            refractory = CFG["gate_frames"]
        if prev_luma is not None and abs(luma - prev_luma) > CFG["gate_luma_delta"]:
            refractory = CFG["gate_frames"]
        prev_fg_ratio, prev_luma = fg_ratio, luma

        # ---- 映像H: オプティカルフロー ----
        if prev_gray is None:
            prev_gray = gray.copy()
            cv.putText(frame, "warming up...", (20, 36),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv.imshow("Nori AV Shannon+", frame)
            if cv.waitKey(1) & 0xFF in (27, ord('q')): break
            continue

        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, **flow_params)
        fx, fy = flow[...,0], flow[...,1]
        mag = np.sqrt(fx*fx + fy*fy)
        mag_max = np.percentile(mag, CFG["flow_clip_percentile"]) + 1e-6
        mag = np.clip(mag, 0, mag_max)
        mag[mag < CFG["flow_deadzone"]] = 0.0
        h_flow, _ = np.histogram(mag, bins=CFG["flow_bins"], range=(0, mag_max))
        H_video = shannon_entropy_from_hist(h_flow)
        H_video_ema = ema(H_video_ema, H_video, CFG["flow_ema"])

        # ---- 音声H ----
        H_audio = None
        if AUDIO_OK:
            try:
                audio_block = audio_q.get_nowait()
                H_audio = audio_shannon(audio_block, bins=CFG["audio_bins"])
                H_audio_ema = ema(H_audio_ema, H_audio, CFG["audio_ema"])
            except queue.Empty:
                pass
        # 音声未取得時は None のまま

        # ---- ファジィ化（低HほどStill寄与）----
        f_video = fuzzy_low(H_video_ema if H_video_ema is not None else 0.0, 0.35, 1.10)
        if H_audio_ema is not None:
            f_audio = fuzzy_low(H_audio_ema, 2.0, 5.0)
        else:
            f_audio = 0.5  # 音声なし=中立

        # ---- 融合（現在の報酬系重みで）----
        wv, wa = rew.wv, rew.wa
        fused = float(np.clip(wv*(1-f_video) + wa*(1-f_audio), 0, 1))
        still_score = 1.0 - fused

        # ---- 予測・報酬更新 ----
        _, e_v = pred.update_and_predict("H_video", H_video_ema if H_video_ema is not None else 0.0)
        _, e_a = pred.update_and_predict("H_audio", H_audio_ema if H_audio_ema is not None else 0.0)
        _, e_s = pred.update_and_predict("Still",  still_score)

        ctrl = rew.step(Hv=H_video_ema if H_video_ema is not None else 0.0,
                        Ha=H_audio_ema if H_audio_ema is not None else None,
                        S=still_score, ev=e_v, ea=e_a, es=e_s)

        # ---- 不応ゲート + 滞留 + ヒステリシス ----
        gate_text = ""
        if refractory > 0:
            refractory -= 1
            state = "MOTION"
            still_counter = 0
            gate_text = f"GATED {refractory}"
        else:
            if state == "MOTION":
                if still_score > (1.0 - ctrl["enter"]):
                    still_counter += 1
                    if still_counter >= ctrl["dwell"]:
                        state = "STILL"; still_counter = 0
                else:
                    still_counter = 0
            else:
                if still_score < (1.0 - ctrl["exit"]):
                    state = "MOTION"

        # ---- UI ----
        color = (0,255,0) if state=="STILL" else (0,0,255)
        cv.putText(frame, f"STATE: {state}", (20, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
        if gate_text:
            cv.putText(frame, gate_text, (20, 52), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv.LINE_AA)

        draw_meter(frame, H_video_ema if H_video_ema is not None else 0.0,
                   0.0, CFG["H_display_max"], 20, 80, 260, 16, "H_video (nats)")
        if H_audio_ema is not None:
            draw_meter(frame, H_audio_ema, 0.0, 6.0, 20, 105, 260, 16, "H_audio (nats)", (0,180,200))
        draw_meter(frame, still_score, 0.0, 1.0, 20, 130, 260, 16, "Stillness score", (0,255,255))

        cv.putText(frame, f"enter:{ctrl['enter']:.2f} exit:{ctrl['exit']:.2f} dwell:{ctrl['dwell']}",
                   (20,160), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,255), 1)
        cv.putText(frame, f"wV:{ctrl['wv']:.2f} wA:{ctrl['wa']:.2f} reward:{ctrl['reward']:.3f} avg:{ctrl['reward_avg']:.3f}",
                   (20,180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,255), 1)
        cv.putText(frame, "Keys: [s]=status [r]=reset [l]=log [q]/ESC=quit",
                   (20, 340), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # 矢印デバッグ（重いならコメントアウト）
        step = 24
        gh, gw = gray.shape
        for yy in range(0, gh, step):
            for xx in range(0, gw, step):
                dx, dy = flow[yy, xx]
                cv.arrowedLine(frame,(xx,yy),(int(xx+dx),int(yy+dy)),(180,180,50),1,tipLength=0.3)

        # CLIキー処理
        key = cv.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('s'):
            print(f"STATE={state} enter={ctrl['enter']:.2f} exit={ctrl['exit']:.2f} "
                  f"dwell={ctrl['dwell']} wV={ctrl['wv']:.2f} wA={ctrl['wa']:.2f} "
                  f"reward={ctrl['reward']:.3f} avg={ctrl['reward_avg']:.3f} "
                  f"Hv={H_video_ema:.3f} Ha={(H_audio_ema if H_audio_ema is not None else 0.0):.3f} "
                  f"still={still_score:.3f}")
        elif key == ord('r'):
            rew = RewardSystem()  # 初期化
            print("RewardSystem reset.")
        elif key == ord('l'):
            log_enable = not log_enable
            print(f"Logging {'ON' if log_enable else 'OFF'}")

        if log_enable:
            sys.stdout.write(
                f"\rSTATE={state} still={still_score:.2f} Hv={H_video_ema:.2f} "
                f"Ha={(H_audio_ema if H_audio_ema is not None else 0.0):.2f} "
                f"enter={ctrl['enter']:.2f} dwell={ctrl['dwell']:3d}"
            )
            sys.stdout.flush()

        cv.imshow("Nori AV Shannon+", frame)
        prev_gray = gray

finally:
    try:
        cap.release()
    except Exception:
        pass
    try:
        if audio_stream is not None:
            audio_stream.stop(); audio_stream.close()
    except Exception:
        pass
    cv.destroyAllWindows()
    print("\n[INFO] Bye.")
