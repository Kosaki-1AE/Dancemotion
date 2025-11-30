# -*- coding: utf-8 -*-
"""
nori_onepass_unified_intent.py
- A(意識) だけだと「意思」感が薄いので、軽量の Intent ループを追加
- 仕組み：
  1) preference P (7次元) … 最近“良かった”特徴を記憶（Stillが安定した瞬間に強化）
  2) intent energy I (0..1) … 一貫した選択が続くと上がる（迷いが少ない＝意思強）
  3) policy 選択 … 'FACE' or 'AV' を T でsoftmax選択（少しだけ探索ノイズ ε ）
  4) 反馈 … I と A でブレンド比/しきい値/温度をさらに増減 → 次の判断に影響

依存:
  - opencv-python, numpy
  - (任意) sounddevice
  - (任意) torch （USE_TORCH=True のときのみ）
"""

import cv2 as cv
import time, sys, queue, random
import numpy as np
from collections import deque
from log_manager import RingStats, CsvLogger, SessionMemory, Tick

# ===== 設定フラグ =====
HUD_POPUP  = True
USE_AUDIO  = True
USE_TORCH  = True  # 無意識ON/OFF
SHOW_TEXT  = False
LOG_ENABLE = True   # ログON/OFF
hist  = RingStats(maxlen=300)                          # 約10〜15秒ぶん
slog  = CsvLogger(enabled=LOG_ENABLE)                  # 1秒ごとCSV
smem  = SessionMemory().load()                         # 前回の傾向を取得
C_GAIN = 1.6   # 意識側の増幅係数
U_DAMP = 0.6   # 無意識側の減衰係数（0～1）

# ===== Awareness Engine の選択 =====
if USE_TORCH:
    from awareness_torch import TorchAwarenessEngine as AwarenessEngine
else:
    from awareness_fast import FastAwarenessEngine as AwarenessEngine

# ===== HUD =====
def make_hud_canvas(w=360, h=220, bg=(32,32,32)):
    hud = np.zeros((h, w, 3), dtype=np.uint8); hud[:] = bg; return hud

def hud_bar(hud, y, txt, val, col):
    x0, x1, bar_w, bar_h = 20, 340, 320, 12
    cv.rectangle(hud, (x0,y), (x1,y+bar_h), (60,60,60), 1)
    v = float(np.clip(val, 0, 1))
    cv.rectangle(hud, (x0,y), (x0+int(bar_w*v), y+bar_h), col, -1)
    cv.putText(hud, f"{txt}: {val:.2f}", (x0, y-2), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv.LINE_AA)

def show_hud(vals):
    hud = make_hud_canvas()
    hud_bar(hud,  20, "Motion", vals["f_motion"], (0,200,0))
    hud_bar(hud,  40, "EyeDir", vals["f_eyedir"], (0,180,200))
    hud_bar(hud,  60, "Blink ", vals["f_blink"],  (200,180,0))
    hud_bar(hud,  80, "Smile ", vals["f_mouth"],  (120,180,255))
    hud_bar(hud, 100, "Tongue", vals["f_tong"],   (180,120,255))
    hud_bar(hud, 120, "Score ", vals["still"],    (0,255,255))
    hud_bar(hud, 140, "A (aware)", vals["A"],     (255,180,120))
    hud_bar(hud, 160, "T (temp)", np.clip((vals["T"]-0.6)/(1.6-0.6),0,1), (180,255,120))
    hud_bar(hud, 180, "I (intent)", vals["I"],    (255,120,180))
    cv.putText(hud, f"POLICY: {vals['policy']}", (20, 205), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
    cv.imshow("Shannon-HUD", hud)

# ===== Audio (optional) =====
try:
    import sounddevice as sd
except Exception:
    sd = None
    USE_AUDIO = False
    print("[WARN] sounddevice が無いので音声は無効化します。", file=sys.stderr)

audio_q = queue.Queue()
audio_stream = None
AUDIO_OK = True

def audio_callback(indata, frames, time_info, status):
    if indata is None or len(indata)==0: return
    x = indata.copy()
    if x.ndim==2 and x.shape[1]>1: x = x.mean(axis=1, keepdims=True)
    audio_q.put(x)

def start_audio(sr=16000, blocksize=256, device=None):
    global audio_stream, AUDIO_OK
    if not USE_AUDIO or sd is None:
        AUDIO_OK = False; return
    try:
        audio_stream = sd.InputStream(channels=1, samplerate=sr, blocksize=blocksize,
                                      device=device, dtype='float32', callback=audio_callback)
        audio_stream.start(); AUDIO_OK = True
    except Exception as e:
        AUDIO_OK = False
        print(f"[WARN] マイク開始に失敗: {e} → 音声は無効化します。", file=sys.stderr)

# ===== Utils =====
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
    cv.rectangle(frame,(20,y),(300,y+12),(60,60,60),1)
    cv.rectangle(frame,(20,y),(20+int(280*np.clip(val,0,1)),y+12),col,-1)
    cv.putText(frame, f"{txt}: {val:.2f}", (20, y-2), cv.FONT_HERSHEY_SIMPLEX, 0.45,(255,255,255),1, cv.LINE_AA)

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

# ===== Config =====
W,H = 480,270
CFG = dict(
    cam_id=0, frame_size=(W,H),
    # flow
    flow_deadzone=0.03, flow_bins=12, flow_clip_percentile=99.0, flow_ema=0.25,
    # audio
    sr=16000, blocksize=256, audio_bins=32, audio_ema=0.35, audio_gain=1.0, audio_device=None,
    # gate
    gate_fg_delta=0.08, gate_luma_delta=12.0, gate_frames=20,
    # hysteresis
    enter=0.50, exit=0.60, dwell=45,
)

THR = dict(
    motion_low=0.35, motion_high=1.10,
    eyed_low=0.35,   eyed_high=1.10,
    blink_spike=0.20,
    mouth_low=0.25,  mouth_high=0.90,
    tongue_low=0.20, tongue_high=0.85
)

FLOW_PARAMS = dict(pyr_scale=0.5, levels=2, winsize=11, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)

# ===== Face helpers =====
def skin_mask_ycrcb(bgr):
    ycrcb = cv.cvtColor(bgr, cv.COLOR_BGR2YCrCb)
    mask = cv.inRange(ycrcb, (0,135,85), (255,180,135))
    mask = cv.medianBlur(mask,5)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3,3),np.uint8),1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE,np.ones((5,5),np.uint8),2)
    return mask

def largest_bbox(mask, min_area=6000):
    cnts,_ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt = max(cnts, key=cv.contourArea)
    if cv.contourArea(cnt) < min_area: return None
    return cv.boundingRect(cnt)

# ===== Intent engine =====
class IntentEngine:
    def __init__(self, feat_dim=7):
        self.P = np.zeros((feat_dim,), dtype=np.float32)  # preference
        self.I = 0.2  # intent energy
        self.last_policy = 'FACE'
        self.still_streak = 0

    def reinforce(self, feat, policy, good=True):
        # Stillが続いたときに“良かった特徴”を強化
        lr = 0.05 if good else -0.03
        delta = lr * feat / (np.linalg.norm(feat)+1e-6)
        if policy == 'FACE':
            self.P[:5] += delta[:5]  # 顔系を多めに学習
        else:
            self.P[5:] += delta[5:]  # AV系を多めに学習
        self.P = np.clip(self.P, -1.5, 1.5)

        # 意図エネルギー更新
        self.I = float(np.clip(self.I + (0.02 if good else -0.03), 0.0, 1.0))

    def choose_policy(self, score_face, score_av, A, T):
        # A高い=集中→FACE寄り事前バイアス / A低い=俯瞰→AV寄り
        bias = 0.3*(A - (1.0 - A))  # -0.3..0.3
        s_face = score_face + bias
        s_av   = score_av   - bias

        # 温度 T でsoftmax
        probs = np.exp(np.array([s_face, s_av]) / max(0.3, T))
        probs = probs / probs.sum()
        # 少し探索：ε は (1-I)*(1-A) で決める（意思も意識も低いときほど探索）
        eps = 0.15 * (1.0 - self.I) * (1.0 - A)
        if random.random() < eps:
            probs = np.array([0.5, 0.5])
        pol = 'FACE' if (random.random() < probs[0]) else 'AV'
        self.last_policy = pol
        return pol, probs[0].item()

# ===== Main =====
def main():
    # smem から初期値ブートストラップ（好みで重み）
    A_bias = float(smem.get("boot_A", 0.5)) * 0.2          # 初期Aに混ぜる率
    I_init = float(smem.get("boot_I", 0.3))
    ENTER_SCALE = float(smem.get("enter_scale", 1.0))
    EXIT_SCALE  = float(smem.get("exit_scale", 1.0))
    EPS_SCALE   = float(smem.get("eps_scale", 1.0))
    DEAD_SCALE  = float(smem.get("deadzone_scale", 1.0))

    cap = cv.VideoCapture(0)
    assert cap.isOpened(), "camera open failed"
    start_audio(CFG["sr"], CFG["blocksize"], CFG["audio_device"])

    aware = AwarenessEngine(feat_dim=7, **({} if not USE_TORCH else {"seq_len": 32}))
    intent = IntentEngine(feat_dim=7) 

    # --- ポリシー選択（= 意識の決定） ---
    policy, p_face = intent.choose_policy(score_face, score_av, A, T)
    p_av = 1.0 - p_face

    # 意識/無意識のラベル
    conscious   = policy                               # "FACE" or "AV"
    unconscious = "AV" if policy == "FACE" else "FACE"

    # （任意）無意識がAVのときは音声の計算頻度を半分に落とす
    do_audio = True
    if (unconscious == "AV") and (frame_idx & 1):      # 奇数フレームはスキップ
        do_audio = False

    # --- 意識/無意識ゲイン ---
    # A(意識)とI(意思)から“決断の強さ”を 0..1 にまとめる
    alpha = 0.6 * A + 0.4 * float(intent.I)

    # 意識側をブースト、無意識側を減衰
    G_CONSC     = 1.2      # 意識側の増幅係数
    G_UNCON_DMP = 0.8      # 無意識側の減衰係数
    MIN_UNCON   = 0.15     # 無意識側の下限（完全ゼロは避ける）

    if policy == "FACE":
        g_face = 1.0 + G_CONSC * alpha
        g_av   = max(MIN_UNCON, 1.0 - G_UNCON_DMP * alpha)
    else:  # policy == "AV"
        g_av   = 1.0 + G_CONSC * alpha
        g_face = max(MIN_UNCON, 1.0 - G_UNCON_DMP * alpha)

    # --- ブレンド重みを再正規化して適用 ---
    # まず信頼度だけで計算した基準重み（wf0, wa0）がある想定
    wf = max(1e-6, wf0) * g_face
    wa = max(1e-6, wa0) * g_av
    s  = wf + wa
    wf, wa = wf / s, wa / s

    still = float(np.clip(wf * still_face + wa * still_av, 0.0, 1.0))


    intent.I = I_init                                      # 意思の初期値

    bg = cv.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
    refractory = 0; prev_fg=None; prev_luma=None
    prev_gray=None

    # frame_idx = 0                                          #ここにこれ入ってると記憶喪失状態になるっぽい
    H_motion_ema=None; H_eye_dir_ema=None; H_blink_prev=None
    blink_hist = deque(maxlen=15)
    H_video_ema=None; H_audio_ema=None

    state="MOTION"; still_counter=0

    fps_target = 20.0; fps_alpha = 0.1
    fps_ema = fps_target; last_time = time.time()

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
        box = largest_bbox(skin, min_area=6000)
        if box is None:
            cv.putText(frame,"No face ROI", (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
            cv.imshow("Shannon-intent", frame)
            if cv.waitKey(1)==27: break
            prev_gray = grayb
            continue

        x,y,w,h = box
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,200,255),2)
        y1 = y + int(h*0.33); y2 = y + int(h*0.66)
        eyeROI  = grayb[y:y1, x:x+w]
        mouthROI= grayb[y2:y+h, x:x+w]
        if eyeROI.size==0 or mouthROI.size==0:
            prev_gray = grayb
            continue

        # ---- Flow ----
        if prev_gray is None:
            prev_gray = grayb
            cv.putText(frame,"warming up...", (20,30), cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2)
            cv.imshow("Shannon-intent", frame); 
            if cv.waitKey(1)==27: break
            continue

        flow = cv.calcOpticalFlowFarneback(prev_gray, grayb, None, **FLOW_PARAMS)
        fx, fy = flow[...,0], flow[...,1]
        mag = np.sqrt(fx*fx + fy*fy)
        mag_max = np.percentile(mag, 99.0) + 1e-6
        dead = CFG["flow_deadzone"] * (0.8 + 0.4*intent.I)  # 意思が強いと微小ノイズを無視
        mag = np.clip(mag, 0, mag_max); mag[mag < dead] = 0.0

        # ---- Face features ----
        h_flow,_ = np.histogram(mag.ravel(), bins=CFG["flow_bins"], range=(0,mag_max))
        Hm = shannon_entropy_from_hist(h_flow)
        H_motion_ema = ema(H_motion_ema, Hm, CFG["flow_ema"])

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

        Hy_eye,_ = np.histogram(eyeROI.ravel(), bins=32, range=(0,255))
        Hy_eye = shannon_entropy_from_hist(Hy_eye)
        if H_blink_prev is None: H_blink_prev = Hy_eye
        dH = max(0.0, Hy_eye - H_blink_prev)
        H_blink_prev = ema(H_blink_prev, Hy_eye, 0.4)
        blink_hist.append(dH)
        blink_spike = float(np.mean(blink_hist))

        sobel_y = cv.Sobel(mouthROI, cv.CV_32F, 0,1, ksize=3)
        vabs = np.abs(sobel_y).ravel()
        H_mouth,_ = np.histogram(vabs, bins=12, range=(0, np.percentile(vabs,99)+1e-6))
        H_mouth = shannon_entropy_from_hist(H_mouth)
        hsv = cv.cvtColor(frame[y2:y+h, x:x+w], cv.COLOR_BGR2HSV)
        Hc,Sc,Vc = cv.split(hsv)
        red = ((Hc < 15) | (Hc > 165)) & (Sc > 40) & (Vc > 40)
        tongue_score = float(np.clip(np.mean(red.astype(np.float32))*3.0 - 0.2, 0, 1))
        H_tong_proxy = 1.0 - tongue_score

        f_motion = fuzzy_low(H_motion_ema, 0.35, 1.10)
        f_eyedir= fuzzy_low(H_eye_dir_ema, 0.35, 1.10)
        f_blink = float(1.0 - np.clip(blink_spike/0.20, 0, 1))
        f_mouth = fuzzy_low(H_mouth, 0.25, 0.90)
        f_tong  = fuzzy_low(H_tong_proxy, 0.20, 0.85)

        still_face = f_motion * f_eyedir * f_blink
        still_face *= 0.7 + 0.3*(f_mouth*0.7 + f_tong*0.3)
        still_face = float(np.clip(still_face, 0, 1))

        # 信頼度
        c_face = float(np.clip((w*h)/(W*H*0.20), 0.0, 1.0))

        # ---- AV ----
        H_video_ema = ema(H_video_ema, Hm, 0.25)
        H_audio = None
        if USE_AUDIO and AUDIO_OK:
            try:
                blk = audio_q.get_nowait()
                XH = audio_shannon(blk, bins=32, gain=1.0)
                H_audio_ema = ema(H_audio_ema, XH, 0.35)
            except queue.Empty:
                pass

        f_video = fuzzy_low(H_video_ema if H_video_ema is not None else 0.0, 0.35, 1.10)
        f_audio = fuzzy_low(H_audio_ema, 2.0, 5.0) if H_audio_ema is not None else 0.5
        fused_av = float(np.clip(0.6*(1-f_video) + 0.4*(1-f_audio), 0, 1))
        still_av = 1.0 - fused_av
        c_av = 1.0 if H_audio_ema is not None else 0.6

        # 一旦の自動ブレンド（信頼度だけ）
        s = c_face + c_av
        wf0, wa0 = (1.0, 0.0) if s<=1e-6 else (c_face/s, c_av/s)
        still0 = float(np.clip(wf0*still_face + wa0*still_av, 0, 1))

        # === Awareness ===
        feat = np.array([f_motion, f_eyedir, f_blink, f_mouth, f_tong, f_video, (f_audio if H_audio_ema is not None else 0.5)], dtype=np.float32)
        A, T_attn = aware.step(feat)
        # A -> T 直写
        T_min, T_max = 0.7, 1.6
        T_from_A = T_min + (1.0 - A) * (T_max - T_min)
        T = 0.5*T_attn + 0.5*T_from_A if USE_TORCH else T_from_A

        # FPS セーフティ（Torch時のみ）
        now = time.time(); dt = now - last_time; last_time = now
        fps = 1.0 / max(dt, 1e-6)
        fps_ema = fps_alpha * fps + (1 - fps_alpha) * fps_ema
        if USE_TORCH and fps_ema < fps_target:
            T *= max(0.5, fps_ema / fps_target)
        
        # ---- ログへpush（短期） ----
        hist.push(dict(A=A, T=T, I=float(intent.I), pol=policy, state=state,
                    still=still, fps=fps_ema,
                    f_motion=f_motion, f_eyedir=f_eyedir, f_blink=f_blink,
                    f_mouth=f_mouth, f_tong=f_tong))

        # ---- 1秒ごとCSV追記 ----
        slog.maybe_write(Tick(
            t=time.time(), A=A, T=T, I=float(intent.I), pol=policy, state=state,
            still=still, fps=fps_ema,
            f_motion=f_motion, f_eyedir=f_eyedir, f_blink=f_blink,
            f_mouth=f_mouth, f_tong=f_tong
        ))

        # === Intent policy 選択 ===
        # preferenceに基づく追加スコア（自分の“好み”）
        pref_gain = 0.2
        score_face = still_face + pref_gain * float(np.dot(intent.P[:5], feat[:5]))
        score_av   = still_av   + pref_gain * float(np.dot(intent.P[5:],  feat[5:]))

        policy, p_face = intent.choose_policy(score_face, score_av, A, T)

        # policyでブレンドをバイアス
        if policy == 'FACE':
            k = 1.0 + 1.5*(A*0.7 + intent.I*0.3)
            w_face = np.exp(k * (c_face + 1e-6)); w_av = np.exp((1.0)*(c_av + 1e-6))
        else:
            k = 1.0 + 1.5*((1.0-A)*0.7 + (1.0-intent.I)*0.3)
            w_face = np.exp((1.0)*(c_face + 1e-6)); w_av = np.exp(k * (c_av + 1e-6))

        wf = w_face / (w_face + w_av); wa = 1.0 - wf
        still = float(np.clip(wf*still_face + wa*still_av, 0, 1))

        # === ヒステリシス（AとIで可変） ===
        enter = np.clip(CFG["enter"] * (0.75 + 0.5*A + 0.25*intent.I), 0.1, 0.95)
        exit_  = np.clip(CFG["exit"]  * (1.25 - 0.5*A - 0.25*intent.I), 0.05, 0.9)
        enter *= ENTER_SCALE
        exit_  *= EXIT_SCALE
        
        # IntentEngine.choose_policy 内の一部（擬似コード）
        eps = 0.15 * (1.0 - self.I) * (1.0 - A)
        eps *= EPS_SCALE   # ←再参照補正
        dead = CFG["flow_deadzone"] * (0.8 + 0.4*intent.I) * DEAD_SCALE
        mag[mag < dead] = 0.0
        if frame_idx < 30:   # 最初の約1秒だけ
            A = float( (1.0-0.2)*A + 0.2*A_bias )


        gate_text = ""
        good_event = False
        if refractory>0:
            refractory -= 1
            state="MOTION"; still_counter=0; gate_text=f"GATED {refractory}"
        else:
            if state=="MOTION":
                if still > (1.0 - enter):
                    still_counter += 1
                    if still_counter >= CFG["dwell"]:
                        state="STILL"; still_counter=0; good_event=True
                else:
                    still_counter=0
            else:
                if still < (1.0 - exit_):
                    state="MOTION"
                else:
                    # STILL 継続中も少し強化
                    good_event=True

        # === 強化学習もどき ===
        if good_event:
            intent.reinforce(feat, policy, good=True)
        else:
            # バタついてるときは弱化
            if fps_ema < fps_target*0.8:
                intent.reinforce(feat, policy, good=False)

        # ===== UI =====
        if SHOW_TEXT:
            cv.putText(frame, f"STATE:{state}  POL:{policy}", (18,26),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,255),2, cv.LINE_AA)
            cv.putText(frame, f"A:{A:.2f}  T:{T:.2f}  FPS:{fps_ema:.1f}", (18,50),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,0),2, cv.LINE_AA)
            if gate_text:
                cv.putText(frame, gate_text, (18,72),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2, cv.LINE_AA)

        if HUD_POPUP:
            show_hud(dict(
                f_motion=f_motion, f_eyedir=f_eyedir, f_blink=f_blink, f_mouth=f_mouth, f_tong=f_tong,
                still=still, A=A, T=T, I=float(intent.I), policy=policy
            ))
        else:
            draw_bar(frame,  90, "f_motion", f_motion, (0,200,0))
            draw_bar(frame, 110, "f_eyedir", f_eyedir,(0,180,200))
            draw_bar(frame, 130, "f_blink ", f_blink, (200,180,0))
            draw_bar(frame, 150, "f_mouth ", f_mouth, (120,180,255))
            draw_bar(frame, 170, "f_tong  ", f_tong,  (180,120,255))
            draw_bar(frame, 190, "Still*",  still,    (0,255,255))
        # 矢印
        step=32
        gh, gw = grayb.shape
        for yy in range(0, gh, step):
            for xx in range(0, gw, step):
                dx, dy = flow[yy, xx]
                col = (255,210,120) if (x<=xx<x+w and y<=yy<y+h) else (255,255,0)
                cv.arrowedLine(frame,(xx,yy),(int(xx+dx),int(yy+dy)),col,1,tipLength=0.3)

        cv.imshow("Shannon-intent", frame)
        prev_gray = grayb
        if cv.waitKey(1) & 0xFF in (27, ord('q')):
            break
    
    # === 終了処理 ===
    try:
        sm = SessionMemory()           # 新規で開いてOK
        sm.load()
        sm.update_from_hist(hist)      # 直近ヒストリーから補正値を学習
        sm.save()
    except Exception as e:
        print("[WARN] session save failed:", e)

    slog.close()

    try: cap.release()
    except: pass
    try:
        if audio_stream is not None: audio_stream.stop(); audio_stream.close()
    except: pass
    frame_idx += 1
    cv.destroyAllWindows()
    print("[INFO] Bye.")

if __name__ == "__main__":
    main()
