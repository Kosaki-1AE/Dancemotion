# -*- coding: utf-8 -*-
# live_pose_helix.py  — HelixCoherence + MediaPipe + 音楽アシスト + 録画 + モード切替

import cv2 as cv
import numpy as np
import time, math, os
from collections import deque
from datetime import datetime

# ====== ML / Torch ======
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== MediaPipe ======
import mediapipe as mp

# ====== 音楽（任意） ======
AUDIO_OK = True
try:
    import threading
    import sounddevice as sd
    import aubio
except Exception:
    AUDIO_OK = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== モード定義 ====
MODE_MUSIC = 1
MODE_HUMAN = 2
MODE_CODANCE = 3
MODE_AUTO = 4
MODE_NAME = {1: "MUSIC", 2: "HUMAN", 3: "CO-DANCE", 4: "AUTO"}

# ========= HelixCoherenceNet （簡略版）=========
class HelixPE(nn.Module):
    def __init__(self, d_out=32, omega=2.0, alpha=0.01):
        super().__init__()
        self.omega=omega; self.alpha=alpha
        self.proj = nn.Linear(5, d_out)
    def forward(self, T, beat=4.0, bpm=None, device=DEVICE):
        t = torch.arange(T, dtype=torch.float32, device=device)
        base = torch.stack([
            torch.cos(self.omega*t),
            torch.sin(self.omega*t),
            self.alpha*t,
            t / max(1, T-1),
            (t % beat)/beat
        ], dim=-1)
        return self.proj(base)

class ChiralCrossAttn(nn.Module):
    def __init__(self, d, nhead=4, drop=0.1):
        super().__init__()
        self.r2e = nn.MultiheadAttention(d, nhead, batch_first=True, dropout=drop)
        self.e2r = nn.MultiheadAttention(d, nhead, batch_first=True, dropout=drop)
        self.Wcw = nn.Linear(d,d); self.Wccw = nn.Linear(d,d)
        self.lmbd = nn.Parameter(torch.tensor(0.5))
        self.nR = nn.LayerNorm(d); self.nE = nn.LayerNorm(d)
    def forward(self, HR, HE, s_sign):
        W = self.Wcw if s_sign.mean().item()>=0 else self.Wccw
        HEc, HRc = W(HE), W(HR)
        Radd,_ = self.e2r(HR, HEc, HEc); Eadd,_ = self.r2e(HE, HRc, HRc)
        return self.nR(HR+self.lmbd*Radd), self.nE(HE+self.lmbd*Eadd)

class DualBranchEncoder(nn.Module):
    def __init__(self, d_inR, d_inE, d_model=256, nhead=4):
        super().__init__()
        self.r_in = nn.Linear(d_inR, d_model)
        self.e_in = nn.Linear(d_inE, d_model)
        self.r_enc = nn.GRU(d_model, d_model//2, num_layers=2, batch_first=True, bidirectional=True)
        self.e_enc = nn.GRU(d_model, d_model//2, num_layers=2, batch_first=True, bidirectional=True)
        self.projR = nn.Linear(d_model, d_model)
        self.projE = nn.Linear(d_model, d_model)
        self.cross = ChiralCrossAttn(d_model, nhead=nhead)
    def forward(self, xR, xE, s_sign):
        HR,_ = self.r_enc(F.gelu(self.r_in(xR)))
        HE,_ = self.e_enc(F.gelu(self.e_in(xE)))
        return self.cross(self.projR(HR), self.projE(HE), s_sign)

class Reservoir(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.z = nn.GRU(d, d, batch_first=True)
        self.u = nn.GRU(d, d, batch_first=True)
        self.mix = nn.Parameter(torch.tensor(0.6))
        self.n = nn.LayerNorm(d)
    def forward(self, H):
        Z,_ = self.z(H); U,_ = self.u(H)
        return self.n(self.mix*Z + (1-self.mix)*U)

class Heads(nn.Module):
    def __init__(self, d, n_classes, dim_r):
        super().__init__()
        self.y = nn.Linear(d, n_classes)
        self.r = nn.Linear(d, dim_r)
        self.sig = nn.Linear(d, 1)
        self.s = nn.Linear(d, 2)
    def forward(self, H):
        return dict(y=self.y(H), r=self.r(H), sig=self.sig(H), s=self.s(H))

class HelixCoherenceNet(nn.Module):
    def __init__(self, d_inR, d_inE, n_classes=12, dim_r=3, d_pe=32, d_model=256):
        super().__init__()
        self.pe = HelixPE(d_out=d_pe)
        self.enc = DualBranchEncoder(d_inR+d_pe, d_inE+d_pe, d_model=d_model)
        self.res = Reservoir(d_model*2)
        self.heads = Heads(d_model*2, n_classes, dim_r)
    def forward(self, xR, xE, s_sign, beat=4.0):
        B,T,_ = xR.shape
        pe = self.pe(T, beat=beat, device=xR.device)[None].expand(B,-1,-1)
        HR,HE = self.enc(torch.cat([xR,pe],-1), torch.cat([xE,pe],-1), s_sign)
        H = torch.cat([HR,HE], -1)
        return self.heads(self.res(H))

# ====== 温度制御（後出し） ======
def tau_by_coherence(delta_t, Tc, tau0=1.5, k=1.0):
    x = float(delta_t)/max(1e-6, Tc); x = max(0.0, min(2.0, x))
    return tau0*math.exp(-k*x)

# ========= MediaPipe Pose 抽出 =========
mp_pose = mp.solutions.pose
PARENT = { # 簡易ツリー（描画用）
    11:12, 12:24, 11:23, 23:24, 13:11, 15:13, 14:12, 16:14, 25:23, 27:25, 26:24, 28:26
}

def extract_pose(frame_bgr, pose):
    h,w,_ = frame_bgr.shape
    rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks: return None
    pts = []
    for lm in res.pose_landmarks.landmark:
        x = lm.x*w; y = lm.y*h; z = lm.z*w  # zは相対なので幅スケール
        v = lm.visibility
        pts.append([x,y,z,v])
    return np.array(pts, dtype=np.float32)  # [33,4]

def normalize_pose(pts):
    # 原点=骨盤中心(LEFT_HIP=23, RIGHT_HIP=24), スケール=肩幅(11-12)
    origin = (pts[23,:3] + pts[24,:3]) / 2.0
    shoulders = np.linalg.norm(pts[11,:3]-pts[12,:3]) + 1e-6
    xyz = (pts[:,:3] - origin[None,:]) / shoulders
    vis = pts[:,3:4]
    return np.concatenate([xyz, vis], axis=1)  # [33,4]

def features_from_pose(buf_xyz, ema=0.6):
    arr = np.stack(buf_xyz, axis=0)  # [T,33,4]
    xyz = arr[..., :3]; vis = arr[..., 3:]
    # R: EMA平滑
    R = np.copy(xyz)
    for t in range(1,R.shape[0]):
        R[t] = ema*R[t-1] + (1-ema)*R[t]
    # E: 速度
    V = np.diff(xyz, axis=0, prepend=xyz[:1])
    Rf = np.concatenate([R.reshape(R.shape[0], -1), vis.reshape(R.shape[0], -1)], axis=-1)
    Ef = np.concatenate([V.reshape(V.shape[0], -1), vis.reshape(V.shape[0], -1)], axis=-1)
    return Rf.astype(np.float32), Ef.astype(np.float32)

# ========= 音楽アシスト =========
if AUDIO_OK:
    class MusicListener:
        def __init__(self, samplerate=44100, hop=512):
            self.sr = samplerate
            self.hop = hop
            self.lock = threading.Lock()
            self.onset = aubio.onset("default", 1024, hop, samplerate)
            self.tempo = aubio.tempo("default", 1024, hop, samplerate)
            self.phase = 0.0; self.bpm = 100.0; self.energy = 0.0
            self.last_beat_t = time.time(); self.enabled = True
            self._stream = None
        def _audio_cb(self, indata, frames, time_info, status):
            if not self.enabled: return
            mono = np.mean(indata, axis=1).astype(np.float32)
            t = self.tempo(mono)
            if t is not None and t > 0:
                with self.lock:
                    self.bpm = 60.0 / float(t)
                    self.last_beat_t = time.time()
                    self.phase = 0.0
            else:
                with self.lock:
                    dt = time.time() - self.last_beat_t
                    beat_len = 60.0 / max(40.0, min(220.0, self.bpm))
                    self.phase = (dt / beat_len) % 1.0
            with self.lock:
                self.energy = 0.9*self.energy + 0.1*float(np.sqrt((mono**2).mean()+1e-12))
        def start(self):
            self._stream = sd.InputStream(callback=self._audio_cb, channels=2,
                                          samplerate=self.sr, blocksize=self.hop)
            self._stream.start()
        def stop(self):
            if self._stream:
                self._stream.stop(); self._stream.close()
        def get(self):
            with self.lock:
                return float(self.phase), float(self.bpm), float(self.energy)
else:
    class MusicListener:
        def __init__(self, *a, **kw): pass
        def start(self): pass
        def stop(self): pass
        def get(self): return 0.0, 100.0, 0.0

def music_bias_vector(r_vec, phase, energy, bpm):
    beat_pulse = math.cos(2*math.pi*phase)  # 1(強拍)→-1
    gain = 0.05*(0.5 + energy) * (1.0 + 0.002*(bpm-100.0))
    return r_vec*(1.0 + gain*beat_pulse)

def kinematic_rollout_music(last_pose_xyz, r_vec, phase, energy, bpm, steps=12, noise=0.03):
    traj = []; cur = last_pose_xyz.copy()
    for s in range(steps):
        bias = music_bias_vector(r_vec, (phase + s/steps)%1.0, energy, bpm)
        drift = bias[None,:]*0.02 + np.random.randn(*cur.shape)*noise*np.linspace(1.0,0.3,cur.shape[0])[:,None]
        cur = cur + drift; traj.append(cur.copy())
    return np.stack(traj, axis=0)

def score_rollout_music(traj, phase0, energy):
    vel = np.linalg.norm(np.diff(traj, axis=0, prepend=traj[:1]), axis=-1).mean()
    smooth = -np.var(np.diff(traj, axis=0), axis=(0,1,2))
    S = traj.shape[0]; idx_peak = int((phase0 % 1.0) * S)
    local_vel = np.linalg.norm(traj[min(S-1,idx_peak)]-traj[max(0,idx_peak-1)])
    beat_align = local_vel
    return 0.45*vel + 0.25*smooth + 0.25*beat_align + 0.05*energy

def choose_best_rollout_music(last_pose_xyz, r_vec, phase, energy, bpm, N=96):
    best_s, best = -1e9, None
    for _ in range(N):
        roll = kinematic_rollout_music(last_pose_xyz, r_vec, phase, energy, bpm,
                                       steps=8, noise=np.random.uniform(0.01,0.05))
        s = score_rollout_music(roll, phase, energy)
        if s>best_s: best_s, best = s, roll
    return best

# ========= 可視化 =========
def draw_skeleton(frame, norm_xyz, color=(50,220,50)):
    h,w,_ = frame.shape
    pts2 = norm_xyz[:,:2].copy()
    scale = w/6.0
    cx,cy = w/2, h/2+50
    pts2 = (pts2*scale + np.array([cx,cy])[None,:]).astype(int)
    for c,p in PARENT.items():
        a = pts2[c]; b = pts2[p]
        cv.line(frame, tuple(a), tuple(b), color, 2)
    for p in pts2:
        cv.circle(frame, tuple(p), 2, color, -1)
    return frame

# ==== 仮想骨格（人がいなくても踊る用） ====
class VirtualPoseState:
    def __init__(self, n_joints=33):
        self.xyz = np.zeros((n_joints, 3), dtype=np.float32)
        self.dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    def step(self, r_vec, noise=0.01):
        drift = r_vec[None,:]*0.02 + np.random.randn(*self.xyz.shape)*noise*np.linspace(1.0,0.3,self.xyz.shape[0])[:,None]
        self.xyz += drift
        return self.xyz

def unit(x):
    n = np.linalg.norm(x) + 1e-6
    return x / n

def sample_music_direction(prev_dir, phase, energy, bpm, jitter=0.2):
    base = music_bias_vector(prev_dir, phase, energy, bpm)
    rand = np.random.randn(3).astype(np.float32) * jitter * (0.5 + energy)
    new_dir = unit(0.85*base + 0.15*rand)
    return new_dir

# ========= メイン =========
def main():
    # ---- 保存まわり ----
    SAVE_DIR = "records"
    os.makedirs(SAVE_DIR, exist_ok=True)
    writer = None
    record_overlay = True
    is_recording = False
    cur_out_path = None

    # ---- パラメタ ----
    N_CLASSES=12; DIM_R=3
    L = 6   # 後出し窓（フレーム）
    Tc = 8  # コヒーレンス時間
    FPS_SMOOTH = 0.9

    # ---- 音同期 ----
    music = MusicListener(); music.start()
    music_assist = True
    bpm_lock = False
    bpm_fixed = 100.0

    # ---- モード ----
    mode = MODE_AUTO   # 初期は自動
    vpose = VirtualPoseState(n_joints=33)
    codance_mix = 0.5  # CO-DANCE のブレンド率（0=人だけ, 1=音だけ）

    # ---- モデル準備 ----
    dR = 33*3 + 33*1  # xyz + vis
    dE = 33*3 + 33*1  # vel + vis
    model = HelixCoherenceNet(d_inR=dR, d_inE=dE, n_classes=N_CLASSES, dim_r=DIM_R).to(DEVICE)
    model.eval()
    # state = torch.load('helix_model.pt', map_location=DEVICE); model.load_state_dict(state)

    # ---- MediaPipe ----
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # ---- カメラ ----
    cap = cv.VideoCapture(0)  # Windowsで掴めないときは cv.VideoCapture(0, cv.CAP_DSHOW)
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) or 1280)
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 720)

    def make_writer(path, fps_guess=30.0):
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        return cv.VideoWriter(path, fourcc, fps_guess, (W, H))

    buf = deque(maxlen=32)
    prob_buf = deque(maxlen=64)
    s_sign = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)  # 初期は右巻
    fps, last = 0.0, time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv.flip(frame, 1)
            raw_frame = frame.copy()

            pts = extract_pose(frame, pose)
            human_present = False
            if pts is not None:
                norm = normalize_pose(pts)  # [33,4]
                human_present = (norm[:,3].mean() > 0.5)
                buf.append(norm)

                if len(buf)>=8:
                    Rf, Ef = features_from_pose(buf, ema=0.6)
                    xR = torch.from_numpy(Rf[None,:,:]).to(DEVICE)
                    xE = torch.from_numpy(Ef[None,:,:]).to(DEVICE)

                    with torch.no_grad():
                        # 音情報
                        phase, bpm_est, energy = music.get()
                        bpm_use = (bpm_fixed if bpm_lock else bpm_est)

                        # モデル推論（人ベースのr_vec）
                        out = model(xR, xE, s_sign)
                        logits = out["y"][:,-1,:]
                        delta_t = min(L, len(buf))
                        tau = tau_by_coherence(delta_t, Tc)
                        _ = F.softmax(logits/tau, dim=-1)[0].detach().cpu().numpy()
                        r_vec_human = out["r"][:,-1,:][0].detach().cpu().numpy()
                        s_hat = F.softmax(out["s"][:,-1,:], dim=-1)[0].detach().cpu().numpy()
                        s_sign = torch.tensor([1.0 if s_hat[1]>=s_hat[0] else -1.0], dtype=torch.float32, device=DEVICE)

                        # 音方向の更新（仮想骨格）
                        vpose.dir = sample_music_direction(vpose.dir, phase, energy, bpm_use)
                        r_vec_music = vpose.dir.copy()

                        # AUTOモードのときに現在モード確定
                        cur_mode = mode
                        if mode == MODE_AUTO:
                            cur_mode = MODE_CODANCE if human_present else MODE_MUSIC

                        # ロールアウト分岐
                        if cur_mode == MODE_HUMAN:
                            # 人の現在姿勢に沿って
                            roll = choose_best_rollout_music(norm[:,:3], r_vec_human, phase, energy, bpm_use, N=96) if music_assist \
                                   else choose_best_rollout_music(norm[:,:3], r_vec_human, 0.0, 0.0, 100.0, N=64)
                            pred_pose = roll[0]

                        elif cur_mode == MODE_MUSIC:
                            # 人がいなくても踊る（仮想骨格）
                            roll = choose_best_rollout_music(vpose.xyz, r_vec_music, phase, energy, bpm_use, N=96)
                            pred_pose = roll[0]
                            vpose.step(r_vec_music, noise=np.random.uniform(0.01,0.03))

                        elif cur_mode == MODE_CODANCE:
                            # 人×音のブレンド
                            r_vec_mix = unit((1.0-codance_mix)*r_vec_human + codance_mix*r_vec_music)
                            roll = choose_best_rollout_music(norm[:,:3], r_vec_mix, phase, energy, bpm_use, N=96)
                            pred_pose = roll[0]

                        else:
                            # 安全策：人モード
                            roll = choose_best_rollout_music(norm[:,:3], r_vec_human, phase, energy, bpm_use, N=96)
                            pred_pose = roll[0]

                        draw_skeleton(frame, pred_pose)

            # FPS表示 + HUD
            now = time.time()
            fps = FPS_SMOOTH*fps + (1-FPS_SMOOTH)*(1.0/max(1e-6, now-last)); last = now
            cv.putText(frame, f"FPS:{fps:.1f}  L={L} Tc={Tc}  chiral:{'R' if s_sign.item()>0 else 'L'}",
                       (10,25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # モードHUD
            cv.putText(frame, f"mode:{MODE_NAME[mode]}  mix:{codance_mix:.2f}",
                       (10,100), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200,220,255), 2)

            # 音HUD
            hud_music = f"music:{'ON' if music_assist else 'OFF'}"
            if AUDIO_OK:
                hud_music += f"  bpm:{int(bpm_fixed) if bpm_lock else 'auto'}"
            else:
                hud_music += " (driver off)"
            cv.putText(frame, hud_music, (10,75), cv.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)

            # 録画ステータス
            status = "REC" if is_recording else "LIVE"
            cv.putText(frame, f"{status}  save={'ovl' if record_overlay else 'raw'}",
                       (10,50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if is_recording else (200,200,200), 2)

            # 録画
            if is_recording and writer is not None:
                frame_to_save = frame if record_overlay else raw_frame
                writer.write(frame_to_save)

            cv.imshow("HelixCoherence Live", frame)
            key = cv.waitKey(1) & 0xFF
            if key==27:  # ESC
                break
            elif key==ord('['): L = max(0, L-1)
            elif key==ord(']'): L = min(24, L+1)
            elif key==ord('-'): Tc = max(1, Tc-1)
            elif key==ord('='): Tc = min(48, Tc+1)
            elif key in (ord('r'), ord('R')):
                if not is_recording:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cur_out_path = os.path.join(SAVE_DIR, f"helix_{ts}.mp4")
                    writer = make_writer(cur_out_path, fps_guess=max(10.0, min(60.0, fps or 30.0)))
                    is_recording = True
                    print(f"[REC] start: {cur_out_path}")
                else:
                    is_recording = False
                    if writer:
                        writer.release(); writer = None
                        print(f"[REC] stop : {cur_out_path}")
                    cur_out_path = None
            elif key in (ord('n'), ord('N')):
                if is_recording:
                    if writer: writer.release()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cur_out_path = os.path.join(SAVE_DIR, f"helix_{ts}.mp4")
                    writer = make_writer(cur_out_path, fps_guess=max(10.0, min(60.0, fps or 30.0)))
                    print(f"[REC] roll new file: {cur_out_path}")
            elif key in (ord('o'), ord('O')):
                record_overlay = not record_overlay
                print(f"[REC] save mode: {'overlay' if record_overlay else 'raw'}")
            elif key in (ord('m'), ord('M')):
                music_assist = not music_assist
                print(f"[MUSIC] assist: {'ON' if music_assist else 'OFF'}")
            elif key in (ord('b'), ord('B')):
                if AUDIO_OK:
                    bpm_lock = not bpm_lock
                    if bpm_lock:
                        _, bpm_est, _ = music.get()
                        bpm_fixed = max(40.0, min(220.0, bpm_est))
                    print(f"[MUSIC] bpm lock: {'ON' if bpm_lock else 'OFF'} ({bpm_fixed:.1f})")
                else:
                    print("[MUSIC] bpm lock requires aubio+sounddevice installed.")
            # モード切替
            elif key == ord('1'):
                mode = MODE_MUSIC; print("[MODE] MUSIC")
            elif key == ord('2'):
                mode = MODE_HUMAN; print("[MODE] HUMAN")
            elif key == ord('3'):
                mode = MODE_CODANCE; print("[MODE] CO-DANCE")
            elif key in (ord('a'), ord('A')):
                mode = MODE_AUTO; print("[MODE] AUTO")
            elif key == ord(','):
                codance_mix = max(0.0, codance_mix - 0.05)
            elif key == ord('.'):
                codance_mix = min(1.0, codance_mix + 0.05)

    finally:
        try: music.stop()
        except Exception: pass
        if writer: writer.release()
        cap.release()
        cv.destroyAllWindows()

if __name__=="__main__":
    main()
