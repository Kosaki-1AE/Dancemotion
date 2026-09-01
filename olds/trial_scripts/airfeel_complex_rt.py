# airfeel_complex_rt.py
# ---------------------------------------------------
# リアルタイム空気感 × 複素責任ベクトル × 責任場4次元テンソル × 歪み × 複素ラグラジアン
# 依存: opencv-python, mediapipe, numpy, matplotlib
# 実行: python airfeel_complex_rt.py
# 終了: 'q'
# ---------------------------------------------------

from collections import deque

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

# ========== MediaPipe 初期化 ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
drawer = mp.solutions.drawing_utils

# ========== 可視化（波形） ==========
window_size = 200
wave_A = deque(maxlen=window_size)
wave_L = deque(maxlen=window_size)

plt.ion()
fig, ax = plt.subplots()
lineA, = ax.plot([], [], lw=2, label="Airfeel A(t)")
lineL, = ax.plot([], [], lw=2, label="Lagrangian Re(L)(t)")
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(0, window_size)
ax.set_title("Airfeel & Complex Lagrangian (real-time)")
ax.legend(loc="upper right")

# ========== 空気感スコア（静:+1 / 動:-1） ==========
def calc_airfeel(landmarks, prev_landmarks):
    if prev_landmarks is None:
        return 0.0
    curr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    prev = np.array([[lm.x, lm.y, lm.z] for lm in prev_landmarks])
    diff = curr - prev
    speed = np.linalg.norm(diff, axis=1).mean()
    still = np.exp(-speed * 50.0)     # 速度小→1に近づく
    motion = 1.0 - still
    return float(still - motion)      # [-1, +1]

# ========== 4成分の実責任成分ベクトル（論理・感覚・時間・社会） ==========
def build_real_features(A_now, A_prev, lmks):
    # 論理: 静寄り → (A+1)/2
    logic = np.clip((A_now + 1.0) / 2.0, 0.0, 1.0)
    # 感覚: 動寄り → 1 - logic
    sense = 1.0 - logic
    # 時間: 変化が小さいほど大（コヒーレンス）→ exp(-k|ΔA|)
    dA = 0.0 if A_prev is None else abs(A_now - A_prev)
    timeC = float(np.exp(-4.0 * dA))
    # 社会: 開放度（両手首距離/両肩距離で正規化）
    social = 0.5
    try:
        Lw = np.array([lmks[15].x, lmks[15].y])
        Rw = np.array([lmks[16].x, lmks[16].y])
        Ls = np.array([lmks[11].x, lmks[11].y])
        Rs = np.array([lmks[12].x, lmks[12].y])
        wrist_dist = np.linalg.norm(Lw - Rw)
        shoulder_dist = np.linalg.norm(Ls - Rs) + 1e-6
        open_ratio = np.clip(wrist_dist / shoulder_dist, 0.0, 2.0)
        social = float(np.clip(open_ratio / 2.0, 0.0, 1.0))
    except Exception:
        pass

    vec = np.array([logic, sense, timeC, social], dtype=float)  # 4次元
    s = vec.sum()
    if s > 1e-8:
        vec = vec / s
    return vec  # 和=1

# ========== 複素責任ベクトル C(t) の構成 ==========
# 実部: 論理・時間の寄与を主（静側）
# 虚部: 感覚・社会の寄与を主（動/同期側）
# 可変の混合率を持たせるための重み行列を定義
W_real = np.array([ [0.70, 0.10, 0.20, 0.00],   # logic  ← L,S,Time,Social
                    [0.15, 0.10, 0.70, 0.05],   # time
                    [0.00, 0.00, 0.00, 0.00],   # dummy (未使用スロット)
                    [0.00, 0.00, 0.00, 0.00] ]) # dummy

W_imag = np.array([ [0.05, 0.70, 0.00, 0.25],   # sense ← L,S,Time,Social
                    [0.05, 0.20, 0.00, 0.75],   # social
                    [0.00, 0.00, 0.00, 0.00],   # dummy
                    [0.00, 0.00, 0.00, 0.00] ]) # dummy

# 実際には4次元の複素ベクトルに落とす（軸: 論理, 感覚, 時間, 社会）
# ここでは簡便に:
#   Re(C) = [r_logic, r_sense, r_time, r_social]
#   Im(C) = [i_logic, i_sense, i_time, i_social]
# を作り、C_k = Re_k + i * Im_k
def build_complex_responsibility(f4):
    # f4 = [logic, sense, time, social]
    # 実部は「静寄り」を強調：logic/time が主
    r_logic  = 0.70*f4[0] + 0.20*f4[2] + 0.10*f4[1]
    r_sense  = 0.10*f4[0] + 0.20*f4[1] + 0.10*f4[3]
    r_time   = 0.15*f4[0] + 0.70*f4[2] + 0.05*f4[1]
    r_social = 0.05*f4[3] + 0.05*f4[1] + 0.05*f4[0]

    # 虚部は「動/同期寄り」を強調：sense/social が主
    i_logic  = 0.05*f4[0] + 0.25*f4[3] + 0.70*f4[1]
    i_sense  = 0.70*f4[1] + 0.20*f4[3] + 0.05*f4[0]
    i_time   = 0.05*f4[2] + 0.10*f4[1] + 0.10*f4[3]
    i_social = 0.75*f4[3] + 0.20*f4[1] + 0.05*f4[0]

    Re = np.array([r_logic, r_sense, r_time, r_social], dtype=float)
    Im = np.array([i_logic, i_sense, i_time, i_social], dtype=float)

    # 正規化（大きさを安定化）
    Re = Re / (np.linalg.norm(Re) + 1e-8)
    Im = Im / (np.linalg.norm(Im) + 1e-8)

    C = Re + 1j*Im  # 複素責任ベクトル (4,)
    return C, Re, Im

# ========== 責任場テンソル R(t) と歪み Aμν、複素ラグラジアン L ==========
# R(t): 実対称（基盤場）。対角は各軸強度、非対角は軸間カップリング。
def build_responsibility_field(f4):
    # 対角（各軸の強度）: 基本1.0に、今の関心分布で微調整
    diag = 0.6 + 0.4 * f4  # 0.6〜1.0
    # 非対角の固定カップリング行列（好みに応じて調整可）
    W = np.array([
        [0.0, 0.35, 0.25, 0.30],
        [0.35, 0.0, 0.25, 0.30],
        [0.25, 0.25, 0.0, 0.40],
        [0.30, 0.30, 0.40, 0.0],
    ], dtype=float)
    # 現在の関心分布に応じた強さ
    off = W * np.outer(f4, f4)
    np.fill_diagonal(off, 0.0)
    R = off + np.diag(diag)
    # 対称性を保証
    R = 0.5*(R + R.T)
    return R  # 実数4x4

# 歪み（空気感テンソル）Aμν = dR/dt + κ * Re(Jc)
# ここで Jc = C ⊗ C*（外積 with 複素共役）→ エルミート 4x4
def build_curvature_and_L(R_prev, R_now, C, kappa=0.8):
    # dR/dt を近似（フレーム差分）
    if R_prev is None:
        dR = np.zeros_like(R_now)
    else:
        dR = R_now - R_prev

    # 複素責任流テンソル（エルミート）
    Jc = np.outer(C, np.conjugate(C))  # 4x4 複素
    Jc_real = np.real(Jc)

    # 歪みテンソル（空気感）
    A = dR + kappa * Jc_real  # 実4x4

    # 複素ラグラジアン: L = 0.5 * Σ R_now * Jc  の実部を可視化
    L_complex = 0.5 * np.sum(R_now * Jc)
    return A, L_complex

# ========== メイン ==========
cap = cv2.VideoCapture(0)
prev_landmarks = None
prev_A = None
prev_R = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    A = 0.0
    C = np.zeros(4, dtype=np.complex128)
    L_val = 0.0
    ReC = np.zeros(4); ImC = np.zeros(4)

    if results.pose_landmarks:
        # 空気感スコア
        A = calc_airfeel(results.pose_landmarks.landmark, prev_landmarks)
        prev_landmarks = results.pose_landmarks.landmark

        # 4成分の実責任特徴（和=1）
        f4 = build_real_features(A, prev_A, results.pose_landmarks.landmark)

        # 複素責任ベクトル
        C, ReC, ImC = build_complex_responsibility(f4)
        C_norm = np.linalg.norm(C)

        # 責任場テンソル（実）
        R_now = build_responsibility_field(f4)

        # 歪み & 複素ラグラジアン
        A_tensor, L_complex = build_curvature_and_L(prev_R, R_now, C, kappa=0.8)
        prev_R = R_now
        prev_A = A

        # 値の可視化用
        L_val = np.real(L_complex)

        # 描画（骨格）
        drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ---- グラフ波形更新 ----
        wave_A.append(A)
        wave_L.append(np.tanh(L_val))  # スケール詰め

        lineA.set_ydata(wave_A); lineA.set_xdata(range(len(wave_A)))
        lineL.set_ydata(wave_L); lineL.set_xdata(range(len(wave_L)))
        ax.draw_artist(ax.patch); ax.draw_artist(lineA); ax.draw_artist(lineL)
        fig.canvas.flush_events()

        # ---- 数値オーバーレイ ----
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (int(w*0.80), 180), (0,0,0), -1)
        cv2.putText(frame, f"A(t): {A:.3f}", (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
        cv2.putText(frame, f"|C(t)|: {C_norm:.3f}  Re(L): {L_val:.3f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        cv2.putText(frame, f"ReC [Lo {ReC[0]:.2f} Se {ReC[1]:.2f} Ti {ReC[2]:.2f} So {ReC[3]:.2f}]",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
        cv2.putText(frame, f"ImC [Lo {ImC[0]:.2f} Se {ImC[1]:.2f} Ti {ImC[2]:.2f} So {ImC[3]:.2f}]",
                    (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
        cv2.putText(frame, "q: quit", (20, 170),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,160,160), 1)

        # Aに応じた色バー（静=青〜動=赤）
        r = int(np.clip((1 - (A+1)/2) * 255, 0, 255))
        b = int(np.clip(((A+1)/2) * 255, 0, 255))
        cv2.rectangle(frame, (0,0), (w,6), (r,50,b), -1)

    # 表示
    cv2.imshow("Airfeel + Complex Responsibility Field", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
