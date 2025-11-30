# airfeel_tensor_rt.py
# ---------------------------------------------------
# リアルタイム空気感 × 責任場4次元テンソル × ラグラジアン（1ファイル完結）
# 依存: opencv-python, mediapipe, numpy, matplotlib
# 実行: python airfeel_tensor_rt.py
# 終了: ウィンドウ選択中に 'q'
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

# ========== 波形描画 ==========
window_size = 200  # 表示フレーム数
scores = deque(maxlen=window_size)
Ls     = deque(maxlen=window_size)

plt.ion()
fig, ax = plt.subplots()
lineA, = ax.plot([], [], lw=2, label="Airfeel A(t)")
lineL, = ax.plot([], [], lw=2, label="Lagrangian L(t)")
ax.set_ylim(-1.2, 1.2)
ax.set_xlim(0, window_size)
ax.set_title("Airfeel & Lagrangian (real-time)")
ax.legend(loc="upper right")

# ========== 空気感スコア（元実装の趣旨を踏襲） ==========
def calc_airfeel(landmarks, prev_landmarks):
    if prev_landmarks is None:
        return 0.0
    curr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    prev = np.array([[lm.x, lm.y, lm.z] for lm in prev_landmarks])
    diff = curr - prev
    speed = np.linalg.norm(diff, axis=1).mean()  # 平均速度
    stillness = np.exp(-speed * 50)   # 小さい速度ほど 1 に近い
    motion = 1 - stillness
    score = stillness - motion        # [-1, +1]
    return float(score)

# ========== 4次元特徴ベクトル J_vec を作る ==========
# 軸: [論理, 感覚, 時間, 社会]
# - 論理: Stillness 寄り (A の正側)
# - 感覚: Motion 寄り (A の負側)
# - 時間: 変化の落ち着き (|ΔA| が小さいほど大) → exp(-k*|ΔA|)
# - 社会: 姿勢の「開き」(両手の距離/肩幅で近似) → 開放感/関与度の簡易 proxy
def build_J_vec(A_now, A_prev, lmks):
    logic  = np.clip((A_now + 1.0) / 2.0, 0.0, 1.0)   # 0..1
    sense  = 1.0 - logic                               # 0..1
    dA     = 0.0 if A_prev is None else abs(A_now - A_prev)
    timeC  = float(np.exp(-4.0 * dA))                  # 変化が小さい=安定=大

    social = 0.5  # デフォルト
    try:
        # 簡易: 左右手首の距離 / 両肩距離 で「開き」を正規化
        # landmarks index (MediaPipe Pose): LEFT/RIGHT_WRIST=15/16, LEFT/RIGHT_SHOULDER=11/12
        Lw = np.array([lmks[15].x, lmks[15].y])
        Rw = np.array([lmks[16].x, lmks[16].y])
        Ls = np.array([lmks[11].x, lmks[11].y])
        Rs = np.array([lmks[12].x, lmks[12].y])
        wrist_dist = np.linalg.norm(Lw - Rw)
        shoulder_dist = np.linalg.norm(Ls - Rs) + 1e-6
        open_ratio = np.clip(wrist_dist / shoulder_dist, 0.0, 2.0)  # だいたい 0〜2
        social = float(np.clip(open_ratio / 2.0, 0.0, 1.0))         # 0..1 に圧縮
    except Exception:
        pass

    J_vec = np.array([logic, sense, timeC, social], dtype=float)
    # 数値安定化のため正規化（合計が0ならそのまま）
    s = J_vec.sum()
    if s > 1e-8:
        J_vec = J_vec / s
    return J_vec  # 和=1 の確率ベクトル風

# ========== 責任場テンソル R と ラグラジアン L ==========
# R(t): ここでは「対角 = 各軸の強度」「非対角 = 軸間カップリング」を簡易生成
# - 対角: diag = base_diag * (0.6 + 0.4*J_vec)  → 今の関心が強い軸を重めに
# - 非対角: W で固定の相互作用行列を用意し、W⊙(J_vec⊗J_vec) で現在の強さを可変に
def build_R_and_L(J_vec):
    base_diag = np.array([1.0, 1.0, 1.0, 1.0])
    diag = base_diag * (0.6 + 0.4 * J_vec)  # 0.6〜1.0

    # 軸間カップリングの雰囲気（任意に調整可）
    # 例: 論理↔感覚は中程度、時間↔社会はやや強め…など
    W = np.array([
        [0.0, 0.35, 0.25, 0.30],
        [0.35, 0.0, 0.25, 0.30],
        [0.25, 0.25, 0.0, 0.40],
        [0.30, 0.30, 0.40, 0.0],
    ], dtype=float)

    # 現在の関心分布に応じたカップリング強度
    outer = np.outer(J_vec, J_vec)             # 4x4
    offdiag = W * outer
    np.fill_diagonal(offdiag, 0.0)             # 対角は後で足す

    R = offdiag + np.diag(diag)                # 4x4 対称行列っぽく

    J = outer                                  # 責任流 J^{μν} = J_vec ⊗ J_vec
    L = 0.5 * float(np.sum(R * J))             # L = 0.5 Σ Rμν Jμν

    return R, J, L

# ========== メインループ ==========
cap = cv2.VideoCapture(0)
prev_landmarks = None
prev_A = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    A = 0.0
    J_vec = np.array([0.25, 0.25, 0.25, 0.25])
    Lval = 0.0

    if results.pose_landmarks:
        A = calc_airfeel(results.pose_landmarks.landmark, prev_landmarks)
        J_vec = build_J_vec(A, prev_A, results.pose_landmarks.landmark)
        R, J, Lval = build_R_and_L(J_vec)
        prev_landmarks = results.pose_landmarks.landmark

        # 骨格描画
        drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 履歴更新
    scores.append(A)
    Ls.append(np.tanh(Lval))  # スケール詰めのため双曲線tanhで[-1,1]へ圧縮

    # ---- グラフ更新 ----
    lineA.set_ydata(scores)
    lineA.set_xdata(range(len(scores)))
    lineL.set_ydata(Ls)
    lineL.set_xdata(range(len(Ls)))
    ax.draw_artist(ax.patch)
    ax.draw_artist(lineA)
    ax.draw_artist(lineL)
    fig.canvas.flush_events()

    # ---- 数値表示（コンソール） ----
    print(f"A(t)={A:.3f} | J=[logic {J_vec[0]:.2f}, sense {J_vec[1]:.2f}, time {J_vec[2]:.2f}, social {J_vec[3]:.2f}] | L={Lval:.3f}")

    # ---- 映像オーバーレイ ----
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (int(w*0.55), 150), (0,0,0), -1)
    cv2.putText(frame, f"A(t): {A:.3f}", (20, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
    cv2.putText(frame, f"L(t): {Lval:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
    cv2.putText(frame, f"J logic:{J_vec[0]:.2f} sense:{J_vec[1]:.2f} time:{J_vec[2]:.2f} social:{J_vec[3]:.2f}",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)

    # スコアに応じて枠色（静=青〜動=赤）を変える簡易演出
    # A ≈ +1 → 青, 0 → 白, -1 → 赤
    r = int(np.clip((1 - (A+1)/2) * 255, 0, 255))
    b = int(np.clip(((A+1)/2) * 255, 0, 255))
    cv2.rectangle(frame, (0,0), (w,5), (r,50,b), -1)

    cv2.imshow("Airfeel + Responsibility Field", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
