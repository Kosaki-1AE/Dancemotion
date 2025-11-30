# airfeel_meaning_order_rt_triple.py
# ---------------------------------------------
# Realtime Meaning-Order Vector Field
# 3 windows:
#   1) I+M (Raw座標：上が原点 / 解析用)
#   2) I+M (Display座標：カメラ直感一致)
#   3) H+M (Entropy背景)
#
# Keys: q=quit, s=save ./out/
# Options: --demo  (webcam不要のデモ)
# ---------------------------------------------

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


# ========= Core: build field =========
def build_meaning_order_field(
    L=8.0, N=90,
    agents=None,
    sources=None,
    wavelength=3.0,
    alpha=1.2, beta=0.9, delta=0.8,
    influence_sigma=2.3,
    quiver_step=4
):
    xs = np.linspace(-L, L, N)
    ys = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, ys)

    if agents is None:
        agents = [
            {"pos": np.array([-3.0, -1.0]), "dir": np.array([ 1.0,  0.2]), "w": 1.0},
            {"pos": np.array([ 2.5, -2.0]), "dir": np.array([-0.6,  0.8]), "w": 0.9},
            {"pos": np.array([-1.0,  2.5]), "dir": np.array([ 0.3, -0.9]), "w": 0.8},
            {"pos": np.array([ 3.0,  2.0]), "dir": np.array([-0.7, -0.2]), "w": 0.7},
        ]

    if sources is None:
        sources = [np.array([0.0,  0.6]), np.array([0.0, -0.6])]

    # --- Responsibility field R ---
    Rx = np.zeros_like(X); Ry = np.zeros_like(Y)
    for ag in agents:
        d = ag["dir"] / (np.linalg.norm(ag["dir"]) + 1e-12)
        dx = X - ag["pos"][0]; dy = Y - ag["pos"][1]
        dist2 = dx*dx + dy*dy
        infl = ag["w"] * np.exp(-dist2/(2*influence_sigma**2))
        Rx += infl * d[0]; Ry += infl * d[1]

    # --- Interference I (double-slit-like) ---
    k = 2*np.pi / wavelength
    def rdist(P): return np.sqrt((X-P[0])**2 + (Y-P[1])**2)
    r1 = rdist(sources[0]); r2 = rdist(sources[1])
    I = np.cos(k*r1) + np.cos(k*r2)
    dIx, dIy = np.gradient(I, xs, ys, edge_order=2)

    # --- Entropy H (from softmax of influences) ---
    raw = []
    for ag in agents:
        dx = X - ag["pos"][0]; dy = Y - ag["pos"][1]
        dist2 = dx*dx + dy*dy
        raw.append(ag["w"] * np.exp(-dist2/(2*influence_sigma**2)))
    raw = np.stack(raw, axis=0)
    raw_shift = raw - np.max(raw, axis=0, keepdims=True)
    p = np.exp(raw_shift); p = p / (np.sum(p, axis=0, keepdims=True) + 1e-12)
    H = -np.sum(p * np.log(p + 1e-12), axis=0)
    dHx, dHy = np.gradient(H, xs, ys, edge_order=2)

    # --- Meaning-Order field M ---
    Mx = alpha*Rx + beta*dIx - delta*dHx
    My = alpha*Ry + beta*dIy - delta*dHy

    step = quiver_step
    return {
        "xs": xs, "ys": ys, "X": X, "Y": Y,
        "I": I, "H": H, "Mx": Mx, "My": My,
        "Xq": X[::step, ::step], "Yq": Y[::step, ::step],
        "Mxq": Mx[::step, ::step], "Myq": My[::step, ::step],
    }

# ========= Map A,J -> field parameters =========
def map_RT_to_field(A, J):
    # J = [logic, sense, time, social]  (sum≈1)
    dirs = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([-1.0, 0.1]),
        np.array([0.0, -1.0])
    ]
    poss = [
        np.array([-3.0, -1.0]),
        np.array([ 2.5, -2.0]),
        np.array([-1.0,  2.5]),
        np.array([ 3.0,  2.0])
    ]
    agents = [{"pos": poss[k], "dir": dirs[k], "w": float(0.6 + 0.8*J[k])} for k in range(4)]
    d = 0.4 + 1.2*float(abs(A))                 # slit separation by |A|
    sources = [np.array([0.0, d/2]), np.array([0.0, -d/2])]
    wavelength = 4.0 - 1.5*float((A+1)/2)      # more motion -> shorter
    return agents, sources, wavelength

# ========= Demo (no webcam) =========
def _demo_signals(T=260):
    t = np.linspace(0, 10, T)
    A = np.tanh(np.sin(2*np.pi*0.15*t))
    J = np.vstack([
        0.5*(A+1)/1.5 + 0.2,
        1 - (0.5*(A+1)/1.5 + 0.2),
        np.exp(-0.8*np.abs(np.gradient(A))),
        0.3 + 0.2*np.sin(2*np.pi*0.05*t + 0.7)
    ]).clip(1e-6, None)
    J = J / J.sum(axis=0, keepdims=True)
    return t, A, J

def demo_loop():
    os.makedirs("out", exist_ok=True)
    t, A, J = _demo_signals()
    fig_raw,  ax_raw  = plt.subplots(figsize=(6,5))
    fig_disp, ax_disp = plt.subplots(figsize=(6,5))
    fig_HM,   ax_HM   = plt.subplots(figsize=(6,5))  # ★ H+M 追加

    for i in range(len(A)):
        agents, sources, wl = map_RT_to_field(A[i], J[:, i])
        f = build_meaning_order_field(agents=agents, sources=sources, wavelength=wl)

        # Raw（I+M, origin 上）
        ax_raw.clear()
        ax_raw.imshow(f["I"], extent=[f["xs"].min(), f["xs"].max(), f["ys"].min(), f["ys"].max()],
                      origin='upper', aspect='equal')
        ax_raw.quiver(f["Xq"], f["Yq"], f["Mxq"], f["Myq"], scale=60)
        ax_raw.set_title(f"I+M Raw (A={A[i]:+.2f})")

        # Display（I+M, origin 下＝直感一致）
        ax_disp.clear()
        ax_disp.imshow(f["I"], extent=[f["xs"].min(), f["xs"].max(), f["ys"].min(), f["ys"].max()],
                       origin='lower', aspect='equal')
        ax_disp.quiver(f["Xq"], f["Yq"], f["Mxq"], f["Myq"], scale=60)
        ax_disp.set_title("I+M Display（直感座標）")

        # H+M（スクショと同じ見え方：origin 下）
        ax_HM.clear()
        ax_HM.imshow(f["H"], extent=[f["xs"].min(), f["xs"].max(), f["ys"].min(), f["ys"].max()],
                     origin='lower', aspect='equal')
        ax_HM.quiver(f["Xq"], f["Yq"], f["Mxq"], f["Myq"], scale=60, color="k")
        ax_HM.set_title("H + M")

        plt.pause(0.01)

    fig_raw.savefig("out/demo_raw.png", dpi=180)
    fig_disp.savefig("out/demo_display.png", dpi=180)
    fig_HM.savefig("out/demo_HM.png", dpi=180)
    print("Saved: out/demo_raw.png, out/demo_display.png, out/demo_HM.png")

# ========= Realtime (webcam) =========
def realtime_loop():
    import cv2
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawer = mp.solutions.drawing_utils

    prev_landmarks, prev_A = None, None

    plt.ion()
    fig_raw,  ax_raw  = plt.subplots(figsize=(6,5))
    fig_disp, ax_disp = plt.subplots(figsize=(6,5))
    fig_HM,   ax_HM   = plt.subplots(figsize=(6,5))  # ★ H+M 追加

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        A, J_vec = 0.0, np.array([0.25,0.25,0.25,0.25])
        if results.pose_landmarks:
            curr = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            if prev_landmarks is None:
                speed = 0.0
            else:
                prev = np.array([[lm.x, lm.y, lm.z] for lm in prev_landmarks])
                speed = np.linalg.norm(curr - prev, axis=1).mean()
            stillness = np.exp(-50.0*speed); motion = 1 - stillness
            A = float(stillness - motion)

            logic  = np.clip((A+1)/2, 0, 1); sense = 1 - logic
            dA = 0.0 if prev_A is None else abs(A-prev_A)
            timeC = float(np.exp(-4.0*dA))
            social = 0.5
            try:
                Lw = np.array([results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y])
                Rw = np.array([results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y])
                Ls = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y])
                Rs = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y])
                open_ratio = np.linalg.norm(Lw - Rw) / (np.linalg.norm(Ls - Rs) + 1e-6)
                social = float(np.clip(open_ratio / 2.0, 0.0, 1.0))
            except: pass

            J_vec = np.array([logic, sense, timeC, social])
            J_vec = J_vec / (J_vec.sum() + 1e-12)

            prev_landmarks, prev_A = results.pose_landmarks.landmark, A
            drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        agents, sources, wl = map_RT_to_field(A, J_vec)
        f = build_meaning_order_field(agents=agents, sources=sources, wavelength=wl)

        # Raw（I+M, origin 上）
        ax_raw.clear()
        ax_raw.imshow(f["I"], extent=[f["xs"].min(), f["xs"].max(), f["ys"].min(), f["ys"].max()],
                      origin='upper', aspect='equal')
        ax_raw.quiver(f["Xq"], f["Yq"], f["Mxq"], f["Myq"], scale=60)
        ax_raw.set_title(f"I+M Raw (A={A:+.2f})")

        # Display（I+M, origin 下）
        ax_disp.clear()
        ax_disp.imshow(f["I"], extent=[f["xs"].min(), f["xs"].max(), f["ys"].min(), f["ys"].max()],
                       origin='lower', aspect='equal')
        ax_disp.quiver(f["Xq"], f["Yq"], f["Mxq"], f["Myq"], scale=60)
        ax_disp.set_title("I+M Display（直感座標）")

        # H+M（origin 下）
        ax_HM.clear()
        ax_HM.imshow(f["H"], extent=[f["xs"].min(), f["xs"].max(), f["ys"].min(), f["ys"].max()],
                     origin='lower', aspect='equal')
        ax_HM.quiver(f["Xq"], f["Yq"], f["Mxq"], f["Myq"], scale=60, color="k")
        ax_HM.set_title("H + M")

        plt.pause(0.01)

        # 上部バー：静→青, 動→赤
        h, w = frame.shape[:2]
        r = int(np.clip((1 - (A+1)/2) * 255, 0, 255))
        b = int(np.clip(((A+1)/2) * 255, 0, 255))
        import cv2
        cv2.rectangle(frame, (0,0), (w,6), (r,50,b), -1)
        cv2.imshow("Webcam (skeleton)", frame)

        k = (cv2.waitKey(1) & 0xFF)
        if k == ord('q'): break
        if k == ord('s'):
            os.makedirs("out", exist_ok=True)
            fig_raw.savefig("out/shot_raw.png", dpi=180)
            fig_disp.savefig("out/shot_display.png", dpi=180)
            fig_HM.savefig("out/shot_HM.png", dpi=180)
            print("Saved: out/shot_raw.png, out/shot_display.png, out/shot_HM.png")

    cap.release(); cv2.destroyAllWindows()

# ========= main =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run without webcam (synthetic)")
    args = parser.parse_args()
    if args.demo: demo_loop()
    else: realtime_loop()
