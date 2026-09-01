# airfeel_meaning_order_rt_v2.py
# -------------------------------------------------------------
# Realtime "Meaning-Order Vector Field" with explicit R(x,y) view
# Modes:
#   python airfeel_meaning_order_rt_v2.py            # realtime (webcam & mediapipe required)
#   python airfeel_meaning_order_rt_v2.py --demo     # demo (no webcam)
#   python airfeel_meaning_order_rt_v2.py --rtonly   # realtime R(x,y) only
#
# Keys:
#   q : quit
#   s : save current figures under ./out/
# -------------------------------------------------------------

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


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

    # ---------------- R(x,y): responsibility vector field ----------------
    Rx = np.zeros_like(X); Ry = np.zeros_like(Y)
    for ag in agents:
        d = ag["dir"] / (np.linalg.norm(ag["dir"]) + 1e-12)
        dx = X - ag["pos"][0]; dy = Y - ag["pos"][1]
        dist2 = dx*dx + dy*dy
        infl = ag["w"] * np.exp(-dist2/(2*influence_sigma**2))
        Rx += infl * d[0]; Ry += infl * d[1]

    # ---------------- I(x,y): double-slit-like interference ----------------
    k = 2*np.pi / wavelength
    def rdist(P):
        return np.sqrt((X-P[0])**2 + (Y-P[1])**2)
    r1 = rdist(sources[0]); r2 = rdist(sources[1])
    I = np.cos(k*r1) + np.cos(k*r2)
    dIx, dIy = np.gradient(I, xs, ys, edge_order=2)

    # ---------------- H(x,y): entropy from softmax of influences -----------
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

    # ---------------- M(x,y): meaning-order vector field -------------------
    Mx = alpha*Rx + beta*dIx - delta*dHx
    My = alpha*Ry + beta*dIy - delta*dHy

    # Downsample for quiver
    step = quiver_step
    return {
        "xs": xs, "ys": ys, "X": X, "Y": Y,
        "Rx": Rx, "Ry": Ry,
        "I": I, "H": H, "Mx": Mx, "My": My,
        "Xq": X[::step, ::step], "Yq": Y[::step, ::step],
        "Mxq": Mx[::step, ::step], "Myq": My[::step, ::step],
        "Rxq": Rx[::step, ::step], "Ryq": Ry[::step, ::step],
        "agents": agents, "sources": sources
    }

def map_RT_to_field(A, J):
    """
    A in [-1,+1], J = [logic, sense, time, social] (sum≈1)
    - Jで各エージェントの重みwを決定
    - |A|でスリット間隔、Aで波長（動き強→短波長）を決定
    """
    dirs = [
        np.array([1.0, 0.0]),   # logic
        np.array([0.0, 1.0]),   # sense
        np.array([-1.0, 0.1]),  # time
        np.array([0.0, -1.0])   # social
    ]
    poss = [
        np.array([-3.0, -1.0]),
        np.array([ 2.5, -2.0]),
        np.array([-1.0,  2.5]),
        np.array([ 3.0,  2.0])
    ]
    agents = []
    for k in range(4):
        agents.append({"pos": poss[k], "dir": dirs[k], "w": float(0.6 + 0.8*J[k])})

    d = 0.4 + 1.2*float(abs(A))                   # slit separation
    sources = [np.array([0.0, d/2]), np.array([0.0, -d/2])]
    wavelength = 4.0 - 1.5*float((A+1)/2)        # more motion -> shorter wavelength
    return agents, sources, wavelength

def _demo_signals(T=240):
    t = np.linspace(0, 10, T)
    A = np.tanh(np.sin(2*np.pi*0.15*t))
    J = np.vstack([
        0.5*(A+1)/1.5 + 0.2,                        # logic
        1 - (0.5*(A+1)/1.5 + 0.2),                  # sense (反転)
        np.exp(-0.8*np.abs(np.gradient(A))),        # time coherence
        0.3 + 0.2*np.sin(2*np.pi*0.05*t + 0.7)      # social
    ]).clip(1e-6, None)
    J = (J / J.sum(axis=0, keepdims=True))
    return t, A, J

def _ensure_out():
    os.makedirs("out", exist_ok=True)

def demo_loop(rtonly=False):
    _ensure_out()
    T = 260
    t, A, J = _demo_signals(T=T)

    if rtonly:
        figR, axR = plt.subplots(figsize=(7,5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        axI, axH, axR = axes

    for i in range(T):
        agents, sources, wl = map_RT_to_field(A[i], J[:, i])
        fields = build_meaning_order_field(agents=agents, sources=sources, wavelength=wl)

        # R-only
        axR.clear()
        axR.quiver(fields["Xq"], fields["Yq"], fields["Rxq"], fields["Ryq"], scale=60)
        axR.set_title("Responsibility Vector Field R(x,y)")

        if not rtonly:
            # I + M
            axI.clear()
            axI.imshow(fields["I"], extent=[fields["xs"].min(), fields["xs"].max(), fields["ys"].min(), fields["ys"].max()], origin='lower', aspect='equal')
            axI.quiver(fields["Xq"], fields["Yq"], fields["Mxq"], fields["Myq"], scale=60)
            axI.set_title(f"I + M  (A={A[i]:+.2f})")
            # H + M
            axH.clear()
            axH.imshow(fields["H"], extent=[fields["xs"].min(), fields["xs"].max(), fields["ys"].min(), fields["ys"].max()], origin='lower', aspect='equal')
            axH.quiver(fields["Xq"], fields["Yq"], fields["Mxq"], fields["Myq"], scale=60)
            axH.set_title("H + M")

        plt.pause(0.01)

    if rtonly:
        figR.savefig("out/demo_R_only.png", dpi=180)
    else:
        fig.savefig("out/demo_tripanel.png", dpi=180)

def realtime_loop(rtonly=False):
    import cv2
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    drawer = mp.solutions.drawing_utils

    prev_landmarks = None
    prev_A = None

    plt.ion()
    if rtonly:
        figR, axR = plt.subplots(figsize=(7,5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        axI, axH, axR = axes

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        A = 0.0
        J_vec = np.array([0.25, 0.25, 0.25, 0.25])
        if results.pose_landmarks:
            curr = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            if prev_landmarks is None:
                speed = 0.0
            else:
                prev = np.array([[lm.x, lm.y, lm.z] for lm in prev_landmarks])
                diff = curr - prev
                speed = np.linalg.norm(diff, axis=1).mean()
            stillness = np.exp(-50.0*speed)
            motion = 1.0 - stillness
            A = float(stillness - motion)

            logic  = np.clip((A + 1.0) / 2.0, 0.0, 1.0)
            sense  = 1.0 - logic
            dA     = 0.0 if prev_A is None else abs(A - prev_A)
            timeC  = float(np.exp(-4.0*dA))

            social = 0.5
            try:
                Lw = np.array([results.pose_landmarks.landmark[15].x, results.pose_landmarks.landmark[15].y])
                Rw = np.array([results.pose_landmarks.landmark[16].x, results.pose_landmarks.landmark[16].y])
                Ls = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y])
                Rs = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y])
                wrist_dist = np.linalg.norm(Lw - Rw)
                shoulder_dist = np.linalg.norm(Ls - Rs) + 1e-6
                open_ratio = np.clip(wrist_dist / shoulder_dist, 0.0, 2.0)
                social = float(np.clip(open_ratio / 2.0, 0.0, 1.0))
            except Exception:
                pass

            J_vec = np.array([logic, sense, timeC, social], dtype=float)
            s = J_vec.sum()
            if s > 1e-8: J_vec /= s

            prev_landmarks = results.pose_landmarks.landmark
            prev_A = A

            # draw skeleton
            drawer.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        agents, sources, wl = map_RT_to_field(A, J_vec)
        fields = build_meaning_order_field(agents=agents, sources=sources, wavelength=wl)

        # R-only
        axR.clear()
        axR.quiver(fields["Xq"], fields["Yq"], fields["Rxq"], fields["Ryq"], scale=60)
        axR.set_title("Responsibility Vector Field R(x,y)")

        if not rtonly:
            # I + M
            axI.clear()
            axI.imshow(fields["I"], extent=[fields["xs"].min(), fields["xs"].max(), fields["ys"].min(), fields["ys"].max()], origin='lower', aspect='equal')
            axI.quiver(fields["Xq"], fields["Yq"], fields["Mxq"], fields["Myq"], scale=60)
            axI.set_title(f"I + M  (A={A:+.2f}; J={np.round(J_vec,2)})")
            # H + M
            axH.clear()
            axH.imshow(fields["H"], extent=[fields["xs"].min(), fields["xs"].max(), fields["ys"].min(), fields["ys"].max()], origin='lower', aspect='equal')
            axH.quiver(fields["Xq"], fields["Yq"], fields["Mxq"], fields["Myq"], scale=60)
            axH.set_title("H + M")

        # overlay static bar: blue(静)→red(動)
        h, w = frame.shape[:2]
        r = int(np.clip((1 - (A+1)/2) * 255, 0, 255))
        b = int(np.clip(((A+1)/2) * 255, 0, 255))
        import cv2
        cv2.rectangle(frame, (0,0), (w,6), (r,50,b), -1)
        cv2.imshow("Webcam (Airfeel skeleton)", frame)

        plt.pause(0.01)
        k = (cv2.waitKey(1) & 0xFF)
        if k == ord('q'):
            break
        if k == ord('s'):
            _ensure_out()
            if rtonly:
                figR.savefig("out/shot_R_only.png", dpi=180)
            else:
                fig.savefig("out/shot_tripanel.png", dpi=180)
            print("Saved snapshots to ./out/")

    cap.release()
    import cv2
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="run without webcam (synthetic sequence)")
    parser.add_argument("--rtonly", action="store_true", help="show only R(x,y) in realtime")
    args = parser.parse_args()
    if args.demo:
        demo_loop(rtonly=args.rtonly)
    else:
        realtime_loop(rtonly=args.rtonly)
