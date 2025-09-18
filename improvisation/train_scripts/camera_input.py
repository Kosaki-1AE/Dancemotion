# -*- coding: utf-8 -*-
# Realtime camera -> responsibility vector (gaze + motion) -> QuantumStillnessEngine

# ===== 追加インポート =====
import time

import cv2
import numpy as np

from base_scripts.qstillness_unified_plus import QuantumStillnessEngine, SimParams
from .scouter_overlay import ScouterHUD  # ← 先頭に追加

# ===== エンジン初期化（callableにしない） =====
eng = QuantumStillnessEngine(SimParams(T=999999))  # 長く回せるように
try:
    from .scouter_overlay import ScouterHUD
except Exception:
    try:
        from scouter_overlay import ScouterHUD
    except Exception:
        class ScouterHUD:
            def __init__(self, *a, **k): pass
            def update(self, **k): pass
            def apply(self, frame): return frame

# ===== MediaPipe（無ければ自動フォールバック） =====
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
except Exception:
    mp_face = None  # 未インストールでも落ちない

def run_camera_session(source=0, width=1280, height=720, window="Dancemotion — Camera"):
    cap = cv2.VideoCapture(source)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    hud = ScouterHUD(title="SCOUTER")
    t0 = time.time(); n=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        n += 1
        fps = n / max(1e-6, time.time()-t0)
        hud.update(fps=fps, energy=0.5, still=0.33)  # ← 実スコアに差し替え可
        frame = hud.apply(frame)
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
    cap.release(); cv2.destroyAllWindows()

# ===== 視線/瞬目/「ま」を推定する軽量クラス =====
class GazeMaEstimator:
    def __init__(self):
        self.face = (
            mp_face.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5
            ) if mp_face else None
        )
        self.prev_time = time.time()
        self.blinks = 0
        self.last_ear_low = False
        self.ema_motion = 0.0   # 動きの滑らかさ（“ま”検出に）

    @staticmethod
    def _ear(eye_pts):
        # Eye Aspect Ratio（6点: 横0-3, 縦1-5/2-4）
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (A + B) / (2.0 * C + 1e-8)

    def __call__(self, frame_bgr, motion_strength):
        """返り: gaze_forward(0..1), blink_rate(Hz), ma_score(0..1)"""
        h, w = frame_bgr.shape[:2]

        # Mediapipe無し or 失敗時 → フォールバック
        if not self.face:
            self.ema_motion = 0.9 * self.ema_motion + 0.1 * motion_strength
            return 0.5, 0.0, float(np.clip(1.0 - self.ema_motion, 0.0, 1.0))

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.face.process(rgb)
        except Exception:
            self.ema_motion = 0.9 * self.ema_motion + 0.1 * motion_strength
            return 0.5, 0.0, float(np.clip(1.0 - self.ema_motion, 0.0, 1.0))

        if not res.multi_face_landmarks:
            self.ema_motion = 0.9 * self.ema_motion + 0.1 * motion_strength
            return 0.5, 0.0, float(np.clip(1.0 - self.ema_motion, 0.0, 1.0))

        lm = res.multi_face_landmarks[0].landmark

        # 目のランドマークindex（MediaPipe）
        left_idx  = [33, 160, 158, 133, 153, 144]
        right_idx = [263, 387, 385, 362, 380, 373]
        iris_l, iris_r = 468, 473  # refined_landmarks 有効時の虹彩中心

        L = np.array([[lm[i].x * w, lm[i].y * h] for i in left_idx],  dtype=np.float32)
        R = np.array([[lm[i].x * w, lm[i].y * h] for i in right_idx], dtype=np.float32)

        # 瞬目検出（EAR）
        ear = (self._ear(L) + self._ear(R)) / 2.0
        EAR_THR = 0.19
        if ear < EAR_THR and not self.last_ear_low:
            self.blinks += 1
            self.last_ear_low = True
        elif ear >= EAR_THR and self.last_ear_low:
            self.last_ear_low = False

        # 視線（虹彩の中心が目の中央に近いほど高スコア）
        if len(lm) > iris_r:
            iris = np.array([
                ( (lm[iris_l].x + lm[iris_r].x) / 2.0 * w,
                  (lm[iris_l].y + lm[iris_r].y) / 2.0 * h )
            ], dtype=np.float32)[0]
            Lc = L[[0, 3]].mean(axis=0); Rc = R[[0, 3]].mean(axis=0)
            eye_center = (Lc + Rc) / 2.0
            eye_span = np.linalg.norm(Rc - Lc) + 1e-6
            d = np.linalg.norm(iris - eye_center) / eye_span
            gaze_forward = float(np.clip(1.0 - d * 1.8, 0.0, 1.0))
        else:
            gaze_forward = 0.5

        # “ま”：動きが小さく視線が安定しているほど高い
        self.ema_motion = 0.9 * self.ema_motion + 0.1 * motion_strength
        ma_score = float(np.clip(0.7 * gaze_forward + 0.3 * (1.0 - self.ema_motion), 0.0, 1.0))

        # 瞬目レート（簡易）
        now = time.time()
        dt = max(now - self.prev_time, 1e-3)
        blink_rate = self.blinks / dt
        if dt > 20.0:  # 定期リセット
            self.prev_time = now
            self.blinks = 0

        return gaze_forward, float(blink_rate), ma_score


def main():
    # run_session(...) の先頭で HUD 準備
    hud = ScouterHUD()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けませんでした")
        return

    # 解像度を固定（フレームサイズ揺れ対策：必要なら調整）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_gray = None
    step = 0
    gaze_est = GazeMaEstimator()

    while True:
        # ---- フレーム取得 ----
        ret, frame = cap.read()
        if not ret:
            break

        # ---- motion_strength を算出（Optical Flow 平均大きさ）----
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None or prev_gray.shape != gray.shape:
                motion_strength = 0.0
            else:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_strength = float(np.mean(mag)) / 10.0  # スケールは環境で微調整
            prev_gray = gray
        except Exception as e:
            # 一時的に失敗しても継続
            motion_strength = 0.0
            prev_gray = None

        # ---- 視線/“ま”の推定 ----
        try:
            gaze_forward, blink_rate, ma_score = gaze_est(frame, motion_strength)
        except Exception:
            gaze_forward, blink_rate, ma_score = 0.5, 0.0, 0.5

        # ---- 責任ベクトルへ反映（安全域）----
        w_b = 0.05  # 瞬目→緊張の寄与
        eng.side.tension += 0.10 * (1.0 - gaze_forward)            # 逸らすほど緊張↑
        eng.side.tension += w_b * max(0.0, blink_rate - 0.25)      # 瞬目多めで緊張↑
        eng.side.wonder  = min(1.0, eng.side.wonder + 0.05 * gaze_forward)

        # 視線が合って静かなら外界ON（“ま”で空気が凝る瞬間）
        eng.side.external_input = 1 if (gaze_forward > 0.8 and motion_strength < 0.15) else 0

        # ---- 1 ステップ進行 ----
        try:
            eng._step(step)
            p_motion = eng._prob_motion()
        except Exception:
            # 状態が飛んだときでも続行
            p_motion = 0.5
        step += 1

        # ---- 画面オーバーレイ表示 ----
        disp = frame.copy()
        # ループ内、disp を作る直前／直後で:
        disp = frame.copy()
        disp = hud.render(disp, feats, p_motion)   # ← ここ1行でスカウター化

        cv2.imshow("Realtime (Popup Launcher)", disp)
        cv2.putText(disp, f"Motion={motion_strength:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(disp, f"Gaze={gaze_forward:.2f}  Blink/s={blink_rate:.2f}  Ma={ma_score:.2f}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(disp, f"P(Motion)={p_motion:.3f}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Realtime Cam × QuantumStillness", disp)

        # qで終了 / 軽く間引き
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
