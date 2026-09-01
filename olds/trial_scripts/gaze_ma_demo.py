# -*- coding: utf-8 -*-
"""
gaze_ma_demo.py — 内蔵カメラで「目線 + ま(Stillness)」をざっくり可視化する遊びスクリプト
依存: pip install opencv-python mediapipe numpy

起動:
    python gaze_ma_demo.py

操作:
    q/ESC : 終了
    c     : キャリブレーション（基準の目線を取り直し）
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2

try:
    import mediapipe as mp
except Exception as e:
    raise SystemExit("mediapipe が見つかりません。`pip install mediapipe` を実行してください。") from e

mp_draw = mp.solutions.drawing_utils
mp_face = mp.solutions.face_mesh

# 目のランドマーク番号（MediaPipe FaceMesh, refine_landmarks=True 前提）
# 左右は「本人の左右」基準。
LM = {
    "L_OUT": 33, "L_IN": 133, "L_UP": 159, "L_DN": 145,   # 左目（本人左）
    "R_OUT": 362, "R_IN": 263, "R_UP": 386, "R_DN": 374,  # 右目（本人右）
    "L_IRIS": [468, 469, 470, 471],
    "R_IRIS": [473, 474, 475, 476],
}

def _pt(landmarks, i, w, h):
    lm = landmarks[i]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def _center(pts: np.ndarray) -> np.ndarray:
    return np.mean(pts, axis=0)

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

@dataclass
class EMA:
    """指数移動平均（ノイズ抑え用）"""
    y: np.ndarray
    alpha: float

    def update(self, x: np.ndarray) -> np.ndarray:
        self.y = (1.0 - self.alpha) * self.y + self.alpha * x
        return self.y

def eye_metrics(landmarks, w, h) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """両目の中心/虹彩中心/開閉度/EAR を返す"""
    Lc = np.stack([_pt(landmarks, LM["L_OUT"], w, h), _pt(landmarks, LM["L_IN"], w, h)])
    Rc = np.stack([_pt(landmarks, LM["R_OUT"], w, h), _pt(landmarks, LM["R_IN"], w, h)])
    L_up, L_dn = _pt(landmarks, LM["L_UP"], w, h), _pt(landmarks, LM["L_DN"], w, h)
    R_up, R_dn = _pt(landmarks, LM["R_UP"], w, h), _pt(landmarks, LM["R_DN"], w, h)

    # 虹彩中心（refine_landmarks 有効時）
    L_iris = np.stack([_pt(landmarks, i, w, h) for i in LM["L_IRIS"]])
    R_iris = np.stack([_pt(landmarks, i, w, h) for i in LM["R_IRIS"]])
    L_iris_c = _center(L_iris); R_iris_c = _center(R_iris)

    L_eye_c = _center(Lc); R_eye_c = _center(Rc)
    eye_c = 0.5 * (L_eye_c + R_eye_c)
    iris_c = 0.5 * (L_iris_c + R_iris_c)

    # 開閉度（上下距離 / 幅） → 簡易EAR
    L_open = _norm(L_up - L_dn) / max(1e-6, _norm(Lc[0] - Lc[1]))
    R_open = _norm(R_up - R_dn) / max(1e-6, _norm(Rc[0] - Rc[1]))
    ear = 0.5 * (L_open + R_open)

    # 目線ベクトル（目中心→虹彩中心）を目幅で正規化（だいたい -1..1 くらい）
    eye_width = 0.5 * (_norm(Lc[0] - Lc[1]) + _norm(Rc[0] - Rc[1]))
    gaze_vec = (iris_c - eye_c) / max(1e-6, eye_width)

    return eye_c, gaze_vec, ear, eye_width

def draw_bar(img, x, y, w, h, val, label, color=(60, 220, 60)):
    """0..1 のバー描画"""
    val = float(np.clip(val, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), 1)
    cv2.rectangle(img, (x, y), (x + int(w * val), y + h), color, -1)
    cv2.putText(img, f"{label}: {val:.2f}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("カメラが開けませんでした（index=0）。Webカメラの接続/権限を確認してください。")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    face = mp_face.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # 虹彩ランドマーク
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # 平滑化フィルタ
    gaze_ema = EMA(np.zeros(2, dtype=np.float32), alpha=0.3)
    head_ema = EMA(np.zeros(2, dtype=np.float32), alpha=0.2)

    baseline_gaze = np.zeros(2, dtype=np.float32)  # キャリブレーション後の基準
    prev_gaze = None
    prev_center = None
    blink_state = 0.0
    t0 = time.time()
    nframe = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        nframe += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face.process(rgb)

        # デフォルト値（顔が無いとき）
        stillness = 0.5
        gaze_show = np.array([0.0, 0.0], dtype=np.float32)
        fps = nframe / max(1e-6, (time.time() - t0))

        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            eye_c, gaze_vec, ear, eye_w = eye_metrics(lm, W, H)

            # 平滑化 + 基準からの変化
            gaze_s = gaze_ema.update(gaze_vec)
            gaze_rel = gaze_s - baseline_gaze
            gaze_show = gaze_rel.copy()

            # 頭の動き量（両目中心の移動）
            head_s = head_ema.update(eye_c)
            head_speed = 0.0
            if prev_center is not None:
                head_speed = _norm(head_s - prev_center) / max(1.0, np.sqrt(W*W + H*H))
                head_speed *= 8.0  # 調整
            prev_center = head_s.copy()

            # 目線の変化量
            gaze_speed = 0.0
            if prev_gaze is not None:
                gaze_speed = float(_norm(gaze_rel - prev_gaze))
            prev_gaze = gaze_rel.copy()

            # 簡易ま(Stillness)合成: 0..1（1=静か）
            # - 頭の動きが小さいほど +
            # - 目線速度が小さいほど +
            # - まばたき時(ear小)は少し下げる
            blink = 1.0 if ear < 0.19 else 0.0  # 閾値は顔サイズ次第で調整
            blink_state = 0.9 * blink_state + 0.1 * blink  # ちらつき防止

            motion = np.clip(head_speed, 0.0, 1.0)
            eye_motion = np.clip(gaze_speed * 2.5, 0.0, 1.0)
            stillness = float(np.clip(1.0 - (0.6 * motion + 0.3 * eye_motion + 0.1 * blink_state), 0.0, 1.0))

            # 目線の可視化
            cx, cy = int(eye_c[0]), int(eye_c[1])
            gx = int(cx + 60 * gaze_rel[0])
            gy = int(cy + 60 * gaze_rel[1])
            cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
            cv2.arrowedLine(frame, (cx, cy), (gx, gy), (0, 255, 0), 2, tipLength=0.25)

            # 目の枠
            for i in (LM["L_OUT"], LM["L_IN"], LM["R_OUT"], LM["R_IN"]):
                p = _pt(lm, i, W, H).astype(int)
                cv2.circle(frame, tuple(p), 2, (200, 200, 50), -1)

        # HUD
        pad = 10
        cv2.putText(frame, "Gaze (x,y): [%.2f, %.2f]" % (gaze_show[0], gaze_show[1]),
                    (pad, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1, cv2.LINE_AA)
        cv2.putText(frame, "FPS: %.1f" % fps, (pad, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1, cv2.LINE_AA)
        draw_bar(frame, pad, 70, 200, 14, stillness, "Stillness", (60, 200, 255))

        cv2.imshow("Gaze + Ma (toy)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == ord('c'):
            baseline_gaze = gaze_ema.y.copy()  # 現在の目線を基準に

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
