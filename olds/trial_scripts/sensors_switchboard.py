# sensors_switchboard.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time

import cv2
import numpy as np

# ===== Optional deps (lazy) =====
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
except Exception:
    mp_face = None

# Essentia/V-JEPA は無くても走るようにする（必要時のみ import）
def _safe_import_essentia():
    try:
        import essentia.standard as es  # noqa
        return True
    except Exception:
        return False

# 目線＆「ま」
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
        self.ema_motion = 0.0

    @staticmethod
    def _ear(eye_pts: np.ndarray) -> float:
        A = np.linalg.norm(eye_pts[1]-eye_pts[5])
        B = np.linalg.norm(eye_pts[2]-eye_pts[4])
        C = np.linalg.norm(eye_pts[0]-eye_pts[3])
        return (A+B)/(2.0*C + 1e-8)

    def __call__(self, frame_bgr: np.ndarray, motion_strength: float):
        h, w = frame_bgr.shape[:2]
        # デフォフォールバック
        def _fallback():
            self.ema_motion = 0.9*self.ema_motion + 0.1*motion_strength
            return dict(
                gaze_forward=0.5, blink_rate=0.0, ma_score=float(np.clip(1.0 - self.ema_motion, 0.0, 1.0)),
                gaze_vec=(0.0, 0.0)
            )
        if not self.face:
            return _fallback()
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.face.process(rgb)
        except Exception:
            return _fallback()
        if not res.multi_face_landmarks:
            return _fallback()

        lm = res.multi_face_landmarks[0].landmark
        left_idx  = [33,160,158,133,153,144]
        right_idx = [263,387,385,362,380,373]
        iris_l, iris_r = 468, 473

        L = np.array([[lm[i].x*w, lm[i].y*h] for i in left_idx],  dtype=np.float32)
        R = np.array([[lm[i].x*w, lm[i].y*h] for i in right_idx], dtype=np.float32)

        ear = (self._ear(L)+self._ear(R))/2.0
        EAR_THR = 0.19
        if ear < EAR_THR and not self.last_ear_low:
            self.blinks += 1; self.last_ear_low = True
        elif ear >= EAR_THR and self.last_ear_low:
            self.last_ear_low = False

        # 虹彩中心→目線ベクトル（目中心基準の正規化オフセット）
        if len(lm) > iris_r:
            iris = np.array([ ( (lm[iris_l].x+lm[iris_r].x)/2*w,
                                (lm[iris_l].y+lm[iris_r].y)/2*h ) ], dtype=np.float32)[0]
            Lc = L[[0,3]].mean(axis=0); Rc = R[[0,3]].mean(axis=0)
            eye_center = (Lc+Rc)/2.0
            span_vec = Rc - Lc
            span = np.linalg.norm(span_vec) + 1e-6
            offset = (iris - eye_center) / span  # [-?, +?] 小さめ
            # forwardスコア: 中央ほど高い
            gaze_forward = float(np.clip(1.0 - 1.8*np.linalg.norm(offset), 0.0, 1.0))
            gaze_vec = (float(offset[0]), float(offset[1]))  # 右:+x, 下:+y（BGR座標系）
        else:
            gaze_forward, gaze_vec = 0.5, (0.0, 0.0)

        self.ema_motion = 0.9*self.ema_motion + 0.1*motion_strength
        ma_score = float(np.clip(0.7*gaze_forward + 0.3*(1.0 - self.ema_motion), 0.0, 1.0))

        now = time.time(); dt = max(now - self.prev_time, 1e-3)
        blink_rate = self.blinks / dt
        if dt > 20.0: self.prev_time = now; self.blinks = 0

        return dict(
            gaze_forward=gaze_forward, blink_rate=float(blink_rate),
            ma_score=ma_score, gaze_vec=gaze_vec
        )

# 切替ハブ
class SensorSwitchboard:
    """
    起動時のフラグで使うセンサーを選択。
    毎フレーム: step_features() を呼ぶと dict を返す。
    """
    def __init__(self, use_gaze=True, use_optflow=True,
                 use_essentia=False, audio_path:str|None=None,
                 use_vjepa=False, video_path:str|None=None,
                 target_T:int|None=None):
        self.use_gaze = use_gaze
        self.use_optflow = use_optflow
        self.use_essentia = use_essentia
        self.use_vjepa = use_vjepa
        self.prev_gray = None

        self.gaze_est = GazeMaEstimator() if use_gaze else None

        # オフライン系は事前に配列化（存在すれば）
        self.audio_series = None
        self.video_series = None
        self.t = 0
        self.T = int(target_T or 0)

        if use_essentia and audio_path:
            from qstillness_io_media import (
                align_media_to_T, extract_audio_features_with_essentia)
            a = extract_audio_features_with_essentia(audio_path)
            self.audio_series = a
            self.T = max(self.T, len(a["onset_strength"]))
        if use_vjepa and video_path:
            from qstillness_io_media import (
                align_media_to_T, extract_video_embeddings_with_vjepa)
            v = extract_video_embeddings_with_vjepa(video_path, fps=8)
            self.video_series = v
            self.T = max(self.T, len(v["motion_mag"]))

        self.aligned = None
        if self.T and (self.audio_series or self.video_series):
            from qstillness_io_media import align_media_to_T
            self.aligned = align_media_to_T(
                self.T, audio=self.audio_series, video=self.video_series
            )

    def step_features(self, frame_bgr: np.ndarray) -> dict:
        """
        毎ステップ返す dict:
          motion_strength, gaze_forward, blink_rate, ma_score, gaze_vec,
          onset, loudness, beat_flag, motion_ext
        """
        h, w = frame_bgr.shape[:2]
        out = {
            "motion_strength": 0.0,
            "gaze_forward": 0.5, "blink_rate": 0.0, "ma_score": 0.5, "gaze_vec": (0.0, 0.0),
            "onset": 0.0, "loudness": 0.0, "beat_flag": 0.0, "motion_ext": 0.0
        }

        # 1) optical-flow
        if self.use_optflow:
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                    out["motion_strength"] = float(np.mean(mag))/10.0
                self.prev_gray = gray
            except Exception:
                self.prev_gray = None

        # 2) gaze/ma
        if self.use_gaze and self.gaze_est:
            g = self.gaze_est(frame_bgr, out["motion_strength"])
            out.update(g)

        # 3) offline audio/video (aligned)
        if self.aligned and self.T:
            idx = min(self.t, self.T-1)
            out["onset"] = float(self.aligned["onset_strength"][idx])
            out["loudness"] = float(self.aligned["loudness"][idx])
            out["beat_flag"] = float(self.aligned["beat_flag"][idx])
            out["motion_ext"] = float(self.aligned["motion_mag"][idx])

        self.t += 1
        return out
