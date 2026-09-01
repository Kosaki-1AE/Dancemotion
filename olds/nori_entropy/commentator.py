# -*- coding: utf-8 -*-
"""
commentator.py
軽量な実況コメントモジュール。映像に重ねる or 別HUDに出すのを選べる。

使用例:
    from commentator import Commentator
    comm = Commentator(mode="overlay")   # "overlay" or "hud"
    ...
    comm.update(state=state, A=A, I=I, policy=policy, still=still,
                f_motion=f_motion, f_eyedir=f_eyedir, f_blink=f_blink,
                f_mouth=f_mouth, f_tong=f_tong, fps=fps_ema)
    frame = comm.draw_on_frame(frame)    # overlay のとき
    comm.show_hud()                      # hud のとき
"""
import cv2 as cv
import numpy as np
from collections import deque

class StreamCommentator:
    def __init__(self, mode:str="overlay", max_lines:int=4, smooth:float=0.2, hud_size=(480,160)):
        assert mode in ("overlay", "hud")
        self.mode = mode
        self.max_lines = max_lines
        self.smooth = smooth
        self.hud_size = hud_size
        self.enabled = True
        self._last_tags = []
        self._A = None
        self._I = None
        self._still = None
        self._fps = None
        self.window_name = "Nori-Comment"

    @staticmethod
    def _tags_from_signals(state, A, I, policy, still, f_motion, f_eyedir, f_blink, f_mouth, f_tong, fps):
        tags = []
        # 意識
        if A > 0.80: tags.append("集中")
        elif A < 0.30: tags.append("俯瞰")
        else: tags.append("切替中")
        # 意思
        if I is not None:
            if I > 0.70: tags.append("意思強")
            elif I < 0.30: tags.append("様子見")
            else: tags.append("迷い少")
        # 方針
        if policy is not None:
            tags.append("FACE重視" if policy=="FACE" else "AV重視")
        # 状態
        if state == "STILL":
            tags.append("静止安定")
        else:
            if f_motion < 0.20 and f_blink > 0.80: tags.append("静止目前")
            elif f_motion > 0.60: tags.append("動き多め")
        # 注意
        if fps is not None and fps < 15: tags.append("処理重い→抑制中")
        if f_eyedir < 0.3 and (policy == "FACE"): tags.append("目線ばらつき")
        if f_mouth < 0.3: tags.append("口周り静か")
        # サマリ
        if I is None: I = 0.0
        if fps is None: fps = 0.0
        summary = f"A:{A:.2f} I:{I:.2f} still:{still:.2f} FPS:{fps:.1f}"
        return tags[:6], summary

    def update(self, **kwargs):
        # kwargs: state, A, I, policy, still, f_motion, f_eyedir, f_blink, f_mouth, f_tong, fps
        self.state  = kwargs.get("state", "MOTION")
        self.A      = float(kwargs.get("A", 0.5))
        self.I      = None if ("I" not in kwargs or kwargs["I"] is None) else float(kwargs["I"])
        self.policy = kwargs.get("policy", None)
        self.still  = float(kwargs.get("still", 0.0))
        self.f_motion = float(kwargs.get("f_motion", 0.0))
        self.f_eyedir = float(kwargs.get("f_eyedir", 0.0))
        self.f_blink  = float(kwargs.get("f_blink", 0.0))
        self.f_mouth  = float(kwargs.get("f_mouth", 0.0))
        self.f_tong   = float(kwargs.get("f_tong", 0.0))
        self.fps      = None if ("fps" not in kwargs or kwargs["fps"] is None) else float(kwargs["fps"])

        tags, summary = self._tags_from_signals(
            self.state, self.A, (self.I if self.I is not None else 0.0), self.policy, self.still,
            self.f_motion, self.f_eyedir, self.f_blink, self.f_mouth, self.f_tong, self.fps
        )

        # 1行目: タグ, 2行目: サマリ
        lines = [" / ".join(tags), summary]

        # スムージング: 数値だけEMA
        def _ema(prev, x, a): return x if prev is None else (a*x + (1-a)*prev)
        self._A    = _ema(self._A, self.A, self.smooth)
        self._I    = _ema(self._I, (self.I if self.I is not None else 0.0), self.smooth)
        self._still= _ema(self._still, self.still, self.smooth)
        self._fps  = _ema(self._fps, (self.fps if self.fps is not None else 0.0), self.smooth)

        self._last_tags = lines

    def draw_on_frame(self, frame):
        """overlayモードのときに、frameへコメントを描画"""
        if not self.enabled or self.mode != "overlay" or not self._last_tags:
            return frame
        y0 = 96
        for i, t in enumerate(self._last_tags):
            col = (50,255,255) if i==0 else (255,255,255)
            th  = 2 if i==0 else 1
            cv.putText(frame, t, (20, y0 + 22*i), cv.FONT_HERSHEY_SIMPLEX, 0.6, col, th, cv.LINE_AA)
        return frame

    def show_hud(self):
        """hudモードのときに、独立ウィンドウへコメントを描画"""
        if not self.enabled or self.mode != "hud" or not self._last_tags:
            return
        w, h = self.hud_size
        hud = np.zeros((h, w, 3), dtype=np.uint8)
        hud[:] = (32,32,32)
        cv.putText(hud, self._last_tags[0], (18, 56),  cv.FONT_HERSHEY_SIMPLEX, 0.65, (50,255,255), 2, cv.LINE_AA)
        cv.putText(hud, self._last_tags[1], (18, 88),  cv.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 1, cv.LINE_AA)
        cv.imshow(self.window_name, hud)

    def toggle(self):
        self.enabled = not self.enabled

    def set_mode(self, mode:str):
        assert mode in ("overlay", "hud")
        self.mode = mode
