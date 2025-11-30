# scouter_overlay.py
import math
import time

import cv2
import numpy as np


def _ema(prev, x, a=0.2):
    return (1-a)*prev + a*x if prev is not None else x

def _color_by_power(p):
    # p: 0..1 で想定。色は緑→黄→赤
    if p < 0.33:  return ( 80, 255,  80)
    if p < 0.66:  return ( 40, 210, 255)
    return ( 50,  80, 255)

class ScouterHUD:
    """
    Dragon Ball スカウター風 HUD
    .render(frame, feats, p_motion) を毎フレーム呼ぶだけ
    feats 必須キー:
      gaze_forward, gaze_vec=(gx,gy), motion_strength, onset, motion_ext
    """
    def __init__(self, w=640, h=480):
        self.w, self.h = w, h
        self.t0 = time.time()
        self.sP = None      # smoothed power
        self.sG = None      # smoothed gaze_forward
        self.lock = 0.0     # lock-on 0..1
        self.trail = []     # gaze軌跡
        self.trail_max = 30
        self.hud_alpha = 0.22

    def _visor(self, img):
        # 半透明バイザー
        overlay = img.copy()
        poly = np.array([
            [0,0],[self.w,0],[self.w,int(self.h*0.25)],
            [int(self.w*0.60),int(self.h*0.33)],[0,int(self.h*0.33)]
        ], np.int32)
        cv2.fillPoly(overlay, [poly], (40, 200, 40))
        cv2.addWeighted(overlay, self.hud_alpha, img, 1-self.hud_alpha, 0, dst=img)

        # 中央に水平スケール線
        y = int(self.h*0.12); step = 40
        for x in range(0, self.w, step):
            cv2.line(img, (x,y), (x+step//2,y), (60,255,60), 1, cv2.LINE_AA)
        cv2.line(img, (0,y), (self.w,y), (60,255,60), 1, cv2.LINE_AA)

    def _bars(self, img, p, g, flow):
        # 左上: Power Level
        power = int(round(9000 * p))  # お約束
        c = _color_by_power(p)
        cv2.putText(img, f"POWER={power}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2, cv2.LINE_AA)

        # ゲージ
        x0, y0, w, h = 20, 50, 260, 16
        cv2.rectangle(img, (x0,y0), (x0+w,y0+h), (60,255,60), 1)
        cv2.rectangle(img, (x0,y0), (x0+int(w*p), y0+h), c, -1)
        cv2.rectangle(img, (x0,y0), (x0+w,y0+h), (20,40,20), 1)

        # 右上: meta readouts
        cv2.putText(img, f"gaze={g:.2f}  flow={flow:.2f}", (self.w-360, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,200), 2, cv2.LINE_AA)

    def _reticle(self, img, feats):
        # 顔中央近辺を想定して画面中央にレティクル
        cx, cy = self.w//2, self.h//2
        r = 68
        cv2.circle(img, (cx,cy), r, (0,255,140), 1, cv2.LINE_AA)
        cv2.circle(img, (cx,cy), 2, (0,255,140), -1, cv2.LINE_AA)
        for a in range(0,360,45):
            x = int(cx + r*math.cos(math.radians(a)))
            y = int(cy + r*math.sin(math.radians(a)))
            cv2.line(img, (x,y), (int((x+cx)/2), int((y+cy)/2)), (0,255,140), 1, cv2.LINE_AA)

        # 目線ベクトル矢印
        gx, gy = feats.get("gaze_vec",(0.0,0.0))
        tip = (int(cx + gx*120), int(cy + gy*120))
        cv2.arrowedLine(img, (cx,cy), tip, (0,255,255), 2, tipLength=0.25)

        # 目線軌跡
        self.trail.append(tip);
        if len(self.trail) > self.trail_max: self.trail.pop(0)
        for i in range(1, len(self.trail)):
            a = self.trail[i-1]; b = self.trail[i]
            alpha = i / self.trail_max
            col = (0, int(255*(1-alpha)), int(255*alpha))
            cv2.line(img, a, b, col, 1, cv2.LINE_AA)

    def _lock_on(self, img, p):
        # パワーが高いとロックオン進行
        target = 1.0 if p > 0.6 else 0.0
        self.lock = _ema(self.lock, target, a=0.15)
        if self.lock < 0.05: return
        # ロック枠
        s = int(40 + 40*self.lock)
        cx, cy = self.w//2, self.h//2
        col = (0, 255, 255) if self.lock < 0.99 else (0, 80, 255)
        cv2.rectangle(img, (cx-s,cy-s), (cx+s,cy+s), col, 2, cv2.LINE_AA)
        if self.lock > 0.99:
            cv2.putText(img, "LOCK", (cx-s, cy-s-10), cv2.FONT_HERSHEY_SIMPLEX, .7, col, 2, cv2.LINE_AA)

    def render(self, frame_bgr, feats, p_motion):
        h, w = frame_bgr.shape[:2]
        self.h, self.w = h, w
        img = frame_bgr

        # スムージング
        self.sP = _ema(self.sP, float(np.clip(p_motion, 0.0, 1.0)), a=0.25)
        self.sG = _ema(self.sG, float(np.clip(feats.get("gaze_forward",0.5),0,1)), a=0.25)

        # 描画
        self._visor(img)
        self._bars(img, self.sP, self.sG, feats.get("motion_strength",0.0))
        self._reticle(img, feats)
        self._lock_on(img, self.sP)

        # 右下の小レーダ（onset / v-mot）
        onset = float(feats.get("onset",0.0))
        vmo   = float(feats.get("motion_ext",0.0))
        x0,y0 = w-150, h-120
        cv2.rectangle(img,(x0,y0),(x0+130,y0+100),(60,255,60),1)
        cv2.putText(img,"RADAR",(x0+24,y0-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(60,255,60),1,cv2.LINE_AA)
        # onset bar
        cv2.putText(img,"on", (x0+6,y0+26), 0, 0.5, (0,255,0),1,cv2.LINE_AA)
        cv2.rectangle(img,(x0+30,y0+16),(x0+120,y0+32),(0,120,0),1)
        cv2.rectangle(img,(x0+30,y0+16),(x0+30+int(90*onset),y0+32),(0,255,0),-1)
        # v-mot bar
        cv2.putText(img,"vm", (x0+6,y0+56), 0, 0.5, (0,255,255),1,cv2.LINE_AA)
        cv2.rectangle(img,(x0+30,y0+46),(x0+120,y0+62),(0,120,120),1)
        cv2.rectangle(img,(x0+30,y0+46),(x0+30+int(90*min(1.0,vmo)),y0+62),(0,255,255),-1)

        # 下帯のテキスト（Scouter風）
        cv2.putText(img, "SCAN: SUBJECT-ID=0001  STATUS=LIVE  MODE=SCOUTER",
                    (20, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60,255,60), 2, cv2.LINE_AA)
        return img
