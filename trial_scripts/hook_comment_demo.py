# -*- coding: utf-8 -*-
"""
hook_comment_demo.py
カメラ処理のメインループ内で Commentator を呼ぶ場所だけのデモ。
"""
import cv2 as cv
from commentator import Commentator

# ダミー信号
A, I, policy = 0.9, 0.6, "FACE"
state, still = "MOTION", 0.2
f_motion=f_eyedir=f_blink=f_mouth=f_tong=0.5
fps=30.0

cap = cv.VideoCapture(0)
comm = Commentator(mode="overlay")  # "hud" でもOK

while True:
    ok, frame = cap.read()
    if not ok: break

    # === 毎フレーム: 値を更新
    comm.update(state=state, A=A, I=I, policy=policy, still=still,
                f_motion=f_motion, f_eyedir=f_eyedir, f_blink=f_blink,
                f_mouth=f_mouth, f_tong=f_tong, fps=fps)

    # === 表示
    frame = comm.draw_on_frame(frame)  # overlay時
    comm.show_hud()                    # hud時

    cv.imshow("demo", frame)
    k = cv.waitKey(1) & 0xFF
    if k in (27, ord('q')):
        break
    elif k in (ord('c'), ord('C')):
        comm.toggle()

cap.release()
cv.destroyAllWindows()
print("[INFO] demo done")
