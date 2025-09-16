# -*- coding: utf-8 -*-
# camera_launcher_gui.py
# ぽちぽちGUIでセンサー選択してリアタイ実行（gaze / optical-flow / Essentia / V-JEPA）

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from qstillness_unified_plus import (QuantumStillnessEngine, SimParams,
                                     delta_to_angle)
from sensors_switchboard import SensorSwitchboard  # 前ターンで渡したファイル

# ===== 実行ループ（GUIで集めた設定を使って起動） =====
def run_session(use_gaze: bool, use_flow: bool, audio_path: str | None,
                video_path: str | None, width: int, height: int):

    use_ess = bool(audio_path)
    use_vj  = bool(video_path)

    # スイッチボード（必要なバックエンドだけ有効化）
    sw = SensorSwitchboard(
        use_gaze=use_gaze,
        use_optflow=use_flow,
        use_essentia=use_ess, audio_path=audio_path,
        use_vjepa=use_vj,     video_path=video_path
    )

    # 量子エンジン
    eng = QuantumStillnessEngine(SimParams(T=999999, jitter_std=0.05))

    # 感性ノブ（適宜調整）
    knobs = dict(
        k_onset=0.15, k_loud=0.10, onset_gate=0.6,
        k_avert=0.10, k_blink=0.05, k_wonder=0.05,
        k_gx=0.40, k_gy=0.40,
        k_motion_int=0.9,   # optical-flow
        k_motion_ext=0.6    # V-JEPA(外部) motion
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        messagebox.showerror("Camera", "カメラが開けませんでした")
        return

    step = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feats = sw.step_features(frame)

            # 目線ベクトルやモーションから“矢”に微調整
            gx, gy = feats["gaze_vec"]
            gaze_bias = np.clip(knobs["k_gx"]*gx + knobs["k_gy"]*(-gy), -0.8, 0.8)
            delta_boost = np.tanh(
                knobs["k_motion_int"]*feats["motion_strength"]
              + knobs["k_motion_ext"]*feats["motion_ext"]
              + gaze_bias
            )

            # 角度を少し与えておく（軽量）
            angle = delta_to_angle(delta_boost, k_theta=0.9)
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(2); qc.ry(angle, 0)
            eng._sv = eng._sv.evolve(qc)

            # 音イベントで外界ON
            if feats["beat_flag"] > 0.5 or feats["onset"] > knobs["onset_gate"]:
                eng.side.external_input = 1

            # 視線が合って静かな時も外界ON
            if feats["gaze_forward"] > 0.8 and feats["motion_strength"] < 0.15:
                eng.side.external_input = 1

            # 視線/瞬目→緊張・驚き
            eng.side.tension += knobs["k_avert"] * (1.0 - feats["gaze_forward"])
            eng.side.tension += knobs["k_blink"] * max(0.0, feats["blink_rate"] - 0.25)
            eng.side.wonder   = min(1.0, eng.side.wonder + knobs["k_wonder"]*feats["gaze_forward"])

            # 1ステップ
            eng._step(step)
            p_motion = eng._prob_motion()
            step += 1

            # 画面オーバーレイ
            disp = frame.copy()
            cv2.putText(disp, f"P(Motion)={p_motion:.3f}", (20, 40), 0, 1, (255,0,0), 2)
            cv2.putText(disp, f"gaze={feats['gaze_forward']:.2f}  blink/s={feats['blink_rate']:.2f}",
                        (20, 75), 0, .8, (0,200,255), 2)
            cv2.putText(disp, f"flow={feats['motion_strength']:.2f}  onset={feats['onset']:.2f}  v-mot={feats['motion_ext']:.2f}",
                        (20, 105), 0, .8, (0,255,0), 2)
            # 目線ベクトルの矢印
            h, w = disp.shape[:2]
            cx, cy = w//2, h//2
            tip = (int(cx + feats["gaze_vec"][0]*120), int(cy + feats["gaze_vec"][1]*120))
            cv2.arrowedLine(disp, (cx,cy), tip, (0,255,255), 2, tipLength=0.25)

            cv2.imshow("Realtime (Popup Launcher)", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ===== GUI =====
def pick_file(entry_widget, types, title):
    path = filedialog.askopenfilename(filetypes=types, title=title)
    if path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, path)

def main():
    root = tk.Tk()
    root.title("QuantumStillness: センサー選択ランチャー")
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=12)
    frm.grid(row=0, column=0, sticky="nsew")

    # チェックボックス
    var_gaze = tk.BooleanVar(value=True)
    var_flow = tk.BooleanVar(value=True)

    ttk.Checkbutton(frm, text="目線/Gaze + \"ま\"", variable=var_gaze).grid(row=0, column=0, sticky="w")
    ttk.Checkbutton(frm, text="オプティカルフロー（動き）", variable=var_flow).grid(row=1, column=0, sticky="w")

    # 音（Essentia）
    ttk.Label(frm, text="音声ファイル（任意 / Essentia）").grid(row=2, column=0, sticky="w", pady=(8,0))
    ent_audio = ttk.Entry(frm, width=48)
    ent_audio.grid(row=3, column=0, sticky="w")
    ttk.Button(frm, text="参照", command=lambda: pick_file(ent_audio,
        [("Audio", "*.mp3 *.wav *.flac *.ogg"), ("All", "*.*")], "音声ファイルを選択")
    ).grid(row=3, column=1, padx=6)

    # 映像（V-JEPA 代替パス）
    ttk.Label(frm, text="映像ファイル（任意 / V-JEPA）").grid(row=4, column=0, sticky="w", pady=(8,0))
    ent_video = ttk.Entry(frm, width=48)
    ent_video.grid(row=5, column=0, sticky="w")
    ttk.Button(frm, text="参照", command=lambda: pick_file(ent_video,
        [("Video", "*.mp4 *.mov *.avi *.mkv"), ("All", "*.*")], "映像ファイルを選択")
    ).grid(row=5, column=1, padx=6)

    # 解像度
    frm2 = ttk.Frame(frm)
    frm2.grid(row=6, column=0, columnspan=2, pady=(10,0), sticky="w")
    ttk.Label(frm2, text="幅").grid(row=0, column=0, padx=(0,4))
    ent_w = ttk.Entry(frm2, width=6); ent_w.insert(0, "640"); ent_w.grid(row=0, column=1)
    ttk.Label(frm2, text="高さ").grid(row=0, column=2, padx=(10,4))
    ent_h = ttk.Entry(frm2, width=6); ent_h.insert(0, "480"); ent_h.grid(row=0, column=3)

    # 実行ボタン
    def on_run():
        try:
            w = int(ent_w.get() or "640"); h = int(ent_h.get() or "480")
        except Exception:
            messagebox.showerror("入力エラー", "幅/高さは整数で指定してください"); return

        root.withdraw()  # ウィンドウを隠す
        try:
            run_session(
                use_gaze=var_gaze.get(),
                use_flow=var_flow.get(),
                audio_path=(ent_audio.get().strip() or None),
                video_path=(ent_video.get().strip() or None),
                width=w, height=h
            )
        finally:
            root.deiconify()

    ttk.Button(frm, text="▶ 実行（qで終了）", command=on_run).grid(row=7, column=0, pady=12, sticky="w")

    ttk.Label(frm, text="Tips: 実行中はカメラウィンドウで q を押すと終了します。").grid(row=8, column=0, columnspan=2, sticky="w")

    root.mainloop()


if __name__ == "__main__":
    main()
