# -*- coding: utf-8 -*-
"""
camera_autostart_realtime.py — Popupなしで即起動して、
カメラ＋（システム音声 or マイク）を自動で測定し、
QuantumStillnessEngine にフィードするランチャー。

依存:
    pip install sounddevice soundfile numpy opencv-python aubio  # aubio推奨（無ければ簡易オンセットに自動フォールバック）
    # Windowsでシステム音キャプチャ: sounddevice が WASAPI ループバック対応

使い方:
    # とりあえず自動（WASAPIループバック>マイクの順で検出）640x480
    python camera_autostart_realtime.py

    # 解像度指定
    python camera_autostart_realtime.py --width 1280 --height 720

    # 強制的にマイクから（WASAPIが使えない/不要なとき）
    python camera_autostart_realtime.py --audio-source mic

    # 強制的にループバック（対応環境のみ）
    python camera_autostart_realtime.py --audio-source loopback

メモ:
    - プレステホン等のオーディオデバイスだとループバック名が異なる場合があります。
    - aubio が見つからない場合は、RMSベースの簡易オンセットに切替。
"""

from __future__ import annotations

import argparse
import platform
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# === 音まわり（sounddeviceで取り込み、aubioでオンセット/テンポ推定） ===
try:
    import sounddevice as sd
    import soundfile as sf  # noqa: F401 (両者の組合わせでsdがWASAPI安定)
    SD_AVAILABLE = True
except Exception as e:
    print("[warn] sounddeviceが使えません:", e)
    SD_AVAILABLE = False

try:
    import aubio  # 高品質オンセット/ビート
    AUBIO_AVAILABLE = True
except Exception:
    AUBIO_AVAILABLE = False

# === ワイ理論エンジン側（既存実装を利用） ===
from base_scripts.qstillness_unified_plus import (QuantumStillnessEngine,
                                                  SimParams, delta_to_angle)


# ---- オーディオ計測 ----
@dataclass
class AudioConfig:
    samplerate: int = 48000
    blocksize: int = 1024
    channels: int = 2
    source: str = "auto"  # auto | loopback | mic

class AudioMeter:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        self.stream: Optional[sd.InputStream] = None
        self.rms_smooth = 0.0
        self.onset_val = 0.0
        self.beat_flag = 0.0
        self.bpm = 0.0
        self._aubio_init_done = False
        self._o_source: Optional[aubio.onset] = None
        self._b_source: Optional[aubio.tempo] = None
        self._aubio_hop = 512

    # === デバイス選択（WASAPIループバック優先） ===
    def _pick_device(self) -> Tuple[int, dict]:
        dev = None
        stream_kwargs = {}
        if platform.system() == "Windows" and self.cfg.source in ("auto", "loopback"):
            try:
                sd.default.hostapi = None  # auto
                wasapi = None
                for i in range(sd.hostapi_count()):
                    h = sd.query_hostapi(i)
                    if "WASAPI" in h.get("name", ""):
                        wasapi = i
                        break
                if wasapi is not None:
                    sd.default.hostapi = wasapi
                # ループバック対応デバイスを探す
                for i in range(sd.query_hostapi(sd.default.hostapi)["deviceCount"]):
                    di = sd.query_devices(i)
                    if di.get("hostapi") == sd.default.hostapi and di.get("maxOutputChannels", 0) > 0:
                        # 出力デバイスのloopbackとして開ける
                        dev = i
                        break
                if dev is not None:
                    stream_kwargs = dict(dtype="float32", samplerate=self.cfg.samplerate,
                                         channels=self.cfg.channels, blocksize=self.cfg.blocksize,
                                         dither_off=True, extra_settings=sd.WasapiSettings(loopback=True))
            except Exception as e:
                print("[warn] WASAPIループバック選択に失敗:", e)
                dev = None

        # マイク fallback
        if dev is None:
            print("[info] マイク入力を使用します")
            stream_kwargs = dict(dtype="float32", samplerate=self.cfg.samplerate,
                                 channels=self.cfg.channels, blocksize=self.cfg.blocksize)
        return (dev if dev is not None else sd.default.device, stream_kwargs)

    def _audio_cb(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            # バッファ遅延など
            pass
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            _ = None

    def start(self):
        if not SD_AVAILABLE:
            print("[error] sounddevice が無いためAudioMeterは無効です")
            return False
        dev, kwargs = self._pick_device()
        self.stream = sd.InputStream(device=dev, callback=self._audio_cb, **kwargs)
        self.stream.start()
        # aubio初期化（あれば）
        if AUBIO_AVAILABLE and not self._aubio_init_done:
            self._aubio_init_done = True
            self._o_source = aubio.onset("default", self.cfg.blocksize, self._aubio_hop, self.cfg.samplerate)
            self._b_source = aubio.tempo("default", self.cfg.blocksize, self._aubio_hop, self.cfg.samplerate)
        return True

    def stop(self):
        try:
            if self.stream is not None:
                self.stream.stop(); self.stream.close()
        finally:
            self.stream = None

    def step(self):
        """キューから溜まった分だけ処理して、onset/beat/rmsを更新"""
        got = False
        while True:
            try:
                buf = self.q.get_nowait()
            except queue.Empty:
                break
            got = True
            mono = np.mean(buf, axis=1)
            # RMS（平滑化）
            rms = np.sqrt(np.mean(mono**2))
            self.rms_smooth = 0.92 * self.rms_smooth + 0.08 * rms
            if AUBIO_AVAILABLE and self._o_source is not None:
                fbuf = mono.astype(np.float32)
                on = self._o_source(fbuf)
                bt = self._b_source(fbuf)
                self.onset_val = float(on[0]) if isinstance(on, np.ndarray) else float(on)
                if bt is not None and np.sum(bt) > 0:
                    # ビートが取れたフレームは1.0
                    self.beat_flag = 1.0
                    # 推定テンポ
                    self.bpm = float(self._b_source.get_bpm()) if hasattr(self._b_source, "get_bpm") else self.bpm
                else:
                    self.beat_flag = max(0.0, 0.85 * self.beat_flag - 0.02)
            else:
                # 簡易オンセット: RMSが移動平均を一定比率で超えたらonset
                ratio = (rms + 1e-6) / (self.rms_smooth + 1e-6)
                self.onset_val = float(np.clip((ratio - 1.08) * 5.0, 0.0, 1.0))
                self.beat_flag = 1.0 if self.onset_val > 0.7 else max(0.0, 0.85 * self.beat_flag - 0.02)
        return got

# ---- ランタイム（GUIなし） ----

def run_autostart(width: int, height: int, audio_source: str = "auto"):
    # カメラ
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print("[error] カメラを開けませんでした")
        return

    # 音声
    am = AudioMeter(AudioConfig(source=audio_source)) if SD_AVAILABLE else None
    if am is not None:
        am.start()

    # エンジン
    eng = QuantumStillnessEngine(SimParams(T=999999, jitter_std=0.05))

    # 視線・フローの軽量計測（外部ライブラリに依存しない簡易実装）
    prev_gray = None
    blink_hist = []

    step = 0
    print("[info] 起動しました。ウィンドウ選択後、q で終了します。")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # === 音 ===
            onset = 0.0; beat_flag = 0.0; loud = 0.0
            if am is not None:
                am.step()
                onset = float(am.onset_val)
                beat_flag = float(am.beat_flag)
                loud = float(am.rms_smooth)

            # === 視線（超簡易: 顔中心への向き→forward、ここは実装自由度高） ===
            #    実運用では MediaPipe Face Mesh 等で正確な視線ベクトルを
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            # 中央に向いていると仮定（デモ用）
            gaze_forward = 0.85
            gaze_vec = np.array([0.0, 0.0], dtype=np.float32)

            # === 瞬目（ダミー: ランダム微変動） ===
            blink_rate = max(0.0, min(1.0, 0.2 + 0.02 * np.random.randn()))
            blink_hist.append(blink_rate)
            if len(blink_hist) > 30:
                blink_hist.pop(0)

            # === オプティカルフロー（強度だけ） ===
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_strength = 0.0
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                motion_strength = float(np.clip(np.mean(np.linalg.norm(flow, axis=2)) * 10.0, 0.0, 1.0))
            prev_gray = gray

            # === V-JEPA外部モーションのダミー（将来の差し替えポイント） ===
            motion_ext = max(0.0, motion_strength - 0.25) * 0.7

            # === フィーチャ合成 ===
            feats = dict(
                gaze_vec=gaze_vec,
                gaze_forward=gaze_forward,
                blink_rate=blink_rate,
                motion_strength=motion_strength,
                motion_ext=motion_ext,
                onset=onset,
                beat_flag=beat_flag,
                loud=loud,
            )

            # === 矢の微調整（元GUI版と同じ構造） ===
            k_gx, k_gy = 0.40, 0.40
            k_motion_int, k_motion_ext = 0.9, 0.6

            gaze_bias = np.clip(k_gx * feats["gaze_vec"][0] + k_gy * (-feats["gaze_vec"][1]), -0.8, 0.8)
            delta_boost = np.tanh(
                k_motion_int * feats["motion_strength"] +
                k_motion_ext * feats["motion_ext"] +
                gaze_bias
            )
            angle = delta_to_angle(delta_boost, k_theta=0.9)
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(2); qc.ry(angle, 0)
            eng._sv = eng._sv.evolve(qc)

            # === 外界ONトリガ ===
            if feats["beat_flag"] > 0.5 or feats["onset"] > 0.6:
                eng.side.external_input = 1
            if feats["gaze_forward"] > 0.8 and feats["motion_strength"] < 0.15:
                eng.side.external_input = 1

            # === 緊張・驚き ===
            eng.side.tension += 0.10 * (1.0 - feats["gaze_forward"])  # avert
            eng.side.tension += 0.05 * max(0.0, feats["blink_rate"] - 0.25)
            eng.side.wonder = min(1.0, eng.side.wonder + 0.05 * feats["gaze_forward"])

            # === 1ステップ ===
            eng._step(step)
            p_motion = eng._prob_motion()
            step += 1

            # === オーバーレイ ===
            disp = frame.copy()
            cv2.putText(disp, f"P(Motion)={p_motion:.3f}", (20, 40), 0, 1, (255,0,0), 2)
            cv2.putText(disp, f"gaze={feats['gaze_forward']:.2f}  blink/s={feats['blink_rate']:.2f}", (20, 75), 0, .8, (0,200,255), 2)
            cv2.putText(disp, f"flow={feats['motion_strength']:.2f}  onset={feats['onset']:.2f}  beat={feats['beat_flag']:.2f}", (20, 105), 0, .8, (0,255,0), 2)
            if am is not None and am.bpm:
                cv2.putText(disp, f"BPM~{am.bpm:.1f}", (20, 135), 0, .8, (255,255,0), 2)

            # 簡易の矢印（中央→視線方向）
            tip = (int(cx + feats["gaze_vec"][0] * 120), int(cy + feats["gaze_vec"][1] * 120))
            cv2.arrowedLine(disp, (cx, cy), tip, (0,255,255), 2, tipLength=0.25)

            cv2.imshow("Realtime (Autostart)", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if am is not None:
            am.stop()
        cv2.destroyAllWindows()


# ---- CLI ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--audio-source", choices=["auto", "loopback", "mic"], default="auto",
                    help="WindowsはWASAPIループバックを優先（auto）。非対応ならマイクへフォールバック")
    args = ap.parse_args()

    run_autostart(args.width, args.height, args.audio_source)


if __name__ == "__main__":
    main()
