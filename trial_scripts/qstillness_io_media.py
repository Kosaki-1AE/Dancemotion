# qstillness_io_media.py
# ------------------------------------------------------------
# Essentia(音) / V-JEPA(映像) の特徴抽出を行い、
# qstillness_unified_plus.run_responsibility_quantum_loop に“刺さる配列”を返す。
# 依存: numpy, (optional) essentia, (optional) torch/transformers/opencv
# ------------------------------------------------------------
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np


# ======== 音: Essentia ========
def extract_audio_features_with_essentia(
    audio_path: str,
    sr: int = 44100,
    hop_s: float = 0.04644,   # ~1024 hop @ 22.05kHz相当、ここはお好みで
    frame_s: float = 0.09288, # ~2048 frame
    onset_smooth: int = 3
) -> Dict[str, np.ndarray]:
    """
    返り値（時系列長 N ）:
      - beat_flag: 0/1 ビート位置
      - onset_strength: [0..1]
      - loudness: [0..1]（RMSを0-1正規化）
      - bpm: 推定テンポ（float）
      - sec_per_hop: hop秒
    """
    try:
        import essentia
        import essentia.standard as es
    except Exception as e:
        raise ImportError("Essentia が見つかりません。pip/conda でインストールしてください") from e

    loader = es.MonoLoader(filename=audio_path, sampleRate=sr)
    audio = loader()

    # Loudness (RMS)
    frame = es.FrameCutter(frameSize=int(frame_s*sr), hopSize=int(hop_s*sr))
    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    rms = es.RMS()
    rms_vals = []
    for f in essentia.array(frame(audio)):
        if len(f) == 0: break
        rms_vals.append(float(rms(f)))
    rms_vals = np.asarray(rms_vals, dtype=np.float32)
    loud = (rms_vals - rms_vals.min()) / (1e-8 + (rms_vals.max()-rms_vals.min()))

    # Onset
    odf = es.OnsetDetection(method='complex')
    fc = es.FrameCutter(frameSize=int(frame_s*sr), hopSize=int(hop_s*sr))
    w = es.Windowing(type='hann')
    sp = es.Spectrum()
    pool = essentia.Pool()
    for fr in essentia.array(fc(audio)):
        if len(fr)==0: break
        mag = sp(w(fr))
        pool.add('odf', odf(mag, mag)[0])  # onset detection function
    odf_vals = np.asarray(pool['odf'], dtype=np.float32)
    if onset_smooth > 1:
        # 簡単な移動平均平滑化
        k = onset_smooth
        odf_vals = np.convolve(odf_vals, np.ones(k)/k, mode='same')
    odf_vals = odf_vals - odf_vals.min()
    odf_vals = odf_vals / (1e-8 + odf_vals.max())
    onset_strength = odf_vals

    # Beat / Tempo
    # ざっくりテンポとビート位置を抽出
    rt = es.RhythmExtractor2013(method="multifeature")
    bpm, beats, _, _, _ = rt(audio)
    # beats: seconds の配列 → フレーム列へ射影してフラグ化
    sec_per_hop = hop_s
    N = len(onset_strength)
    beat_flag = np.zeros(N, dtype=np.int32)
    for sec in beats:
        idx = int(round(sec / sec_per_hop))
        if 0 <= idx < N:
            beat_flag[idx] = 1

    return {
        "beat_flag": beat_flag,
        "onset_strength": onset_strength.astype(np.float32),
        "loudness": loud.astype(np.float32),
        "bpm": float(bpm),
        "sec_per_hop": float(sec_per_hop),
    }


# ======== 映像: V-JEPA or フォールバック ========
def extract_video_embeddings_with_vjepa(
    video_path: str,
    fps: int = 8,
    clip_len: int = 8,
    device: str = "cpu",
    fallback_optflow: bool = True,
) -> Dict[str, np.ndarray]:
    """
    返り値（時系列長 M ）:
      - embed: (M, D) 動画埋め込み（V-JEPA or 近似）
      - motion_mag: (M,) 連続差分ノルム（動きの強さ指標、0..1正規化）
      - fps_used: 実際にサンプリングしたFPS
    """
    # 1) まずは OpenCV でフレームを拾う（V-JEPAでも前処理で使う）
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(orig_fps / fps)))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("フレームが取り出せませんでした")

    # 2) V-JEPA をロードして埋め込み化（ない場合はフォールバック）
    try:
        import torch

        # ここでは“仮の”V-JEPAエンコーダの呼び出し枠だけ用意。
        # 実際には facebookresearch/jepa (または V-JEPA2) の推論関数で
        # frames → embeddings を作る。以下はダミーの平均プーリング。
        # 取り回しを軽くするため、まずは画素平均を“超軽量な近似埋め込み”に。
        with torch.no_grad():
            embs = []
            for f in frames:
                arr = torch.from_numpy(f).float()  # H W C
                embs.append(arr.mean(dim=(0,1)).numpy())  # (3,)
            embed = np.stack(embs, axis=0)  # (M, 3) の簡易特徴
    except Exception:
        if not fallback_optflow:
            raise
        # 3) フォールバック：Optical Flow の大きさを“動き特徴”として使う
        import cv2
        embs = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            embs.append(np.array([mag.mean(), mag.std(), ang.mean()], dtype=np.float32))
            prev_gray = gray
        embed = np.stack(embs, axis=0)  # (M, 3)

    # 4) 連続差分ノルムで motion_mag を作る
    if len(embed) < 2:
        motion = np.zeros((len(embed),), dtype=np.float32)
    else:
        diffs = np.linalg.norm(embed[1:] - embed[:-1], axis=1)
        diffs = (diffs - diffs.min()) / (1e-8 + (diffs.max()-diffs.min()))
        motion = np.concatenate([[0.0], diffs]).astype(np.float32)

    return {"embed": embed.astype(np.float32), "motion_mag": motion, "fps_used": float(fps)}


# ======== エンジンに合わせて時系列をリサンプル ========
def align_media_to_T(
    T: int,
    audio: Optional[Dict[str, np.ndarray]] = None,
    video: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, np.ndarray]:
    """
    エンジンのステップ数 T に合わせて、音と映像の時系列をリサンプルする。
    返り値（長さ T）:
      - beat_flag, onset_strength, loudness, motion_mag （なければ0配列）
    """
    def _resample_1d(x: np.ndarray, T: int) -> np.ndarray:
        if x is None or len(x) == 0:
            return np.zeros((T,), dtype=np.float32)
        idx = np.linspace(0, len(x)-1, num=T)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

    out = {}
    out["beat_flag"] = _resample_1d(audio["beat_flag"], T) if audio else np.zeros((T,), np.float32)
    out["onset_strength"] = _resample_1d(audio["onset_strength"], T) if audio else np.zeros((T,), np.float32)
    out["loudness"] = _resample_1d(audio["loudness"], T) if audio else np.zeros((T,), np.float32)
    out["motion_mag"] = _resample_1d(video["motion_mag"], T) if video else np.zeros((T,), np.float32)

    # beat は 0/1 へ丸め直す
    out["beat_flag"] = (out["beat_flag"] > 0.5).astype(np.float32)
    return out
