"""
dance_gsmc_from_video.py

動画から骨格＋音を解析して、
G / M / C / S と 「成長ステージ」を出すプロトタイプ。

必要ライブラリ (pip):
    pip install opencv-python mediapipe librosa numpy scipy

別途必要:
    ffmpeg コマンド（PATH が通っていること）

※ 本コードはプロトタイプ想定。
   閾値やスコアリングロジックは環境やダンサーに合わせて調整推奨。
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Dict

import cv2
import librosa
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks

# =========================
#  データ構造
# =========================

@dataclass
class MotionAnalysisResult:
    fps: float
    times: np.ndarray          # shape: (T,)   秒単位
    motion_energy: np.ndarray  # shape: (T,)
    hit_times: np.ndarray      # shape: (H,)


@dataclass
class AudioBeatResult:
    sr: int
    beat_times: np.ndarray     # shape: (B,)


@dataclass
class GSMCScores:
    G: int
    M: int
    C: int
    S: int
    stage: int          # 成長ステージ (0〜4)
    stage_label: str    # その説明テキスト
    detail: str         # ログテキスト


# =========================
#  動画 → 骨格 → モーションエナジー
# =========================

def extract_motion_energy_from_video(video_path: str,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5) -> MotionAnalysisResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    landmarks_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        else:
            if landmarks_list:
                coords = landmarks_list[-1].copy()
            else:
                coords = np.zeros((33, 3), dtype=np.float32)

        landmarks_list.append(coords)

    cap.release()
    pose.close()

    landmarks = np.stack(landmarks_list, axis=0)  # (T, 33, 3)
    T = landmarks.shape[0]
    times = np.arange(T) / fps

    # フレーム間差分→速度→平均ノルム＝モーションエナジー
    diff = np.diff(landmarks, axis=0)          # (T-1, 33, 3)
    vel = np.linalg.norm(diff, axis=2)         # (T-1, 33)
    motion_energy = vel.mean(axis=1)           # (T-1,)

    motion_energy = np.concatenate([[0.0], motion_energy])

    me_mean = motion_energy.mean()
    me_std = motion_energy.std() + 1e-6
    height_threshold = me_mean + 0.5 * me_std

    peaks, _ = find_peaks(
        motion_energy,
        height=height_threshold,
        distance=int(fps * 0.05)
    )
    hit_times = times[peaks]

    return MotionAnalysisResult(
        fps=fps,
        times=times,
        motion_energy=motion_energy,
        hit_times=hit_times,
    )


# =========================
#  音声 → ビート解析（ffmpeg 版）
# =========================

def extract_audio_and_beats(video_path: str) -> AudioBeatResult:
    """
    ffmpeg で一旦 wav にしてから librosa で読み込むルート。
    ※ ffmpeg コマンドがインストールされている必要あり。
    """
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "audio_tmp.wav")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            wav_path,
        ]

        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        y, sr = librosa.load(wav_path, sr=None, mono=True)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    return AudioBeatResult(sr=sr, beat_times=beat_times)


# =========================
#  ヒットとビートの関係解析
# =========================

def compute_timing_stats(hit_times: np.ndarray,
                         beat_times: np.ndarray) -> Dict[str, float]:
    """
    各ヒットに対して、最も近いビートとの時間差を計算。
    その平均・標準偏差などを返す。
    """
    if len(hit_times) == 0 or len(beat_times) == 0:
        return {
            "mean_abs_offset": np.inf,
            "std_offset": np.inf,
            "onbeat_ratio": 0.0,
            "count": 0,
        }

    offsets = []
    onbeat_count = 0
    onbeat_threshold = 0.07  # 70ms 以内ならオンビート扱い

    for ht in hit_times:
        idx = np.argmin(np.abs(beat_times - ht))
        diff = ht - beat_times[idx]
        offsets.append(diff)
        if abs(diff) <= onbeat_threshold:
            onbeat_count += 1

    offsets = np.array(offsets)
    mean_abs_offset = float(np.mean(np.abs(offsets)))
    std_offset = float(np.std(offsets))
    onbeat_ratio = float(onbeat_count / len(hit_times))

    return {
        "mean_abs_offset": mean_abs_offset,
        "std_offset": std_offset,
        "onbeat_ratio": onbeat_ratio,
        "count": int(len(hit_times)),
    }


# =========================
#  間（Stillness）の解析
# =========================

def compute_stillness_stats(motion_energy: np.ndarray,
                            times: np.ndarray,
                            hit_times: np.ndarray,
                            pre_window: float = 0.3) -> Dict[str, float]:
    """
    各ヒットの直前 pre_window 秒のモーションエナジー平均を計算。
    それを「どれだけ止まれているか」の指標とする。
    """
    if len(hit_times) == 0:
        return {
            "mean_pre_energy": np.inf,
            "std_pre_energy": np.inf,
            "count": 0,
        }

    pre_energies = []

    for ht in hit_times:
        mask = (times >= ht - pre_window) & (times < ht)
        if not np.any(mask):
            continue
        pre_energies.append(motion_energy[mask].mean())

    if not pre_energies:
        return {
            "mean_pre_energy": np.inf,
            "std_pre_energy": np.inf,
            "count": 0,
        }

    pre_energies = np.array(pre_energies)
    return {
        "mean_pre_energy": float(pre_energies.mean()),
        "std_pre_energy": float(pre_energies.std()),
        "count": int(len(pre_energies)),
    }


# =========================
#  GSMC スコアリング ＋ 成長ステージ
# =========================

def score_gsmc(motion: MotionAnalysisResult,
               audio: AudioBeatResult) -> GSMCScores:
    timing_stats = compute_timing_stats(motion.hit_times, audio.beat_times)
    stillness_stats = compute_stillness_stats(
        motion.motion_energy, motion.times, motion.hit_times
    )

    lines = []

    me = motion.motion_energy
    me_mean = me.mean()
    me_std = me.std() + 1e-6
    peak_energy = me.max()

    normalized_peak = (peak_energy - me_mean) / me_std

    # --- M: 実行（Motion） ---
    if normalized_peak > 4:
        M_score = 3
    elif normalized_peak > 3:
        M_score = 2
    elif normalized_peak > 1.0:
        M_score = 1
    else:
        M_score = 0

    lines.append(f"[M] peak_energy_z = {normalized_peak:.2f} → M = {M_score}")

    # --- C: 整合（Coherence） ---
    mean_abs_offset = timing_stats["mean_abs_offset"]
    onbeat_ratio = timing_stats["onbeat_ratio"]

    if not np.isfinite(mean_abs_offset):
        C_score = 0
        lines.append("[C] ビート情報が不足しているため C=0 としました。")
    else:
        # 初心者〜1年生向けにゆるめた閾値
        if mean_abs_offset < 0.06 and onbeat_ratio > 0.6:
            C_score = 3
        elif mean_abs_offset < 0.12 and onbeat_ratio > 0.4:
            C_score = 2
        elif mean_abs_offset < 0.25 or onbeat_ratio > 0.20:
            C_score = 1
        else:
            C_score = 0

        lines.append(
            f"[C] mean_abs_offset={mean_abs_offset*1000:.1f}ms, "
            f"onbeat_ratio={onbeat_ratio:.2f} → C = {C_score}"
        )

    # --- S: 間（Stillness） ---
    mean_pre_energy = stillness_stats["mean_pre_energy"]

    if not np.isfinite(mean_pre_energy):
        S_score = 0
        lines.append("[S] ヒットが検出できなかったため S=0 としました。")
    else:
        ratio = mean_pre_energy / (me_mean + 1e-6)

        # ★ Stillness in Motion 重視バージョン
        #   2.0 未満なら「とりあえずちょっと止まれてる」扱いにする
        if ratio < 0.8:
            S_score = 3
        elif ratio < 1.3:
            S_score = 2
        elif ratio < 2.0:
            S_score = 1
        else:
            S_score = 0

        lines.append(
            f"[S] pre_energy_ratio={ratio:.2f} (小さいほど『止まれている』) → S = {S_score}"
        )

    # --- G: 意図（Genesys） ---
    if not np.isfinite(mean_abs_offset):
        G_score = 0
        std_offset = np.inf
        lines.append("[G] タイミング情報が不足しているため G=0 としました。")
    else:
        std_offset = timing_stats["std_offset"]
        if std_offset < 0.03 and onbeat_ratio < 0.6:
            G_score = 3
        elif std_offset < 0.06:
            G_score = 2
        elif std_offset < 0.15:
            G_score = 1
        else:
            G_score = 0

        lines.append(
            f"[G] std_offset={std_offset*1000:.1f}ms → G = {G_score}"
        )

    # --------------------
    # 成長ステージ判定（Stillness 優先版）
    # --------------------
    stage = 0
    stage_label = "Lv.0: まだデータ不足（ヒットやビートが安定していない段階）"

    hit_count = timing_stats["count"]
    if hit_count >= 3 and M_score >= 1:
        stage = 1
        stage_label = "Lv.1: とりあえず『動けている』段階"

        # ★ まず Stillness の成長を優先して見る
        if S_score >= 1:
            stage = 2
            stage_label = "Lv.2: Stillness の片鱗が出てきた段階（止まる／タメるが部分的にできる）"

            if C_score >= 1:
                stage = 3
                stage_label = "Lv.3: Stillness in Motion が見え始めた段階（止まり＋ビート同期）"

                if G_score >= 1:
                    stage = 4
                    stage_label = "Lv.4: 意図的なズラしまで使える段階（SIM を任意にコントロール）"

    lines.append("")
    lines.append(f"[STAGE] 成長ステージ: Lv.{stage}")
    lines.append(f"         {stage_label}")

    detail = "\n".join(lines)

    return GSMCScores(
        G=G_score,
        M=M_score,
        C=C_score,
        S=S_score,
        stage=stage,
        stage_label=stage_label,
        detail=detail,
    )



# =========================
#  メイン
# =========================

def analyze_dance_video(video_path: str) -> GSMCScores:
    print(f"動画解析開始: {video_path}")
    motion = extract_motion_energy_from_video(video_path)
    print("  → 骨格／モーションエナジー抽出完了")
    audio = extract_audio_and_beats(video_path)
    print("  → 音声ビート抽出完了")
    scores = score_gsmc(motion, audio)
    print("  → GSMC スコア算出完了")
    return scores


if __name__ == "__main__":
    print("どの動画で解析する？（ファイルのパスを入力してね）")
    video_path = input("video path > ").strip()

    if not os.path.exists(video_path):
        print("❌ そのファイル見つからんかった…パスを確認してみて！")
        raise SystemExit(1)

    scores = analyze_dance_video(video_path)

    print("\n==============================")
    print("  GSMC スコア（成長版・暫定）")
    print("==============================")
    print(f"G (意図) ：{scores.G} / 3")
    print(f"M (実行) ：{scores.M} / 3")
    print(f"C (整合) ：{scores.C} / 3")
    print(f"S (間)   ：{scores.S} / 3")
    print("------------------------------")
    print(f"成長ステージ: Lv.{scores.stage}")
    print(scores.stage_label)
    print("------------------------------")
    print(scores.detail)
    print("------------------------------")
    print("※ しきい値は 1年生〜中級を想定した暫定版なので、実データを見ながら微調整してOK。")
