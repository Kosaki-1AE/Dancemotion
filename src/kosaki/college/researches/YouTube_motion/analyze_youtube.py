import csv
import os
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# ---- フォント設定（日本語表示に対応） ----
rcParams['font.family'] = 'MS Gothic'

# ---- ディレクトリ準備 ----
base_dir = Path(__file__).resolve().parent
video_dir = base_dir / "video"
log_dir = base_dir / "log"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# ---- 解析対象の動画ファイル（手動でvideoフォルダに入れておく） ----
video_filename = input("🎥 解析したい動画ファイル名（例: my_video.mp4）を入力してください：\n> ").strip()
video_path = video_dir / video_filename

# ---- タイムスタンプとファイル準備 ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
audio_path = video_dir / f"extracted_audio_{timestamp}.wav"
csv_path = log_dir / f"youtube_analysis_{timestamp}.csv"
output_path = video_dir / f"youtube_result_{timestamp}.mp4"

# ---- 音声抽出（ffmpeg使用） ----
print("🎧 音声を抽出中（ffmpeg使用）...")
subprocess.call([
    "ffmpeg",
    "-y",
    "-i", str(video_path),
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", "22050",
    "-ac", "1",
    str(audio_path)
])
print("✅ 音声を保存しました：", audio_path)

# ---- 音声読み込み＆音量分析 ----
print("🔍 音声分析中...")
y, sr = librosa.load(audio_path, sr=22050)
hop_length = 512
volume_series = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
volume_series = np.clip(volume_series / np.max(volume_series), 0, 1)

# ---- 映像準備 ----
cap = cv2.VideoCapture(str(video_path))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---- 映像出力（字幕付き動画） ----
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

# ---- CSV出力ファイルの準備 ----
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time_sec", "motion", "volume", "stillness_score", "label"])

# ---- ラベル関数 ----
def classify(score):
    if score > 0.7:
        return "Silent"
    elif score > 0.4:
        return "Neutral"
    else:
        return "Noisy"

# ---- 初期化 ----
prev_frame = None
motion_buffer = deque(maxlen=100)
vol_buffer = deque(maxlen=100)
score_buffer = deque(maxlen=100)

# ---- グラフ表示準備 ----
plt.ion()
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# ---- メイン処理ループ ----
frame_idx = 0
print("📊 動画解析開始...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame is None:
        prev_frame = blur
        continue

    diff = cv2.absdiff(prev_frame, blur)
    motion = np.sum(diff) / 1000000
    motion = np.clip(motion, 0, 1)
    prev_frame = blur

    vol_idx = int(frame_idx * hop_length / fps)
    vol = float(volume_series[vol_idx]) if vol_idx < len(volume_series) else 0.0

    score = (1 - vol) * (1 - motion)
    label = classify(score)
    current_time = frame_idx / fps

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"{current_time:.2f}", motion, vol, score, label])

    motion_buffer.append(motion)
    vol_buffer.append(vol)
    score_buffer.append(score)

    axs[0].cla()
    axs[0].plot(motion_buffer)
    axs[0].set_title("Motion Intensity")

    axs[1].cla()
    axs[1].plot(vol_buffer)
    axs[1].set_title("Volume")

    axs[2].cla()
    axs[2].plot(score_buffer, color='purple')
    axs[2].set_title("Stillness in Motion")

    plt.tight_layout()
    plt.pause(0.001)

    display_text = f"{current_time:.1f}s - Stillness: {score:.2f} ({label})"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    out.write(frame)

    frame_idx += 1

# ---- 終了処理 ----
cap.release()
out.release()
cv2.destroyAllWindows()
print("✅ 解析完了！")
print(f"📄 CSVログ: {csv_path}")
print(f"🎞 分析映像: {output_path}")
