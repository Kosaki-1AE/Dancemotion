import csv
import os
import pathlib
import subprocess
import time
from collections import deque
from datetime import datetime

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
from matplotlib import rcParams

# ---- 日本語フォント設定（matplotlib用） ----
rcParams['font.family'] = 'MS Gothic'

# ---- 音声設定 ----
SAMPLING_RATE = 22050
BLOCK_SIZE = 2048
volume_level = 0.0
audio_buffer = deque(maxlen=BLOCK_SIZE)

# ---- ログ・動画・音声フォルダの準備 ----
base_dir = pathlib.Path(__file__).resolve().parent.parent
log_dir = base_dir / "log"
video_dir = base_dir / "video"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

# ---- タイムスタンプ ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---- 音声ファイル保存用リストとデバイス選択 ----
audio_record = []

devices = sd.query_devices()
mic_device = None
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0 and ("usb" in dev['name'].lower() or "external" in dev['name'].lower()):
        mic_device = i
        break
if mic_device is None:
    mic_device = sd.default.device[0]

# ---- 音声録音用 ----
def audio_record_callback(indata, frames, time, status):
    audio_buffer.extend(indata[:, 0])
    audio_record.append(indata.copy())
    global volume_level
    volume_level = np.linalg.norm(indata[:, 0]) / frames

audio_stream = sd.InputStream(samplerate=SAMPLING_RATE, channels=1, callback=audio_record_callback, device=mic_device, blocksize=BLOCK_SIZE)
audio_stream.start()

# ---- 映像保存 ----
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 仮のFPS（あとで上書き）
video_filename = video_dir / f"stillness_video_{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(video_filename), fourcc, 1.0, (frame_width, frame_height))

# ---- CSVファイル準備 ----
csv_filename = log_dir / f"stillness_log_{timestamp}.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time", "motion", "volume", "stillness_score", "label"])

# ---- 初期化 ----
prev_frame = None
motion_buffer = deque(maxlen=100)
vol_buffer = deque(maxlen=100)
score_buffer = deque(maxlen=100)
label_buffer = deque(maxlen=100)

plt.ion()
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

def classify_stillness(score):
    if score > 0.7:
        return "Silent"
    elif score > 0.4:
        return "Neutral"
    else:
        return "Noisy"

# ---- FPS測定用 ----
frame_count = 0
start_time = time.time()

# ---- メインループ ----
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame is None:
        prev_frame = blur
        continue

    frame_delta = cv2.absdiff(prev_frame, blur)
    motion_intensity = np.sum(frame_delta) / 1000000
    motion_intensity = np.clip(motion_intensity, 0, 1)
    prev_frame = blur

    vol = np.clip(volume_level, 0, 1)
    score = (1 - vol) * (1 - motion_intensity)
    label = classify_stillness(score)

    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_time, motion_intensity, vol, score, label])

    motion_buffer.append(motion_intensity)
    vol_buffer.append(vol)
    score_buffer.append(score)
    label_buffer.append(label)

    axs[0].cla()
    axs[0].plot(motion_buffer)
    axs[0].set_title("Motion Intensity")

    axs[1].cla()
    axs[1].plot(vol_buffer)
    axs[1].set_title("Sound Volume")

    axs[2].cla()
    axs[2].plot(score_buffer, color='purple')
    axs[2].set_title("Stillness in Motion")

    axs[3].cla()
    if len(audio_buffer) >= BLOCK_SIZE:
        y = np.array(audio_buffer, dtype=np.float32)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLING_RATE, n_mels=40, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        axs[3].imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
        axs[3].set_title("Mel-Spectrogram")

    plt.tight_layout()
    plt.pause(0.01)

    display_text = f"Stillness in Motion: {score:.2f} ({label})"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.imshow("Live", frame)
    out.write(frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---- 終了処理 ----
cap.release()
out.release()
cv2.destroyAllWindows()

audio_stream.stop()
audio_stream.close()
audio_data = np.concatenate(audio_record, axis=0)
audio_filename = video_dir / f"stillness_audio_{timestamp}.wav"
sf.write(audio_filename, audio_data, SAMPLING_RATE)

# ---- 実測FPS計算 ----
elapsed_time = time.time() - start_time
actual_fps = frame_count / elapsed_time
print(f"🎯 実測FPS: {actual_fps:.2f}")

# ---- ffmpegで音声付き動画に再エンコード（FPS反映） ----
final_filename = video_dir / f"stillness_final_{timestamp}.mp4"
subprocess.call([
    "ffmpeg",
    "-y",
    "-r", f"{actual_fps:.2f}",
    "-i", str(video_filename),
    "-i", str(audio_filename),
    "-c:v", "copy",
    "-c:a", "aac",
    "-strict", "experimental",
    str(final_filename)
])
