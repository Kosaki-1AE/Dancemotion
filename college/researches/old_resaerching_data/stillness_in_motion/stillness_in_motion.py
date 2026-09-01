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

rcParams['font.family'] = 'MS Gothic'

SAMPLING_RATE = 22050
BLOCK_SIZE = 2048
volume_level = 0.0
audio_buffer = deque(maxlen=BLOCK_SIZE)

base_dir = pathlib.Path(__file__).resolve().parent.parent
log_dir = base_dir / "log"
video_dir = base_dir / "video"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
audio_record = []

devices = sd.query_devices()
mic_device = next((i for i, dev in enumerate(devices)  if dev['max_input_channels'] > 0 and ("usb" in dev['name'].lower() or "external" in dev['name'].lower())), sd.default.device[0])

def audio_record_callback(indata, frames, time_info, status):
    audio_buffer.extend(indata[:, 0])
    audio_record.append(indata.copy())
    global volume_level
    volume_level = np.linalg.norm(indata[:, 0]) / frames

audio_stream = sd.InputStream(samplerate=SAMPLING_RATE, channels=1, callback=audio_record_callback, device=mic_device, blocksize=BLOCK_SIZE)
audio_stream.start()

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_filename = video_dir / f"stillness_video_{timestamp}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(video_filename), fourcc, 1.0, (frame_width, frame_height))

csv_filename = log_dir / f"stillness_log_{timestamp}.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time", "motion", "volume", "beat_strength", "motion_delta", "stillness_score", "label"])

prev_frame = None
motion_buffer = deque(maxlen=100)
vol_buffer = deque(maxlen=100)
score_buffer = deque(maxlen=100)
beat_buffer = deque(maxlen=100)
motion_delta_buffer = deque(maxlen=100)

plt.ion()
fig, axs = plt.subplots(5, 1, figsize=(8, 12))

def classify_stillness(score):
    if score > 0.7:
        return "Silent"
    elif score > 0.4:
        return "Neutral"
    else:
        return "Noisy"

frame_count = 0
start_time = time.time()

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

    # 音声：ビート強度の推定
    if len(audio_buffer) >= BLOCK_SIZE:
        y = np.array(audio_buffer, dtype=np.float32)
        onset_env = librosa.onset.onset_strength(y=y, sr=SAMPLING_RATE)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SAMPLING_RATE)
        beat_strength = np.mean(onset_env[beats]) if len(beats) > 0 else 0.0
        beat_strength = np.clip(beat_strength / 10, 0, 1)
    else:
        beat_strength = 0.0

    # 映像：動きの変化（衝撃）を算出
    if len(motion_buffer) > 1:
        motion_delta = abs(motion_intensity - motion_buffer[-1])
        motion_delta = np.clip(motion_delta, 0, 1)
    else:
        motion_delta = 0.0

    # スコア計算（すべて掛け算で統合）
    score = (1 - vol) * (1 - motion_intensity) * (1 - beat_strength) * (1 - motion_delta)
    label = classify_stillness(score)

    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_time, motion_intensity, vol, beat_strength, motion_delta, score, label])

    motion_buffer.append(motion_intensity)
    vol_buffer.append(vol)
    score_buffer.append(score)
    beat_buffer.append(beat_strength)
    motion_delta_buffer.append(motion_delta)

    axs[0].cla()
    axs[0].plot(motion_buffer)
    axs[0].set_title("Motion Intensity")

    axs[1].cla()
    axs[1].plot(vol_buffer)
    axs[1].set_title("Sound Volume")

    axs[2].cla()
    axs[2].plot(beat_buffer, color='orange')
    axs[2].set_title("Beat Strength")

    axs[3].cla()
    axs[3].plot(motion_delta_buffer, color='red')
    axs[3].set_title("Motion Delta (Impact)")

    axs[4].cla()
    axs[4].plot(score_buffer, color='purple')
    axs[4].set_title("Stillness in Motion")

    plt.tight_layout()
    plt.pause(0.01)

    display_text = f"Stillness in Motion: {score:.2f} ({label})"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.imshow("Live", frame)
    out.write(frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
audio_stream.stop()
audio_stream.close()
audio_data = np.concatenate(audio_record, axis=0)
audio_filename = video_dir / f"stillness_audio_{timestamp}.wav"
sf.write(audio_filename, audio_data, SAMPLING_RATE)

elapsed_time = time.time() - start_time
actual_fps = frame_count / elapsed_time
print(f"🎯 実測FPS: {actual_fps:.2f}")

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
