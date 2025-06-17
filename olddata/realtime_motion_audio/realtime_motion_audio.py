import csv
from collections import deque
from datetime import datetime

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib import rcParams

# ---- 日本語フォント設定（matplotlib用） ----
rcParams['font.family'] = 'MS Gothic'  # または 'IPAexGothic'

# ---- 音声設定 ----
SAMPLING_RATE = 22050
BLOCK_SIZE = 2048
volume_level = 0.0
audio_buffer = deque(maxlen=BLOCK_SIZE)

# ---- 音声コールバック ----
def audio_callback(indata, frames, time, status):
    global volume_level, audio_buffer
    data = indata[:, 0]
    audio_buffer.extend(data)
    volume_level = np.linalg.norm(data) / frames

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLING_RATE, blocksize=BLOCK_SIZE)
stream.start()

# ---- カメラ設定 ----
cap = cv2.VideoCapture(0)
prev_frame = None

motion_buffer = deque(maxlen=100)
vol_buffer = deque(maxlen=100)
score_buffer = deque(maxlen=100)
label_buffer = deque(maxlen=100)

# ---- CSVファイル準備 ----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"stillness_log_{timestamp}.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["time", "motion", "volume", "stillness_score", "label"])

# ---- 可視化 ----
plt.ion()
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

def classify_stillness(score):
    if score > 0.7:
        return "静寂"
    elif score > 0.4:
        return "中間"
    else:
        return "騒がしい"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- 映像処理（動きの強さ） ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame is None:
        prev_frame = blur
        continue
    frame_delta = cv2.absdiff(prev_frame, blur)
    motion_intensity = np.sum(frame_delta) / 1000000
    motion_intensity = np.clip(motion_intensity, 0, 1)
    prev_frame = blur

    # ---- 音量 ----
    vol = np.clip(volume_level, 0, 1)

    # ---- 空気感スコア ----
    score = (1 - vol) * (1 - motion_intensity)
    label = classify_stillness(score)

    # ---- ログ保存 ----
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    with open(csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([current_time, motion_intensity, vol, score, label])

    # ---- バッファ更新 ----
    motion_buffer.append(motion_intensity)
    vol_buffer.append(vol)
    score_buffer.append(score)
    label_buffer.append(label)

    # ---- グラフ表示 ----
    axs[0].cla()
    axs[0].plot(motion_buffer)
    axs[0].set_title("動きの強さ")

    axs[1].cla()
    axs[1].plot(vol_buffer)
    axs[1].set_title("音の大きさ")

    axs[2].cla()
    axs[2].plot(score_buffer, color='purple')
    axs[2].set_title("動きの中にある静けさの割合(空気感の発生地点)")

    # ---- スペクトログラム（Mel）表示 ----
    axs[3].cla()
    if len(audio_buffer) >= BLOCK_SIZE:
        y = np.array(audio_buffer, dtype=np.float32)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLING_RATE, n_mels=40, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        axs[3].imshow(mel_db, aspect='auto', origin='lower', cmap='magma')
        axs[3].set_title("Melスペクトログラム")

    plt.tight_layout()
    plt.pause(0.01)

    # ---- 映像表示 ----
    display_text = f"Stillness in Motion: {score:.2f} ({label})"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.imshow("Live", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---- 終了処理 ----
cap.release()
cv2.destroyAllWindows()
stream.stop()
stream.close()
