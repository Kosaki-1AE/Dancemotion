import cv2
import librosa
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks


# === 音声をマイクから録音してビートを抽出 ===
def record_and_extract_beats(duration=10, sr=22050):
    print(f"[INFO] マイクから {duration}秒 録音中...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("[INFO] 録音完了、ビート解析中...")

    y = recording.flatten()
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    print(f"[INFO] 検出されたビート数: {len(beat_times)}")
    return beat_times

# === 動画から右手首の動作タイミング抽出 ===
def extract_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    y_positions = []

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            y_positions.append(wrist.y)
        else:
            y_positions.append(np.nan)

    cap.release()
    y_array = np.array(y_positions)
    y_array = np.nan_to_num(y_array, nan=np.nanmean(y_array))

    peaks, _ = find_peaks(-y_array, distance=frame_rate * 0.4)
    movement_times = peaks / frame_rate
    print(f"[INFO] 検出された動作数: {len(movement_times)}")
    return movement_times

# === ズレ統計量 ===
def compute_offsets(beat_times, movement_times):
    differences = []
    for m_time in movement_times:
        closest_beat = min(beat_times, key=lambda b: abs(b - m_time))
        differences.append(abs(closest_beat - m_time))

    avg_offset = np.mean(differences)
    std_offset = np.std(differences)
    return differences, avg_offset, std_offset

# === 可視化 ===
def plot_results(beat_times, movement_times, differences):
    plt.figure(figsize=(10, 4))
    plt.vlines(beat_times, 0, 1, color='blue', label='マイク音ビート')
    plt.vlines(movement_times, 0, 1, color='red', label='ダンス動作')
    plt.legend()
    plt.title("マイク音とダンス動作のズレ")
    plt.xlabel("時間 [秒]")
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 3))
    plt.hist(differences, bins=20, color='purple', alpha=0.7)
    plt.title("ズレの分布")
    plt.xlabel("ズレ時間（秒）")
    plt.ylabel("回数")
    plt.tight_layout()
    plt.show()

# === メイン処理 ===
if __name__ == "__main__":
    video_path = "your_dance.mp4"
    duration = 10  # 録音時間（秒）

    beat_times = record_and_extract_beats(duration=duration)
    movement_times = extract_movement(video_path)
    differences, avg_offset, std_offset = compute_offsets(beat_times, movement_times)

    print(f"\n--- 結果 ---")
    print(f"平均ズレ: {avg_offset:.3f} 秒")
    print(f"標準偏差: {std_offset:.3f} 秒")

    plot_results(beat_times, movement_times, differences)
