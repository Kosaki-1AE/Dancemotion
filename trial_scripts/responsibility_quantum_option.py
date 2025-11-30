import tempfile

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
from essentia.standard import MusicExtractor
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


# =========================
# MediaPipe: リアルタイム骨格ベクトル
# =========================
def get_mediapipe_vector(frame, mp_pose, pose_detector):
    results = pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return np.array([0.0, 0.0, 0.0, 0.0])
    landmarks = results.pose_landmarks.landmark
    vec = np.array([
        landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x,
        landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y,
        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x,
        landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP].y
    ])
    return vec


# =========================
# Essentia: 一定秒数の音声解析
# =========================
def get_essentia_vector(duration=2, samplerate=44100):
    tmpfile = tempfile.mktemp(suffix=".wav")
    print(f"録音中... {duration}秒")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    sf.write(tmpfile, audio, samplerate)

    extractor = MusicExtractor(lowlevelStats=['mean'], rhythmStats=['mean'], tonalStats=['mean'])
    features, _ = extractor(tmpfile)

    bpm = features['rhythm.bpm']
    centroid = features['lowlevel.spectral_centroid.mean']
    mfcc1 = features['lowlevel.mfcc.mean'][0]
    mfcc2 = features['lowlevel.mfcc.mean'][1]

    return np.array([bpm, centroid, mfcc1, mfcc2])


# =========================
# Qiskit: 責任量子クラス
# =========================
class ResponsibilityQuantum:
    def __init__(self, R_self=1.0, R_other=1.0):
        self.R_self = R_self
        self.R_other = R_other

    def build_circuit(self, alpha, beta):
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha /= norm
        beta /= norm

        qc = QuantumCircuit(1, 1)
        qc.initialize([alpha, beta], 0)

        # 責任ベクトルを量子回転に反映
        theta = (self.R_self - self.R_other) * np.pi / 4
        qc.ry(theta, 0)
        qc.measure(0, 0)
        return qc

    def simulate(self, alpha, beta, shots=512):
        qc = self.build_circuit(alpha, beta)
        sim = Aer.get_backend("aer_simulator")   # ← qasm_simulator じゃなくてこっち
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()  # ← assemble 不要
        return result.get_counts()


# =========================
# メイン: リアルタイム実行ループ
# =========================
def main():
    rq = ResponsibilityQuantum(R_self=1.2, R_other=0.8)

    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose()

    cap = cv2.VideoCapture(0)  # Webカメラ

    # Matplotlib 初期設定
    plt.ion()
    fig, ax = plt.subplots()
    bars = ax.bar(["0 (Motion)", "1 (Music)"], [0, 0], color=["blue", "orange"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("確率")
    ax.set_title("リアルタイム量子観測ヒストグラム")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe骨格
            M_vec = get_mediapipe_vector(frame, mp_pose, pose_detector)

            # Essentia音楽（数秒録音）
            E_vec = get_essentia_vector(duration=2)

            # 複素数係数生成
            alpha = complex(M_vec[0], M_vec[1])
            beta = complex(E_vec[0], E_vec[1])

            # Qiskitシミュレーション
            counts = rq.simulate(alpha, beta, shots=512)

            # 確率に変換
            p0 = counts.get("0", 0) / 512
            p1 = counts.get("1", 0) / 512

            # グラフ更新
            bars[0].set_height(p0)
            bars[1].set_height(p1)
            plt.pause(0.01)

            # コンソール表示
            print("観測結果:", counts)

            # Webカメラ表示
            cv2.imshow("MediaPipe Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
