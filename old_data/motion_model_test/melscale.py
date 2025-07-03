import librosa
import matplotlib.pyplot as plt
import numpy as np


# 音声データを読み込み
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)  # 音声ファイルを読み込み
    return y, sr

# メルスケール化されたスペクトログラムを取得
def mel_spectrum(y, sr):
    # スペクトログラムの計算
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    return S

# メルスケールをログスケールに変換（オプション）
def mel_log_spectrum(S):
    return librosa.power_to_db(S, ref=np.max)

# メルスケールスペクトログラムをプロット
def plot_mel_spectrum(S, sr, title='Mel Spectrogram'):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

# 実行例
file_path = 'path_to_your_audio_file.wav'  # ここに音声ファイルのパスを入れる
y, sr = load_audio(file_path)
S = mel_spectrum(y, sr)
plot_mel_spectrum(S, sr)

# メルスペクトログラムを取得し、計算結果を表示
mel_log_S = mel_log_spectrum(S)
print("Mel-log Spectrogram Shape:", mel_log_S.shape)
