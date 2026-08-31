# expression_wave_explorer.py
import os
import re
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RadioButtons, Slider
from mpl_toolkits.mplot3d import Axes3D


# === ZIPファイルを読み込んでスコア化 ===
def score_journal(text):
    def word_score(words):
        return sum(text.count(w) for w in words) / max(len(text), 1)

    sentences = re.split(r'[。！？]', text)
    avg_sentence_length = np.mean([len(s.strip()) for s in sentences if s.strip()])
    language_score = min(avg_sentence_length / 50, 1.0)

    visual_score = min(word_score(["color", "light", "image", "see", "scenery", "painting", "viewpoint", "contour"]) * 100, 1.0)
    rhythm_score = min(word_score(["rhythm", "tempo", "sway", "wave", "repeat"]) * 100, 1.0)
    auditory_score = min(word_score(["sound", "hear", "voice", "echo", "noise"]) * 100, 1.0)
    abstract_score = min(word_score(["concept", "meaning", "existence", "unconscious", "sensation", "perception", "thought", "space"]) * 100, 1.0)

    return round(language_score, 2), round(visual_score, 2), round(rhythm_score, 2), round(auditory_score, 2), round(abstract_score, 2)

# ZIPファイルを読み取り、スコアをDataFrameにまとめる
zip_path = "Jurnal.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    file_list = [f for f in z.namelist() if f.endswith(".txt")]
    data = []
    for fname in file_list:
        with z.open(fname) as f:
            text = f.read().decode('utf-8')
            scores = score_journal(text)
            data.append((fname, *scores))

score_df = pd.DataFrame(data, columns=["Filename", "Language", "Visual", "Rhythm", "Auditory", "Abstract"])

# === 軸ペア候補 ===
axis_sets = {
    "Language-Visual-Rhythm": ("Language", "Visual", "Rhythm"),
    "Auditory-Abstract-Language": ("Auditory", "Abstract", "Language"),
    "Rhythm-Visual-Abstract": ("Rhythm", "Visual", "Abstract"),
    "Language-Abstract-Rhythm": ("Language", "Abstract", "Rhythm"),
    "Auditory-Rhythm-Visual": ("Auditory", "Rhythm", "Visual")
}

# === 初期ベクトル長 ===
vector_scale = 0.15

# === 描画関数 ===
def update_plot(selected):
    ax.clear()
    ax.set_title(f"3D Expression Frequency Map: {selected}")

    x_label, y_label, z_label = axis_sets[selected]
    x = score_df[x_label].values
    y = score_df[y_label].values
    z = score_df[z_label].values

    abstract = score_df["Abstract"].values
    auditory = score_df["Auditory"].values
    colors = [(0.2, 0.2, float(a)) for a in abstract]
    linewidths = 1 + auditory * 5

    ax.quiver(x, y, z, x - 0.5, y - 0.5, z - 0.5,
              length=vector_scale_slider.val, normalize=True, color=colors, linewidths=linewidths)

    for i, label in enumerate(score_df["Filename"]):
        filename = os.path.splitext(os.path.basename(label))[0]
        match = re.search(r"(\d{6})", filename)
        short_label = match.group(1) if match else filename
        ax.text(x[i], y[i], z[i], short_label, size=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    plt.draw()

# === プロット初期化 ===
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.2)

# ラジオボタンで視点切り替え
ax_radio = plt.axes([0.02, 0.3, 0.2, 0.5])
radio = RadioButtons(ax_radio, list(axis_sets.keys()))
radio.on_clicked(update_plot)

# スライダーでベクトル長調整
ax_slider = plt.axes([0.25, 0.05, 0.6, 0.03])
vector_scale_slider = Slider(ax_slider, 'Vector Length', 0.05, 0.5, valinit=vector_scale)
vector_scale_slider.on_changed(lambda val: update_plot(radio.value_selected))

# 最初のプロット
update_plot("Language-Visual-Rhythm")
plt.show()
