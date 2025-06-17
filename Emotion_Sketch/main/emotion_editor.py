import json
import os
import tkinter as tk
from tkinter import simpledialog

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

EMOTION_FILE = "emotions.json"

def load_emotions():
    if os.path.exists(EMOTION_FILE):
        with open(EMOTION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_emotions(emotion_map):
    with open(EMOTION_FILE, "w", encoding="utf-8") as f:
        json.dump(emotion_map, f, ensure_ascii=False, indent=4)

def play_emotion(freq, volume, duration):
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = np.sin(2 * np.pi * freq * t) * volume
    sd.play(tone, fs)
    sd.wait()

modifier_map = {
    "鋭く": [200, 0.1, -0.3],
    "重く": [-300, -0.1, 0.5],
    "派手に": [150, 0.2, -0.1],
    "透明感": [300, -0.1, 0.3],
    "深く": [-200, -0.1, 0.4],
    "明るく": [100, 0.1, -0.1],
    "暗く": [-100, -0.1, 0.2]
}

def apply_natural_command(command, emotion_map):
    for emotion in emotion_map.keys():
        if emotion in command:
            for mod, delta in modifier_map.items():
                if mod in command:
                    freq, vol, dur = emotion_map[emotion]
                    df, dv, dd = delta
                    new_freq = max(50, freq + df)
                    new_vol = min(1.0, max(0.0, vol + dv))
                    new_dur = max(0.1, dur + dd)
                    emotion_map[emotion] = [round(new_freq,2), round(new_vol,2), round(new_dur,2)]
                    return f"{emotion} を {mod} → 周波数: {new_freq}, 音量: {new_vol}, 長さ: {new_dur}"
    return "感情または修飾語が見つかりませんでした。"

def create_gui():
    emotion_map = load_emotions()

    def update_listbox():
        listbox.delete(0, tk.END)
        for emotion in emotion_map:
            listbox.insert(tk.END, f"{emotion} : {emotion_map[emotion]}")

    def on_add():
        name = simpledialog.askstring("感情名", "新しい感情の名前は？")
        if name and name not in emotion_map:
            freq = simpledialog.askfloat("周波数(Hz)", f"{name}の周波数を入力:")
            vol = simpledialog.askfloat("音量(0.0~1.0)", f"{name}の音量を入力:")
            dur = simpledialog.askfloat("長さ(秒)", f"{name}の長さを入力:")
            if freq and vol and dur:
                emotion_map[name] = [freq, vol, dur]
                save_emotions(emotion_map)
                update_listbox()

    def on_edit():
        selection = listbox.curselection()
        if not selection:
            return
        key = list(emotion_map.keys())[selection[0]]
        freq, vol, dur = emotion_map[key]
        freq = simpledialog.askfloat("周波数(Hz)", f"{key}の周波数を入力:", initialvalue=freq)
        vol = simpledialog.askfloat("音量(0.0~1.0)", f"{key}の音量を入力:", initialvalue=vol)
        dur = simpledialog.askfloat("長さ(秒)", f"{key}の長さを入力:", initialvalue=dur)
        if freq and vol and dur:
            emotion_map[key] = [freq, vol, dur]
            save_emotions(emotion_map)
            update_listbox()

    def on_plot():
        fig, ax = plt.subplots()
        names = list(emotion_map.keys())
        freqs = [emotion_map[name][0] for name in names]
        vols = [emotion_map[name][1] for name in names]
        durs = [emotion_map[name][2] for name in names]

        ax.scatter(freqs, vols, s=[d*200 for d in durs], alpha=0.6)
        for i, name in enumerate(names):
            ax.text(freqs[i] + 10, vols[i] + 0.02, name, fontsize=12)

        def on_click(event):
            if event.inaxes != ax:
                return
            for i in range(len(freqs)):
                if abs(event.xdata - freqs[i]) < 20 and abs(event.ydata - vols[i]) < 0.05:
                    freq, vol, dur = emotion_map[names[i]]
                    play_emotion(freq, vol, dur)
                    break

        fig.canvas.mpl_connect('button_press_event', on_click)

        ax.set_xlabel("周波数 (Hz)")
        ax.set_ylabel("音量")
        ax.set_title("感情マッピング (周波数 vs 音量 + 長さサイズ)")

        fig_canvas = tk.Toplevel()
        fig_canvas.title("感情グラフ")
        canvas = FigureCanvasTkAgg(fig, master=fig_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def on_command():
        command = simpledialog.askstring("自然文コマンド", "感情編集コマンドを入力してね：")
        if command:
            result = apply_natural_command(command, emotion_map)
            save_emotions(emotion_map)
            update_listbox()
            simpledialog.messagebox.showinfo("結果", result)

    root = tk.Tk()
    root.title("感情音マッピングエディタ（音＋自然文対応）")

    listbox = tk.Listbox(root, width=50, height=15)
    listbox.pack()

    tk.Button(root, text="編集", command=on_edit).pack(fill='x')
    tk.Button(root, text="追加", command=on_add).pack(fill='x')
    tk.Button(root, text="グラフ表示", command=on_plot).pack(fill='x')
    tk.Button(root, text="自然文コマンド", command=on_command).pack(fill='x')

    update_listbox()
    root.mainloop()

create_gui()
