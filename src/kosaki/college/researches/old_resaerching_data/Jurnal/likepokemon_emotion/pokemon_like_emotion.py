import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

# 初期化
emotion_weights = np.array([1.0, 1.0, 1.0, 1.0])  # 喜, 怒, 哀, 楽
emotion_dict = {}

# GUIアプリ本体
class EmotionApp:
    def __init__(self, master):
        self.master = master
        master.title("自我持ちキャラ")

        self.canvas = tk.Canvas(master, width=300, height=300)
        self.canvas.pack()

        # キャラ画像
        self.images = {
            "neutral": ImageTk.PhotoImage(Image.open("normal.png")),
            "neutral_shifted": ImageTk.PhotoImage(Image.open("normal_shifted.png")),
            "happy": ImageTk.PhotoImage(Image.open("happy.png")),
            "happy_shifted": ImageTk.PhotoImage(Image.open("happy_shifted.png")),
            "angry": ImageTk.PhotoImage(Image.open("angry.png")),
            "angry_shifted": ImageTk.PhotoImage(Image.open("angry_shifted.png")),
            "sad": ImageTk.PhotoImage(Image.open("sad.png")),
            "sad_shifted": ImageTk.PhotoImage(Image.open("sad_shifted.png")),
        }
        self.current_mood = "neutral"
        self.is_shifted = False
        self.image_on_canvas = self.canvas.create_image(150, 150, image=self.images[self.current_mood])

        self.entry = tk.Entry(master)
        self.entry.pack()

        self.button = tk.Button(master, text="話しかける", command=self.process_input)
        self.button.pack()

        self.label = tk.Label(master, text="")
        self.label.pack()

        self.animate()  # アニメーション開始

    def map_word_to_emotion(self, word):
        if word not in emotion_dict:
            emotion_dict[word] = np.random.uniform(-0.2, 0.2, 4)
        return emotion_dict[word]

    def compute_emotion_score(self, vec):
        return np.dot(vec, emotion_weights)

    def process_input(self):
        word = self.entry.get()
        vec = self.map_word_to_emotion(word)
        score = self.compute_emotion_score(vec)
        self.label.config(text=f"感情スコア: {score:.2f}")
        self.update_image(vec)

    def update_image(self, vec):
        idx = np.argmax(vec)
        if idx == 0:
            self.current_mood = "happy"
        elif idx == 1:
            self.current_mood = "angry"
        elif idx == 2:
            self.current_mood = "sad"
        else:
            self.current_mood = "neutral"
        self.canvas.itemconfig(self.image_on_canvas, image=self.images[self.current_mood])

    def animate(self):
        mood = self.current_mood
        if self.is_shifted:
            mood_key = mood
        else:
            mood_key = mood + "_shifted"
        self.canvas.itemconfig(self.image_on_canvas, image=self.images[mood_key])
        self.is_shifted = not self.is_shifted
        self.master.after(500, self.animate)


root = tk.Tk()
app = EmotionApp(root)
root.mainloop()