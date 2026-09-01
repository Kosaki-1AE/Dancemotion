# emotion_vectorizer.py
# 感情ベクトル変換モジュール（Plutchikベース）

import numpy as np

class EmotionVectorizer:
    def __init__(self):
        self.emotion_map = {
            "joy": [0.9, 0.5, 0.2],
            "trust": [0.8, 0.6, 0.3],
            "fear": [0.2, 0.3, 0.9],
            "sadness": [0.1, 0.2, 0.6],
            "anger": [0.9, 0.2, 0.1],
            "disgust": [0.3, 0.2, 0.5],
            "anticipation": [0.5, 0.6, 0.2],
            "surprise": [0.7, 0.3, 0.4],
            "neutral": [0.4, 0.4, 0.4]
        }

    def get_vector(self, emotion_label: str, intensity: float = 1.0) -> np.ndarray:
        base_vector = self.emotion_map.get(emotion_label.lower(), self.emotion_map["neutral"])
        return np.array([v * intensity for v in base_vector])

    def list_emotions(self):
        return list(self.emotion_map.keys())

if __name__ == "__main__":
    ev = EmotionVectorizer()
    print("joy:", ev.get_vector("joy", 0.7))
