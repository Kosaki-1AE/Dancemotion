# run_model.py

import numpy as np
from emotion_model import EmotionProcessor

# 空気感ベクトル例（緊張、期待、静寂、活気、不安）
atmosphere = np.array([0.7, 0.2, 0.5, 0.1, 0.3])

eproc = EmotionProcessor()

emotion = eproc.estimate_emotion(atmosphere)
meaning = eproc.reframe_meaning(emotion, frontal_control=0.2)
output = eproc.generate_output(meaning)

print("=== 空気感ベクトル ===")
print(atmosphere)

print("\n=== 推定された感情ベクトル ===")
print(emotion)

print("\n=== 意味付け後のベクトル ===")
print(meaning)

print("\n=== 最終的な動作出力 ===")
print(output)
