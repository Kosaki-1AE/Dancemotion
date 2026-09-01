import numpy as np


# 線形変換（責任ベクトルを投影するイメージ）これで規定だけ整えまっす
def linear_transform(x, W, b):
    return np.dot(x, W) + b

# ReLU（正の責任だけ通す）愛を発生させる責任ベクトルがこっち
def relu(x):
    return np.maximum(0, x)

# 逆ReLU（負の責任だけ通す）えぐみを発生させる責任ベクトルがこっち
def neg_relu(x):
    return np.minimum(0, x)

# 入力ベクトル（責任ベクトル的なもの）一応は外部から与えられるもの
x = np.array([1.0, -2.0, 3.0])

# 重み行列とバイアス
W = np.array([[0.5, -1.0, 0.3],
              [0.8,  0.2, -0.5],
              [-0.6, 0.4,  1.0]])
b = np.array([0.1, -0.2, 0.3])

# 線形変換（こっちが本命ね）
z = linear_transform(x, W, b)

# 活性化関数の両方を適用
out_relu = relu(z)
out_neg_relu = neg_relu(z)
out_neg_abs = np.abs(out_neg_relu)   # 👈 絶対値に変換
# 統合（愛 + エグみを正にしたもの）これは処理として必要になるかもと思って入れてます
out_combined = out_relu + out_neg_abs

print("入力ベクトル:", x)
print("線形変換後:", z)
print("ReLU結果 (正の責任：愛):", out_relu)
print("逆ReLU結果 (負の責任：えぐみ):", out_neg_relu)
print("逆ReLUの絶対値 (負の強さ):", out_neg_abs)  # ここが「エグみの強さ」
print("ReLUと逆ReLUの和（全体の責任）:", out_relu + out_neg_relu)  # 元の線形変換結果に戻るはず
print("ReLUと逆ReLUの差（安全地帯の幅の広さ）:", out_relu - out_neg_relu)  # これは統合のやつと同じことしてるのよね
print("積(商はあんま意味ない可能性がある):", out_relu * out_neg_relu)
print()
print("なお愛とえぐみの比較がこちら:")

for i, (pos, neg) in enumerate(zip(out_relu, out_neg_abs)):
    if pos > neg:
        result = "愛が優勢"
    elif pos < neg:
        result = "エグみが優勢"
    else:
        result = "拮抗"
    print(f"成分{i}: 正={pos:.2f}, 負(強さ)={neg:.2f} → {result}")
print()
print("よってダンサーがどういう配分に見えるかというと（感覚派の人間が曲聞いた時の流れ的に「こうじゃね？」って思うのを数値化すると）:", out_combined)