# ResponsibilityAllow_silu.py
import numpy as np


# 線形変換（責任ベクトルの投影）
def linear_transform(x, W, b):
    return np.dot(x, W) + b

# SiLU (Swish): x * sigmoid(x)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

# ===== 入力（例） =====
x = np.array([1.0, -2.0, 3.0])

# 重み行列とバイアス（例）
W = np.array([[0.5, -1.0, 0.3],
              [0.8,  0.2, -0.5],
              [-0.6, 0.4,  1.0]])
b = np.array([0.1, -0.2, 0.3])

# 1) 線形変換
z = linear_transform(x, W, b)

# 2) SiLU を適用（負側も“弱く”通す）
out_silu = silu(z)

# 3) 解析：正寄与 / 負寄与（強さ）
#    ReLU/逆ReLUと同様の見方を維持しつつ、SiLU後の値で評価
pos_part = np.maximum(0.0, out_silu)       # 正の寄与（愛）
neg_part = np.minimum(0.0, out_silu)       # 負の寄与（えぐみ：符号付き）
neg_strength = np.abs(neg_part)            # 負の“強さ”（絶対値）
combined_strength = pos_part + neg_strength  # 総合の強さ（符号情報を外した合成）

print("入力ベクトル:", x)
print("線形変換後 z:", z)
print("SiLU(z):      ", out_silu)
print("正の寄与(愛): ", pos_part)
print("負の寄与(えぐみ):", neg_part)
print("負の強さ(|えぐみ|):", neg_strength)
print("総合の強さ(愛 + |えぐみ|):", combined_strength)
print()
print("愛とえぐみの比較（SiLU後の寄与ベース）:")

for i, (pos, neg_abs) in enumerate(zip(pos_part, neg_strength)):
    if pos > neg_abs:
        result = "愛が優勢"
    elif pos < neg_abs:
        result = "エグみが優勢"
    else:
        result = "拮抗"
    print(f"成分{i}: 正={pos:.4f}, 負(強さ)={neg_abs:.4f} → {result}")

# 参考：SiLUは ReLU と違って負側も0に潰さず“弱く”残す。
#       そのため pos/neg のバランス評価や総合強さが滑らかに変化し、情報を捨てにくい。
print()
print("よってダンサーがどういう配分に見えるかというと（感覚派の人間が曲聞いた時の流れ的に「こうじゃね？」って思うのを数値化すると）:", combined_strength)