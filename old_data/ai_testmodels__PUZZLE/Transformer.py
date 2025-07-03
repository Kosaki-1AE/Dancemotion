import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# 入力ベクトル（3トークン、各次元4）
x = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])

# 重み初期化
W_q = np.random.randn(4, 4)
W_k = np.random.randn(4, 4)
W_v = np.random.randn(4, 4)

# Q, K, V 計算
Q = x @ W_q
K = x @ W_k
V = x @ W_v

# Attentionスコア計算
scores = Q @ K.T / np.sqrt(Q.shape[-1])  # スケーリング付き内積
weights = softmax(scores)                # softmaxで重み
output = weights @ V                    # 加重平均で出力

print("Attention weights:")
print(np.round(weights, 2))
print("Attention output:")
print(np.round(output, 2))
