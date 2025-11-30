import numpy as np

# ====== シンプルなニューラルネット ======
# 入力層 2 → 隠れ層 2 → 出力層 1

# 活性化関数（シグモイド）
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# トレーニングデータ (XORの例)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([[0],[1],[1],[0]])

# パラメータ初期化
np.random.seed(42)
W1 = np.random.rand(2, 2)  # 入力→隠れ層の重み
W2 = np.random.rand(2, 1)  # 隠れ層→出力の重み
b1 = np.zeros((1, 2))      # 隠れ層バイアス
b2 = np.zeros((1, 1))      # 出力層バイアス
output = sigmoid(np.dot(a1, W1) + b1)
output = sigmoid(np.dot(output, W2) + b2)

# 学習率
lr = 0.1

# ====== 学習ループ ======
for epoch in range(10000):
    # --- 順伝播 ---
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)  # 隠れ層出力

    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # 出力層出力

    # --- 誤差計算 ---
    error = y - a2

    # --- 逆伝播 ---
    d_a2 = error * sigmoid_derivative(a2)
    d_W2 = np.dot(a1.T, d_a2)
    d_b2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)
    d_W1 = np.dot(X.T, d_a1)
    d_b1 = np.sum(d_a1, axis=0, keepdims=True)

    # --- パラメータ更新 ---
    USE_LOCAL_BACKPROP = False  # Trueなら「直前版」
    if USE_LOCAL_BACKPROP:
        # 直前版: 各層で即座にパラメータ更新
        W2 += lr * d_W2
        b2 += lr * d_b2
    else:
        # 通常版: まとめてパラメータ更新
        W1 += lr * d_W1
        b1 += lr * d_b1
        W2 += lr * d_W2
        b2 += lr * d_b2

    # --- ログ出力 ---
    if epoch % 2000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ====== 結果 ======
print("\n最終出力:")
print(a2)
