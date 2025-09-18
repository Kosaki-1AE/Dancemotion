# -*- coding: utf-8 -*-
# ResponsibilityAllow_acts.py — 活性化関係だけ＆出力フォーマットはReLU版とおそろい

# -*- coding: utf-8 -*-
from typing import Callable, Dict
import numpy as np

# === 正側活性 ===
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0.0, x, alpha * x)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def silu(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-x))
    return x * s

def gelu(x: np.ndarray) -> np.ndarray:
    # Hendrycks & Gimpel 近似
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * (x**3))))

# === 負版を作るファクトリ ===
def negify(f: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def _neg(x: np.ndarray) -> np.ndarray:
        return -f(-x)
    _neg.__name__ = f"neg_{f.__name__}"
    return _neg

# === 負版 実体 ===
neg_relu       = negify(relu)
def neg_leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return -leaky_relu(-x, alpha=alpha)
neg_sigmoid    = negify(sigmoid)
neg_tanh       = negify(tanh)    # tanhは奇関数→実質同じ
neg_silu       = negify(silu)
neg_gelu       = negify(gelu)

# === レジストリ ===
_BASE: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": relu, "leaky_relu": leaky_relu, "sigmoid": sigmoid, "tanh": tanh,
    "silu": silu, "gelu": gelu,
    "neg_relu": neg_relu, "neg_leaky_relu": neg_leaky_relu, "neg_sigmoid": neg_sigmoid,
    "neg_tanh": neg_tanh, "neg_silu": neg_silu, "neg_gelu": neg_gelu,
}

# === 公開シンボル ===
__all__ = [
    "relu", "leaky_relu", "sigmoid", "tanh", "silu", "gelu",
    "negify",
    "neg_relu", "neg_leaky_relu", "neg_sigmoid", "neg_tanh", "neg_silu", "neg_gelu",
    "_BASE",
]

ACTS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
for name, fn in _BASE.items():
    ACTS[name] = fn
    ACTS[f"neg_{name}"] = negify(fn)

def get_activation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in ACTS:
        raise KeyError(f"unknown activation: {name}")
    return ACTS[name]

# ========= 線形変換 =========
def linear_transform(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.dot(x, W) + b

# ========= デモ：出力をReLU版フォーマットで表示 =========
if __name__ == "__main__":
    # --- 回す活性のリスト（正/負をペアで） ---
    POS_NAMES = ["relu", "leaky_relu", "sigmoid", "tanh", "silu", "gelu"]
    NEG_NAMES = [f"neg_{n}" for n in POS_NAMES]

    # 入力ベクトル
    x = np.array([1.0, -2.0, 3.0])

    # 重み行列とバイアス（任意の値でOK）
    W = np.array([[0.5, -1.0, 0.3],
                  [0.8,  0.2, -0.5],
                  [-0.6, 0.4,  1.0]])
    b = np.array([0.1, -0.2, 0.3])

    # 線形変換（固定）
    z = linear_transform(x, W, b)

    # 各ペアを順に評価・表示
    for POS_ACT_NAME, NEG_ACT_NAME in zip(POS_NAMES, NEG_NAMES):
        pos_fn = get_activation(POS_ACT_NAME)
        neg_fn = get_activation(NEG_ACT_NAME)

        # 活性化（正＝愛 / 負＝えぐみ）
        out_pos = pos_fn(z)
        out_neg = neg_fn(z)
        out_neg_abs = np.abs(out_neg)            # えぐみの“強さ”（正値にする）
        out_combined = out_pos + out_neg_abs     # 愛 + えぐみ強さ（知覚的な総量イメージ）

        # ====== 出力（ReLU版と同じ見た目）======
        print("="*72)
        print(f"使用活性（正/負）: {POS_ACT_NAME} / {NEG_ACT_NAME}")
        print("入力ベクトル:", x)
        print("線形変換後:", z)
        print("正の責任（愛）:", out_pos)
        print("負の責任（えぐみ）:", out_neg)
        print("負の強さ（|えぐみ|）:", out_neg_abs)
        print("愛とえぐみの合算（愛 + |えぐみ|）:", out_combined)
        print("正負の和（検算・原点復帰）:", out_pos + out_neg)  # 対称ペアなら z に一致
        print("正負の差（安全地帯の幅）:", out_pos - out_neg)
        print()
        print("なお愛とえぐみの比較がこちら:")
        for i, (pos, negs) in enumerate(zip(out_pos, out_neg_abs)):
            if pos > negs:
                result = "愛が優勢"
            elif pos < negs:
                result = "エグみが優勢"
            else:
                result = "拮抗"
            print(f"成分{i}: 正={pos:.2f}, 負(強さ)={negs:.2f} → {result}")
        print()
        print("よってダンサーがどういう配分に見えるか（感覚派の“流れ”の数値化）:", out_combined)
        print()
