# -*- coding: utf-8 -*-
# complex_ops.py — 複素責任ベクトル用の演算まとめ

import numpy as np

# ===== 基本ベクトル生成 =====
def make_complex_vector(d: int, mode: str = "imag") -> np.ndarray:
    """
    複素責任ベクトルの初期化
    mode="imag" → 純虚数基底
    mode="rand" → ランダム複素基底
    """
    if mode == "imag":
        v = 1j * np.eye(1, d, 0).flatten()  # e1 に i を置く
    elif mode == "rand":
        v = np.random.randn(d) + 1j*np.random.randn(d)
    else:
        v = np.zeros(d, dtype=np.complex128)
    return v.astype(np.complex128)

# ===== 複素線形変換 =====
def complex_linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    複素責任ベクトルに対する線形変換
    """
    return np.dot(x, W.astype(np.complex128)) + b

# ===== 虚数活性化関数 =====
def i_relu(x: np.ndarray) -> np.ndarray:
    """実部と虚部に独立にReLU適用"""
    return np.maximum(0.0, x.real) + 1j * np.maximum(0.0, x.imag)

def i_sigmoid(x: np.ndarray) -> np.ndarray:
    """複素sigmoid (実部/虚部に別々適用)"""
    sig = lambda y: 1.0 / (1.0 + np.exp(-y))
    return sig(x.real) + 1j*sig(x.imag)

# ===== 分解 =====
def split_real_imag(x: np.ndarray):
    """
    複素責任ベクトルを実部(観測可能)と虚部(潜在)に分ける
    """
    return x.real, x.imag
