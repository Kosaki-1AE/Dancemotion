# responsibility_allow/analyze.py

from typing import Callable, Dict, List, Optional

import numpy as np
from complex_ops import split_real_imag
from contrib import split_contrib
from fluct import apply_psych_fluctuation
from linops import linear_transform

Act = Callable[[np.ndarray], np.ndarray]

def analyze_activation_complex(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    pos_fn: Act,
    neg_fn: Act,
    *,
    tau: float = 1.0,
    fluct_mode: str = "none",
    fluct_kwargs: Optional[Dict] = None,
    center: str | float = "auto",
    contrib_mode: str = "separate",  # ← 追加: "separate" or "strength"
    verbose: bool = False,
) -> Dict:
    if fluct_kwargs is None:
        fluct_kwargs = {}

    # 線形変換して複素にキャスト
    z = linear_transform(x, W, b).astype(np.complex128)
    z_real, z_imag = split_real_imag(z)

    # 実部で従来通りの処理
    out_pos = pos_fn(z_real)
    out_neg = neg_fn(z_real)

    # ← ここが変更点：解釈モードを渡す
    pos_part, neg_strength = split_contrib(out_pos, out_neg, center=center, mode=contrib_mode)

    pos_sum = float(pos_part.sum())
    neg_sum = float(neg_strength.sum())
    delta   = pos_sum - neg_sum

    p_pos = 1.0 / (1.0 + np.exp(-delta / max(tau, 1e-6)))
    p_hat, label = apply_psych_fluctuation(p_pos, mode=fluct_mode, **fluct_kwargs)

    if verbose:
        print(f"[complex] delta={delta:.3f} p_pos={p_pos:.3f} p_hat={p_hat:.3f} label={label}")

    return {
        "z_real": z_real,
        "z_imag": z_imag,    # ←虚部（裏側ログ）
        "out_pos": out_pos,
        "out_neg": out_neg,
        "pos_sum": pos_sum,
        "neg_sum": neg_sum,
        "delta": delta,
        "p_pos": p_pos,
        "p_hat": p_hat,
        "label": int(label),
        "pos_part": pos_part,
        "neg_strength": neg_strength,
        "hidden": z_imag,
    }

def will_event_complex(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    pos_fn: Act,
    neg_fn: Act,
    *,
    theta: float = 0.6,
    tau: float = 1.0,
    fluct_mode: str = "logit_gauss",
    fluct_kwargs: Optional[Dict] = None,
    center: str | float = "auto",
    contrib_mode: str = "separate",  # ← 追加
) -> Dict:
    res = analyze_activation_complex(
        x, W, b, pos_fn, neg_fn,
        tau=tau,
        fluct_mode=fluct_mode,
        fluct_kwargs=fluct_kwargs or {},
        center=center,
        contrib_mode=contrib_mode,  # ← 追加
        verbose=False,
    )
    polarity  = 1 if res["pos_sum"] >= res["neg_sum"] else -1
    intensity = abs(res["delta"])
    commit    = bool(res["p_hat"] >= theta)
    return {
        "commit": commit,
        "p_hat": float(res["p_hat"]),
        "theta": float(theta),
        "polarity": polarity,
        "intensity": float(intensity),
        "detail": res,   # res["hidden"] に虚部が入る
    }

def analyze_activation(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    pos_fn: Act,
    neg_fn: Act,
    *,
    tau: float = 1.0,
    topk: Optional[int] = None,
    fluct_mode: str = "none",
    fluct_kwargs: Optional[Dict] = None,
    center: str | float = "auto",
    contrib_mode: str = "separate",  # ← 追加
    name_pos: Optional[str] = None,
    name_neg: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    if fluct_kwargs is None:
        fluct_kwargs = {}

    z = linear_transform(x, W, b)
    out_pos = pos_fn(z)
    out_neg = neg_fn(z)

    # ← ここが変更点
    pos_part, neg_strength = split_contrib(out_pos, out_neg, center=center, mode=contrib_mode)

    pos_sum = float(pos_part.sum())
    neg_sum = float(neg_strength.sum())
    delta = pos_sum - neg_sum

    p_pos = 1.0 / (1.0 + np.exp(-delta / max(tau, 1e-6)))
    p_hat, label = apply_psych_fluctuation(p_pos, mode=fluct_mode, **fluct_kwargs)

    highlights: List[Dict] = []
    if topk:
        # mode="separate" でも "strength" でも「強いほう」を拾えるよう対称な指標を使用
        strength = np.abs(pos_part) + np.abs(neg_strength)
        idx = np.argsort(-strength)[:topk]
        for i in idx:
            p, n = float(pos_part[i]), float(neg_strength[i])
            verdict = "愛が強い" if p > n else ("えぐみが強い" if p < n else "拮抗")
            highlights.append({"idx": int(i), "pos": p, "neg": n, "verdict": verdict})

    if verbose:
        print(f"\n=== {name_pos or 'Pos'} & {name_neg or 'Neg'} ===")
        print(f"delta={delta:.3f}  p_pos={p_pos:.3f}  p_hat={p_hat:.3f}  label={label}")

    return {
        "z": z,
        "out_pos": out_pos,
        "out_neg": out_neg,
        "pos_sum": pos_sum,
        "neg_sum": neg_sum,
        "delta": delta,
        "p_pos": p_pos,
        "p_hat": p_hat,
        "label": int(label),
        "pos_part": pos_part,
        "neg_strength": neg_strength,
        "topk": highlights,
    }

def will_event(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    pos_fn: Act,
    neg_fn: Act,
    *,
    theta: float = 0.6,
    tau: float = 1.0,
    fluct_mode: str = "logit_gauss",
    fluct_kwargs: Optional[Dict] = None,
    center: str | float = "auto",
    contrib_mode: str = "separate",  # ← 追加
) -> Dict:
    res = analyze_activation(
        x, W, b, pos_fn, neg_fn,
        tau=tau,
        topk=None,
        fluct_mode=fluct_mode,
        fluct_kwargs=fluct_kwargs or {},
        center=center,
        contrib_mode=contrib_mode,  # ← 追加
        name_pos=getattr(pos_fn, "__name__", "pos"),
        name_neg=getattr(neg_fn, "__name__", "neg"),
        verbose=False,
    )
    polarity = 1 if res["pos_sum"] >= res["neg_sum"] else -1
    intensity = abs(res["delta"])
    commit = bool(res["p_hat"] >= theta)
    return {
        "commit": commit,
        "p_hat": float(res["p_hat"]),
        "theta": float(theta),
        "polarity": polarity,
        "intensity": float(intensity),
        "detail": res,
    }
