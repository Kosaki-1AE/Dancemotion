# responsibility_allow/analyze.py
import numpy as np
from typing import Callable, Dict, Optional, List
from linops import linear_transform
from contrib import split_contrib
from fluct import apply_psych_fluctuation
from complex_ops import split_real_imag

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
    pos_part, neg_strength = split_contrib(out_pos, out_neg, center=center)

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
) -> Dict:
    res = analyze_activation_complex(
        x, W, b, pos_fn, neg_fn,
        tau=tau,
        fluct_mode=fluct_mode,
        fluct_kwargs=fluct_kwargs or {},
        center=center,
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
# === 追記ここまで ===

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
    name_pos: Optional[str] = None,
    name_neg: Optional[str] = None,
    verbose: bool = False,
) -> Dict:
    if fluct_kwargs is None:
        fluct_kwargs = {}

    z = linear_transform(x, W, b)
    out_pos = pos_fn(z)
    out_neg = neg_fn(z)

    pos_part, neg_strength = split_contrib(out_pos, out_neg, center=center)

    pos_sum = float(pos_part.sum())
    neg_sum = float(neg_strength.sum())
    delta = pos_sum - neg_sum

    p_pos = 1.0 / (1.0 + np.exp(-delta / max(tau, 1e-6)))
    p_hat, label = apply_psych_fluctuation(p_pos, mode=fluct_mode, **fluct_kwargs)

    highlights: List[Dict] = []
    if topk:
        strength = pos_part + neg_strength
        idx = np.argsort(-strength)[:topk]
        for i in idx:
            p, n = pos_part[i], neg_strength[i]
            verdict = "愛が強い" if p > n else ("えぐみが強い" if p < n else "拮抗")
            highlights.append({"idx": int(i), "pos": float(p), "neg": float(n), "verdict": verdict})

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
) -> Dict:
    res = analyze_activation(
        x, W, b, pos_fn, neg_fn,
        tau=tau,
        topk=None,
        fluct_mode=fluct_mode,
        fluct_kwargs=fluct_kwargs or {},
        center=center,
        name_pos=getattr(pos_fn, "__name__", "pos"),
        name_neg=getattr(neg_fn, "__name__", "neg"),
        verbose=False,
    )
    polarity = 1 if res["pos_sum"] >= res["neg_sum"] else -1
    intensity = abs(res["delta"])
    commit = bool(res["p_hat"] >= theta)
    return {"commit": commit, "p_hat": float(res["p_hat"]), "theta": float(theta),
            "polarity": polarity, "intensity": float(intensity), "detail": res}
