# responsibility_allow/flow.py　責任ベクトル（実数・複素数）に変換＆即興的に“進む／待つ”判断の更新と学習する
from typing import Callable, Dict, Optional

import numpy as np
from analyze import analyze_activation, analyze_activation_complex
from complex_ops import complex_linear, make_complex_vector, split_real_imag

Act = Callable[[np.ndarray], np.ndarray]

# ===============================
# エンコーダ
# ===============================
class HashEncoder:
    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.signs = rng.choice([-1.0, 1.0], size=dim)

    def vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.split():
            h = abs(hash(tok)) % self.dim
            v[h] += self.signs[h]
        n = np.linalg.norm(v) or 1.0
        return v / n

# ===============================
# ヘッド
# ===============================
class FlowHead:
    def __init__(self, in_dim: int = 256, lr: float = 0.2, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(in_dim, 2))
        self.lr = lr

    def predict_dir(self, m: np.ndarray) -> np.ndarray:
        y = m @ self.W
        n = np.linalg.norm(y) or 1.0
        return y / n

    def update(self, m: np.ndarray, target_dir: np.ndarray, pred_dir: np.ndarray, gain: float = 1.0):
        g = (target_dir - pred_dir) * gain
        self.W += self.lr * np.outer(m, g)

# ===============================
# 実数責任ベクトル版
# ===============================
class FlowState:
    def __init__(self, dim: int = 256, theta0: float = 0.6, ema: float = 0.05):
        self.enc = HashEncoder(dim=dim)
        self.head = FlowHead(in_dim=dim)
        self.last_dir = np.zeros(2, dtype=np.float32)
        self.theta = theta0
        self.theta_ema = ema

    def step(self, msg: str, base_delta: float, decider_fn: Callable[[float], Dict]) -> Dict:
        m = self.enc.vec(msg)
        d_hat = self.head.predict_dir(m)

        extra_push = 0.4 * float(d_hat @ (self.last_dir if np.any(self.last_dir) else d_hat))
        out = decider_fn(base_delta + extra_push)

        if out["commit"]:
            target = (1.0 if out["polarity"] >= 0 else -1.0) * d_hat
            gain = 1.0 + 0.5 * abs(out["p_hat"] - self.theta)
            self.last_dir = 0.8 * self.last_dir + 0.2 * d_hat
        else:
            target = 0.5 * (self.last_dir if np.any(self.last_dir) else d_hat)
            gain = 0.3
            self.last_dir = 0.95 * self.last_dir
        self.head.update(m, target, d_hat, gain=gain)

        goal_rate = 0.55
        err = (1.0 if out["commit"] else 0.0) - goal_rate
        self.theta = float(np.clip(self.theta + (-self.theta_ema) * err, 0.4, 0.8))

        out.update({"theta_now": self.theta, "d_hat": d_hat.tolist()})
        return out

# ===============================
# 複素責任ベクトル版
# ===============================
class FlowStateComplex:
    def __init__(self, dim: int = 256, theta0: float = 0.6, ema: float = 0.05):
        self.enc = HashEncoder(dim=dim)
        self.head = FlowHead(in_dim=dim)
        self.R_hat = make_complex_vector(dim, mode="imag")  # 複素責任ベクトル
        self.last_dir = np.zeros(2, dtype=np.float32)
        self.theta = theta0
        self.theta_ema = ema

    def step(self, msg: str, base_delta: float, decider_fn: Callable[[float], Dict]) -> Dict:
        m = self.enc.vec(msg)
        d_hat = self.head.predict_dir(m)

        # 複素責任ベクトル更新
        W = np.random.randn(len(self.R_hat), len(self.R_hat))
        b = np.random.randn(len(self.R_hat))
        self.R_hat = complex_linear(self.R_hat, W, b)
        R_obs, R_hidden = split_real_imag(self.R_hat)

        extra_push = 0.4 * float(d_hat @ (self.last_dir if np.any(self.last_dir) else d_hat))
        out = decider_fn(base_delta + extra_push)

        if out["commit"]:
            target = (1.0 if out["polarity"] >= 0 else -1.0) * d_hat
            gain = 1.0 + 0.5 * abs(out["p_hat"] - self.theta)
            self.last_dir = 0.8 * self.last_dir + 0.2 * d_hat
        else:
            target = 0.5 * (self.last_dir if np.any(self.last_dir) else d_hat)
            gain = 0.3
            self.last_dir = 0.95 * self.last_dir
        self.head.update(m, target, d_hat, gain=gain)

        goal_rate = 0.55
        err = (1.0 if out["commit"] else 0.0) - goal_rate
        self.theta = float(np.clip(self.theta + (-self.theta_ema) * err, 0.4, 0.8))

        out.update({
            "theta_now": self.theta,
            "d_hat": d_hat.tolist(),
            "R_obs": R_obs.tolist(),
            "R_hidden": R_hidden.tolist()
        })
        return out

# ===============================
# イベント判定コア（実数）
# ===============================
def make_ev_decider_core(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    *,
    pos_fn: Act,
    neg_fn: Act,
    tau: float = 1.0,
    center: str | float = "auto",
    fluct_mode: str = "logit_gauss",
    fluct_kwargs: Optional[Dict] = None,
    theta_init: float = 0.6,
) -> Callable[[float], Dict]:
    fl_kwargs = {"sigma": 0.35, "seed": 42}
    if fluct_kwargs:
        fl_kwargs.update(fluct_kwargs)

    def _score(delta_extra: float = 0.0) -> Dict:
        res = analyze_activation(
            x, W, b, pos_fn, neg_fn,
            tau=tau, topk=None,
            fluct_mode=fluct_mode, fluct_kwargs=fl_kwargs,
            center=center, verbose=False,
            name_pos=getattr(pos_fn, "__name__", "pos"),
            name_neg=getattr(neg_fn, "__name__", "neg"),
        )
        delta_mod = res["delta"] + delta_extra
        p_hat = 1.0 / (1.0 + np.exp(-delta_mod / max(tau, 1e-6)))
        commit = bool(p_hat >= theta_init)
        polarity = 1 if res["pos_sum"] >= res["neg_sum"] else -1
        return {"p_hat": float(p_hat), "commit": commit, "polarity": polarity}

    return _score

# ===============================
# イベント判定コア（複素）
# ===============================
def make_ev_decider_core_complex(
    x: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    *,
    pos_fn: Act,
    neg_fn: Act,
    tau: float = 1.0,
    center: str | float = "auto",
    fluct_mode: str = "logit_gauss",
    fluct_kwargs: Optional[Dict] = None,
    theta_init: float = 0.6,
) -> Callable[[float], Dict]:
    fl_kwargs = {"sigma": 0.35, "seed": 42}
    if fluct_kwargs:
        fl_kwargs.update(fluct_kwargs)

    def _score(delta_extra: float = 0.0) -> Dict:
        res = analyze_activation_complex(
            x, W, b, pos_fn, neg_fn,
            tau=tau,
            fluct_mode=fluct_mode,
            fluct_kwargs=fl_kwargs,
            center=center,
        )
        delta_mod = res["delta"] + delta_extra
        p_hat = 1.0 / (1.0 + np.exp(-delta_mod / max(tau, 1e-6)))
        commit = bool(p_hat >= theta_init)
        polarity = 1 if res["pos_sum"] >= res["neg_sum"] else -1
        return {
            "p_hat": float(p_hat),
            "commit": commit,
            "polarity": polarity,
            "hidden": res["hidden"].tolist()
        }

    return _score
