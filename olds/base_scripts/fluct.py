# responsibility_allow/fluct.py　心理的ゆらぎを加えるとこぉろ
from typing import Optional, Tuple

import numpy as np


def apply_psych_fluctuation(
    p_pos: float,
    mode: str = "none",
    *,
    sigma: float = 0.05,
    kappa: float = 50.0,
    thr_mu: float = 0.5,
    thr_sigma: float = 0.05,
    eps: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[float, int]:
    rng = np.random.default_rng(seed)

    def _logit(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    def _sigm(z):
        return 1.0 / (1.0 + np.exp(-z))

    p = float(np.clip(p_pos, 0.0, 1.0))

    if mode == "none":
        return p, int(p >= 0.5)

    if mode == "gauss_p":
        ph = np.clip(p + rng.normal(0.0, sigma), 0.0, 1.0)
        return ph, int(ph >= 0.5)

    if mode == "logit_gauss":
        zh = _logit(p) + rng.normal(0.0, sigma)
        ph = _sigm(zh)
        return ph, int(ph >= 0.5)

    if mode == "beta":
        a = max(1e-6, p * kappa)
        b = max(1e-6, (1 - p) * kappa)
        ph = float(rng.beta(a, b))
        return ph, int(ph >= 0.5)

    if mode == "rand_threshold":
        thr = float(np.clip(rng.normal(thr_mu, thr_sigma), 0.0, 1.0))
        return p, int(p >= thr)

    if mode == "eps_flip":
        base = int(p >= 0.5)
        if rng.random() < eps:
            base = 1 - base
        return p, base

    if mode == "sample":
        y = int(rng.random() < p)
        return p, y

    raise ValueError(f"unknown mode: {mode}")
