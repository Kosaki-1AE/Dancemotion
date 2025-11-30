# -*- coding: utf-8 -*-
# ResponsibilityAllow_acts.py  —  Randomマルチターン観測 + Flowオンライン学習 統合版
import numpy as np
from typing import Dict, Tuple, List

# ========= 基本活性 =========
def relu(x): return np.maximum(0.0, x)
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def tanh(x): return np.tanh(x)
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def silu(x): return x * sigmoid(x)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * (x**3))))

# ========= “負版”： f_neg(x) = - f(-x) =========
def negify(act_fn): return lambda x: -act_fn(-x)
neg_relu = negify(relu)
def neg_leaky_relu(x, alpha=0.01): return -leaky_relu(-x, alpha)
neg_sigmoid = negify(sigmoid)
neg_silu = negify(silu)
neg_gelu = negify(gelu)

# ========= 心理ゆらぎ =========
def apply_psych_fluctuation(
    p_pos, mode="none",
    sigma=0.05, kappa=50.0,
    thr_mu=0.5, thr_sigma=0.05,
    eps=0.0, seed=None
):
    rng = np.random.default_rng(seed)
    def _logit(p): p = np.clip(p,1e-12,1-1e-12); return np.log(p/(1-p))
    def _sigm(x): return 1.0/(1.0+np.exp(-x))
    p = float(np.clip(p_pos, 0.0, 1.0))

    if mode == "none":
        return p, (1 if p >= 0.5 else 0)
    if mode == "gauss_p":
        ph = np.clip(p + rng.normal(0.0, sigma), 0.0, 1.0)
        return ph, (1 if ph >= 0.5 else 0)
    if mode == "logit_gauss":
        zh = _logit(p) + rng.normal(0.0, sigma)
        ph = _sigm(zh); return ph, (1 if ph >= 0.5 else 0)
    if mode == "beta":
        a = max(1e-6, p * kappa); b = max(1e-6, (1-p) * kappa)
        ph = float(rng.beta(a,b)); return ph, (1 if ph >= 0.5 else 0)
    if mode == "rand_threshold":
        thr = float(np.clip(rng.normal(thr_mu, thr_sigma), 0.0, 1.0))
        return p, (1 if p >= thr else 0)
    if mode == "eps_flip":
        base = (1 if p >= 0.5 else 0)
        if rng.random() < eps: base = 1 - base
        return p, base
    if mode == "sample":
        y = int(np.random.random() < p)
        return p, y
    raise ValueError(f"unknown mode: {mode}")

def _split_contrib(out_pos: np.ndarray, out_neg: np.ndarray, center="auto"):
    if center == "auto":
        is_probish = (out_pos.min() >= 0.0) and (out_pos.max() <= 1.0) and (out_neg.max() <= 0.0)
        center = 0.5 if is_probish else 0.0
    if center == 0.5:
        pos_part = np.maximum(0.0, out_pos - 0.5)
        neg_strength = np.maximum(0.0, (-out_neg) - 0.5)
    else:
        pos_part = np.maximum(0.0, out_pos)
        neg_strength = np.maximum(0.0, -np.minimum(0.0, out_neg))
    return pos_part, neg_strength

# ========= 中核：寄与→Δ→確率 =========
def analyze_activation(
    x: np.ndarray, W: np.ndarray, b: np.ndarray,
    pos_fn, neg_fn,
    name_pos="Pos", name_neg="Neg",
    tau=1.0, topk=None,
    fluct_mode="none", fluct_kwargs=None,
    center="auto", verbose=False
):
    if fluct_kwargs is None: fluct_kwargs = {}
    z = np.dot(x, W) + b
    out_pos = pos_fn(z); out_neg = neg_fn(z)
    pos_part, neg_strength = _split_contrib(out_pos, out_neg, center=center)

    pos_sum = float(pos_part.sum())
    neg_sum = float(neg_strength.sum())
    delta   = pos_sum - neg_sum
    p_pos   = 1.0/(1.0+np.exp(-delta/max(tau,1e-6)))
    p_hat, label = apply_psych_fluctuation(p_pos, mode=fluct_mode, **fluct_kwargs)

    highlights = []
    if topk:
        strength = pos_part + neg_strength
        idx = np.argsort(-strength)[:topk]
        for i in idx:
            p, n = pos_part[i], neg_strength[i]
            verdict = "愛が強い" if p > n else ("えぐみが強い" if p < n else "拮抗")
            highlights.append({"idx": int(i), "pos": float(p), "neg": float(n), "verdict": verdict})

    if verbose:
        print(f"\n=== {name_pos} & {name_neg} ===")
        print(f"delta={delta:.3f}  p_pos={p_pos:.3f}  p_hat={p_hat:.3f}  label={label}")

    return {
        "z": z, "out_pos": out_pos, "out_neg": out_neg,
        "pos_sum": pos_sum, "neg_sum": neg_sum, "delta": delta,
        "p_pos": p_pos, "p_hat": p_hat, "label": int(label),
        "pos_part": pos_part, "neg_strength": neg_strength,
        "topk": highlights
    }

# ========= 意思発生（イベント化） =========
def will_event(
    x, W, b, pos_fn, neg_fn,
    theta=0.6, tau=1.0,
    fluct_mode="logit_gauss", fluct_kwargs=None,
    center="auto"
):
    res = analyze_activation(
        x, W, b, pos_fn, neg_fn,
        name_pos=pos_fn.__name__, name_neg=neg_fn.__name__,
        tau=tau, topk=None,
        fluct_mode=fluct_mode, fluct_kwargs=fluct_kwargs or {},
        center=center, verbose=False
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
        "detail": res
    }

# ========= Online “Flow”：最初ゼロ知識→勝手に流れを掴む =========
class HashEncoder:
    def __init__(self, dim=256, seed=0):
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

class FlowHead:
    def __init__(self, in_dim=256, lr=0.2, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(in_dim, 2))
        self.lr = lr
    def predict_dir(self, m: np.ndarray) -> np.ndarray:
        y = m @ self.W
        n = np.linalg.norm(y) or 1.0
        return y / n
    def update(self, m: np.ndarray, target_dir: np.ndarray, pred_dir: np.ndarray, gain=1.0):
        g = (target_dir - pred_dir) * gain
        self.W += self.lr * np.outer(m, g)

class FlowState:
    def __init__(self, dim=256, theta0=0.6, ema=0.05):
        self.enc = HashEncoder(dim=dim)
        self.head = FlowHead(in_dim=dim)
        self.last_dir = np.zeros(2, dtype=np.float32)
        self.theta = theta0
        self.theta_ema = ema
    def step(self, msg: str, base_delta: float, decider_fn) -> Dict:
        m = self.enc.vec(msg)
        d_hat = self.head.predict_dir(m)
        extra_push = 0.4 * float(d_hat @ (self.last_dir if np.any(self.last_dir) else d_hat))
        out = decider_fn(base_delta + extra_push)  # p_hat/commit/polarity 決定

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
        self.theta = float(np.clip(self.theta + (-self.theta_ema)*err, 0.4, 0.8))
        out.update({"theta_now": self.theta, "d_hat": d_hat.tolist()})
        return out

# ========= will_event と Flow を繋ぐ簡易デコーダ =========
def make_ev_decider_core(x, W, b, pos_fn=silu, neg_fn=neg_silu,
                         tau=1.0, center="auto",
                         fluct_mode="logit_gauss", fluct_kwargs=None,
                         theta_init=0.6):
    fl_kwargs = {"sigma":0.35, "seed":42}
    if fluct_kwargs: fl_kwargs.update(fluct_kwargs)

    def _score(delta_extra: float = 0.0):
        res = analyze_activation(
            x, W, b, pos_fn, neg_fn,
            name_pos=pos_fn.__name__, name_neg=neg_fn.__name__,
            tau=tau, topk=None,
            fluct_mode=fluct_mode, fluct_kwargs=fl_kwargs,
            center=center, verbose=False
        )
        delta_mod = res["delta"] + delta_extra
        p_hat = 1.0/(1.0 + np.exp(-delta_mod/max(tau,1e-6)))
        commit = bool(p_hat >= theta_init)
        polarity = 1 if res["pos_sum"] >= res["neg_sum"] else -1
        return {"p_hat": float(p_hat), "commit": commit, "polarity": polarity}
    return _score

# ========= デモ実行 =========
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 共有パラメタ（適当に調整OK）
    D = 6                      # 特徴次元（デモなので小さめ）
    x = rng.normal(0, 1, size=(D,))
    W = rng.normal(0, 0.8, size=(D, D))
    b = rng.normal(0, 0.2, size=(D,))

    # -------- (A) ランダム刺激で多ターン観測 --------
    print("\n=== (A) Random Multi-Turn Observation ===")
    for t in range(10):
        x_rand = rng.normal(0, 1, size=(D,))                  # 毎ターン刺激を変える
        out = will_event(
            x_rand, W, b, pos_fn=silu, neg_fn=neg_silu,
            theta=0.62, tau=1.0,
            fluct_mode="logit_gauss", fluct_kwargs={"sigma":0.35, "seed":t},
            center="auto"
        )
        print(f"t={t:02d}  commit={out['commit']}  p_hat={out['p_hat']:.3f}  "
              f"polarity={out['polarity']}  intensity={out['intensity']:.3f}")

    # -------- (B) Flow：ゼロ知識→勝手に流れ把握 --------
    print("\n=== (B) Flow Online Learning (Correspondence-like) ===")
    decider = make_ev_decider_core(
        x, W, b, pos_fn=silu, neg_fn=neg_silu,
        tau=1.0, center="auto",
        fluct_mode="logit_gauss", fluct_kwargs={"sigma":0.4, "seed":123},
        theta_init=0.60
    )
    flow = FlowState(dim=256, theta0=0.60, ema=0.05)

    # 文通風の仮メッセージ列（本番はここを自由入力に置換）
    msgs = [
        {"phase":"Explore",     "role":"you", "text":"目的：Stillness×矢×ノリエントロピーの相互作用を検証する。"},
        {"phase":"Hypothesize", "role":"you", "text":"仮説：Stillness↑で|Δ|↑、整合↑でcommit率↑。"},
        {"phase":"Counter",     "role":"you", "text":"反例：場の方向を急反転させてもcommitは維持されるか？"},
        {"phase":"Design",      "role":"ai",  "text":"実験設計：条件を変化させt=60までログ収集。"},
        {"phase":"Experiment",  "role":"ai",  "text":"実験：2×2条件でcommit率と|Δ|を算出。"},
        {"phase":"Evaluate",    "role":"you", "text":"評価：単調性/相関/反例を要約し仮説を判定。"},
        {"phase":"Commit",      "role":"you", "text":"決定：v0モデル採用。次は層間相互情報を追加。"},
    ]

    last_base_delta = 0.0
    for t, m in enumerate(msgs):
        out = flow.step(m["text"], base_delta=last_base_delta, decider_fn=decider)
        last_base_delta = 0.2 * last_base_delta + 0.1 * (1 if out["commit"] else -0.2)

        print(
            f"t={t:02d}  phase={m['phase']:<11s} role={m['role']:<3s}  "
            f"commit={out['commit']}  p_hat={out['p_hat']:.3f}  "
            f"theta={out['theta_now']:.2f}  d_hat={np.round(out['d_hat'],3)}  "
            f"msg='{m['text'][:24]}{'…' if len(m['text'])>24 else ''}'"
        )