# improv_engine_ms.py
# -*- coding: utf-8 -*-
"""
即興ダンス生成ミニエンジン（ミリ秒次元版）
- Δt = 0.001〜0.01秒で更新（人間の無意識反応スケール）
- Genesys（量子揺らぎ）＝ ms 更新
- Coherence（空気感）＝ 20〜50ms更新
- Motion（行動）＝ 拍で更新
- Stillness（評価）＝ 200〜300ms更新
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List

# ===== ▼ パラメータ =====
DT = 0.002                  # 2ms（必要なら0.001にしてOK）
COHERENCE_INTERVAL = 0.030  # 30msで更新
MOTION_INTERVAL = 0.500     # 0.5s → BPM同期で後で置換
STILLNESS_INTERVAL = 0.250  # 250msで評価
RUNTIME = 60                # 全体60秒走らせる

SEED = 42
random.seed(SEED)

LOGISTIC_R = 3.77
CHAOS_MIX = 0.25
EWMA_ALPHA = 0.20
PRINT_DETAIL = True

# ===== ▼ モチーフ =====
MOTIFS = {
    "freeze":      {"layer": "Stillness",  "energy": 0.1,  "surprise": 0.2,  "align": 0.9},
    "micro_hold":  {"layer": "Stillness",  "energy": 0.2,  "surprise": 0.3,  "align": 0.8},
    "breath_lock": {"layer": "Stillness",  "energy": 0.25, "surprise": 0.25, "align": 0.85},

    "wave":        {"layer": "Coherence",  "energy": 0.4,  "surprise": 0.5,  "align": 0.7},
    "glide":       {"layer": "Coherence",  "energy": 0.45, "surprise": 0.45,"align": 0.7},
    "iso":         {"layer": "Coherence",  "energy": 0.5,  "surprise": 0.5,  "align": 0.65},

    "hit":         {"layer": "Motion",     "energy": 0.8,  "surprise": 0.6,  "align": 0.5},
    "tut":         {"layer": "Motion",     "energy": 0.7,  "surprise": 0.55, "align": 0.55},
    "footwork":    {"layer": "Motion",     "energy": 0.75, "surprise": 0.65,"align": 0.5},
}

LAYERS = ["Stillness", "Coherence", "Motion"]

def softmax(d: Dict[str, float]):
    mx = max(d.values())
    ex = {k: math.exp(v - mx) for k,v in d.items()}
    s = sum(ex.values())
    return {k: v/s for k, v in ex.items()}

def choose_weighted(w):
    r = random.random()
    cum = 0
    for k, v in w.items():
        cum += v
        if r <= cum:
            return k
    return list(w.keys())[-1]

# ===== ▼ カオス関数 =====
class Logistic:
    def __init__(self, x0=0.473, r=LOGISTIC_R):
        self.x = x0
        self.r = r
    def step(self):
        self.x = self.r * self.x * (1 - self.x)
        return self.x

# ===== ▼ Genesys（ms更新） =====
@dataclass
class Genesys:
    base_logits: Dict[str, float] = field(default_factory=dict)
    chaos: Logistic = field(default_factory=Logistic)

    def init_uniform(self):
        for k, meta in MOTIFS.items():
            base = {"Stillness":0.3, "Coherence":0.2, "Motion":0.1}[meta["layer"]]
            self.base_logits[k] = base + random.uniform(-0.02,0.02)

    def distribution(self):
        chaos_x = self.chaos.step()
        chaos_vec = {
            k: abs(math.sin(chaos_x * (i+1) * math.pi))
            for i, k in enumerate(self.base_logits)
        }
        s = sum(chaos_vec.values())
        chaos_vec = {k: v/s for k,v in chaos_vec.items()}
        mix = {
            k: (1-CHAOS_MIX)*self.base_logits[k] + CHAOS_MIX*chaos_vec[k]
            for k in self.base_logits
        }
        return softmax(mix)

# ===== ▼ Coherence（ファジィ、30msごと） =====
@dataclass
class FuzzyCoherence:
    def eval(self, e, s, a):
        low  = lambda x: max(0, 1-2*x)
        mid  = lambda x: max(0, 1-abs(2*x-1))
        high = lambda x: max(0, 2*x-1)

        still = max(low(e)*high(a), low(e)*mid(a), mid(e)*high(a))
        burst = max(high(e)*mid(a), high(e)*low(a), high(s))
        blend = max(mid(e)*mid(a), mid(s), low(s)*mid(a))

        tot = still + burst + blend + 1e-9
        return {
            "stillness": still/tot,
            "burst": burst/tot,
            "blend": blend/tot,
        }

# ===== ▼ Motion（0.5秒ごと or 1拍ごと） =====
@dataclass
class MotionSelector:
    layer_bias: Dict[str,float] = field(default_factory=lambda: {"Stillness":0,"Coherence":0,"Motion":0})
    motif_bonus: Dict[str,float] = field(default_factory=dict)

    def pick(self, dist, fz):
        layer_w = {
            "Stillness": 0.8*fz["stillness"] + 0.4*fz["blend"],
            "Coherence": 0.8*fz["blend"] + 0.2*fz["stillness"] + 0.2*fz["burst"],
            "Motion":    0.8*fz["burst"] + 0.4*fz["blend"],
        }
        score = {}
        for k,p in dist.items():
            L = MOTIFS[k]["layer"]
            bonus = self.motif_bonus.get(k, 0)
            score[k] = p + 0.6*layer_w[L] + 0.2*bonus + self.layer_bias[L]
        return choose_weighted(softmax(score))

# ===== ▼ 報酬（微小） =====
def reward(prev, now):
    r = 0.0
    if prev == now: r -= 0.05
    r += 0.2 * MOTIFS[now]["surprise"]
    return r

# ===== ▼ メイン（msループ） =====
def run_ms_engine():
    genesys = Genesys()
    genesys.init_uniform()
    fuzzy = FuzzyCoherence()
    motion = MotionSelector(motif_bonus={k:0 for k in MOTIFS})

    t = 0.0
    prev_action = None
    last_coh = 0.0
    last_mot = 0.0
    last_stl = 0.0

    print("=== Start Improvisation (ms scale) ===")

    while t < RUNTIME:

        # --- Genesys（毎ms） ---
        dist = genesys.distribution()

        # --- Coherence（30ms更新） ---
        if t - last_coh >= COHERENCE_INTERVAL:
            e = sum(MOTIFS[k]["energy"]   * p for k,p in dist.items())
            s = sum(MOTIFS[k]["surprise"] * p for k,p in dist.items())
            a = sum(MOTIFS[k]["align"]    * p for k,p in dist.items())
            fz = fuzzy.eval(e, s, a)
            last_coh = t

        # --- Motion（0.5秒ごと、後で拍に置換） ---
        if t - last_mot >= MOTION_INTERVAL:
            act = motion.pick(dist, fz)
            last_mot = t

            # ログ
            if PRINT_DETAIL:
                print(f"[t={t:6.3f}s] act={act:12s} layer={MOTIFS[act]['layer']:10s}")

            # 報酬
            r = reward(prev_action, act)
            L = MOTIFS[act]["layer"]

            motion.motif_bonus[act] = (1-EWMA_ALPHA)*motion.motif_bonus[act] + EWMA_ALPHA*r
            motion.layer_bias[L] = (1-EWMA_ALPHA)*motion.layer_bias[L] + EWMA_ALPHA*(r*0.3)

            prev_action = act

        # --- Stillness評価（250ms） ---
        if t - last_stl >= STILLNESS_INTERVAL:
            # 特に何もしないけど、後で安定度メトリクス入れる
            last_stl = t

        # --- 時間進行 ---
        t += DT
        time.sleep(DT * 0.0001)  # CPU優しめ（削除可）大元0.2秒ね。

    print("=== End ===")

if __name__ == "__main__":
    run_ms_engine()
