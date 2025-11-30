# improv_engine.py
# -*- coding: utf-8 -*-
"""
即興ダンス生成ミニエンジン
- Flow: 4-8-16-8-4 拍（拍=ステップ）を基本ループに
- Genesys: アイデア分布(重ね合わせ)をロジスティック写像で微揺らぎ
- Coherence: ファジィ制御（Energy/Surprise/Alignment -> Stillness/Burst/Blend）
- Motion: ルール&確率でモチーフ選択（Stillness/Coherence/Motionの3層）
- Feedback: 報酬に応じて重みをEWMA更新
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ====== ユーザ調整パラメータ ======
FLOW_STEPS = [4, 8, 16, 8, 4]          # 君のフロー
TEMPO_BPM = 92                         # 1拍=60/BPM 秒
SEED = 42
LOGISTIC_R = 3.77                      # カオス寄り(3.7~3.9) / 低めで安定(3.2~3.5)
CHAOS_MIX = 0.25                       # 0~1: 重ね合わせ分布に混ぜるカオス量
EWMA_ALPHA = 0.25                      # 報酬の学習率
PRINT_DETAIL = True

random.seed(SEED)

# ====== モチーフ辞書（必要に応じて増やしてOK） ======
MOTIFS = {
    "freeze":      {"layer": "Stillness",  "energy": 0.1, "surprise": 0.2, "align": 0.9},
    "micro_hold":  {"layer": "Stillness",  "energy": 0.2, "surprise": 0.3, "align": 0.8},
    "breath_lock": {"layer": "Stillness",  "energy": 0.25,"surprise": 0.25,"align": 0.85},

    "wave":        {"layer": "Coherence",  "energy": 0.4, "surprise": 0.5, "align": 0.7},
    "glide":       {"layer": "Coherence",  "energy": 0.45,"surprise": 0.45,"align": 0.7},
    "iso":         {"layer": "Coherence",  "energy": 0.5, "surprise": 0.5, "align": 0.65},

    "hit":         {"layer": "Motion",     "energy": 0.8, "surprise": 0.6, "align": 0.5},
    "tut":         {"layer": "Motion",     "energy": 0.7, "surprise": 0.55,"align": 0.55},
    "footwork":    {"layer": "Motion",     "energy": 0.75,"surprise": 0.65,"align": 0.5},
}

# レイヤ順序（軽→重）
LAYERS = ["Stillness", "Coherence", "Motion"]

# ====== 便利関数 ======
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def softmax(d: Dict[str, float]) -> Dict[str, float]:
    mx = max(d.values())
    exps = {k: math.exp(v - mx) for k, v in d.items()}
    s = sum(exps.values())
    return {k: v/s for k, v in exps.items()}

def choose_weighted(weights: Dict[str, float]) -> str:
    r = random.random()
    cum = 0.0
    for k, w in weights.items():
        cum += w
        if r <= cum:
            return k
    return list(weights.keys())[-1]

# ====== カオス: ロジスティック写像 ======
class Logistic:
    def __init__(self, x0: float = 0.473, r: float = LOGISTIC_R):
        self.x = x0
        self.r = r

    def step(self) -> float:
        self.x = self.r * self.x * (1 - self.x)
        return self.x

# ====== Genesys: 重ね合わせ(分布) ======
@dataclass
class Genesys:
    base_logits: Dict[str, float] = field(default_factory=dict)
    chaos: Logistic = field(default_factory=Logistic)
    chaos_mix: float = CHAOS_MIX

    def init_uniform(self, motifs: Dict[str, dict]):
        # レイヤに軽いバイアス: Stillness>Coherence>Motion (初手静)
        for name, meta in motifs.items():
            base = {"Stillness": 0.3, "Coherence": 0.2, "Motion": 0.1}[meta["layer"]]
            self.base_logits[name] = base + random.uniform(-0.02, 0.02)

    def distribution(self) -> Dict[str, float]:
        # カオス微揺らぎ: 各モチーフにx∈(0,1)のランダム重みをブレンド
        chaos_x = self.chaos.step()
        chaos_vec = {k: abs(math.sin(chaos_x * (i+1) * math.pi)) for i, k in enumerate(self.base_logits)}
        # 正規化してblend
        csum = sum(chaos_vec.values())
        chaos_vec = {k: v/csum for k, v in chaos_vec.items()}
        mix = {k: (1 - self.chaos_mix) * self.base_logits[k] + self.chaos_mix * chaos_vec[k]
               for k in self.base_logits}
        return softmax(mix)

# ====== Coherence: ファジィ制御 ======
@dataclass
class FuzzyCoherence:
    # 3入力: energy/surprise/align -> 3出力: stillness/burst/blend
    def eval(self, e: float, s: float, a: float) -> Dict[str, float]:
        # シンプルな三角/台形メンバーシップ（手作り）
        low  = lambda x: max(0.0, 1.0 - 2.0*x)         # ~[0,0.5]
        mid  = lambda x: max(0.0, 1.0 - abs(2.0*x-1))  # ~peak at 0.5
        high = lambda x: max(0.0, 2.0*x-1.0)           # ~[0.5,1]

        # ルール例
        rule_still = max(low(e)*high(a), low(e)*mid(a), mid(e)*high(a))
        rule_burst = max(high(e)*mid(a), high(e)*low(a), high(s))
        rule_blend = max(mid(e)*mid(a), mid(s), low(s)*mid(a))

        # 正規化
        total = rule_still + rule_burst + rule_blend + 1e-9
        return {
            "stillness": rule_still/total,
            "burst":     rule_burst/total,
            "blend":     rule_blend/total
        }

# ====== Motion: 行動選択 ======
@dataclass
class MotionSelector:
    layer_bias: Dict[str, float] = field(default_factory=lambda: {"Stillness": 0.0,"Coherence": 0.0,"Motion": 0.0})
    motif_bonus: Dict[str, float] = field(default_factory=dict)

    def pick(self, dist: Dict[str, float], fuzzy_out: Dict[str, float]) -> str:
        # レイヤ重み
        layer_w = {
            "Stillness": 0.8*fuzzy_out["stillness"] + 0.4*fuzzy_out["blend"],
            "Coherence": 0.8*fuzzy_out["blend"]     + 0.2*fuzzy_out["stillness"] + 0.2*fuzzy_out["burst"],
            "Motion":    0.8*fuzzy_out["burst"]     + 0.4*fuzzy_out["blend"],
        }
        # 各モチーフの最終重み
        final = {}
        for k, p in dist.items():
            L = MOTIFS[k]["layer"]
            bonus = self.motif_bonus.get(k, 0.0)
            final[k] = p + self.layer_bias.get(L, 0.0) + 0.6*layer_w[L] + 0.2*bonus
        # softmaxして1つ選択
        final = softmax(final)
        return choose_weighted(final)

# ====== Feedback: 簡単な報酬 ======
def simple_reward(prev: str, now: str, step_idx: int, flow_idx: int) -> float:
    """ 例:
    - フローの谷(4や8)でStillness系→高報酬
    - 山(16)でMotion系→高報酬
    - 同一モチーフ連打は微ペナルティ
    """
    L = MOTIFS[now]["layer"]
    length = FLOW_STEPS[flow_idx]
    at_peak = (length == 16)
    at_valley = (length in (4, 8)) and not at_peak

    r = 0.0
    if at_peak and L == "Motion": r += 1.0
    if at_valley and L == "Stillness": r += 0.8
    if L == "Coherence": r += 0.4

    if prev == now: r -= 0.2
    # ほんの少しサプライズを加点
    r += 0.3 * MOTIFS[now]["surprise"]
    return r

# ====== メインループ ======
def run_session(cycles: int = 1):
    genesys = Genesys()
    genesys.init_uniform(MOTIFS)
    fuzzy = FuzzyCoherence()
    motion = MotionSelector(motif_bonus={k:0.0 for k in MOTIFS})

    timeline: List[Tuple[float,str,str]] = []
    prev = None

    beat_sec = 60.0 / TEMPO_BPM
    t_cursor = 0.0

    for c in range(cycles):
        for i, length in enumerate(FLOW_STEPS):
            for step in range(length):
                # 1) Genesys: 分布取得(重ね合わせ+カオス)
                dist = genesys.distribution()

                # 2) Coherence: ファジィ評価（分布の加重平均からシーン特徴を作る）
                e = sum(MOTIFS[k]["energy"]   * p for k, p in dist.items())
                s = sum(MOTIFS[k]["surprise"] * p for k, p in dist.items())
                a = sum(MOTIFS[k]["align"]    * p for k, p in dist.items())
                fz = fuzzy.eval(e, s, a)

                # 3) Motion: モチーフ決定
                act = motion.pick(dist, fz)

                # 4) Feedback: 報酬→重み更新
                r = simple_reward(prev, act, step, i)
                # モチーフ個別ボーナスとレイヤバイアスを更新(EWMA)
                motion.motif_bonus[act] = (1-EWMA_ALPHA)*motion.motif_bonus.get(act,0.0) + EWMA_ALPHA*r
                lay = MOTIFS[act]["layer"]
                motion.layer_bias[lay]  = (1-EWMA_ALPHA)*motion.layer_bias.get(lay,0.0) + EWMA_ALPHA*(r*0.5)

                # ログ
                timeline.append((t_cursor, act, lay))
                if PRINT_DETAIL:
                    print(f"[t={t_cursor:6.2f}s | beat {step+1:02d}/{length} in {FLOW_STEPS[i]}] "
                          f"{act:12s}  layer={lay:10s}  E={e:.2f} S={s:.2f} A={a:.2f}  r={r:+.2f}")

                # 次の拍へ
                prev = act
                t_cursor += beat_sec

    # 最終サマリ
    counts = {L:0 for L in LAYERS}
    for _, _, L in timeline:
        counts[L] += 1
    total = len(timeline)
    print("\n=== Summary ===")
    for L in LAYERS:
        pct = 100.0*counts[L]/total
        print(f"{L:10s}: {counts[L]:4d} steps ({pct:5.1f}%)")
    print(f"Total steps: {total}, Total time: {t_cursor:.1f}s @ {TEMPO_BPM} BPM")

if __name__ == "__main__":
    run_session(cycles=2)  # 4-8-16-8-4 を2周
