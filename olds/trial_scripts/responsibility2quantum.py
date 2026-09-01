# responsibility2quantum.py
# ----------------------------------------------------
# ResponsibilityAllow_acts → QuantumStillnessEngine の“配線”ブリッジ。
# 依存: numpy, qiskit, あなたの 2 ファイル:
#   - ResponsibilityAllow_acts.py
#   - qstillness_engine.py（前に作ったエンジン）
import math
from typing import Dict, Optional

import numpy as np

from qstillness_engine import QuantumStillnessEngine, SimParams
from ResponsibilityAllow_acts import (analyze_activation, neg_relu,
                                      neg_sigmoid, relu, sigmoid)


# --- 変換ユーティリティ ---
def delta_to_angle(delta: float, k_theta: float = 0.8) -> float:
    """δ→q0のRy角。過激化防止にtanh→クリップ。"""
    th = k_theta * np.tanh(delta) * math.pi
    return float(np.clip(th, -math.pi, math.pi))

def prob_to_bias(p_hat: float) -> float:
    """確信度→ancilla bias（0..1）。0.5からの距離を2倍して上限1。"""
    return float(np.clip(2.0 * abs(p_hat - 0.5), 0.0, 1.0))

def strength_to_mix(pos_part: np.ndarray, neg_strength: np.ndarray, k_m: float = 0.25) -> float:
    """寄与強度の合計からミックス量（0..1）を作る。"""
    s = float(pos_part.sum() + neg_strength.sum())
    return float(np.clip(k_m * s, 0.0, 1.0))

# --- 特徴の生成（デモ用）---
def next_features(x: np.ndarray, noise=0.15) -> np.ndarray:
    """デモ：徐々に揺らぐ入力ベクトル。外界ノイズ想定。"""
    return x + np.random.normal(0.0, noise, size=x.shape)

# --- メイン統合 ---
def run_responsibility_quantum_loop(
    T: int = 250,
    acts_mode: str = "relu",           # "relu" or "sigmoid"（他に差し替え可）
    seed: Optional[int] = 7
) -> Dict:
    if seed is not None:
        np.random.seed(seed)

    # 1) 初期化：特徴/重み（デモ）
    x = np.array([1.0, -0.5, 0.8, 0.2])
    W = np.array([[ 0.6, -0.2, 0.3, -0.5],
                  [ 0.1,  0.4, -0.7, 0.2],
                  [-0.3,  0.2, 0.5,  0.1],
                  [ 0.7, -0.6, 0.2,  0.4]])
    b = np.array([0.05, -0.1, 0.2, 0.0])

    # 2) 量子エンジン
    params = SimParams(T=T, tension_threshold=1.0, jitter_std=0.05)
    eng = QuantumStillnessEngine(params)

    # 3) 走査
    timeline = {
        "p_motion": [], "delta": [], "p_hat": [], "label": [],
        "mix": [], "angle": [], "bias": [], "events": []
    }

    for t in range(T):
        # 3-1) 特徴を更新（外界ゆらぎ）
        if t > 0:
            x = next_features(x, noise=0.12)

        # 3-2) 活性関数の選択
        if acts_mode == "relu":
            fn_pos, fn_neg = relu, neg_relu
            cent = "auto"
        elif acts_mode == "sigmoid":
            fn_pos, fn_neg = sigmoid, neg_sigmoid
            cent = 0.5
        else:
            raise ValueError("unknown acts_mode")

        # 3-3) 心理側の集約（君の根源API）
        ana = analyze_activation(
            x=x, W=W, b=b,
            pos_fn=fn_pos, neg_fn=fn_neg,
            name_pos="Pos", name_neg="Neg",
            tau=1.0, topk=None,
            fluct_mode="logit_gauss", fluct_kwargs={"sigma":0.35, "seed":None},
            center=cent, verbose=False
        )

        delta   = ana["delta"]
        p_hat   = ana["p_hat"]
        label   = ana["label"]
        pos_part, neg_strength = ana["pos_part"], ana["neg_strength"]

        # 3-4) 量子側へマッピング
        # a) 矢の角度（意思の強さ）
        angle = delta_to_angle(delta, k_theta=0.9)
        # ここでは q0 に回転を与える（Ry(angle)）
        # 既存エンジンのステップ前に“Stillness小操作”として注入
        # （簡単化のため: 小さな操作として RY(angle) を混ぜる）
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.ry(angle, 0)
        eng._sv = eng._sv.evolve(qc)

        # b) 閾値超え＝稲妻（label==1 をトリガに）
        if label == 1:
            bias = prob_to_bias(p_hat)
            eng._lightning_pulse(bias)
            eng.log.events.append(f"[t={t}] LIGHTNING(by bridge) bias={bias:.2f}")
            # ヒューマン側を軽く刺激
            eng.side.tension = 0.0
            eng.side.fear    = min(1.0, eng.side.fear + 0.4)
            eng.side.wonder  = min(1.0, eng.side.wonder + 0.2)
            eng.side.external_input = 1
            eng.side.sound_timer = max(eng.side.sound_timer, eng.params.sound_hold_steps)
        else:
            # 稲妻無しでも音の帰還ミックスは弱めにかけられる
            mix = strength_to_mix(pos_part, neg_strength, k_m=0.12)
            eng._sound_return_mix(mix)

        # 3-5) エンジンの自然ダイナミクスを 1 ステップ進める
        eng._step(t)

        # 3-6) ログ
        timeline["p_motion"].append(eng._prob_motion())
        timeline["delta"].append(float(delta))
        timeline["p_hat"].append(float(p_hat))
        timeline["label"].append(int(label))
        timeline["mix"].append(float(strength_to_mix(pos_part, neg_strength)))
        timeline["angle"].append(float(angle))
        timeline["bias"].append(float(prob_to_bias(p_hat)) if label == 1 else 0.0)
        if len(eng.log.events) and (timeline["events"][-1] if timeline["events"] else None) != eng.log.events[-1]:
            timeline["events"].append(eng.log.events[-1])

    return {"timeline": timeline, "events": timeline["events"], "engine_log": eng.log}
    
if __name__ == "__main__":
    out = run_responsibility_quantum_loop(T=180, acts_mode="relu", seed=42)
    print("events(sample):", out["events"][:5])
    pm = np.mean(out["timeline"]["p_motion"])
    print("mean P(Motion):", round(float(pm), 3))
