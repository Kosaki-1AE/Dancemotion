# qstillness_unified_plus.py
# ------------------------------------------------------------
# 1ファイル統合: QuantumStillnessEngine + 責任→量子ブリッジ
# 依存: numpy, qiskit, ResponsibilityAllow_acts.py（同ディレクトリ）
# 使い方:
#   python qstillness_unified_plus.py
#   from qstillness_unified_plus import QuantumStillnessEngine, run_responsibility_quantum_loop
# ------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# === ここ重要: ユーザー提供の根源モジュールを使う ===
# （無ければ ImportError を出して気づけるように素直に失敗させる）
from ResponsibilityAllow_acts import (  # activations; core analysis
    analyze_activation, apply_psych_fluctuation, gelu, leaky_relu, neg_gelu,
    neg_leaky_relu, neg_relu, neg_sigmoid, neg_silu, relu, sigmoid, silu, tanh)

# ↑ ユーザーのオリジナル実装をそのまま呼ぶ。:contentReference[oaicite:1]{index=1}


# ========== 1) 量子エンジン ==========
@dataclass
class SimParams:
    T: int = 250
    tension_alpha: float = 0.06
    tension_leak: float = 0.02
    tension_threshold: float = 1.0
    fear_decay: float = 0.90
    wonder_gain: float = 0.10
    sound_hold_steps: int = 12
    jitter_std: float = 0.06
    auto_arrow_scale: float = 0.0
    seed: Optional[int] = None

@dataclass
class Log:
    p_motion: List[float] = field(default_factory=list)
    tension:  List[float] = field(default_factory=list)
    fear:     List[float] = field(default_factory=list)
    wonder:   List[float] = field(default_factory=list)
    events:   List[str]   = field(default_factory=list)
    memory:   List[str]   = field(default_factory=list)

@dataclass
class HumanSide:
    tension: float = 0.0
    fear: float = 0.0
    wonder: float = 0.0
    external_input: int = 0
    sound_timer: int = 0

class QuantumStillnessEngine:
    """
    q0: Agent (|0>=Stillness, |1>=Motion)
    q1: Ancilla (Lightning trigger)
    公開API: reset / set_state / get_state /
             _apply_small_stillness / _arrow_rotation /
             _lightning_pulse / _sound_return_mix / _prob_motion /
             _step / run
    """
    def __init__(self, params: Optional[SimParams] = None):
        self.params = params or SimParams()
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
        self.log = Log()
        self.side = HumanSide()
        self._sv = self._init_state()

    def _init_state(self) -> Statevector:
        return Statevector.from_label('00')  # |q1 q0> の順（little-endian）

    def reset(self):
        self.log = Log()
        self.side = HumanSide()
        self._sv = self._init_state()

    def set_state(self, state: Statevector):
        if not isinstance(state, Statevector):
            raise TypeError("state must be Statevector")
        if state.dim != 4:
            raise ValueError("state must be 2-qubit (dim=4) Statevector")
        self._sv = state

    def get_state(self) -> Statevector:
        return self._sv

    # ---- 量子操作ユーティリティ ----
    def _apply_small_stillness(self, eps: float = 0.0):
        if abs(eps) < 1e-12: return
        qc = QuantumCircuit(2)
        qc.rz(eps, 1)  # ancilla 側に微相
        self._sv = self._sv.evolve(qc)

    def _arrow_rotation(self, theta: float):
        if abs(theta) < 1e-12: return
        qc = QuantumCircuit(2)
        qc.ry(float(np.clip(theta, -math.pi, math.pi)), 0)  # q0へ“矢”
        self._sv = self._sv.evolve(qc)

    def _lightning_pulse(self, ancilla_bias: float = 1.0):
        b = float(np.clip(ancilla_bias, 0.0, 1.0))
        theta = math.pi * b
        qc = QuantumCircuit(2)
        qc.ry(theta, 1)  # ancilla tilt
        qc.cx(1, 0)      # if ancilla==1 then flip agent
        self._sv = self._sv.evolve(qc)

    def _sound_return_mix(self, mix: float = 0.25):
        m = float(np.clip(mix, 0.0, 1.0))
        if m <= 1e-12: return
        angle = (math.pi / 2) * m
        qc = QuantumCircuit(2)
        qc.ry(angle, 0); qc.h(0); qc.ry(-angle, 0)
        self._sv = self._sv.evolve(qc)

    def _prob_motion(self) -> float:
        probs = self._sv.probabilities([0])  # [P(q0=0), P(q0=1)]
        return float(probs[1])

    def _step(self, t: int):
        p, s = self.params, self.side

        # 任意: 緊張に比例して“矢”を自動注入
        if p.auto_arrow_scale != 0.0:
            theta = float(np.clip(p.auto_arrow_scale * s.tension, -1.0, 1.0)) * math.pi
            self._arrow_rotation(theta)

        if s.external_input == 0:
            s.tension += p.tension_alpha
            s.fear = min(1.0, s.fear + 0.20)
            self._apply_small_stillness(0.0)
        else:
            s.tension = max(0.0, s.tension - p.tension_leak)
            s.fear *= p.fear_decay
            s.wonder = min(1.0, s.wonder + p.wonder_gain)
            self._sound_return_mix(0.25)

        # 閾値超で稲妻
        if s.tension >= p.tension_threshold:
            bias_noise = float(np.clip(np.random.normal(0.0, p.jitter_std), -0.5, 0.5))
            bias = float(np.clip(1.0 - bias_noise, 0.0, 1.0))
            self._lightning_pulse(bias)
            self.log.events.append(f"[t={t}] LIGHTNING (bias={bias:.2f})")
            self.log.memory.append(f"{t}: ⚡ 稲妻がStillnessを切り裂く")

            s.tension = 0.0
            s.fear = 1.0
            s.wonder = 0.5
            s.external_input = 1
            s.sound_timer = p.sound_hold_steps

        # 音の帰還ウィンドウ管理
        if s.external_input == 1:
            s.sound_timer -= 1
            if s.sound_timer <= 0:
                s.external_input = 0

        # ログ
        self.log.p_motion.append(self._prob_motion())
        self.log.tension.append(s.tension)
        self.log.fear.append(s.fear)
        self.log.wonder.append(s.wonder)

    def run(self) -> Log:
        for t in range(self.params.T):
            self._step(t)
        return self.log

# ====== 追記: 多モーダル駆動ランナー ======
def run_multimodal_session(
    T: int,
    acts_mode: str = "relu",
    seed: Optional[int] = 7,
    fluct_mode: str = "logit_gauss",
    fluct_kwargs: Optional[Dict] = None,
    x0: Optional[np.ndarray] = None, W: Optional[np.ndarray] = None, b: Optional[np.ndarray] = None,
    params: Optional[SimParams] = None,
    *,
    media_series: Optional[Dict[str, np.ndarray]] = None,
    alpha_audio_tension: Tuple[float,float] = (0.15, 0.10),  # (onset, loud)
    audio_onset_gate: float = 0.6,
    beta_motion_delta: float = 0.8,      # 動き→delta 係数
    beta_motion_mix: float = 0.25,       # 動き→mix 係数
    expose_events: bool = False
) -> Dict:
    """
    media_series は qstillness_io_media.align_media_to_T() の出力を想定:
      - beat_flag, onset_strength, loudness, motion_mag （長さTの配列）
    """
    if seed is not None:
        np.random.seed(seed)
    if fluct_kwargs is None:
        fluct_kwargs = {"sigma": 0.35, "seed": None}
    if acts_mode not in _ACTS_TABLE:
        raise ValueError(f"unknown acts_mode: {acts_mode}")
    pos_fn, neg_fn, center = _ACTS_TABLE[acts_mode]

    # 既定特徴
    x = x0 if x0 is not None else np.array([1.0, -0.5, 0.8, 0.2])
    W = W  if W  is not None else np.array([
        [ 0.6, -0.2, 0.3, -0.5],
        [ 0.1,  0.4, -0.7, 0.2],
        [-0.3,  0.2, 0.5,  0.1],
        [ 0.7, -0.6, 0.2,  0.4]
    ])
    b = b  if b  is not None else np.array([0.05, -0.1, 0.2, 0.0])

    eng = QuantumStillnessEngine(params or SimParams(T=T, tension_threshold=1.0, jitter_std=0.05))
    timeline = {"p_motion": [], "delta": [], "p_hat": [], "label": [],
                "mix": [], "angle": [], "bias": []}
    lightning_count = 0

    # メディア系列
    beat = media_series["beat_flag"] if (media_series and "beat_flag" in media_series) else np.zeros(T, np.float32)
    onset = media_series["onset_strength"] if (media_series and "onset_strength" in media_series) else np.zeros(T, np.float32)
    loud  = media_series["loudness"] if (media_series and "loudness" in media_series) else np.zeros(T, np.float32)
    motion= media_series["motion_mag"] if (media_series and "motion_mag" in media_series) else np.zeros(T, np.float32)

    for t in range(T):
        # --- 音で外界フラグ＆緊張をドライブ ---
        # ビート or 強オンセットで external_input=1
        eng.side.external_input = 1 if (beat[t] > 0.5 or onset[t] > audio_onset_gate) else 0
        # 緊張に音の強度を足す
        eng.side.tension += alpha_audio_tension[0]*float(onset[t]) + alpha_audio_tension[1]*float(loud[t])

        # --- 既存の“責任”分析 ---
        if t > 0:
            x = next_features(x, noise=0.12)
        ana = analyze_activation(
            x=x, W=W, b=b, pos_fn=pos_fn, neg_fn=neg_fn,
            name_pos="Pos", name_neg="Neg",
            tau=1.0, topk=None,
            fluct_mode=fluct_mode, fluct_kwargs=fluct_kwargs,
            center=center, verbose=False
        )
        delta = ana["delta"]; p_hat = ana["p_hat"]; label = ana["label"]
        pos_part, neg_strength = ana["pos_part"], ana["neg_strength"]

        # --- 映像の“動き”で delta と mix をブースト ---
        motion_z = np.tanh( (float(motion[t]) - 0.5) * 2.0 )  # [-1,1] に再圧縮
        delta = delta + beta_motion_delta * motion_z
        mix_extra = np.clip(beta_motion_mix * abs(motion_z), 0.0, 1.0)

        # --- 矢（Ry角） ---
        angle = delta_to_angle(delta, k_theta=0.9)
        qc = QuantumCircuit(2); qc.ry(angle, 0)
        eng._sv = eng._sv.evolve(qc)

        # --- 稲妻（内部のみ） or 音の帰還ミックス ---
        if label == 1:
            bias = prob_to_bias(p_hat)
            eng._lightning_pulse(bias)
            if expose_events:
                eng.log.events.append(f"[t={t}] LIGHTNING(by media) bias={bias:.2f}")
            lightning_count += 1
            eng.side.tension = 0.0
            eng.side.fear    = min(1.0, eng.side.fear + 0.4)
            eng.side.wonder  = min(1.0, eng.side.wonder + 0.2)
            eng.side.external_input = 1
            eng.side.sound_timer = max(eng.side.sound_timer, eng.params.sound_hold_steps)
        else:
            mix = strength_to_mix(pos_part, neg_strength, k_m=0.12)
            eng._sound_return_mix(np.clip(mix + mix_extra, 0.0, 1.0))

        # --- 1ステップ進行 & ログ ---
        eng._step(t)
        timeline["p_motion"].append(eng._prob_motion())
        timeline["delta"].append(float(delta))
        timeline["p_hat"].append(float(p_hat))
        timeline["label"].append(int(label))
        timeline["mix"].append(float(strength_to_mix(pos_part, neg_strength)))
        timeline["angle"].append(float(angle))
        timeline["bias"].append(float(prob_to_bias(p_hat)) if label == 1 else 0.0)

    mean_p_motion = float(np.mean(timeline["p_motion"]))
    return {"timeline": timeline, "engine_log": eng.log,
            "lightning_count": lightning_count,
            "mean_p_motion": mean_p_motion,
            "params": eng.params}

# ========== 2) マッピング（責任→量子） ==========
def delta_to_angle(delta: float, k_theta: float = 0.9) -> float:
    """δ→q0のRy角（安全にtanhしてクリップ）"""
    th = k_theta * np.tanh(delta) * math.pi
    return float(np.clip(th, -math.pi, math.pi))

def prob_to_bias(p_hat: float) -> float:
    """確信度→ancilla bias（0..1）"""
    return float(np.clip(2.0 * abs(p_hat - 0.5), 0.0, 1.0))

def strength_to_mix(pos_part: np.ndarray, neg_strength: np.ndarray, k_m: float = 0.12) -> float:
    """寄与強度→Sound Mix（0..1）"""
    s = float(pos_part.sum() + neg_strength.sum())
    return float(np.clip(k_m * s, 0.0, 1.0))

def next_features(x: np.ndarray, noise=0.12) -> np.ndarray:
    """外界ゆらぎのデモ"""
    return x + np.random.normal(0.0, noise, size=x.shape)


# ========== 3) 実行ループ（活性をフルに選べる） ==========
_ACTS_TABLE = {
    # name: (pos_fn, neg_fn, center)
    "relu":        (relu,        neg_relu,        "auto"),
    "leaky(0.1)": (lambda z: leaky_relu(z, alpha=0.1),
                   lambda z: neg_leaky_relu(z, alpha=0.1), "auto"),
    "silu":        (silu,        neg_silu,        "auto"),
    "gelu":        (gelu,        neg_gelu,        "auto"),
    "sigmoid":     (sigmoid,     neg_sigmoid,     0.5),
    # tanhは奇関数なのでneg_tanh==tanh。必要なら追加してOK。
}

def run_responsibility_quantum_loop(
    T: int = 250,
    acts_mode: str = "relu",        # 上の _ACTS_TABLE のキーから選択
    seed: Optional[int] = 7,
    fluct_mode: str = "logit_gauss",# ResponsibilityAllow_actsの揺らぎモードをそのまま指定
    fluct_kwargs: Optional[Dict] = None,
    # 特徴量の初期値/線形変換（デモ用。実運用は実データで差し替え）
    x0: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    params: Optional[SimParams] = None
) -> Dict:
    if seed is not None:
        np.random.seed(seed)
    if fluct_kwargs is None:
        fluct_kwargs = {"sigma": 0.35, "seed": None}

    if acts_mode not in _ACTS_TABLE:
        raise ValueError(f"unknown acts_mode: {acts_mode}")

    pos_fn, neg_fn, center = _ACTS_TABLE[acts_mode]

    # ---- デモ用の既定（必要なら引数で上書き）----
    x = x0 if x0 is not None else np.array([1.0, -0.5, 0.8, 0.2])
    W = W  if W  is not None else np.array([
        [ 0.6, -0.2, 0.3, -0.5],
        [ 0.1,  0.4, -0.7, 0.2],
        [-0.3,  0.2, 0.5,  0.1],
        [ 0.7, -0.6, 0.2,  0.4]
    ])
    b = b  if b  is not None else np.array([0.05, -0.1, 0.2, 0.0])

    eng = QuantumStillnessEngine(params or SimParams(T=T, tension_threshold=1.0, jitter_std=0.05))

    timeline = {
        "p_motion": [], "delta": [], "p_hat": [], "label": [],
        "mix": [], "angle": [], "bias": [], "events": []
    }

    for t in range(T):
        if t > 0:
            x = next_features(x, noise=0.12)

        ana = analyze_activation(
            x=x, W=W, b=b,
            pos_fn=pos_fn, neg_fn=neg_fn,
            name_pos="Pos", name_neg="Neg",
            tau=1.0, topk=None,
            fluct_mode=fluct_mode, fluct_kwargs=fluct_kwargs,
            center=center, verbose=False
        )

        delta   = ana["delta"]
        p_hat   = ana["p_hat"]
        label   = ana["label"]
        pos_part, neg_strength = ana["pos_part"], ana["neg_strength"]

        # “矢”角 → 量子回転（q0）
        angle = delta_to_angle(delta, k_theta=0.9)
        qc = QuantumCircuit(2); qc.ry(angle, 0)
        eng._sv = eng._sv.evolve(qc)

        # ラベルで稲妻トリガ
        if label == 1:
            bias = prob_to_bias(p_hat)
            eng._lightning_pulse(bias)
            eng.log.events.append(f"[t={t}] LIGHTNING(by bridge) bias={bias:.2f}")
            # 人側へ軽い衝撃
            eng.side.tension = 0.0
            eng.side.fear    = min(1.0, eng.side.fear + 0.4)
            eng.side.wonder  = min(1.0, eng.side.wonder + 0.2)
            eng.side.external_input = 1
            eng.side.sound_timer = max(eng.side.sound_timer, eng.params.sound_hold_steps)
        else:
            mix = strength_to_mix(pos_part, neg_strength, k_m=0.12)
            eng._sound_return_mix(mix)

        # 人×量子の自然ダイナミクス
        eng._step(t)

        # ログ
        timeline["p_motion"].append(eng._prob_motion())
        timeline["delta"].append(float(delta))
        timeline["p_hat"].append(float(p_hat))
        timeline["label"].append(int(label))
        timeline["mix"].append(float(strength_to_mix(pos_part, neg_strength)))
        timeline["angle"].append(float(angle))
        timeline["bias"].append(float(prob_to_bias(p_hat)) if label == 1 else 0.0)
        if len(eng.log.events) and (timeline["events"][-1] if timeline["events"] else None) != eng.log.events[-1]:
            timeline["events"].append(eng.log.events[-1])

    return {"timeline": timeline, "events": timeline["events"], "engine_log": eng.log, "params": eng.params}

# --- 4. drop-in replacement: run_responsibility_quantum_loop と __main__ ---

def run_responsibility_quantum_loop(
    T: int = 250,
    acts_mode: str = "relu",
    seed: Optional[int] = 7,
    fluct_mode: str = "logit_gauss",
    fluct_kwargs: Optional[Dict] = None,
    x0: Optional[np.ndarray] = None,
    W: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    params: Optional[SimParams] = None,
    *,
    expose_events: bool = False  # ← 追加: Trueにすると従来通りevents文字列も残す
) -> Dict:
    if seed is not None:
        np.random.seed(seed)
    if fluct_kwargs is None:
        fluct_kwargs = {"sigma": 0.35, "seed": None}

    if acts_mode not in _ACTS_TABLE:
        raise ValueError(f"unknown acts_mode: {acts_mode}")

    pos_fn, neg_fn, center = _ACTS_TABLE[acts_mode]

    # 既定のデモ行列
    x = x0 if x0 is not None else np.array([1.0, -0.5, 0.8, 0.2])
    W = W  if W  is not None else np.array([
        [ 0.6, -0.2, 0.3, -0.5],
        [ 0.1,  0.4, -0.7, 0.2],
        [-0.3,  0.2, 0.5,  0.1],
        [ 0.7, -0.6, 0.2,  0.4]
    ])
    b = b  if b  is not None else np.array([0.05, -0.1, 0.2, 0.0])

    eng = QuantumStillnessEngine(params or SimParams(T=T, tension_threshold=1.0, jitter_std=0.05))

    timeline = {
        "p_motion": [], "delta": [], "p_hat": [], "label": [],
        "mix": [], "angle": [], "bias": []
    }
    lightning_count = 0  # ← 追加: 稲妻回数カウンタ

    for t in range(T):
        if t > 0:
            x = next_features(x, noise=0.12)

        ana = analyze_activation(
            x=x, W=W, b=b,
            pos_fn=pos_fn, neg_fn=neg_fn,
            name_pos="Pos", name_neg="Neg",
            tau=1.0, topk=None,
            fluct_mode=fluct_mode, fluct_kwargs=fluct_kwargs,
            center=center, verbose=False
        )

        delta   = ana["delta"]
        p_hat   = ana["p_hat"]
        label   = ana["label"]
        pos_part, neg_strength = ana["pos_part"], ana["neg_strength"]

        # “矢”角 → 量子回転（q0）
        angle = delta_to_angle(delta, k_theta=0.9)
        qc = QuantumCircuit(2); qc.ry(angle, 0)
        eng._sv = eng._sv.evolve(qc)

        # ラベルで稲妻トリガ（表には出さない）
        if label == 1:
            bias = prob_to_bias(p_hat)
            eng._lightning_pulse(bias)
            if expose_events:
                eng.log.events.append(f"[t={t}] LIGHTNING(by bridge) bias={bias:.2f}")
            lightning_count += 1  # ← カウントだけ増やす

            # 人側へ軽い衝撃
            eng.side.tension = 0.0
            eng.side.fear    = min(1.0, eng.side.fear + 0.4)
            eng.side.wonder  = min(1.0, eng.side.wonder + 0.2)
            eng.side.external_input = 1
            eng.side.sound_timer = max(eng.side.sound_timer, eng.params.sound_hold_steps)
        else:
            mix = strength_to_mix(pos_part, neg_strength, k_m=0.12)
            eng._sound_return_mix(mix)

        # 人×量子の自然ダイナミクス
        eng._step(t)

        # ログ（確率など）
        timeline["p_motion"].append(eng._prob_motion())
        timeline["delta"].append(float(delta))
        timeline["p_hat"].append(float(p_hat))
        timeline["label"].append(int(label))
        timeline["mix"].append(float(strength_to_mix(pos_part, neg_strength)))
        timeline["angle"].append(float(angle))
        timeline["bias"].append(float(prob_to_bias(p_hat)) if label == 1 else 0.0)

    mean_p_motion = float(np.mean(timeline["p_motion"]))  # ← 追加: 平均確率

    return {
        "timeline": timeline,
        "engine_log": eng.log,             # eventsはexpose_events=True時のみ詳細あり
        "lightning_count": lightning_count,
        "mean_p_motion": mean_p_motion,
        "params": eng.params
    }


if __name__ == "__main__":
    out = run_responsibility_quantum_loop(
        T=180,
        acts_mode="relu",
        seed=42,
        fluct_mode="logit_gauss",
        fluct_kwargs={"sigma":0.35, "seed":None},
        expose_events=False  # ← ここをTrueにすると従来のevents文字列も保存
    )
    print(f"lightning_count: {out['lightning_count']}")
    print(f"mean P(Motion): {out['mean_p_motion']:.3f}")
