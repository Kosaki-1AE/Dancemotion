# qstillness_engine.py
# ------------------------------------------------------------
# QuantumStillnessEngine: 「Stillness → Lightning → Motion → Sound Return」
# ・q0 (agent): |0>=Stillness, |1>=Motion
# ・q1 (ancilla): Lightning trigger (稲妻)
# 依存: numpy, qiskit (quantum_info.Statevector / QuantumCircuit)
# ------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


# ========== パラメータ / ログ / 人側状態 ==========
@dataclass
class SimParams:
    # シミュレーション長
    T: int = 250

    # 緊張ダイナミクス
    tension_alpha: float = 0.06   # 外界刺激なし時の緊張の増分
    tension_leak: float  = 0.02   # 外界刺激あり時の緊張減衰
    tension_threshold: float = 1.0  # 稲妻の発火閾値

    # 感情ダイナミクス
    fear_decay: float   = 0.90
    wonder_gain: float  = 0.10

    # サウンド帰還ウィンドウ（外界入力ONの保持）
    sound_hold_steps: int = 12

    # 稲妻のバイアス揺らぎ
    jitter_std: float = 0.06

    # “矢（意思）”の回転を自動注入する場合のスケール（任意で使う）
    auto_arrow_scale: float = 0.0  # 0で無効（外側から矢角を与える派生用途向け）

    # 乱数シード（Statevector進化には不要だが、稲妻バイアスの揺らぎに使用）
    seed: Optional[int] = None


@dataclass
class Log:
    # 量子状態由来
    p_motion: List[float] = field(default_factory=list)  # P(q0=1)
    # 人側ダイナミクス
    tension:  List[float] = field(default_factory=list)
    fear:     List[float] = field(default_factory=list)
    wonder:   List[float] = field(default_factory=list)
    # イベント＆メモリ（詩的ログなど）
    events:   List[str]   = field(default_factory=list)
    memory:   List[str]   = field(default_factory=list)


@dataclass
class HumanSide:
    tension: float = 0.0
    fear: float = 0.0
    wonder: float = 0.0
    external_input: int = 0   # 0/1: サウンド帰還窓（外界入力）フラグ
    sound_timer: int = 0      # 外界入力ONの残りステップ


# ========== エンジン本体 ==========
class QuantumStillnessEngine:
    """
    q0: Agent (|0>=Stillness, |1>=Motion)
    q1: Ancilla (Lightning trigger)

    公開的に使ってOKな主なAPI:
      - reset()
      - set_state(state: Statevector)
      - get_state() -> Statevector
      - _apply_small_stillness(eps: float = 0.0)         # 小さな静けさ操作（微相）
      - _lightning_pulse(ancilla_bias: float = 1.0)      # 稲妻: ancillaを傾け → CXでq0反転
      - _sound_return_mix(mix: float = 0.25)             # 稲妻後の“音の帰還”っぽい混合
      - _prob_motion() -> float                          # P(q0=1)
      - _step(t: int)                                    # 1ステップ進める（人側×量子の更新）
      - run() -> Log                                     # Tステップ実行
    """
    def __init__(self, params: Optional[SimParams] = None):
        self.params = params or SimParams()
        if self.params.seed is not None:
            np.random.seed(self.params.seed)

        self.log = Log()
        self.side = HumanSide()
        self._sv = self._init_state()

    # ---------- 量子初期化・Getter/Setter ----------
    def _init_state(self) -> Statevector:
        # Qiskitの基底順は little-endian（右がq0）なので |q1 q0>
        # 初期は |00> にしておく
        return Statevector.from_label('00')

    def reset(self):
        """人側・量子側を完全リセット。"""
        self.log = Log()
        self.side = HumanSide()
        self._sv = self._init_state()

    def set_state(self, state: Statevector):
        """外部で作ったStatevectorを注入したいときに。"""
        if not isinstance(state, Statevector):
            raise TypeError("state must be a qiskit.quantum_info.Statevector")
        if state.dim != 4:
            raise ValueError("state must be 2-qubit (dim=4) Statevector")
        self._sv = state

    def get_state(self) -> Statevector:
        """現在のStatevectorを返す。"""
        return self._sv

    # ---------- 量子ユーティリティ ----------
    def _apply_small_stillness(self, eps: float = 0.0):
        """
        微小位相回転（存在の揺らぎにほんの少し“クセ”をつける）。
        eps=0 ならNo-op。
        """
        if abs(eps) < 1e-12:
            return
        qc = QuantumCircuit(2)
        # ここでは agent(q0) ではなく ancilla(q1) 側に位相を置く選択。
        # 「空気の膜」が薄く位相を持つイメージ。
        qc.rz(eps, 1)
        self._sv = self._sv.evolve(qc)

    def _arrow_rotation(self, theta: float):
        """
        “責任の矢”をq0に与える回転（Ry）。
        外部から意思角を注入したい時に使う（オプション）。
        """
        if abs(theta) < 1e-12:
            return
        qc = QuantumCircuit(2)
        qc.ry(float(np.clip(theta, -math.pi, math.pi)), 0)
        self._sv = self._sv.evolve(qc)

    def _lightning_pulse(self, ancilla_bias: float = 1.0):
        """
        稲妻：ancillaにバイアスをかけ（|1>寄りなら強い）→ CXでq0を反転。
        ancilla_bias ∈ [0,1] を想定。1 に近いほど発火しやすい。
        """
        b = float(np.clip(ancilla_bias, 0.0, 1.0))
        theta = math.pi * b
        qc = QuantumCircuit(2)
        qc.ry(theta, 1)  # ancilla tilt
        qc.cx(1, 0)      # if ancilla==1 then flip agent
        self._sv = self._sv.evolve(qc)

    def _sound_return_mix(self, mix: float = 0.25):
        """
        “音の帰還”：外界と内界の干渉を軽く混ぜる（Hっぽい回転を挟む）。
        mix ∈ [0,1]。0 → 何もしない、1 → Hを最大限に近い形で混合。
        """
        m = float(np.clip(mix, 0.0, 1.0))
        if m <= 1e-12:
            return
        angle = (math.pi / 2) * m
        qc = QuantumCircuit(2)
        qc.ry(angle, 0)
        qc.h(0)
        qc.ry(-angle, 0)
        self._sv = self._sv.evolve(qc)

    def _prob_motion(self) -> float:
        """
        q0（agent）が |1> となる確率を返す。
        Statevector.probabilities([0]) を使って周辺化。
        """
        # 戻りは [P(q0=0), P(q0=1)]
        probs = self._sv.probabilities([0])
        return float(probs[1])

    # ---------- 1ステップ（人×量子の共進化） ----------
    def _step(self, t: int):
        p = self.params
        s = self.side

        # ---- 自動の“矢”注入（任意機能：デフォ無効）----
        if p.auto_arrow_scale != 0.0:
            # 緊張が高いほど矢が強くなるような簡易モデル
            theta = float(np.clip(p.auto_arrow_scale * s.tension, -1.0, 1.0)) * math.pi
            self._arrow_rotation(theta)

        # ---- 人側の入力に応じた量子操作 ----
        if s.external_input == 0:
            # 外界刺激なし：静けさは蓄積し、恐れがじわっと上がる
            s.tension += p.tension_alpha
            s.fear     = min(1.0, s.fear + 0.20)
            self._apply_small_stillness(0.0)
        else:
            # 外界刺激あり：緊張は漏れ、恐れは減衰し、不思議（驚き/好奇）増大
            s.tension = max(0.0, s.tension - p.tension_leak)
            s.fear   *= p.fear_decay
            s.wonder  = min(1.0, s.wonder + p.wonder_gain)
            # 稲妻の余韻として音の帰還を混ぜる
            self._sound_return_mix(0.25)

        # ---- 閾値超えで稲妻 ----
        if s.tension >= p.tension_threshold:
            # 稲妻のバイアスは “今この瞬間は発火しやすい” 方向へ（少しランダムに）
            bias_noise = float(np.clip(np.random.normal(0.0, p.jitter_std), -0.5, 0.5))
            bias = float(np.clip(1.0 - bias_noise, 0.0, 1.0))
            self._lightning_pulse(bias)

            # ログ & メモリ
            self.log.events.append(f"[t={t}] LIGHTNING (bias={bias:.2f})")
            self.log.memory.append(f"{t}: ⚡ 稲妻がStillnessを切り裂く")

            # 人側の状態遷移
            s.tension = 0.0
            s.fear    = 1.0
            s.wonder  = 0.5
            s.external_input = 1
            s.sound_timer = p.sound_hold_steps

        # ---- 音の帰還ウィンドウ管理 ----
        if s.external_input == 1:
            s.sound_timer -= 1
            if s.sound_timer <= 0:
                s.external_input = 0

        # ---- ログ更新 ----
        self.log.p_motion.append(self._prob_motion())
        self.log.tension.append(s.tension)
        self.log.fear.append(s.fear)
        self.log.wonder.append(s.wonder)

    # ---------- 実行 ----------
    def run(self) -> Log:
        """
        params.T ステップだけ _step を回す。Log を返す。
        """
        for t in range(self.params.T):
            self._step(t)
        return self.log


# ====== 参考: 単体テスト/お試し（実行は任意） ======
if __name__ == "__main__":
    # 実行は任意。君の方針「コード提示のみでOK」に合わせて、
    # ここは最小限のデモ出力だけにしている。
    eng = QuantumStillnessEngine(SimParams(T=50, seed=0))
    log = eng.run()
    print("sample P(Motion) head:", [round(x, 3) for x in log.p_motion[:5]])
    print("events:", log.events[:3])
