# responsibility_coherence_qiskit.py
# ------------------------------------------------------------
# 日本語文脈モデル ↔ 量子コヒーレンスの対応づけデモ
#  - 責任ベクトルの安定度 alpha (0..1) と 経験値 exp (>0) を
#    量子T1/T2に写像して、コヒーレンス時間を模擬。
#  - |+> 状態を用意し、アイドル(Delay)を入れながら
#    X測定期待値 <X> の減衰を観測＝「文脈の揺らぎ保持」を可視化。
#
# 依存:
#   pip install qiskit qiskit-aer
#
# 使い方:
#   python responsibility_coherence_qiskit.py
#   （パラメタは main 内の PARAMS を編集）
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import SXGate
from qiskit.quantum_info import Statevector

# 以前の AER 実行ブロックの代わり：
def run_ideal(qc: QuantumCircuit):
    sv = Statevector.from_label('0' * qc.num_qubits)
    sv = sv.evolve(qc)              # 回路を適用
    probs = sv.probabilities_dict() # 観測確率
    return probs
# ===================== パラメタ定義 =====================

@dataclass
class RespParams:
    alpha: float = 0.7   # 責任ベクトル安定度 (0..1)  高いほど安定＝コヒーレンス長い
    exp: float = 1000.0  # 経験値 (>0)              大きいほどコヒーレンス長い
    base_T1: float = 50e-6   # 物理ベースライン T1 [s]
    base_T2: float = 40e-6   # 物理ベースライン T2 [s]
    dt: float = 0.222e-9     # シミュレータの dt [s] (Aerの既定~0.222ns相当)
    idle_step_ns: float = 50.0  # 1ステップのアイドル時間 [ns]
    steps_list: List[int] = None  # 試行するアイドルステップ数のリスト

    def __post_init__(self):
        if self.steps_list is None:
            # 文脈が伸びるにつれての減衰カーブを見たいので段階的に
            self.steps_list = [0, 5, 10, 20, 40, 80, 120, 160, 240, 320]


# ===================== マッピング設計 =====================
# しゃきしゃき理論の素案:
#   T_coh ∝ alpha * log(1 + exp)
#   これを量子の T1/T2 に反映（T2 <= 2*T1 の制約に配慮しつつスケール）
#   直感: alpha↑, exp↑ → T1/T2 を引き上げ → コヒーレンス長く

def map_resp_to_T1T2(p: RespParams) -> Tuple[float, float]:
    # 正規化された拡張係数（1以上に拡張）
    k = 1.0 + p.alpha * math.log(1.0 + max(p.exp, 1.0))
    # 過度に巨大化しないようにソフトクリップ
    k = min(k, 1.0 + 6.0)  # 最大 ~7倍

    T1 = p.base_T1 * k
    # T2 は T1 より短くなる傾向（位相が壊れやすい）を残す
    # alpha が高いほど T2/T1 を引き上げる（＝揺らぎに強い）
    t2_ratio = 0.5 + 0.45 * p.alpha  # 0.5〜0.95
    T2 = min(T1 * t2_ratio, 2.0 * T1)  # 物理制約 T2 <= 2*T1 を満たす
    return T1, T2


# ===================== ノイズモデル構築 =====================

def build_noise_model(T1: float, T2: float, dt: float, idle_ns: float) -> Tuple[NoiseModel, int]:
    """Delay(アイドル)に熱緩和ノイズを与えるNoiseModelを作成"""
    noise = NoiseModel()

    # 1アイドルステップの長さを dt に合わせてスロット数へ
    idle_seconds = idle_ns * 1e-9
    duration = int(round(idle_seconds / dt))
    duration = max(duration, 1)

    # 熱緩和エラー（T1,T2, duration*dt）を Delay に紐付け
    # Qiskitでは理想Delay自体に直接エラー付与しづらいので、id(I)やu1-likeに付ける手もあるが
    # ここでは「sx」「x」「id」を“アイドル近似”として同等時間扱いにして付与する。
<<<<<<< HEAD
    err = thermal_relaxation_error(T1=T1, T2=T2, time=duration * dt, excited_state_population=0.0)
=======
    ｓerr = thermal_relaxation_error(T1=T1, T2=T2, time=duration * dt, excited_state_population=0.0)
>>>>>>> a5140286e999700ffe781eb3e42556fa56f3cb89
    for g in ["id", "sx", "x"]:
        noise.add_quantum_error(err, g, [0])

    return noise, duration


# ===================== 実験回路 =====================

def coherence_probe_circuit(idle_duration: int, reps: int = 1) -> QuantumCircuit:
    """
    |+> を用意し、(Delay ≒ id/sx を用いたアイドル近似) → H測定で <X> を観測。
    idle_duration スロット×reps 回だけ“待つ”。
    """
    qc = QuantumCircuit(1, 1)
    # |0> --H--> |+>
    qc.h(0)

    # アイドルの近似（等価時間のゲートで穴埋め）
    # ここでは duration を満たすだけ sx を並べる簡易実装（1個あたりのdurationは後でtranspile影響を受ける）
    for _ in range(reps):
        for _ in range(idle_duration):
            qc.id(0)  # id に熱緩和を付けてある前提

    # X基底で測る（= H → Z測定）
    qc.h(0)
    qc.measure(0, 0)
    return qc


def run_sweep(params: RespParams):
    T1, T2 = map_resp_to_T1T2(params)
<<<<<<< HEAD
    noise, duration = build_noise_model(T1, T2, params.dt, params.idle_step_ns)

    sim = AerSimulator(noise_model=noise)
    results = []

    for steps in params.steps_list:
        qc = coherence_probe_circuit(idle_duration=duration, reps=steps)
        tqc = transpile(qc, sim, optimization_level=0)
        job = sim.run(tqc, shots=4096)
        res = job.result()
        counts = res.get_counts()

        p0 = counts.get("0", 0) / 4096.0
        p1 = counts.get("1", 0) / 4096.0
        # X測定で |+> は理想的に 0 が 100%（H→Z測定）
        # 減衰で均等化に近づく → <X> = p(0) - p(1)
        exp_X = p0 - p1
=======

    results = []
    for steps in params.steps_list:
        t = (steps * params.idle_step_ns) * 1e-9  # 総待機時間 [s]
        exp_X = math.exp(-t / T2)                 # <X>(t) = e^{-t/T2}
        p0 = (1.0 + exp_X) / 2.0                  # X基底での0確率
        p1 = 1.0 - p0
>>>>>>> a5140286e999700ffe781eb3e42556fa56f3cb89

        results.append({
            "steps": steps,
            "idle_ns_total": steps * params.idle_step_ns,
            "exp_X": exp_X,
            "p0": p0,
            "p1": p1,
        })

    return {
        "T1_s": T1,
        "T2_s": T2,
        "idle_step_ns": params.idle_step_ns,
        "dt_s": params.dt,
<<<<<<< HEAD
        "duration_slots_per_step": duration,
=======
        "duration_slots_per_step": None,
>>>>>>> a5140286e999700ffe781eb3e42556fa56f3cb89
        "sweep": results,
    }


# ===================== 実行例 =====================

def pretty_print(report: dict):
    print("# ==== コヒーレンス報告 ====")
    print(f"T1  = {report['T1_s']*1e6:.2f} us")
    print(f"T2  = {report['T2_s']*1e6:.2f} us")
    print(f"dt  = {report['dt_s']*1e9:.3f} ns/slot")
    print(f"idle_step = {report['idle_step_ns']:.1f} ns/step"
          f"  -> {report['duration_slots_per_step']} slots/step")
    print()
    print("steps\tidle[ns]\t<X>\tp0\tp1")
    for r in report["sweep"]:
        print(f"{r['steps']}\t{r['idle_ns_total']:.1f}\t\t{r['exp_X']:.4f}\t{r['p0']:.4f}\t{r['p1']:.4f}")


def main():
    # ===== ここをいじれば “慣れ/安定度” の効果が見える =====
    PARAMS = RespParams(
        alpha=0.7,       # 責任ベクトル安定度（0..1）: 高いほどコヒーレンス長い
        exp=1000.0,      # 経験値: 大きいほどコヒーレンス長い
        base_T1=50e-6,   # ベースライン（装置ごとに調整可）
        base_T2=40e-6,
        dt=0.222e-9,     # Aerの既定相当
        idle_step_ns=50.0,  # 1ステップで待つ時間
        steps_list=None
    )

    report = run_sweep(PARAMS)
    pretty_print(report)

    # 参考：別設定を比較したい場合は複数回回す
    # for a in [0.3, 0.5, 0.7, 0.9]:
    #     for e in [10.0, 100.0, 1000.0, 10000.0]:
    #         rp = RespParams(alpha=a, exp=e)
    #         rep = run_sweep(rp)
    #         print("\n=== alpha=", a, "exp=", e, "===")
    #         pretty_print(rep)


if __name__ == "__main__":
    main()
