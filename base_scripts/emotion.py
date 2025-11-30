# emotion_qiskit_min_legacy.py
# ------------------------------------------------------------
# 「気持ち＝量子状態」ミニマム実装（Legacy互換 / Sampler不使用）
# 1qubit: 自分（|0>=Joy, |1>=Fear）
# 2qubit: 自分(0)×相手(1)（RXXで可変エンタングル, RZZで相関相）
# 可視化：Z基底の確率 / Blochベクトル
# ------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Tuple, Dict

from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RXXGate, RZZGate


# ---------- 基本マッピング ----------
# |0> = Joy（喜） , |1> = Fear（不安）
# w_joy + w_fear = 1 のとき、θ = 2 * asin(sqrt(w_fear))
def weights_to_angles(w_joy: float, w_fear: float, phase: float = 0.0) -> Tuple[float, float]:
    w_joy = max(0.0, min(1.0, w_joy))
    w_fear = max(0.0, min(1.0, w_fear))
    s = max(1e-12, (w_joy + w_fear))
    w_joy, w_fear = w_joy / s, w_fear / s
    theta = 2.0 * math.asin(math.sqrt(w_fear))
    phi = phase
    return theta, phi


def build_single_emotion(w_joy: float, w_fear: float, phase: float = 0.0) -> QuantumCircuit:
    """1量子ビットで気持ち（喜/不安の重ね合わせ）を作る"""
    theta, phi = weights_to_angles(w_joy, w_fear, phase)
    qc = QuantumCircuit(1, name="Emotion(You)")
    qc.ry(theta, 0)
    if abs(phi) > 1e-12:
        qc.rz(phi, 0)
    return qc


@dataclass
class RelationParams:
    you_joy: float
    you_fear: float
    other_joy: float
    other_fear: float
    phase_you: float = 0.0
    phase_other: float = 0.0
    ent_strength: float = 0.0  # 0~1
    corr_phase: float = 0.0    # RZZ 相関位相


def build_relation_circuit(p: RelationParams) -> QuantumCircuit:
    """2量子ビット（q0=You, q1=Other）。RXXで絡み、RZZで相関の“クセ”"""
    theta_y, phi_y = weights_to_angles(p.you_joy, p.you_fear, p.phase_you)
    theta_o, phi_o = weights_to_angles(p.other_joy, p.other_fear, p.phase_other)

    qc = QuantumCircuit(2, name="Relation(You×Other)")

    qc.ry(theta_y, 0)
    if abs(phi_y) > 1e-12:
        qc.rz(phi_y, 0)

    qc.ry(theta_o, 1)
    if abs(phi_o) > 1e-12:
        qc.rz(phi_o, 1)

    gamma = (math.pi / 4.0) * max(0.0, min(1.0, p.ent_strength))
    if gamma > 1e-9:
        qc.append(RXXGate(2 * gamma), [0, 1])

    if abs(p.corr_phase) > 1e-9:
        qc.append(RZZGate(p.corr_phase), [0, 1])

    return qc


# ---------- 解析ユーティリティ（Sampler不使用） ----------
def probs_z_basis(qc: QuantumCircuit, shots: int = 10_000) -> Dict[str, float]:
    """Z基底での確率を Aer qasm_simulator で取得（レガシー互換）"""
    backend = Aer.get_backend("qasm_simulator")
    measured = qc.copy()
    measured.measure_all()
    job = execute(measured, backend=backend, shots=shots)
    result = job.result()
    counts = result.get_counts(measured)
    total = sum(counts.values()) or 1
    # 返るbitstringは通常「上位ビットが左（q1q0）」っぽく見えることが多い
    return {k: v / total for k, v in sorted(counts.items(), key=lambda x: x[0])}


def bloch_of_first_qubit(qc: QuantumCircuit):
    """状態ベクトルから1量子ビット目(q0)のBloch (x,y,z) を推定"""
    sv = Statevector.from_instruction(qc)
    import numpy as np
    rho = np.outer(sv.data, np.conjugate(sv.data))  # 密度行列

    dim = int(round(math.log2(rho.shape[0])))
    if dim == 1:
        rho_1 = rho
    elif dim == 2:
        # 部分トレースで q1 をトレースアウト → q0 の部分密度行列
        # ブロック行列を足し合わせる簡易実装（|0>,|1>で射影）
        rho_1 = rho[0:2, 0:2] + rho[2:4, 2:4]
    else:
        raise ValueError("このデモは最大2量子ビット想定です。")

    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    x = float((sigma_x @ rho_1).trace().real)
    y = float((sigma_y @ rho_1).trace().real)
    z = float((sigma_z @ rho_1).trace().real)
    return x, y, z


# ---------- デモ ----------
def demo_single():
    print("=== Single Emotion: 50% Joy / 50% Fear (中立のゆらぎ) ===")
    qc = build_single_emotion(0.5, 0.5, phase=0.0)
    p = probs_z_basis(qc, shots=8192)
    bloch = bloch_of_first_qubit(qc)
    print("Z-basis probs (|0>=Joy, |1>=Fear):", p)
    print("Bloch(You): x,y,z =", tuple(round(v, 3) for v in bloch))
    print()

    print("=== Jump-ish: Joy 90% / Fear 10%（“腹決め”寄り） ===")
    qc = build_single_emotion(0.9, 0.1, phase=0.0)
    p = probs_z_basis(qc, shots=8192)
    bloch = bloch_of_first_qubit(qc)
    print("Z-basis probs:", p)
    print("Bloch(You): x,y,z =", tuple(round(v, 3) for v in bloch))
    print()


def demo_relation():
    print("=== Relation: 互いに 50/50、エンタングル弱 ===")
    rp = RelationParams(
        you_joy=0.5, you_fear=0.5,
        other_joy=0.5, other_fear=0.5,
        phase_you=0.0, phase_other=0.0,
        ent_strength=0.3,
        corr_phase=0.0
    )
    qc = build_relation_circuit(rp)
    p = probs_z_basis(qc, shots=8192)
    bloch = bloch_of_first_qubit(qc)
    print("Z-basis probs (bitstring ~ q1q0):", p)
    print("Bloch(You): x,y,z =", tuple(round(v, 3) for v in bloch))
    print("  解釈: 00=共にJoy, 01=You Fear/Other Joy, 10=You Joy/Other Fear, 11=共にFear")
    print()

    print("=== Relation: You=Joy寄り, Other=Fear寄り, 強エンタングル + 相関位相 ===")
    rp2 = RelationParams(
        you_joy=0.85, you_fear=0.15,
        other_joy=0.2, other_fear=0.8,
        phase_you=0.2, phase_other=-0.1,
        ent_strength=0.9,
        corr_phase=0.6
    )
    qc2 = build_relation_circuit(rp2)
    p2 = probs_z_basis(qc2, shots=8192)
    bloch2 = bloch_of_first_qubit(qc2)
    print("Z-basis probs (bitstring ~ q1q0):", p2)
    print("Bloch(You): x,y,z =", tuple(round(v, 3) for v in bloch2))
    print("  解釈: 強い絡み合いにより、相手の状態とあなたの出力確率が影響し合う")
    print()


if __name__ == "__main__":
    demo_single()
    demo_relation()
