# -*- coding: utf-8 -*-
"""
Beat Quantum on Qiskit — 拍量子仮説の量子実装ミニマム
- 各拍を1量子試行としてモデル化
- Ry(θ) で HIT(鳴る) の確率を制御、測定で HIT/REST をサンプル
- Rz(φ) で「責任の矢」による位相バイアスを注入（時間オフセットに反映）
- 生成結果は「リズム時刻表」として出力

要件:
  pip install qiskit qiskit-aer

実行:
  python beat_quantum_qiskit.py
"""

from dataclasses import dataclass
import math
import numpy as np
from typing import List, Tuple

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import Aer
from qiskit.compiler import transpile


# ====== モデル・パラメータ ======
@dataclass
class BeatParams:
    tempo_bpm: float = 120.0          # テンポ
    bars: int = 4                     # 小節数
    beats_per_bar: int = 4            # 拍/小節 (4/4想定)
    base_hit_prob: float = 0.8        # 基本のHIT確率（= 鳴る確率）
    stillness_prob: float = 0.0       # Stillness（休符化）を追加で高める確率（ベース確率から引く）
    humanize_prob_jitter: float = 0.05  # 確率に与えるノイズ（標準偏差）
    responsibility_phase_ms: float = 0.0  # 系統的な位相（+は遅れ、-は前ノリ）[ms]
    humanize_phase_ms: float = 5.0       # 位相ノイズ（標準偏差）[ms]
    global_phase_rad: float = 0.0        # 全体位相。必要なら曲全体の“溜め/走り”をまとめて表現
    seed: int = 42


# ====== ユーティリティ ======
def beat_duration_sec(bpm: float) -> float:
    return 60.0 / bpm  # 四分音符の長さ[s]

def prob_to_ry_theta(p_hit: float) -> float:
    """
    |1> を観測する確率 p に対して Ry(θ) の θ を返す。
    初期状態 |0> に Ry(θ) を当てると、|1> 観測確率は sin^2(θ/2)。
      p = sin^2(θ/2)  =>  θ = 2 * arcsin(sqrt(p))
    """
    p = np.clip(p_hit, 0.0, 1.0)
    return 2.0 * math.asin(math.sqrt(p))

def ms_to_rad(ms: float, bpm: float) -> float:
    """
    便宜的に「位相→時間」の対応を取るための規格化。
    ここでは1拍(四分音符)を 2π とみなし、時間オフセットmsを位相ラジアンに写像。
    """
    beat_ms = beat_duration_sec(bpm) * 1000.0
    return 2.0 * math.pi * (ms / beat_ms)

def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# ====== 回路生成 ======
def build_rhythm_circuit(params: BeatParams) -> Tuple[QuantumCircuit, List[float], List[float]]:
    """
    各拍ごとに:
      1) HIT確率 p_i を決める（ベース確率 - Stillness + ノイズ）
      2) Ry(θ_i) で p_i を埋め込み
      3) Rz(φ_i) で責任の矢 + ノイズの位相を注入（時間オフセットへ反映）
      4) 測定 → 1=HIT, 0=REST
      5) reset して次の拍へ
    """
    rng = np.random.default_rng(params.seed)

    n_beats = params.bars * params.beats_per_bar
    q = QuantumRegister(1, "q")
    c = ClassicalRegister(n_beats, "c")  # 1ビット/拍
    qc = QuantumCircuit(q, c, name="BeatQuantumRhythm")

    # 各拍の確率・位相（古典的に後で使うため保存）
    thetas: List[float] = []
    phis: List[float] = []

    for i in range(n_beats):
        # --- 確率（HIT確率） ---
        p = params.base_hit_prob - params.stillness_prob
        p += rng.normal(0.0, params.humanize_prob_jitter)  # 人間味ノイズ
        p = clamp01(p)

        theta = prob_to_ry_theta(p)  # Ry角
        qc.ry(theta, q[0])

        # --- 位相（責任の矢 + ノイズ + 全体位相）---
        phase_ms = params.responsibility_phase_ms + rng.normal(0.0, params.humanize_phase_ms)
        phi = ms_to_rad(phase_ms, params.tempo_bpm) + params.global_phase_rad
        qc.rz(phi, q[0])

        # --- 測定 ---
        qc.measure(q[0], c[i])

        # --- 次拍に備えてリセット ---
        qc.reset(q[0])

        thetas.append(theta)
        phis.append(phi)

    return qc, thetas, phis


# ====== サンプリング & 時刻表作成 ======
def sample_rhythm(qc: QuantumCircuit, shots: int = 1) -> List[str]:
    """
    n拍ぶんの測定結果を文字列で返す（"0101..."）
    shots>1 にすると複数パターンを同時生成可能
    """
    backend = Aer.get_backend("aer_simulator")
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots, seed_simulator=123).result()
    counts = result.get_counts()

    # Aerは結果ビット列を「c[n-1]...c[0]」のMSB→LSB順で返すので拍順に反転
    def normalize(bitstr: str) -> str:
        # 例: '0101' -> c3 c2 c1 c0 の順。ここでは [c0..c3] に反転
        return bitstr[::-1]

    if isinstance(counts, dict):
        # shots=1ならユニークキーが1つ
        # shots>1なら分布から一番多いものを採用 or 全展開
        # ここでは全展開する（出現回数ぶん返す）
        seqs = []
        for bitstr, cnt in counts.items():
            seqs += [normalize(bitstr)] * cnt
        return seqs
    else:
        # あり得ないが型ガード
        return []


def make_timetable(bitstr: str, params: BeatParams, per_beat_phase_ms: List[float]) -> List[dict]:
    """
    測定列(bitstr)に対し、各拍の開始時刻＆オフセットを計算して時刻表を返す
      entry: {index, beat_time_sec, phase_ms, event(HIT/REST)}
    """
    beat_sec = beat_duration_sec(params.tempo_bpm)
    table = []
    for i, b in enumerate(bitstr):
        base_t = i * beat_sec
        # 位相は per_beat_phase_ms[i] に相当（Rzで入れたもの）
        # ms_to_rad と逆変換…でもラジアン→msは線形なので保存済みmsを渡す方が素直
        # ここでは per_beat_phase_ms を直渡ししている想定。
        # 実装簡素化のため呼び出し側で ms を渡す。
        # ここでは placeholder、実際は呼出側が埋める。
        phase_ms = per_beat_phase_ms[i]
        event = "HIT" if b == "1" else "REST"
        t_offset = phase_ms / 1000.0
        table.append({
            "index": i,
            "beat_time_sec": base_t,
            "phase_ms": phase_ms,
            "event_time_sec": max(0.0, base_t + t_offset),
            "event": event
        })
    return table


# ====== デモ実行 ======
if __name__ == "__main__":
    params = BeatParams(
        tempo_bpm=120.0,
        bars=4,
        beats_per_bar=4,
        base_hit_prob=0.85,
        stillness_prob=0.10,
        humanize_prob_jitter=0.05,
        responsibility_phase_ms=8.0,   # ほんの少し“溜め”
        humanize_phase_ms=6.0,
        global_phase_rad=0.0,
        seed=20251002
    )

    # 回路を作る
    qc, thetas, phis = build_rhythm_circuit(params)

    # 参考: 各拍の「位相(ms)」も保存しておく（Rzで入れたφをmsに戻す）
    # ms_to_rad を使ったので、逆に rad→ms はこの線形を戻せばOK
    def rad_to_ms(phi_rad: float, bpm: float) -> float:
        beat_ms = beat_duration_sec(bpm) * 1000.0
        return (phi_rad / (2.0 * math.pi)) * beat_ms

    per_beat_phase_ms = [rad_to_ms(phi, params.tempo_bpm) for phi in phis]

    # 量子サンプリング（1テイク）
    seqs = sample_rhythm(qc, shots=1)
    bitstr = seqs[0] if seqs else ""

    # 時刻表に落とす
    table = make_timetable(bitstr, params, per_beat_phase_ms)

    # 出力
    print("=== Beat Quantum Rhythm (Qiskit) ===")
    print(f"tempo: {params.tempo_bpm} bpm, bars: {params.bars}, beats/bar: {params.beats_per_bar}")
    print(f"hit_prob≈{params.base_hit_prob} - stillness≈{params.stillness_prob} (± jitter)")
    print(f"responsibility_phase_ms≈{params.responsibility_phase_ms} (± {params.humanize_phase_ms})")
    print(f"bitstring (0=REST, 1=HIT): {bitstr}")
    print("-- timetable --")
    for row in table:
        i = row["index"]
        bt = row["beat_time_sec"]
        et = row["event_time_sec"]
        ph = row["phase_ms"]
        ev = row["event"]
        print(f"beat {i:02d} | base={bt:5.3f}s | phase={ph:+6.2f} ms | event@{et:5.3f}s | {ev}")

    # 追加: QASM図を見たいときはアンコメント
    # print(qc.draw("text"))
