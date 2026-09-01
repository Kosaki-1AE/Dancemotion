import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit_aer import Aer


class ResponsibilityQuantum:
    def __init__(self, R_self=1.0, R_other=1.0):
        """
        R_self : 自分への責任
        R_other: 他者／場への責任
        """
        self.R_self = R_self
        self.R_other = R_other

    def build_circuit(self, alpha, beta):
        """
        量子回路を構築
        alpha, beta: 複素数係数
        """
        # 複素数係数を正規化
        norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
        alpha /= norm
        beta /= norm

        qc = QuantumCircuit(1, 1)  # 1 qubit, 1 classical bit

        # 状態ベクトルを初期化（|0>がA, |1>がB）
        qc.initialize([alpha, beta], 0)

        # 責任ベクトルを回転に反映
        theta = (self.R_self - self.R_other) * np.pi / 4
        qc.ry(theta, 0)

        # 測定
        qc.measure(0, 0)
        return qc

    def simulate(self, alpha, beta, shots=1024):
        """
        回路をシミュレーションして結果を返す
        """
        qc = self.build_circuit(alpha, beta)

        # Aer シミュレーター（assemble不要の新しい呼び方）
        sim = Aer.get_backend("aer_simulator")
        tqc = transpile(qc, sim)
        result = sim.run(tqc, shots=shots).result()
        counts = result.get_counts()

        return counts, qc


# ===== 実行例 =====
if __name__ == "__main__":
    rq = ResponsibilityQuantum(R_self=1.2, R_other=0.8)

    alpha = 1.0 + 0.3j   # A側の係数
    beta  = 0.7 + 0.6j   # B側の係数

    counts, qc = rq.simulate(alpha, beta, shots=1000)

    print("観測結果:", counts)
    qc.draw("mpl")
    plt.show()

    # ヒストグラムで可視化
    plot_histogram(counts)
    plt.show()
