from qiskit import QuantumCircuit, Aer, execute
import numpy as np

# === 1. 重ね合わせ状態を準備（責任の矢の候補） ===
# ここでは3つの選択肢を2量子ビットで表現
n_qubits = 2
qc = QuantumCircuit(n_qubits, n_qubits)

# |00> → (H x H) で全ての選択肢の重ね合わせ状態に
qc.h(0)
qc.h(1)

# === 2. 責任の矢の方向付け（戦略ゲート） ===
# 例えば、ある選択肢（|01>や|10>）に少し重みを寄せる
theta = np.pi/6  # バイアス角
qc.ry(theta, 0)   # qubit0に回転
qc.ry(-theta, 1)  # qubit1に逆回転

# === 3. 未測定のまま「行動」をシミュレーション ===
# 実際にはここで外部の「行動関数」を呼ぶイメージ
# （この時点では確率しか決まっていない）
# action = f(quantum_state)  ← 擬似的にここで動かす

# === 4. 最後に測定してどの行動が現実化したかを見る ===
qc.measure([0,1], [0,1])

# === 5. 実行 ===
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts()

print("量子状態から選ばれた行動分布:")
print(counts)
print("\n回路図:")
print(qc)
