import numpy as np

from QuantumGrain import QuantumChain

# 1粒あたりの行動の基底
basis = ["攻める", "守る", "様子見"]

# 行動ごとの責任と空気感（ここは適当に例）
responsibility = np.array([0.9, 0.6, 0.3])
coherence_score = np.array([0.8, 0.5, 0.4])

# 🔟粒のチェーンを作成
chain = QuantumChain.uniform_chain(
    length=10,
    basis_actions=basis,
    responsibility=responsibility,
    coherence_score=coherence_score,
)

# Motion 用のユニタリ（ここはとりあえず単位行列でOK）
U = np.eye(len(basis), dtype=np.complex128)

# 何ステップか時間発展させてみる
for t in range(5):
    chain.step(
        beta=1.0,          # 責任バイアスの強さ
        delta_theta=0.1,   # Stillness 位相
        coupling_k=0.3,    # 隣とのリンク強度
        U=U,               # Motion ゲート（今は単位）
    )
    print(f"step {t}: mean Nori Entropy = {chain.mean_nori_entropy():.4f}")

# 最後にチェーン全体を観測
actions = chain.measure_all()
print("collapse 後の行動一覧:")
for i, a in enumerate(actions):
    print(f"  粒 {i}: {a}")
