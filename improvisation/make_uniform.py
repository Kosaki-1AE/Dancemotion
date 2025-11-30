# 1粒を作る
grain = QuantumGrain.uniform(
    basis_actions=["攻める", "守る", "様子見"],
    responsibility=np.array([0.9, 0.7, 0.3]),    # 責任の矢っぽい分布
    coherence_score=np.array([0.8, 0.6, 0.2]),   # 空気感とのフィット度
)

# Genesys: 責任でゆがめる
grain.apply_responsibility_bias(beta=1.5)

# Stillness: 位相を少し回す
grain.apply_stillness_phase(delta_theta=0.3)

# Motion: 適当なユニタリで「思考のジャンプ」
# （ここは本気出すならちゃんとユニタリ作る）
U = np.eye(3, dtype=np.complex128)
grain.evolve_unitary(U)

# ノリエントロピーを評価
N = grain.nori_entropy()

# 観測（Coherence）：実際にどの行動にcollapseしたか
action = grain.measure()
