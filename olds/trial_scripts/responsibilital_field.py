import numpy as np

# 軸: 論理・感覚・時間・社会
axes = ["論理", "感覚", "時間", "社会"]

# === 初期の責任場 R (ランダム生成 + 対称化) ===
R = np.random.uniform(-1, 1, (4, 4))
R = 0.5 * (R + R.T)

# === 責任流 J ===
# 個人の責任ベクトル (例)
J_vec = np.array([0.7, 0.9, 0.6, 0.8])
J = np.outer(J_vec, J_vec)

# === 空気感テンソル A = κ * J + dR/dt ===
# dR/dt を単純化 → 小さな乱数で場のゆらぎを表現
dR_dt = np.random.normal(0, 0.05, (4, 4))
dR_dt = 0.5 * (dR_dt + dR_dt.T)

kappa = 0.8  # 場と流の結合強度

A = dR_dt + kappa * J

# === 可観測スカラー: 空気感の総合スコア ===
# ここでは Frobenius norm (テンソル全体の大きさ) を使う
airfeel_score = np.linalg.norm(A, 'fro')

# === 出力 ===
print("責任場 R (初期):")
print(np.round(R, 3))

print("\n責任流 J:")
print(np.round(J, 3))

print("\n空気感テンソル A = dR/dt + κ*J:")
print(np.round(A, 3))

print("\n空気感スコア |A| =", round(airfeel_score, 3))
