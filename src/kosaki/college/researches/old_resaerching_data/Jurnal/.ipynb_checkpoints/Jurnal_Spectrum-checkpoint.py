import matplotlib.pyplot as plt

# 仮想データ（例：名前・周波数・カラーコード）
labels = [
    "跳ねた願望の波", "もどかしさ交差波", "自己欲求のまる波", "迷いのグルーヴ波",
    "観照記録波", "整理中のカクカク波", "整理昇華波", "解放志向波",
    "均衡突破波", "構造化モード波", "相対重力波", "関係終息波"
]
frequencies = [41.14, 30.19, 35.62, 33.15, 22.94, 31.32, 27.00, 39.69, 39.60, 34.38, 24.59, 54.40]
colors = [
    "#FFD700", "#708090", "#90EE90", "#87CEEB", "#6A5ACD", "#D3D3D3",
    "#98FB98", "#20B2AA", "#CD5C5C", "#228B22", "#4B0082", "#708090"
]

# 描画
plt.figure(figsize=(12, 6))
bars = plt.bar(labels, frequencies, color=colors, edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Virtual Spectrum / Hz")
plt.title("Graph")
plt.tight_layout()
plt.show()
