# -*- coding: utf-8 -*-
"""
Quantum-Aha Engine (MVP版)
 - ノード: 単語（形態素解析結果を想定）
 - 辺   : 直前要素との関係
 - 判定 : ベクトル距離×角度の一致度を計算
"""

import networkx as nx
import numpy as np

# ====== Step 1: グラフ構造の用意 ======
G = nx.DiGraph()

# ノード（単語＋タグ）
# 例: ("AI", {"label": "情報学"}), ("脳", {"label": "神経科学"})
nodes = [
    ("AI", {"label": "情報学"}),
    ("脳", {"label": "神経科学"}),
    ("量子", {"label": "物理学"}),
    ("感情", {"label": "心理学"}),
]
G.add_nodes_from(nodes)

# 辺（直前関係＋距離＋角度）
# 距離はスカラー、角度はラジアンで指定
edges = [
    ("AI", "量子", {"distance": 1.0, "angle": 0.0}),
    ("脳", "感情", {"distance": 1.0, "angle": np.pi/2}),
    ("量子", "脳", {"distance": 1.0, "angle": np.pi/4}),
]
G.add_edges_from(edges)

# ====== Step 2: ベクトル化関数 ======
def edge_to_vector(edge_data):
    """距離と角度から2Dベクトルに変換"""
    d, theta = edge_data["distance"], edge_data["angle"]
    return np.array([d * np.cos(theta), d * np.sin(theta)])

# ====== Step 3: 一致度計算 ======
def match_score(vec1, vec2):
    """コサイン類似度で一致度を計算"""
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / norm if norm != 0 else 0

# ====== Step 4: 探索 ======
def find_best_match(G, start):
    if start not in G:
        print(f"[WARN] startノードが無い: {start}")
        return []

    out_edges = list(G.out_edges(start, data=True))
    if len(out_edges) < 2:
        print(f"[INFO] {start} から出る辺が {len(out_edges)} 本。比較できん（最低2本必要）")
        print("edges:", out_edges)
        return []

    """startノードから出る辺の一致度を調べる"""
    if start not in G:
        return None

    start_edges = G.out_edges(start, data=True)
    results = []
    for u, v, data in start_edges:
        vec = edge_to_vector(data)
        # 次のノードが持つラベルと一致度を記録
        results.append((v, G.nodes[v]["label"], vec))

    # ベクトル間一致度をチェック（ペアで比較）
    matches = []
    for i in range(len(results)):
        for j in range(i+1, len(results)):
            v1, label1, vec1 = results[i]
            v2, label2, vec2 = results[j]
            score = match_score(vec1, vec2)
            matches.append(((v1, label1), (v2, label2), score))

    return matches

# ====== Step 5: 実行例 ======
start = "量子"
matches = find_best_match(G, start)
if not matches:
    print("[RESULT] 一致ペア無し（上のINFO/WARNメッセージ参照）")
else:
    # スコア降順で表示
    matches.sort(key=lambda x: x[2], reverse=True)
    for (n1, l1), (n2, l2), s in matches:
        print(f"{start}→{n1}({l1})  vs  {start}→{n2}({l2})  :  一致度={s:.3f}")
    # トップだけ強調
    (n1, l1), (n2, l2), s = matches[0]
    print(f"\n[TOP] 次動き候補：{start}→{n1} と {start}→{n2} が最も整合（{s:.3f}）")

