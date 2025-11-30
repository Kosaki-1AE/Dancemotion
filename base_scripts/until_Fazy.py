# -*- coding: utf-8 -*-
"""
Fuzzy-Pipeline: Shannon → Perceptual + Conditional → Fuzzy (仮安定点)
- しゃきしゃきモデルの「不確実性→足場決め」までを一気通貫で実装
- 依存: numpy のみ
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# =========================
# 第1段階: シャノン（不確実性の全体量を把握）
# =========================

def shannon_entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> float:
    """
    H(P) = -Σ p_i log2 p_i
    - p: 確率分布（正・合計1）
    - 数値安定化のため微小値 eps を加える
    """
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))

def shannon_entropy_from_counts(counts: np.ndarray, eps: float = 1e-12) -> float:
    """
    観測頻度からシャノンエントロピーを推定
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    return shannon_entropy_from_probs(counts / total, eps=eps)

# =========================
# 第2段階: パーセプチュアル + 条件付き
#   - パーセプチュアル: 重要部分だけを扱う（知覚選別・圧縮）
#   - 条件付き: ある情報Xが与えられた後の残余不確実性 H(Y|X)
# =========================

def perceptual_select(counts: np.ndarray, topk: int | None = None, energy_ratio: float | None = None) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """
    知覚選別: 「目立つ（重要）」カテゴリだけ残す
    - counts: 1次元の観測頻度ベクトル
    - topk: 上位kカテゴリのみ残す（例: 3）
    - energy_ratio: 上位から累積でこの比率まで残す（例: 0.9）
    戻り値:
      - kept_counts: 選別後カウント（落とした分はまとめて"その他"にプール）
      - stats: {'keep_fraction': 保持カテゴリ数/元カテゴリ数, 'mass_fraction': 残した確率質量の比}
      - keep_mask: どれを残したかのbool配列（same length as counts）
    """
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        # 全部ゼロならそのまま返す
        return counts.copy(), {'keep_fraction': 1.0, 'mass_fraction': 1.0}, np.ones_like(counts, dtype=bool)

    # 重要度: 単純に頻度の高い順（必要ならMIや勾配などに置換可能）
    order = np.argsort(-counts)
    sorted_c = counts[order]
    probs = sorted_c / total
    keep_idx = np.arange(len(counts))

    if energy_ratio is not None:
        csum = np.cumsum(probs)
        k = int(np.searchsorted(csum, energy_ratio, side='left')) + 1
        topk = k if topk is None else min(topk, k)

    if topk is None:
        topk = len(counts)

    sel = order[:topk]
    drop = order[topk:]

    kept = counts.copy()
    # 落とした分は「その他」バケットとして末尾に集約（UI的に一つに潰す想定）
    other_mass = counts[drop].sum()
    kept[drop] = 0.0

    # その他バケットが無い実装ならここでスキップ。今回は便宜上、最大要素に足す。
    if len(sel) > 0 and other_mass > 0:
        kmax = sel[0]
        kept[kmax] += other_mass

    keep_mask = np.zeros_like(counts, dtype=bool)
    keep_mask[sel] = True

    mass_fraction = (counts[sel].sum()) / total
    keep_fraction = len(sel) / len(counts)

    stats = dict(keep_fraction=float(keep_fraction), mass_fraction=float(mass_fraction))
    return kept, stats, keep_mask

def conditional_entropy_from_joint(joint: np.ndarray, eps: float = 1e-12) -> float:
    """
    条件付きエントロピー H(Y|X) を2次元同時分布から計算
      - joint[x, y] >= 0
      - 正規化は内部で実施
    定義: H(Y|X) = Σ_x p(x) H(Y|X=x)
    """
    J = np.asarray(joint, dtype=float)
    total = J.sum()
    if total <= 0:
        return 0.0
    Pxy = J / total
    Px = Pxy.sum(axis=1, keepdims=True)  # shape: (X,1)
    # 数値安定化
    Px = np.clip(Px, eps, 1.0)
    Py_given_x = Pxy / Px
    # 各xでのH(Y|X=x)
    Hy_given_x = -np.sum(np.clip(Py_given_x, eps, 1.0) * np.log2(np.clip(Py_given_x, eps, 1.0)), axis=1)
    # 重み付き平均
    H = float(np.sum(Hy_given_x * Px.squeeze(-1)))
    return H

# =========================
# 第3段階: ファジィ論理（グレーのまま動くための仮安定点）
#   - 入力: 
#       U_total  : 全体不確実性（シャノン）
#       U_resid  : 残余不確実性（条件付き）
#       keep_frac: 知覚選別で残した「カテゴリ率」
#       mass_frac: 知覚選別で保持した「確率質量率」
#   - 出力:
#       stability_score in [0,1]（高いほど「一旦この足場でいける」）
# =========================

# ---- メンバーシップ関数（台形/三角形ベース） ----

def tri_mf(x, a, b, c):
    """三角型メンバーシップ（a<=b<=c）"""
    if x <= a or x >= c: 
        return 0.0
    if x == b: 
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)

def trap_mf(x, a, b, c, d):
    """台形メンバーシップ（a<=b<=c<=d）"""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a + 1e-12)
    return (d - x) / (d - c + 1e-12)

# ---- 入力ファジィ化の設計（経験的な素直パラメタ） ----
def fuzzify_inputs(U_total: float, U_resid: float, keep_frac: float, mass_frac: float) -> Dict[str, Dict[str, float]]:
    """
    各入力の「low/med/high」メンバーシップ度を返す
    - 不確実性は 0~U_max で正規化して扱う想定（ここではU_max=log2(N)などを外部で調整可）
    - keep_frac, mass_frac は 0~1
    """
    def fuzzify_uncertainty(u):
        # 低: 0~0.3, 中: 0.2~0.7, 高: 0.5~1.0 くらいの素直な切り方（被覆させる）
        return {
            'low' : trap_mf(u, -0.05, 0.0, 0.2, 0.35),
            'med' : tri_mf(u, 0.2, 0.45, 0.7),
            'high': trap_mf(u, 0.55, 0.7, 1.0, 1.05),
        }
    def fuzzify_fraction(f):
        # 分布保持率: 低: ~0.4, 中: ~0.7, 高: ~1.0
        return {
            'low' : trap_mf(f, -0.05, 0.0, 0.25, 0.45),
            'med' : tri_mf(f, 0.35, 0.55, 0.75),
            'high': trap_mf(f, 0.65, 0.8, 1.0, 1.05),
        }
    return {
        'U_total': fuzzify_uncertainty(U_total),
        'U_resid': fuzzify_uncertainty(U_resid),
        'keep_frac': fuzzify_fraction(keep_frac),
        'mass_frac': fuzzify_fraction(mass_frac),
    }

# ---- ルールベース（Mamdani, min-max合成） ----
# 直観ルール例：
#  R1: 残余不確実性が低く かつ 保持率(mass/keep)が高 → 安定 高
#  R2: 残余が中 かつ 保持が中 → 安定 中
#  R3: 残余が高 または 全体不確実性が高 かつ 保持が低 → 安定 低
#  R4: 全体が低 かつ 残余が低 → 安定 高（そもそも簡単）
#  R5: 全体が高だが保持が高 かつ 残余が中以下 → 安定 中（圧縮が効いてる）
#  R6: 残余が高 でも keep_fracが高 & mass_fracが高 → 安定 中（強気に行く足場）

def aggregate_rules(fz: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    低・中・高 の各出力集合の強度（αカット）を返す
    """
    U_t = fz['U_total']
    U_r = fz['U_resid']
    K = fz['keep_frac']
    M = fz['mass_frac']

    out = {'low': 0.0, 'med': 0.0, 'high': 0.0}

    # R1
    r1 = min(U_r['low'], max(K['high'], M['high']))
    out['high'] = max(out['high'], r1)

    # R2
    r2 = min(U_r['med'], min(K['med'], M['med']))
    out['med'] = max(out['med'], r2)

    # R3
    r3 = max(U_r['high'], min(U_t['high'], max(K['low'], M['low'])))
    out['low'] = max(out['low'], r3)

    # R4
    r4 = min(U_t['low'], U_r['low'])
    out['high'] = max(out['high'], r4)

    # R5
    r5 = min(U_t['high'], max(K['high'], M['high']), max(U_r['low'], U_r['med']))
    out['med'] = max(out['med'], r5)

    # R6
    r6 = min(U_r['high'], min(K['high'], M['high']))
    out['med'] = max(out['med'], r6)

    return out

# ---- 出力メンバーシップ（安定度スコア 0~1） ----
def stability_output_mf(level: str, x: float) -> float:
    """
    出力のメンバーシップ:
      low:   台形(0.0, 0.0, 0.2, 0.45)
      med:   三角(0.3, 0.5, 0.7)
      high:  台形(0.55, 0.8, 1.0, 1.0)
    """
    if level == 'low':
        return trap_mf(x, 0.0, 0.0, 0.2, 0.45)
    if level == 'med':
        return tri_mf(x, 0.3, 0.5, 0.7)
    if level == 'high':
        return trap_mf(x, 0.55, 0.8, 1.0, 1.0)
    return 0.0

def defuzzify_centroid(alpha: Dict[str, float], num: int = 201) -> float:
    """
    重心法でスカラーにデフラズィ化
    - alpha: {'low': μ, 'med': μ, 'high': μ}（各出力集合の強度）
    - 出力宇宙は [0,1]
    """
    xs = np.linspace(0.0, 1.0, num=num)
    mus = np.zeros_like(xs)
    for i, x in enumerate(xs):
        mu_low  = min(alpha['low'],  stability_output_mf('low',  x))
        mu_med  = min(alpha['med'],  stability_output_mf('med',  x))
        mu_high = min(alpha['high'], stability_output_mf('high', x))
        mus[i] = max(mu_low, mu_med, mu_high)
    denom = mus.sum() + 1e-12
    return float((xs * mus).sum() / denom)

# =========================
# 便利ユーティリティ
# =========================

def normalize_entropy(value: float, max_entropy: float, clip: bool = True) -> float:
    """
    エントロピーを [0,1] に正規化
    """
    if max_entropy <= 0:
        return 0.0
    v = value / max_entropy
    if clip:
        v = float(np.clip(v, 0.0, 1.0))
    return v

# =========================
# デモ（__main__）
# =========================

if __name__ == "__main__":
    # --- 想定データ ---
    # 例: 6カテゴリの観測頻度（ラベルA~F）
    counts = np.array([30, 25, 12, 10, 8, 5], dtype=float)
    N = len(counts)
    H_total = shannon_entropy_from_counts(counts)
    H_max   = np.log2(N)                 # 最大エントロピー（等確率のとき）
    H_total_n = normalize_entropy(H_total, H_max)

    # パーセプチュアル選別（上位90%の確率質量を残す or 上位3カテゴリだけ残す）
    kept_counts, stats, keep_mask = perceptual_select(counts, topk=3, energy_ratio=0.9)
    keep_frac = stats['keep_fraction']   # 残したカテゴリ率
    mass_frac = stats['mass_fraction']   # 残した確率質量率

    # 条件付きエントロピー: 例として X=「文脈3種類」, Y=「上の6カテゴリ」
    # 適当な同時分布（本番は実データでOK）
    rng = np.random.default_rng(42)
    joint = rng.integers(0, 50, size=(3, N)).astype(float)
    H_resid = conditional_entropy_from_joint(joint)   # H(Y|X)
    H_resid_n = normalize_entropy(H_resid, np.log2(N))

    print("=== Stage1: Shannon ===")
    print(f"H_total = {H_total:.3f} bits (H_max={H_max:.3f}) -> normalized={H_total_n:.3f}")
    print("\n=== Stage2: Perceptual Select + Conditional ===")
    print(f"keep_fraction = {keep_frac:.3f}, mass_fraction = {mass_frac:.3f}")
    print(f"H_resid = {H_resid:.3f} bits -> normalized={H_resid_n:.3f}")

    # --- ファジィ論理で仮安定点を決定 ---
    fz_inputs = fuzzify_inputs(H_total_n, H_resid_n, keep_frac, mass_frac)
    alpha = aggregate_rules(fz_inputs)
    stability = defuzzify_centroid(alpha)

    print("\n=== Stage3: Fuzzy (Provisional Stability) ===")
    print(f"rule_strengths = {alpha}")
    print(f"stability_score (0~1) = {stability:.3f}")

    # しきい値例：
    if stability >= 0.66:
        decision = "GO（この足場で進める）"
    elif stability >= 0.4:
        decision = "HOLD（もう少し様子見・軽く追加観測）"
    else:
        decision = "GATHER（情報収集を優先）"

    print(f"→ 仮決定: {decision}")
