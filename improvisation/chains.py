from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ここは「さっき作った QuantumGrain クラス」が定義済み前提
from QuantumGrain import QuantumGrain


@dataclass
class QuantumChain:
    """
    QuantumGrain を縦に10個とか並べて、
    全体として時間発展・カップリング・ノリエントロピーを見たい用のクラス。
    """
    grains: List[QuantumGrain] = field(default_factory=list)

    # -------------------
    # 10個まとめて初期化
    # -------------------
    @classmethod
    def uniform_chain(
        cls,
        length: int,
        basis_actions: List[str],
        responsibility: Optional[np.ndarray] = None,
        coherence_score: Optional[np.ndarray] = None,
    ) -> "QuantumChain":
        """
        同じ basis_actions / 初期パラメータ の粒を length 個並べたチェーンを作る。
        """
        grains = []
        for _ in range(length):
            g = QuantumGrain.uniform(
                basis_actions=basis_actions,
                responsibility=responsibility,
                coherence_score=coherence_score,
            )
            grains.append(g)
        return cls(grains=grains)

    # -------------------
    # チェーン全体への一括操作
    # -------------------
    def apply_responsibility_bias_all(self, beta: float = 1.0) -> None:
        """全粒に Genesys 的な責任バイアスをかける"""
        for g in self.grains:
            g.apply_responsibility_bias(beta=beta)

    def apply_stillness_phase_all(self, delta_theta: float) -> None:
        """全粒に Stillness 的な位相回転をかける"""
        for g in self.grains:
            g.apply_stillness_phase(delta_theta=delta_theta)

    def evolve_unitary_all(self, U: np.ndarray) -> None:
        """全粒を同じユニタリ U で Motion 進化させる"""
        for g in self.grains:
            g.evolve_unitary(U)

    # -------------------
    # 隣同士を“ゆるくリンク”させるカップリング
    # -------------------
    def couple_neighbors(self, k: float = 0.2) -> None:
        """
        隣同士の粒の状態が少し混ざる（空気感が伝播するイメージ）。
        k はカップリング強度（0〜1推奨）。
        new_amp_i = (1-k)*amp_i + (k/2)*(amp_{i-1} + amp_{i+1})
        端っこは片側だけ。
        """
        n = len(self.grains)
        if n == 0:
            return

        # 元の振幅をコピー（同時更新するため）
        original_amps = [g.amplitudes.copy() for g in self.grains]

        for i, g in enumerate(self.grains):
            amp = original_amps[i].copy()
            mix = np.zeros_like(amp)

            # 自分
            mix += (1.0 - k) * original_amps[i]

            # 左隣
            if i > 0:
                mix += (k / 2.0) * original_amps[i - 1]

            # 右隣
            if i < n - 1:
                mix += (k / 2.0) * original_amps[i + 1]

            g.amplitudes = mix
            g.normalize()
            g.log(f"Coupled with neighbors (k={k:.3f})")

    # -------------------
    # チェーン全体のノリエントロピー
    # -------------------
    def total_nori_entropy(self) -> float:
        """
        全粒のノリエントロピーの合計。
        （平均が欲しければ / len(self.grains) してもOK）
        """
        if not self.grains:
            return 0.0
        return sum(g.nori_entropy() for g in self.grains)

    def mean_nori_entropy(self) -> float:
        """平均ノリエントロピー"""
        if not self.grains:
            return 0.0
        return self.total_nori_entropy() / len(self.grains)

    # -------------------
    # 1ステップの「時間発展」をまとめてやる
    # -------------------
    def step(
        self,
        beta: float = 1.0,
        delta_theta: float = 0.0,
        coupling_k: float = 0.2,
        U: Optional[np.ndarray] = None,
    ) -> None:
        """
        1タイムステップ分の更新：
          1. Genesys: 責任バイアス
          2. Stillness: 位相
          3. カップリング: 隣との干渉
          4. Motion: ユニタリ進化
        """
        # 1. 責任で揺らぎを傾ける
        self.apply_responsibility_bias_all(beta=beta)

        # 2. Stillness の位相を全体にかける
        if delta_theta != 0.0:
            self.apply_stillness_phase_all(delta_theta=delta_theta)

        # 3. 隣同士の状態を少し混ぜる（空気感伝播）
        if coupling_k > 0.0:
            self.couple_neighbors(k=coupling_k)

        # 4. Motion: 各粒を同じ U で進化（任意）
        if U is not None:
            self.evolve_unitary_all(U)

    # -------------------
    # チェーン全体を観測する
    # -------------------
    def measure_all(self) -> List[str]:
        """
        全粒を観測し、それぞれどの行動に collapse したかを返す。
        """
        results = []
        for g in self.grains:
            a = g.measure()
            results.append(a)
        return results
