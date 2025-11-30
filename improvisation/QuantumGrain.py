from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class QuantumGrain:
    """
    「量子の1粒」＝
      - 行動の重ね合わせ状態 |psi>
      - 責任の矢（各行動への責任スコア）
      - 空気感・Stillnessとのフィット度
    を1ユニットにまとめたクラス。
    """
    basis_actions: List[str]                    # |a_0>, |a_1>, ...
    amplitudes: np.ndarray                      # 複素振幅ベクトル α_i
    responsibility: np.ndarray                  # 各行動の責任スコア r_i ∈ [0,1]
    coherence_score: np.ndarray                 # 各行動の空気感スコア c_i ∈ [0,1]
    stillness_phase: float = 0.0                # グローバル位相 θ（Stillness）

    history: List[str] = field(default_factory=list)  # ログ用（任意）

    # -------------------
    # 基本プロパティ
    # -------------------
    def __post_init__(self):
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.complex128)
        self.responsibility = np.asarray(self.responsibility, dtype=np.float64)
        self.coherence_score = np.asarray(self.coherence_score, dtype=np.float64)

        n = len(self.basis_actions)
        if not (len(self.amplitudes) == len(self.responsibility) == len(self.coherence_score) == n):
            raise ValueError("basis_actions / amplitudes / responsibility / coherence_score の長さが揃っていません。")

        self.normalize()
        self.log("Initialized QuantumGrain")

    @property
    def probabilities(self) -> np.ndarray:
        """p_i = |α_i|^2"""
        return np.abs(self.amplitudes) ** 2

    @property
    def shannon_entropy(self) -> float:
        """通常のシャノンエントロピー H = -Σ p log p"""
        p = self.probabilities
        p_safe = p[p > 0]
        return float(-np.sum(p_safe * np.log(p_safe)))

    # -------------------
    # ログ
    # -------------------
    def log(self, msg: str) -> None:
        self.history.append(msg)

    # -------------------
    # 正規化
    # -------------------
    def normalize(self) -> None:
        """||psi|| = 1 になるように振幅を正規化"""
        norm = np.linalg.norm(self.amplitudes)
        if norm == 0:
            # すべて0の場合は一様分布にしておく
            self.amplitudes = np.ones_like(self.amplitudes, dtype=np.complex128) / np.sqrt(len(self.amplitudes))
        else:
            self.amplitudes /= norm

    # -------------------
    # Stillness: 位相操作
    # -------------------
    def apply_stillness_phase(self, delta_theta: float) -> None:
        """
        Stillness層に相当。
        グローバル位相 e^{iθ} を掛ける（確率は変わらないが、干渉に効く）。
        """
        self.stillness_phase += delta_theta
        phase = np.exp(1j * delta_theta)
        self.amplitudes *= phase
        self.log(f"Applied stillness phase: Δθ={delta_theta:.4f}")

    # -------------------
    # Genesys: 責任のバイアスをかける
    # -------------------
    def apply_responsibility_bias(self, beta: float = 1.0) -> None:
        """
        Genesys層に相当。
        責任スコア r_i で確率振幅をゆがめる：
          α_i' ∝ α_i * exp(beta * r_i)
        beta > 0: 責任高い行動ほど選ばれやすくなる
        """
        weights = np.exp(beta * self.responsibility)
        self.amplitudes *= weights
        self.normalize()
        self.log(f"Applied responsibility bias with beta={beta:.3f}")

    # -------------------
    # Motion: ユニタリ変換で時間発展
    # -------------------
    def evolve_unitary(self, U: np.ndarray) -> None:
        """
        Motion層に相当。
        ユニタリ行列 U をかけて時間発展 |psi'> = U |psi>。
        U は (n x n) の行列。
        """
        U = np.asarray(U, dtype=np.complex128)
        if U.shape != (len(self.amplitudes), len(self.amplitudes)):
            raise ValueError("U のサイズが状態ベクトルと一致していません。")

        self.amplitudes = U @ self.amplitudes
        self.normalize()
        self.log("Evolved state by unitary U (Motion)")

    # -------------------
    # Coherence: 観測 & collapse
    # -------------------
    def measure(self, rng: Optional[np.random.Generator] = None) -> str:
        """
        Coherence層に相当。
        観測によって1つの行動 |a_k> にcollapseし、その状態に固定する。
        返り値：選ばれた行動ラベル。
        """
        if rng is None:
            rng = np.random.default_rng()

        p = self.probabilities
        idx = int(rng.choice(len(p), p=p))
        chosen_action = self.basis_actions[idx]

        # collapse: |psi> = |a_k>
        new_amplitudes = np.zeros_like(self.amplitudes)
        new_amplitudes[idx] = 1.0 + 0j
        self.amplitudes = new_amplitudes
        self.normalize()

        self.log(f"Measured and collapsed to action: {chosen_action}")
        return chosen_action

    # -------------------
    # ノリエントロピー
    # -------------------
    def nori_entropy(self) -> float:
        """
        ノリエントロピー N を計算する。
        定義（暫定）：
          H = シャノンエントロピー
          C_eff = Σ p_i * r_i * c_i
          N = H * C_eff
        """
        p = self.probabilities
        H = self.shannon_entropy
        C_eff = float(np.sum(p * self.responsibility * self.coherence_score))
        N = H * C_eff
        self.log(f"Computed Nori Entropy: H={H:.4f}, C_eff={C_eff:.4f}, N={N:.4f}")
        return N

    # -------------------
    # ユーティリティ：一様状態から生成
    # -------------------
    @classmethod
    def uniform(
        cls,
        basis_actions: List[str],
        responsibility: Optional[np.ndarray] = None,
        coherence_score: Optional[np.ndarray] = None,
    ) -> "QuantumGrain":
        """
        すべての行動が等しい重みで重ね合わさった初期状態を作成する。
        責任と空気感スコアは指定なければ一様(=0.5)。
        """
        n = len(basis_actions)
        amps = np.ones(n, dtype=np.complex128) / np.sqrt(n)

        if responsibility is None:
            responsibility = np.full(n, 0.5, dtype=np.float64)
        if coherence_score is None:
            coherence_score = np.full(n, 0.5, dtype=np.float64)

        return cls(
            basis_actions=basis_actions,
            amplitudes=amps,
            responsibility=responsibility,
            coherence_score=coherence_score,
        )

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
