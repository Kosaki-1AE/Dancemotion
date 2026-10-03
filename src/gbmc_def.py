import math

import numpy as np


class KQIEngine:
    """
    KQI理論 (Kansei Quantum Information Theory) に基づくGSMC統合モデルエンジン。
    
    定義:
    G = r * omega
    B = {sigma, tau} (where tau = t * rho)
    Q = (r * omega)^2 / sigma
    M = H(Q - theta)
    C = limit_{n->k} (1 + tau/n)^n
    D = Q * C (Gate controlled by M)
    """

    def __init__(self):
        # 厳密な計算のために極小値を定義 (ゼロ除算防止用)
        self.EPSILON = 1e-9

    def heaviside(self, x):
        """ヘヴィサイドの階段関数 H(x)"""
        return 1.0 if x >= 0 else 0.0

    def calculate_coherence(self, tau, k):
        """
        Coherence (C) の計算
        k = 0      -> Static Mode (変化なし)
        k = 1      -> Linear Mode (線形積み上げ)
        k = 'inf'  -> Resonance Mode (指数関数的爆発)
        """
        if k == 0:
            return 1.0
        elif k == 1:
            return 1.0 + tau
        elif k == 'inf' or k == float('inf'):
            # lim_{n->inf} (1 + x/n)^n = e^x
            return math.exp(tau)
        else:
            # 任意の k に対する一般解が必要なら実装するが、
            # 定義上 {0, 1, inf} なのでそれ以外はエラーか1とする
            return 1.0

    def process_frame(self, r, omega, sigma, t, rho, k, theta):
        """
        1フレーム分の入力データを受け取り、GSMCプロセスを実行してD(Depth)を返す。
        
        Args:
            r (float): Physical Range (身体的拡張)
            omega (float): Angular Velocity (回転速度)
            sigma (float): Variance/Noise (散乱係数)
            t (float): Time duration
            rho (float): Density (密度)
            k (int/str): Mode selector {0, 1, 'inf'}
            theta (float): Threshold (閾値)
        
        Returns:
            dict: 各段階の計算結果 (G, Q, M, C, D)
        """
        
        # 1. Genesys (G) - 発生
        G = r * omega
        
        # 2. Base (B) - 場の定義
        # sigmaは入力、tauを計算
        tau = t * rho
        
        # 安全策: sigmaが0の場合の処理（理論上無限大になるが、プログラム上は最大値でクリップなど）
        safe_sigma = max(sigma, self.EPSILON)

        # 3. Quality (Q) - エネルギー生成
        # Q = (r * omega)^2 / sigma
        # つまり Q = G^2 / sigma
        Q = (G ** 2) / safe_sigma
        
        # 4. Motion (M) - 判定ゲート
        # M = H(Q - theta)
        M = self.heaviside(Q - theta)
        
        # 5. Coherence (C) - 共鳴
        C = self.calculate_coherence(tau, k)
        
        # 6. Depth (D) - 統合出力
        # M=1 の時のみ D = Q * C が観測される
        # M=0 の時は 0 (あるいは潜在値) とする
        raw_D = Q * C
        final_D = raw_D * M

        return {
            "G": G,
            "tau": tau,
            "Q": Q,
            "M": M,
            "C": C,
            "D": final_D,
            "status": "MOTION ACTIVATED" if M == 1 else "STILLNESS"
        }

# --- 実行テスト (Simulation) ---

if __name__ == "__main__":
    engine = KQIEngine()

    print("--- KQI System Simulation Start ---")
    
    # シナリオ: だんだん動きが激しくなり、ゾーン(k=inf)に入る
    test_scenarios = [
        # r, omega, sigma, t, rho, k, theta
        (1.0,  0.5,  5.0,  1.0, 0.1, 0,     10.0), # 1. 始動前 (Qが低い、sigma高い)
        (2.0,  1.0,  2.0,  2.0, 0.5, 0,     10.0), # 2. 徐々に加速 (まだsigmaある)
        (5.0,  3.0,  0.5,  3.0, 1.0, 1,     10.0), # 3. ブレイクスルー (Q急上昇、k=1モード)
        (5.0,  3.0,  0.1,  4.0, 2.0, 'inf', 10.0), # 4. ゾーン突入 (sigma極小、k=inf、共鳴爆発)
    ]

    for i, inputs in enumerate(test_scenarios):
        r, omega, sigma, t, rho, k, theta = inputs
        result = engine.process_frame(r, omega, sigma, t, rho, k, theta)
        
        print(f"\n[Step {i+1}] Input: r={r}, w={omega}, σ={sigma}, k={k}")
        print(f"  -> Q (Energy): {result['Q']:.4f}")
        print(f"  -> M (Gate)  : {result['M']} ... {result['status']}")
        print(f"  -> C (Field) : {result['C']:.4f}")
        print(f"  -> D (Depth) : {result['D']:.4f}")
        print(f"  -> D (Depth) : {result['D']:.4f}")