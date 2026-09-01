# singular_module.py
import random


class SingularCalculator:
    def __init__(self, epsilon=1e-6, mode="both"):
        """
        epsilon : 特異点判定のしきい値
        mode    : "collapse", "create", "both" から選択
                collapse = 崩壊ルート
                create   = 創造ルート
                both     = ランダムで選ぶ
        """
        self.epsilon = epsilon
        self.mode = mode

    def compute(self, n, R, d):
        """
        f(x) = (n * R) / d を計算
        ただし d が 0 に近い場合は特異点処理を行う
        """
        if abs(d) < self.epsilon:
            return self.handle_singularity(n, R, d)
        else:
            return (n * R) / d

    def handle_singularity(self, n, R, d):
        """
        特異点処理
        Ωをどう扱うかを選択
        """
        if self.mode == "collapse":
            # 崩壊ルート → 強制的に0に潰す
            return 0.0
        elif self.mode == "create":
            # 創造ルート → 新しい次元のベクトルを生む（ここでは乱数で代用）
            return random.uniform(-1, 1) * (n + R)
        elif self.mode == "both":
            # ランダムに崩壊/創造を選ぶ
            if random.random() < 0.5:
                return 0.0
            else:
                return random.uniform(-1, 1) * (n + R)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
