# permit_absorption_demo.py
# --------------------------------------------
# 「許可(P)」「全被り(A)」の2ノブを使って、
# 攻め/回収/連鎖失敗のダイナミクスを最小シミュレーション。
# あなたの ResponsibilityAllow_acts.py の “心理ゆらぎ”関数をそのまま利用します。
#
# 使い方:
#   1) このファイルと ResponsibilityAllow_acts.py を同じフォルダに置く
#   2) python permit_absorption_demo.py
# --------------------------------------------

import importlib.util
import math
import random
import sys

import numpy as np

# ========== ユーザーのモジュールをロード ==========
# 同じフォルダに ResponsibilityAllow_acts.py を置いてね
mod_path = "ResponsibilityAllow_acts.py"
spec = importlib.util.spec_from_file_location("respacts", mod_path)
respacts = importlib.util.module_from_spec(spec)
sys.modules["respacts"] = respacts
spec.loader.exec_module(respacts)

# ========== ユーティリティ ==========
def sigmoid(x): 
    return 1.0/(1.0+math.exp(-x))

def softmax(x, temp=1.0):
    x = np.array(x, dtype=float)
    x = x / max(1e-8, temp)  # 温度でスコアをスケーリング
    x -= x.max()             # 安定化
    ex = np.exp(x)
    return ex / ex.sum()

# ========== P/A コントローラ ==========
class PermitAbsorptionController:
    """
    P: 許可度（0〜1）。攻め/探索をどれだけ解禁するか。
    A: 全被り（吸収）パワー。大きいほど失敗ペナルティが軽く見える＆温度上昇。
    """
    def __init__(self, alpha0=-0.2, alpha1=0.8, alpha2=0.8, alpha3=1.0, 
                 gamma=1.0, base_T=1.0, P0=0.3):
        # Pの更新式: sigmoid(alpha0 + alpha1*A_pool + alpha2*Recovery - alpha3*Cascade)
        self.alpha0, self.alpha1, self.alpha2, self.alpha3 = alpha0, alpha1, alpha2, alpha3
        # Aの効き: 罰則軽減/温度上げ の強さ
        self.gamma = gamma
        self.base_T = base_T
        self.P = P0

    def update_P(self, A_pool, recovery_ratio, cascade_ratio):
        z = self.alpha0 + self.alpha1*A_pool + self.alpha2*recovery_ratio - self.alpha3*cascade_ratio
        self.P = sigmoid(z)
        return self.P

    def effective_temp(self, A_pool):
        # Aが大きいほど実質温度が上がって（=分母が小さくなって）探索しやすくなる
        return self.base_T / (1.0 + self.gamma * A_pool)

    def effective_lambda(self, lam, A_pool):
        # Aが大きいほど失敗ペナルティ重みが下がる
        return lam / (1.0 + self.gamma * A_pool)

# ========== 抽象タスク定義 ==========
# 3つの戦略: safe / mid / attack
ACTIONS = [
    {"name": "safe",   "G": 0.6, "C": 0.1, "nov": 0.00},  # 実利は小さいが安全
    {"name": "mid",    "G": 0.8, "C": 0.4, "nov": 0.40},  # 中庸
    {"name": "attack", "G": 1.2, "C": 0.8, "nov": 1.00},  # 攻め: 新規性が高いがコスト高
]

def step_once(ctrl, lambdas, A_values, rng):
    """
    1ステップ: 全員が同時に行動を選び、成功/失敗を観測。
    - ctrl.P が大きいほど novelty（探索）にボーナス
    - A_pool が大きいほど温度↑・罰則重み↓
    """
    N = len(lambdas)
    A_pool = sum(A_values) / max(1, N)  # 平均A（簡易 proxy）
    temp = ctrl.effective_temp(A_pool)

    chosen = []
    for i in range(N):
        lam_eff = ctrl.effective_lambda(lambdas[i], A_pool)
        scores = []
        for a in ACTIONS:
            # 許可Pが高いと新規性ボーナスを上乗せ
            novelty_gain = ctrl.P * a["nov"]
            raw_util = a["G"] + novelty_gain - lam_eff * a["C"]

            # --- 心理ゆらぎ（あなたのモジュールを使用） ---
            # util → 確率空間(sigmoid) → ロジット空間でノイズ → 逆写像してスコア化
            p = sigmoid(raw_util)
            p_hat, _ = respacts.apply_psych_fluctuation(p, mode="logit_gauss", sigma=0.15)
            # logit(p) = ln(p/(1-p))
            perturbed = math.log(max(1e-12, p_hat) / max(1e-12, 1.0 - p_hat))
            scores.append(perturbed)

        probs = softmax(scores, temp=temp)
        idx = int(np.random.choice(len(ACTIONS), p=probs))
        chosen.append(idx)

    # 成否のサンプリング（ここではコストCが高いほど失敗しやすい仮定）
    outcomes = []
    for idx in chosen:
        a = ACTIONS[idx]
        fail_p = 0.15 + 0.7*a["C"]  # だいたい [0.15, 0.71] くらい
        ok = rng.random() > fail_p
        outcomes.append(ok)

    return {"A_pool": A_pool, "actions": chosen, "success": outcomes, "temp": temp}

def simulate(ctrl, N=6, T=300, high_absorber=False, seed=42):
    """
    N人・Tステップのシミュレーション。
    high_absorber=True だと、1人だけ A が高い“全被り役”を置く。
    返り値: 攻め率/回収率/連鎖失敗率/最終P
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    lambdas = [1.0 for _ in range(N)]      # 個人の失敗ペナルティ感度
    A_values = [0.0 for _ in range(N)]     # 個人A
    if high_absorber:
        A_values[0] = 2.5                  # 0番を“全被り”に

    last_fail_any = False
    recovery_count = 0
    recovery_opps  = 0
    cascade_count  = 0
    attack_count   = 0

    for t in range(T):
        res = step_once(ctrl, lambdas, A_values, rng)
        # KPI: 攻め率
        attack_count += sum(1 for idx in res["actions"] if ACTIONS[idx]["name"] == "attack")

        # 失敗が場に一人でもあったか
        fail_any = not all(res["success"])

        # 直前に失敗があれば → 今回失敗ゼロになっていれば回復 (=回収)
        if last_fail_any:
            recovery_opps += 1
            if not fail_any:
                recovery_count += 1

        # 失敗が連続すれば連鎖
        if last_fail_any and fail_any:
            cascade_count += 1

        last_fail_any = fail_any

        # 観測された回収率/連鎖率を用いて P を更新
        rec_ratio = recovery_count / max(1, recovery_opps)
        cas_ratio = cascade_count / max(1, T)
        ctrl.update_P(res["A_pool"], rec_ratio, cas_ratio)

    atk_rate = attack_count / (T * N)
    rec_rate = recovery_count / max(1, recovery_opps)
    cas_rate = cascade_count / max(1, T)

    return {
        "attack_rate": atk_rate,
        "recovery_rate": rec_rate,
        "cascade_rate": cas_rate,
        "final_P": ctrl.P,
    }

def run_two_conditions():
    out = {}
    # 条件A：全被りなし
    ctrlA = PermitAbsorptionController(alpha0=-0.2, alpha1=0.8, alpha2=0.9, alpha3=1.2, gamma=1.0, base_T=1.0)
    out["no_absorber"] = simulate(ctrlA, N=6, T=300, high_absorber=False, seed=42)

    # 条件B：全被りあり
    ctrlB = PermitAbsorptionController(alpha0=-0.2, alpha1=0.8, alpha2=0.9, alpha3=1.2, gamma=1.0, base_T=1.0)
    out["with_absorber"] = simulate(ctrlB, N=6, T=300, high_absorber=True, seed=42)

    return out

if __name__ == "__main__":
    results = run_two_conditions()
    print("=== Results ===")
    for k, v in results.items():
        print(k, ":", v)
