# -*- coding: utf-8 -*-
# chat_flow.py  —  自走（勝手に喋る）＋ ユーザー入力に即応するフローボット
import queue
import sys
import threading
import time

import numpy as np


# ====== 基本活性と“負版” ======
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def silu(x):    return x * sigmoid(x)
def negify(fn): return lambda x: -fn(-x)
neg_silu = negify(silu)

# ====== 心理ゆらぎ ======
def p_fluct(p, sigma=0.40):
    # logit空間ノイズ
    p = float(np.clip(p, 1e-12, 1-1e-12))
    z = np.log(p/(1-p)) + np.random.normal(0, sigma)
    return 1.0/(1.0+np.exp(-z))

# ====== 中核：寄与→Δ→p_hat ======
def analyze_activation(x, W, b, tau=1.0):
    z = x @ W + b
    pos = silu(z); neg = neg_silu(z)
    # ReLU的な分解
    pos_part = np.maximum(0.0, pos)
    neg_strength = np.maximum(0.0, -np.minimum(0.0, neg))
    pos_sum = float(pos_part.sum()); neg_sum = float(neg_strength.sum())
    delta = pos_sum - neg_sum
    p = 1.0/(1.0+np.exp(-delta/max(tau,1e-6)))
    return dict(delta=delta, p=p, pos_sum=pos_sum, neg_sum=neg_sum)

def will_event(x, W, b, theta=0.60, tau=1.0, sigma=0.40):
    res = analyze_activation(x, W, b, tau=tau)
    p_hat = p_fluct(res["p"], sigma=sigma)
    commit = p_hat >= theta
    polarity = 1 if res["pos_sum"] >= res["neg_sum"] else -1
    return dict(commit=commit, p_hat=p_hat, polarity=polarity, delta=res["delta"])

# ====== “流れ”を学ぶ超軽量ヘッド ======
class HashEncoder:
    def __init__(self, dim=256, seed=0):
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.signs = rng.choice([-1.0, 1.0], size=dim)
    def vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.split():
            h = abs(hash(tok)) % self.dim
            v[h] += self.signs[h]
        n = np.linalg.norm(v) or 1.0
        return v / n

class FlowHead:
    def __init__(self, in_dim=256, lr=0.2, seed=1):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.1, size=(in_dim, 2))
        self.lr = lr
    def predict(self, m):
        y = m @ self.W
        n = np.linalg.norm(y) or 1.0
        return y / n   # 2D
    def update(self, m, target_dir, pred_dir, gain=1.0):
        g = (target_dir - pred_dir) * gain
        self.W += self.lr * np.outer(m, g)

class FlowState:
    def __init__(self, dim=256, theta0=0.60, ema=0.05):
        self.enc = HashEncoder(dim=dim)
        self.head = FlowHead(in_dim=dim)
        self.last_dir = np.zeros(2, dtype=np.float32)
        self.theta = theta0
        self.theta_ema = ema
    def step(self, msg, base_push, decide_fn):
        m = self.enc.vec(msg)
        d_hat = self.head.predict(m)
        extra = 0.4 * float(d_hat @ (self.last_dir if np.any(self.last_dir) else d_hat))
        out = decide_fn(base_push + extra)  # returns commit, p_hat, polarity
        if out["commit"]:
            target = (1 if out["polarity"]>=0 else -1) * d_hat
            gain = 1.0 + 0.5 * abs(out["p_hat"] - self.theta)
            self.last_dir = 0.8*self.last_dir + 0.2*d_hat
        else:
            target = 0.5 * (self.last_dir if np.any(self.last_dir) else d_hat)
            gain = 0.3
            self.last_dir = 0.95*self.last_dir
        self.head.update(m, target, d_hat, gain=gain)
        # 発生率を目標(0.55)に寄せる θ 自動調整
        goal = 0.55
        err = (1.0 if out["commit"] else 0.0) - goal
        self.theta = float(np.clip(self.theta + (-self.theta_ema)*err, 0.4, 0.8))
        out.update(theta=self.theta, d_hat=d_hat.tolist())
        return out

# ====== “基盤状態”：x, W, b を持つ ======
class Core:
    def __init__(self, dim=6, seed=0):
        rng = np.random.default_rng(seed)
        self.D = dim
        self.W = rng.normal(0, 0.8, size=(dim, dim))
        self.b = rng.normal(0, 0.2, size=(dim,))
        self.rng = rng
    def stim(self):  # ランダム刺激
        return self.rng.normal(0, 1, size=(self.D,))

# ====== メッセージ生成（簡易） ======
def gen_reply(polarity, p_hat):
    # polarity=+1: 前進/肯定、-1: 保留/否定、p_hatが高いほど断定口調
    strong = p_hat >= 0.75
    if polarity >= 0:
        return "よし、いこう。次は小さく実験してみよ。" if strong else "うん、やってみる方向で。軽めに動くね。"
    else:
        return "今はためよ。材料もう少し集める。" if strong else "一旦キープ。検討用の問いを増やす。"

CANDIDATES = [
    "Stillnessの臨界はどこ？",
    "責任の矢をどう立てる？",
    "反例で揺らして確かめたい。",
    "小さく一歩踏み出す案を出す。",
    "俯瞰して目的を再定義する？",
]

# ====== ここから CLI ループ ======
class ChatLoop:
    def __init__(self):
        self.core = Core(dim=6, seed=0)
        self.flow = FlowState(theta0=0.60, ema=0.05)
        self.user_q = queue.Queue()
        self.stop = threading.Event()
        self.parrot_echo = True   # ★ 入力を即オウム返しする
        self.base_push = 0.0
        # “基盤”デコイダー（x は都度ランダム刺激）
        def make_decider(x):
            def _dec(delta_extra=0.0):
                r = will_event(x, self.core.W, self.core.b, theta=self.flow.theta,
                               tau=1.0, sigma=0.40)
                # extra を Δ に足したとして p を再計算（簡易）
                delta_mod = r["delta"] + delta_extra
                p_hat = 1.0/(1.0 + np.exp(-delta_mod))
                commit = p_hat >= self.flow.theta
                return dict(commit=commit, p_hat=p_hat, polarity=r["polarity"])
            return _dec
        self.make_decider = make_decider

    def print_bot(self, text):
        sys.stdout.write(f"\n🤖 Bot: {text}\n> "); sys.stdout.flush()

    def print_sys(self, text):
        sys.stdout.write(f"\n[sys] {text}\n> "); sys.stdout.flush()

    def agent_thread(self):
        # 勝手に喋る（一定間隔）。ユーザー入力が来たらそっち優先。
        while not self.stop.is_set():
            try:
                # 0.3秒ごとにユーザー入力チェック（あれば即返す）
                msg = self.user_q.get(timeout=0.3)
                # --- ユーザー発言に反応 ---
                x = self.core.stim()
                out = self.flow.step(msg, base_push=self.base_push,
                                     decide_fn=self.make_decider(x))
                # ★ parrot モード中は“即時オウム返し”が出てるので、
                #    ここでは学習・内部状態更新だけにして追加発話はしない
                if not self.parrot_echo:
                    reply = gen_reply(out["polarity"], out["p_hat"])
                    self.print_bot(reply + f"  (p={out['p_hat']:.2f}, θ={out['theta']:.2f})")
                self.base_push = 0.2*self.base_push + (0.1 if out["commit"] else -0.02)
                reply = gen_reply(out["polarity"], out["p_hat"])
                self.base_push = 0.2*self.base_push + (0.1 if out["commit"] else -0.02)
                self.print_bot(reply + f"  (p={out['p_hat']:.2f}, θ={out['theta']:.2f})")
            except queue.Empty:
                # 入力が無ければ、自走でひと言
                x = self.core.stim()
                probe = np.random.choice(CANDIDATES)
                out = self.flow.step(probe, base_push=self.base_push,
                                     decide_fn=self.make_decider(x))
                if out["commit"]:
                    reply = gen_reply(out["polarity"], out["p_hat"])
                    self.base_push = 0.2*self.base_push + 0.08
                    self.print_bot(reply + f"  (self, p={out['p_hat']:.2f})")
                else:
                    # Stillness維持：たまに思索メモだけ落とす
                    if np.random.rand() < 0.25:
                        self.print_sys("…内省中（材料待ち）")
                # 自走のテンポ
                time.sleep(1.8)

    def run(self):
        self.print_sys("こちらは勝手に喋ってますんでご自由に話し始めてね(試作品に付き内容噛み合わんかもしれんがご了承。)")
        sys.stdout.write("> "); sys.stdout.flush()
        t = threading.Thread(target=self.agent_thread, daemon=True)
        t.start()
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                msg = line.strip()
                if msg == "":
                    continue
                if msg.lower() in {"exit", "quit"}:
                    self.stop.set(); break
                # ★ オウム返しを即時表示（ここが“瞬間”）
                if self.parrot_echo:
                    self.print_bot(f"「{msg}」")   # ← ここで即返す

                # 学習や次の応答のためにキューへ（agent_thread 側で処理）
                self.user_q.put(msg)
                self.user_q.put(msg)
        except KeyboardInterrupt:
            self.stop.set()
        t.join(timeout=1.0)
        self.print_sys("終了。")
        
if __name__ == "__main__":
    ChatLoop().run()
