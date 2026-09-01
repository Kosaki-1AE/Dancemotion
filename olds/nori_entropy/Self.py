# -*- coding: utf-8 -*-
# Nori-SLM Minimal Core (Awareness × LIFO-Unconscious × Adaptive Entropy)
# 依存: pip install torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 1) 小型Self-Attention（ボルツマン視点） ==========
class TinyAttn(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, T=1.0, causal=True):
        # x: [B, L, D]
        Q, K, V = self.q(x), self.k(x), self.v(x)
        D = Q.size(-1)
        scores = (Q @ K.transpose(-1, -2)) / (D ** 0.5)
        if causal:
            L = x.size(1)
            mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores / T, dim=-1)     # ← Boltzmann重み
        out = attn @ V
        return self.proj(out), attn, (Q, K, V)

# ========== 2) 無意識スタック（LIFOでKV保持） ==========
class UnconsciousStack:
    """
    無意識 = 選択されなかった情報の短期保管庫（LIFO）。
    ・push: K,Vを積む（古いものは破棄）
    ・pop_similar: 現在の関心q_vecに近いものだけ想起（LIFO優先）
    """
    def __init__(self, max_items: int = 32):
        self.max_items = max_items
        self.stack = []  # list of {'K': [B,l,D], 'V':[B,l,D]}

    def push(self, K: torch.Tensor, V: torch.Tensor):
        with torch.no_grad():
            self.stack.append({'K': K.detach(), 'V': V.detach()})
            if len(self.stack) > self.max_items:
                self.stack.pop(0)  # 最古を捨てる（LIFOキープ）

    def pop_similar(self, q_vec: torch.Tensor, top: int = 1, sim_thresh: float = 0.6):
        """
        q_vec: [B, D]（例: 末尾トークンのQの代表）
        return: (K_u, V_u) or None
        """
        if not self.stack:
            return None
        # 新しい順に類似度評価（LIFO）
        cand = []
        for i, kv in enumerate(reversed(self.stack)):
            # 代表キー（平均）で簡略
            K_rep = kv['K'].mean(dim=1)  # [B, D]
            s = F.cosine_similarity(q_vec, K_rep, dim=-1).mean()  # バッチ平均
            cand.append((s.item(), len(self.stack) - 1 - i))
        cand.sort(reverse=True, key=lambda x: x[0])
        idxs = [idx for s, idx in cand[:top] if s >= sim_thresh]
        if not idxs:
            return None
        idx = idxs[0]
        kv = self.stack.pop(idx)
        return kv['K'], kv['V']

# ========== 3) Adaptive制御（シャノンで温度Tとゲート閾値を調整） ==========
class AdaptiveController(nn.Module):
    """
    注意分布のシャノンHで“俯瞰/集中”を推定し、温度Tと想起ゲート閾値を調整。
    Awareness(A)を手動指定する場合は awareness_control() を利用。
    """
    def __init__(self, T_min=0.6, T_max=1.6, sim_lo=0.5, sim_hi=0.9):
        super().__init__()
        self.T_min, self.T_max = T_min, T_max
        self.sim_lo, self.sim_hi = sim_lo, sim_hi

    @staticmethod
    def shannon_entropy(prob: torch.Tensor, eps=1e-9):
        p = prob.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)  # [..., L]→[...]

    def forward(self, attn_last: torch.Tensor):
        """
        attn_last: [B, L]（末尾トークンの注意分布）
        return: (T, sim_thresh)
        """
        H = self.shannon_entropy(attn_last)                   # [B]
        Hn = (H - H.min()) / (H.max() - H.min() + 1e-9)      # 0-1正規化
        # 設計選択：拡散(Hn↑)→俯瞰→T↑ / 集中(Hn↓)→T↓
        T = (self.T_min + (self.T_max - self.T_min) * Hn.mean()).detach().cpu().item()
        # 拡散時は似てなくても想起しやすい（閾値↓） or 逆、は好みで
        sim_thresh = (self.sim_hi - (self.sim_hi - self.sim_lo) * Hn.mean()).detach().cpu().item()
        return T, sim_thresh


# （任意）AスライダーでT/τを直制御したい場合のヘルパ
def awareness_control(H01: float = None, A_manual: float = None,
                      T_min=0.6, T_max=1.6, tau_min=0.01, tau_max=0.10):
    """
    H01: [0,1] 正規化済みH（未指定ならA_manualのみで決定）
    A_manual: [0,1]（1=超集中, 0=超俯瞰）
    return: (A, T, tau)
    """
    if A_manual is None:
        if H01 is None:
            raise ValueError("Either H01 or A_manual must be provided.")
        A = 1.0 - float(H01)  # Hが大きい=拡散→A低（俯瞰）
    else:
        A = float(A_manual)
    T = T_min + (T_max - T_min) * (1 - A)
    tau = tau_max - (tau_max - tau_min) * (1 - A)
    return A, T, tau

# ========== 4) 全体セル：選択⇆俯瞰＋LIFO想起（MVP） ==========
class NoriSLMCell(nn.Module):
    """
    ・Attentionで“いま”の選択（意識）
    ・下位/未選択を無意識スタックへ（LIFO）
    ・シャノンHで温度/閾値を適応
    ・必要時のみ無意識から想起し再アテンション（Adaptive往復）
    """
    def __init__(self, d_model: int, stack_max: int = 32):
        super().__init__()
        self.attn = TinyAttn(d_model)
        self.adapt = AdaptiveController()
        self.stack = UnconsciousStack(max_items=stack_max)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model),
                                nn.GELU(),
                                nn.Linear(4*d_model, d_model))
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, awareness_A: float = None):
        """
        x: [B, L, D]
        awareness_A: 手動A（0=俯瞰,1=集中）。Noneなら自動（Hで決める）
        """
        # 1) 通常アテンション（俯瞰→選択）
        out, attn, (Q, K, V) = self.attn(self.ln(x))
        ctx = x + out
        ctx = ctx + self.ff(self.ln(ctx))  # 方向性（“傾き”）

        # 2) 末尾注意分布を取得（H算出＆スタックpush用の簡略化）
        last_attn = attn[:, -1]  # [B, L]
        # 無意識へpush（ここでは簡略に“全部”を保存。必要なら下位onlyに変更）
        self.stack.push(K, V)

        # 3) 自動 or 手動の適応制御
        if awareness_A is None:
            T, sim_th = self.adapt(last_attn)
        else:
            # 手動AからT/τを得る（τはここでは使わないが外部で利用可）
            _, T, _ = awareness_control(A_manual=awareness_A)
            # 手動A: 集中(A=1)なら想起しづらく、俯瞰(A=0)なら想起しやすい閾値に
            sim_lo, sim_hi = 0.5, 0.9
            sim_th = sim_hi - (sim_hi - sim_lo) * (1 - awareness_A)

        # 4) 無意識から必要時のみ想起（LIFO＋類似）
        q_vec = Q[:, -1].mean(dim=0, keepdim=True)  # [1, D] 簡略代表
        popped = self.stack.pop_similar(q_vec, top=1, sim_thresh=sim_th)
        if popped is not None:
            K_u, V_u = popped
            K_cat = torch.cat([K, K_u], dim=1)
            V_cat = torch.cat([V, V_u], dim=1)
            scores = (Q @ K_cat.transpose(-1, -2)) / (Q.size(-1) ** 0.5)
            attn2 = F.softmax(scores / T, dim=-1)  # Adaptive（温度適応）
            ctx = ctx + (attn2 @ V_cat)

        return ctx  # [B, L, D]

# ========== 5) 使い方（参考） ==========
if __name__ == "__main__":
    B, L, D = 1, 64, 256
    x = torch.randn(B, L, D)
    cell = NoriSLMCell(D, stack_max=16)

    # 自動（注意HでT/閾値が回る）
    for _ in range(2):
        x = cell(x)

    # 手動（意識レベルAを直接指定：0=俯瞰, 1=集中）
    x = cell(x, awareness_A=0.2)  # 俯瞰寄せ（ノイズも拾いやすい）
    x = cell(x, awareness_A=0.9)  # 集中寄せ（芯だけに）
    # ここから線形+softmaxで次トークン分布などに接続すればミニ言語モデル化可能
    print(x.shape, x.dtype, x.device)
    print(x.requires_grad)         # True なら勾配追跡中
    y = x.sum()
    y.backward()                   # 例
    print("leaf param grad example:", some_param.grad is not None)
