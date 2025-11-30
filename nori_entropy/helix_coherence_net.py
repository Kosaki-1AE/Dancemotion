# helix_coherence_net.py
# -*- coding: utf-8 -*-
import math, random, time, json, os
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 1) ユーティリティ
# =========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); 
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def cosine_warmup(step, total, minv=1e-3):
    x = step / max(1,total)
    return minv + 0.5*(1-minv)*(1+math.cos(math.pi*x))

# =========================
# 2) ヘリカル埋め込み
# =========================
class HelixPE(nn.Module):
    """
    Helical Position Encoding:
      h(t) = [cos(omega*t), sin(omega*t), alpha*t, t/T, (t%beat)/beat]
    拍情報があれば渡して上書き可能。
    """
    def __init__(self, d_out=32, omega=2.0, alpha=0.01):
        super().__init__()
        self.omega = omega
        self.alpha = alpha
        self.proj = nn.Linear(5, d_out)
    def forward(self, T:int, beat:Optional[float]=None, bpm:Optional[float]=None):
        t = torch.arange(T, dtype=torch.float32, device=DEVICE)
        cos_sin = torch.stack([torch.cos(self.omega*t), torch.sin(self.omega*t)], dim=-1)  # [T,2]
        ascend  = self.alpha*t[...,None]                                                   # [T,1]
        tnorm   = (t / max(1,T-1))[...,None]                                              # [T,1]
        if beat is None: beat = 4.0
        beat_phase = ((t % beat)/beat)[...,None]                                          # [T,1]
        base = torch.cat([cos_sin, ascend, tnorm, beat_phase], dim=-1)                    # [T,5]
        return self.proj(base)                                                            # [T,d_out]

# =========================
# 3) 二重分岐エンコーダ + キラル結合
# =========================
class ChiralCrossAttn(nn.Module):
    """
    右巻/左巻でキー/バリューを線形変換し分岐間アテンションを結合
    """
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.r2e = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.e2r = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.Wcw  = nn.Linear(d_model, d_model)  # s=+1
        self.Wccw = nn.Linear(d_model, d_model)  # s=-1
        self.lmbd = nn.Parameter(torch.tensor(0.5))
        self.normR = nn.LayerNorm(d_model)
        self.normE = nn.LayerNorm(d_model)
    def forward(self, HR, HE, s_sign: torch.Tensor):
        # s_sign: shape [B] with +1/-1 (float)
        B,T,D = HR.shape
        # expand transform per batch sign
        W = self.Wcw if (s_sign.mean().item()>=0) else self.Wccw
        HEc = W(HE)
        HRc = W(HR)
        # e2r
        R_add,_ = self.e2r(HR, HEc, HEc)
        E_add,_ = self.r2e(HE, HRc, HRc)
        HRp = self.normR(HR + self.lmbd*R_add)
        HEp = self.normE(HE + self.lmbd*E_add)
        return HRp, HEp

class DualBranchEncoder(nn.Module):
    """
    R-Branch: 低周波/安定 (GRU/Transformer可)
    E-Branch: 高周波/揺らぎ
    """
    def __init__(self, d_inR, d_inE, d_model=256, nhead=4):
        super().__init__()
        self.r_in = nn.Linear(d_inR, d_model)
        self.e_in = nn.Linear(d_inE, d_model)
        self.r_enc = nn.GRU(d_model, d_model//2, num_layers=2, batch_first=True, bidirectional=True)
        self.e_enc = nn.GRU(d_model, d_model//2, num_layers=2, batch_first=True, bidirectional=True)
        self.projR = nn.Linear(d_model, d_model)
        self.projE = nn.Linear(d_model, d_model)
        self.cross = ChiralCrossAttn(d_model, nhead=nhead)
    def forward(self, xR, xE, s_sign):
        # xR,xE: [B,T,d_in*]
        HR,_ = self.r_enc(F.gelu(self.r_in(xR)))
        HE,_ = self.e_enc(F.gelu(self.e_in(xE)))
        HR, HE = self.cross(self.projR(HR), self.projE(HE), s_sign)
        return HR, HE

# =========================
# 4) コヒーレンス・レザバー（Z:Stillness, U:Fluctuation）
# =========================
class Reservoir(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.z = nn.GRU(d_model, d_model, batch_first=True)
        self.u = nn.GRU(d_model, d_model, batch_first=True)
        self.mix = nn.Parameter(torch.tensor(0.6))
        self.norm = nn.LayerNorm(d_model)
    def forward(self, H):  # H=[B,T,D]
        Z,_ = self.z(H); U,_ = self.u(H)
        Hf = self.mix*Z + (1-self.mix)*U
        return self.norm(Hf)

# =========================
# 5) ヘッド群
# =========================
class Heads(nn.Module):
    def __init__(self, d_model, n_classes, dim_r):
        super().__init__()
        self.head_y   = nn.Linear(d_model, n_classes)
        self.head_r   = nn.Linear(d_model, dim_r)
        self.head_sig = nn.Linear(d_model, 1)
        self.head_s   = nn.Linear(d_model, 2)  # 右巻/左巻
    def forward(self, Hf):
        return dict(
            y   = self.head_y(Hf),
            r   = self.head_r(Hf),
            sig = self.head_sig(Hf),
            s   = self.head_s(Hf),
        )

# =========================
# 6) 全体モデル
# =========================
class HelixCoherenceNet(nn.Module):
    def __init__(self, d_inR=128, d_inE=128, d_pe=32, d_model=256, n_classes=12, dim_r=3, nhead=4):
        super().__init__()
        self.pe = HelixPE(d_out=d_pe)
        self.enc = DualBranchEncoder(d_inR+d_pe, d_inE+d_pe, d_model=d_model, nhead=nhead)
        self.res = Reservoir(d_model*2)  # HR,HE を結合してから入れる
        self.heads = Heads(d_model*2, n_classes, dim_r)
    def forward(self, xR, xE, s_sign, beat=None, bpm=None):
        # xR,xE: [B,T,feat], s_sign: [B] (+1/-1 float)
        B,T,_ = xR.shape
        pe = self.pe(T, beat=beat, bpm=bpm).to(xR.device)              # [T,d_pe]
        pe = pe[None].expand(B,-1,-1)                                  # [B,T,d_pe]
        HR, HE = self.enc(torch.cat([xR,pe],dim=-1), torch.cat([xE,pe],dim=-1), s_sign)
        H = torch.cat([HR,HE], dim=-1)                                 # [B,T,2D]
        Hf = self.res(H)
        return self.heads(Hf)                                          # dict of logits

# =========================
# 7) 損失（Helix整合 & Nori-Entropy含む）
# =========================
def helix_consistency_loss(HR:torch.Tensor, HE:torch.Tensor, s_sign:torch.Tensor):
    # 右巻/左巻で位相が反転してるほど良し → 相関の符号を s_sign に合わせる
    # 簡略: 時間差分の内積平均の符号を合わせる
    dHR = HR[:,1:,:] - HR[:,:-1,:]
    dHE = HE[:,1:,:] - HE[:,:-1,:]
    cos = F.cosine_similarity(dHR, dHE, dim=-1).mean(dim=1)  # [B]
    target = s_sign  # +1 or -1
    return F.mse_loss(cos, target)

def nori_entropy_regularizer(logits:torch.Tensor, low=0.15, high=0.85):
    # シャノンエントロピーが低すぎ(過信)・高すぎ(無統制)を罰する
    p = F.softmax(logits, dim=-1) + 1e-8
    H = -(p*torch.log(p)).sum(dim=-1).mean()
    # 目標エントロピー範囲に近づける（簡易正則化）
    # 範囲中心を mid に設定
    mid = -(low*math.log(low)+(1-low)*math.log(1-low))
    return (H - mid).abs()

# =========================
# 8) 後出しジャンケン推論（Late Binding）
# =========================
def temperature_by_coherence(delta_t, Tc, tau0=1.5, k=1.0):
    # Δt/Tc が大きいほど（待てるほど）温度を下げてシャープ化
    x = max(0.0, min(2.0, float(delta_t)/max(1e-6,Tc)))
    return tau0*math.exp(-k*x)

@torch.no_grad()
def late_binding_decode(model, xR, xE, s_sign, L=6, Tc=8, beat=None, bpm=None):
    """
    xR,xE: [1,T,feat] ストリーム入力。L: 後出し許容量(フレーム)
    逐次でTフレーム出すが、確定は t-L 時点。
    """
    model.eval()
    B, T, _ = xR.shape
    outs = []
    for t in range(T):
        # ミニバッファで先読み
        t0 = max(0, t-L)
        xbR = xR[:, t0:t+1, :]; xbE = xE[:, t0:t+1, :]
        logits = model(xbR.to(DEVICE), xbE.to(DEVICE), s_sign.to(DEVICE), beat=beat, bpm=bpm)["y"]  # [1, len, C]
        # 確定対象は先頭フレーム
        delta_t = min(L, t+1)  # いま利用できた先読み
        tau = temperature_by_coherence(delta_t, Tc)
        p = F.softmax(logits[:,0,:]/tau, dim=-1)
        outs.append(p.cpu())
    return torch.stack(outs, dim=1)  # [1,T,C] (確率列)

# =========================
# 9) ダミーデータローダ（差し替え前提）
# =========================
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, N=128, T=64, dR=128, dE=128, n_classes=12):
        self.N=N; self.T=T; self.dR=dR; self.dE=dE; self.n_classes=n_classes
    def __len__(self): return self.N
    def __getitem__(self, i):
        xR = np.random.randn(self.T, self.dR).astype(np.float32)
        xE = np.random.randn(self.T, self.dE).astype(np.float32)
        y  = np.random.randint(0, self.n_classes, size=(self.T,), dtype=np.int64)
        r  = np.random.randn(self.T, 3).astype(np.float32)
        sig= np.clip(np.random.randn(self.T,1)*0.2+0.5, 0.0, 1.0).astype(np.float32)
        s  = np.random.choice([0,1])  # 右巻/左巻
        s_sign = np.array([1.0 if s==1 else -1.0], dtype=np.float32)
        return dict(
            xR=xR, xE=xE, y=y, r=r, sig=sig, s=s, s_sign=s_sign
        )

def collate(batch):
    # パディング無しの同一長前提スケルトン
    keys = batch[0].keys()
    out = {}
    for k in keys:
        arr = [torch.from_numpy(v[k]) if isinstance(v[k], np.ndarray) else (torch.tensor(v[k]) if not torch.is_tensor(v[k]) else v[k]) for v in batch]
        out[k] = torch.stack(arr, dim=0)
    return out

# =========================
# 10) 学習ループ
# =========================
def train_epoch(model, loader, opt, sched=None, lambda_dict=None):
    model.train()
    total = 0.0
    for step, batch in enumerate(loader):
        xR = batch["xR"].to(DEVICE); xE = batch["xE"].to(DEVICE)
        y  = batch["y"].to(DEVICE);  r  = batch["r"].to(DEVICE)
        sig= batch["sig"].to(DEVICE); s = batch["s"].to(DEVICE)        # [B,T,1] / [B]
        s_sign = batch["s_sign"].squeeze(1).to(DEVICE)                  # [B]
        out = model(xR, xE, s_sign)
        # ロス
        Ly = F.cross_entropy(out["y"].reshape(-1, out["y"].shape[-1]), y.reshape(-1))
        Lr = F.mse_loss(out["r"], r)
        Lsig = F.mse_loss(out["sig"], sig)
        Ls = F.cross_entropy(out["s"][:,-1,:], s.long())  # 終端でねじれ判定（簡略）
        # Helix整合: encoder内部特徴を再計算（軽量化のため一段前出力を使うならモデル修正が必要）
        # ここでは近似として y ロジットの時間差分を代用（実運用はEnc出力を返す設計にしてOK）
        dlog = out["y"][:,1:,:]-out["y"][:,:-1,:]
        # R/E差の近似代理（簡略）: クラス方向と責任方向の整合で符号合わせ
        # 実案件では Enc 出力 (HR,HE) を返して入れること推奨
        helix = ((dlog[:,:,:1]).mean(dim=(1,2))).clamp(-1,1)  # 疑似相関
        Lh = F.mse_loss(helix, s_sign)
        Ln = nori_entropy_regularizer(out["y"])
        lmb = dict(y=1.0, r=0.5, sig=0.2, s=0.3, h=0.1, n=0.05)
        if lambda_dict: lmb.update(lambda_dict)
        loss = lmb["y"]*Ly + lmb["r"]*Lr + lmb["sig"]*Lsig + lmb["s"]*Ls + lmb["h"]*Lh + lmb["n"]*Ln
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if sched: sched.step()
        total += loss.item()
    return total/ max(1, step+1)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total, n, acc = 0.0, 0, 0
    for batch in loader:
        xR = batch["xR"].to(DEVICE); xE = batch["xE"].to(DEVICE)
        y  = batch["y"].to(DEVICE)
        s_sign = batch["s_sign"].squeeze(1).to(DEVICE)
        out = model(xR,xE,s_sign)
        Ly = F.cross_entropy(out["y"].reshape(-1, out["y"].shape[-1]), y.reshape(-1))
        total += Ly.item()
        pred = out["y"].argmax(-1)
        acc += (pred==y).float().mean().item()
        n += 1
    return dict(loss=total/max(1,n), acc=acc/max(1,n))

# =========================
# 11) エントリポイント（テスト実行）
# =========================
def main():
    set_seed(7)
    # ハイパラ
    N_CLASSES=12; DIM_R=3; T=64; dR=128; dE=128
    model = HelixCoherenceNet(d_inR=dR, d_inE=dE, d_model=256, n_classes=N_CLASSES, dim_r=DIM_R).to(DEVICE)
    train_ds = DummyDataset(N=64, T=T, dR=dR, dE=dE, n_classes=N_CLASSES)
    val_ds   = DummyDataset(N=16, T=T, dR=dR, dE=dE, n_classes=N_CLASSES)
    train_ld = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
    val_ld   = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: cosine_warmup(step, total=200))
    for ep in range(3):
        tr = train_epoch(model, train_ld, opt, sched)
        ev = eval_epoch(model, val_ld)
        print(f"[ep{ep}] train={tr:.4f}  val={ev}")
    # 疑似ストリームの後出しデコード
    batch = next(iter(val_ld))
    xR = batch["xR"][:1].to(DEVICE); xE=batch["xE"][:1].to(DEVICE); s_sign=batch["s_sign"][:1,0].to(DEVICE)
    probs = late_binding_decode(model, xR, xE, s_sign, L=6, Tc=8)
    print("stream probs shape:", probs.shape)

if __name__=="__main__":
    main()
