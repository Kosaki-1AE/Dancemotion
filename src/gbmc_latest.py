import random
import math
import time
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

# ============================================================
# 0) 前提: Minimal Buffer Implementation
# ============================================================
@dataclass
class Item:
    payload: Any
    tags: List[str]
    arousal: float
    novelty: float
    confidence: float
    timestamp: float = field(default_factory=time.time)

    def as_dict(self):
        return {
            "payload": self.payload,
            "tags": self.tags,
            "metrics": {"arousal": self.arousal, "novelty": self.novelty, "conf": self.confidence},
            "ts": self.timestamp
        }

class Buffer:
    def __init__(self, cap: int = 64):
        self.cap = cap
        self.data: List[Item] = []

    def push(self, payload, tags=None, arousal=0.0, novelty=0.0, confidence=0.5):
        it = Item(payload, tags or [], arousal, novelty, confidence)
        self.data.append(it)
        if len(self.data) > self.cap:
            self.data.pop(0)
        return it

    def view(self, n=10):
        return self.data[-n:]

# ============================================================
# 1) KQI Engine (The Logic Core) + Summer Vacation Patch
# ============================================================
class KQIEngine:
    """GSMC理論に基づく計算エンジン (夏休み対応版)"""
    def __init__(self):
        self.EPSILON = 1e-9

    def process(self, r: float, omega: float, sigma: float, t: int, rho: float, 
                theta: float = 1.0, k_mode='inf', accumulated_cost: float = 0.0):
        
        # ---【Summer Vacation Protocol】---
        # コスト(努力値)が閾値を超えている場合、物理法則を書き換える
        # 本来なら sigma(ノイズ/抵抗) が高い場面でも、熟練度により「抵抗なし」として処理する
        
        is_summer_vacation = False
        vacation_factor = 1.0
        
        # コスト閾値設定 (適当に100としておく、主様の裁量で調整可)
        COST_THRESHOLD = 100.0
        
        if accumulated_cost > COST_THRESHOLD:
            is_summer_vacation = True
            # トンネル効果発動：抵抗(Sigma)を強制的に最小化する
            # コストが高ければ高いほど、より深い抵抗も無視できるイメージ
            vacation_factor = 0.01 # 抵抗を1/100にするチート
        
        # --------------------------------

        # 1. Genesys
        G = r * omega

        # 2. Base (Sigma check with Vacation Logic)
        # 夏休みモードなら sigma に vacation_factor が掛かり、抵抗が消滅する
        effective_sigma = max(sigma * vacation_factor, 0.1)

        # 3. Quality (構造的品質)
        # 抵抗が下がれば Quality は爆発的に上がる
        Q = (G ** 2) / effective_sigma

        # 4. Motion (Gate)
        M = 1.0 if Q >= theta else 0.0

        # 5. Coherence
        tau = t * rho
        if k_mode == 'inf':
            C = math.exp(min(tau * 0.1, 10.0))
        elif k_mode == 1:
            C = 1.0 + tau
        else:
            C = 1.0

        # 6. Depth
        D = Q * C * M

        return {
            "G": G, "Q": Q, "M": M, "C": C, "D": D, 
            "sigma_raw": sigma, 
            "sigma_effective": effective_sigma,
            "is_vacation": is_summer_vacation
        }

# ============================================================
# 2) MineLife (The Body / Eco-system)
# ============================================================
@dataclass
class MineLifeConfig:
    w: int = 16
    h: int = 16
    mine_ratio: float = 0.12
    birth: Tuple[int, ...] = (3,)
    survive: Tuple[int, ...] = (2, 3)
    reveal_per_tick: int = 2
    danger_spread: float = 0.35
    seed: Optional[int] = None

class MineLife:
    def __init__(self, cfg: MineLifeConfig):
        self.cfg = cfg
        if cfg.seed is not None: random.seed(cfg.seed)
        self.t = 0
        self.alive = [[1 if random.random() < 0.25 else 0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.mine = [[1 if random.random() < cfg.mine_ratio else 0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.revealed = [[0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.risk = [[0.0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self._update_risk()

    def _n8(self, y, x):
        out = []
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==0 and dx==0: continue
                ny, nx = y+dy, x+dx
                if 0<=ny<self.cfg.h and 0<=nx<self.cfg.w: out.append((ny, nx))
        return out

    def _count(self, grid, y, x):
        return sum(grid[ny][nx] for (ny, nx) in self._n8(y, x))

    def _update_risk(self):
        base = [[min(1.0, self._count(self.mine, y, x)/8.0) for x in range(self.cfg.w)] for y in range(self.cfg.h)]
        new_r = [[0.0]*self.cfg.w for _ in range(self.cfg.h)]
        for y in range(self.cfg.h):
            for x in range(self.cfg.w):
                neigh = self._n8(y, x)
                avg = sum(base[ny][nx] for ny, nx in neigh)/max(1, len(neigh))
                new_r[y][x] = base[y][x]*(1.0-self.cfg.danger_spread) + avg*self.cfg.danger_spread
        self.risk = new_r

    def _life_step(self):
        nxt = [[0]*self.cfg.w for _ in range(self.cfg.h)]
        for y in range(self.cfg.h):
            for x in range(self.cfg.w):
                n = self._count(self.alive, y, x)
                state = self.alive[y][x]
                nxt[y][x] = 1 if (state==1 and n in self.cfg.survive) or (state==0 and n in self.cfg.birth) else 0
        self.alive = nxt

    def _auto_reveal(self):
        cand = [(y, x) for y in range(self.cfg.h) for x in range(self.cfg.w) if self.revealed[y][x] == 0]
        if not cand: return []
        cand.sort(key=lambda p: self.risk[p[0]][p[1]]) 
        picks = []
        for _ in range(min(self.cfg.reveal_per_tick, len(cand))):
            p = cand.pop(-1) if random.random() > 0.7 else cand.pop(0)
            self.revealed[p[0]][p[1]] = 1
            picks.append((p[0], p[1], self.mine[p[0]][p[1]], self._count(self.mine, p[0], p[1])))
        return picks

    def tick(self) -> Dict[str, Any]:
        self.t += 1
        self._life_step()
        self._update_risk()
        reveals = self._auto_reveal()

        alive_ratio = sum(sum(r) for r in self.alive) / (self.cfg.w * self.cfg.h)
        revealed_ratio = sum(sum(r) for r in self.revealed) / (self.cfg.w * self.cfg.h)
        risk_mean = sum(sum(r) for r in self.risk) / (self.cfg.w * self.cfg.h)
        mine_hits = sum(1 for (_, _, hit, _) in reveals if hit == 1)

        tension = min(1.0, 0.15 + 0.6 * risk_mean + 0.25 * mine_hits)
        novelty = max(0.0, 1.0 - revealed_ratio) * 0.7 + abs(0.5 - alive_ratio) * 0.3

        tags = ["MineLife"]
        if tension > 0.7: tags.append("high_tension")
        if alive_ratio > 0.55: tags.append("dense_motion")

        return {
            "t": self.t,
            "alive_ratio": alive_ratio,
            "revealed_ratio": revealed_ratio,
            "risk_mean": risk_mean,
            "tension": tension,
            "novelty": novelty,
            "reveals": reveals,
            "tags": tags,
            "mine_hits": mine_hits
        }

# ============================================================
# 3) The Integration: KQI-Powered Brain Buffer + XP System
# ============================================================
class BrainBufferOneFile:
    """
    MineLife (無意識/身体) -> KQI Engine (意識/論理) -> Buffer (記憶)
    """
    def __init__(self, buffer_cap: int = 64, mine_cfg: Optional[MineLifeConfig] = None):
        self.buf = Buffer(cap=buffer_cap)
        self.mine = MineLife(mine_cfg or MineLifeConfig())
        self.kqi = KQIEngine()

        self.kqi_theta = 0.5  
        self.kqi_mode = 'inf' 
        
        # ★ 追加要素: 累積コスト(XP)
        # これが「どれだけ努力/コストを払ったか」の指標になる
        self.accumulated_cost = 0.0

    def add_cost(self, amount: float):
        """意図的にコスト（努力値）を支払うメソッド"""
        self.accumulated_cost += amount
        print(f"  [Info] Cost Paid! Total: {self.accumulated_cost:.2f}")

    def tick(self, n: int = 1) -> List[Dict[str, Any]]:
        outs = []
        for _ in range(max(0, n)):
            s = self.mine.tick()
            
            # 生きているだけで少しずつコスト(経験値)は溜まることにする
            self.accumulated_cost += 1.5 

            r_val = s["alive_ratio"] * 10.0
            omega_val = s["novelty"] * 5.0
            
            # リスク＝ノイズ（Sigma）
            sigma_val = s["risk_mean"] * 5.0 + (s["mine_hits"] * 10.0)

            t_val = s["t"]
            rho_val = s["alive_ratio"]

            # KQI Process (累積コストを渡す)
            kqi_res = self.kqi.process(
                r=r_val,
                omega=omega_val,
                sigma=sigma_val,
                t=t_val,
                rho=rho_val,
                theta=self.kqi_theta,
                k_mode=self.kqi_mode,
                accumulated_cost=self.accumulated_cost # ここでコストを渡す！
            )

            full_payload = {
                "source": "MineLife",
                "raw_stats": {k: s[k] for k in ["t", "alive_ratio", "risk_mean", "tension"]},
                "kqi_analysis": kqi_res,
                "events": s["reveals"]
            }

            final_tags = s["tags"] + ["KQI_Processed"]
            if kqi_res["M"] > 0:
                final_tags.append("MOTION_ACTIVE")
            if kqi_res["D"] > 100.0:
                final_tags.append("DEEP_RESONANCE")
            
            # 夏休みタグの追加
            if kqi_res["is_vacation"]:
                final_tags.append("SUMMER_VACATION_MODE")

            self.buf.push(
                payload=full_payload,
                tags=final_tags,
                arousal=min(1.0, kqi_res["D"] / 50.0),
                novelty=s["novelty"],
                confidence=kqi_res["M"]
            )

            outs.append({"tick": s["t"], "kqi": kqi_res, "tags": final_tags})

        return outs

    def view(self, n: int = 5):
        return [it.as_dict() for it in self.buf.view(n)]

# ============================================================
# テスト実行
# ============================================================
if __name__ == "__main__":
    brain = BrainBufferOneFile(mine_cfg=MineLifeConfig(w=10, h=10, mine_ratio=0.15))

    print("--- 1. 初期状態 (凡人モード) ---")
    results = brain.tick(n=5)
    for res in results:
        print(f"Tick {res['tick']:02d} | Sigma(抵抗):{res['kqi']['sigma_effective']:.2f} | Tags: {res['tags']}")

    print("\n--- 2. 努力の積み重ね (コスト支払い) ---")
    brain.add_cost(200.0) # 閾値(100)を超えるコストを一括支払い

    print("\n--- 3. 覚醒状態 (束の間の夏休みモード) ---")
    # ここからは物理法則が書き換わり、Sigma(抵抗)が強制的に下げられる
    results = brain.tick(n=5)
    for res in results:
        kqi = res["kqi"]
        vacation_status = "★ON★" if kqi['is_vacation'] else "OFF"
        print(f"Tick {res['tick']:02d} | Sigma(抵抗):{kqi['sigma_effective']:.4f} (Raw:{kqi['sigma_raw']:.2f}) | Vacation:{vacation_status}")