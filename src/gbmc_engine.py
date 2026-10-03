import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 0) 前提: Minimal Buffer Implementation (仮置き)
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
# 1) KQI Engine (The Logic Core)
# ============================================================
class KQIEngine:
    """GSMC理論に基づく計算エンジン"""
    def __init__(self):
        self.EPSILON = 1e-9

    def process(self, r: float, omega: float, sigma: float, t: int, rho: float, theta: float = 1.0, k_mode='inf'):
        # 1. Genesys
        G = r * omega #omega = minesweeper's landmine
        
        # 2. Base (Sigma check)
        safe_sigma = max(sigma, 0.1) # 0.1を下限として安全マージン確保

        # 3. Quality
        Q = (G ** 2) / safe_sigma

        # 4. Motion (Gate)
        M = 1.0 if Q >= theta else 0.0 #theta = minesweeper's threshold

        # 5. Coherence
        tau = t * rho
        if k_mode == 'inf':
            # tauが大きすぎるとexpが爆発するのでクリップするか、対数スケールで調整
            # ここでは味付けとして tau * 0.1 を入力とする
            C = math.exp(min(tau * 0.1, 10.0)) 
        elif k_mode == 1:
            C = 1.0 + tau
        else:
            C = 1.0

        # 6. Depth
        D = Q * C * M
        
        return {"G": G, "Q": Q, "M": M, "C": C, "D": D}

# ============================================================
# 2) MineLife (The Body / Eco-system)
#    (User's provided code, slightly compacted)
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

class MineLife: #if NTU use, embeddied here.
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
                neigh = self._n8(y, x) #if GCN use, embeddied here.
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
        cand.sort(key=lambda p: self.risk[p[0]][p[1]]) # Sort by risk (safer first)
        picks = []
        for _ in range(min(self.cfg.reveal_per_tick, len(cand))):
            # 30% chance to pick from high risk (tail), 70% from low risk (head)
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
        
        # MineLife Internal Metrics
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
# 3) The Integration: KQI-Powered Brain Buffer
# ============================================================
class BrainBufferOneFile:
    """
    MineLife (無意識/身体) -> KQI Engine (意識/論理) -> Buffer (記憶)
    """
    def __init__(self, buffer_cap: int = 64, mine_cfg: Optional[MineLifeConfig] = None):
        self.buf = Buffer(cap=buffer_cap)

        self.mine = MineLife(mine_cfg or MineLifeConfig())
        
        self.kqi = KQIEngine() # ここにKQIエンジンを搭載

        # KQIパラメータ設定
        self.kqi_theta = 0.5  # Motion閾値
        self.kqi_mode = 'inf' # 共鳴モード

    def tick(self, n: int = 1) -> List[Dict[str, Any]]:
        outs = []
        for _ in range(max(0, n)):
            # 1. MineLifeの状態更新 (Body Update)
            s = self.mine.tick()
            
            # 2. 変数マッピング (Sensory Input -> KQI Variables)
            # r (Range/Power): 生命密度が高いほどパワーがある (0.0 - 10.0 scale)
            r_val = s["alive_ratio"] * 10.0
            
            # omega (Velocity): 新規性が高い（変化が激しい）ほど回転が速い (0.0 - 5.0 scale)
            omega_val = s["novelty"] * 5.0
            
            # sigma (Noise): リスクが高い、または地雷を踏んだ瞬間に急増する
            sigma_val = s["risk_mean"] * 5.0 + (s["mine_hits"] * 10.0)
            
            # t, rho: 時間と密度
            t_val = s["t"]
            rho_val = s["alive_ratio"]

            # 3. KQI Engine Process (Cognitive Processing)
            kqi_res = self.kqi.process(
                r=r_val, 
                omega=omega_val, 
                sigma=sigma_val, 
                t=t_val, 
                rho=rho_val, 
                theta=self.kqi_theta, 
                k_mode=self.kqi_mode
            )

            # 4. 統合ペイロードの作成
            # MineLifeの生データ + KQIの解釈データ
            full_payload = {
                "source": "MineLife",
                "raw_stats": {k: s[k] for k in ["t", "alive_ratio", "risk_mean", "tension"]},
                "kqi_analysis": kqi_res, # ここに D, Q, M が入る
                "events": s["reveals"]
            }
            
            # タグ付けの強化 (KQIの結果に基づいてタグを追加)
            final_tags = s["tags"] + ["KQI_Processed"]
            if kqi_res["M"] > 0:
                final_tags.append("MOTION_ACTIVE")
            if kqi_res["D"] > 100.0:
                final_tags.append("DEEP_RESONANCE")

            # 5. BufferへPush (Memory Storage)
            # KQIの算出値(D)を arousal (覚醒度) として扱うのが自然
            self.buf.push(
                payload=full_payload,
                tags=final_tags,
                arousal=min(1.0, kqi_res["D"] / 50.0), # Dを正規化して覚醒度へ
                novelty=s["novelty"],
                confidence=kqi_res["M"] # Motionゲートが開いている＝確信がある
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
    
    print("--- 脳内シミュレーション開始 (KQI Integrated) ---")
    
    # 10ターン回してみる
    results = brain.tick(n=10)
    
    for res in results:
        kqi = res["kqi"]
        print(f"Tick {res['tick']:02d} | "
              f"G:{kqi['G']:.2f} σ:{kqi['Q']:.2f} " # Qは便宜上sigmaの逆数的な位置づけだがここではQを表示
              f"-> Motion:{kqi['M']} "
              f"-> Depth(D):{kqi['D']:.4f} "
              f"| Tags: {res['tags']}")