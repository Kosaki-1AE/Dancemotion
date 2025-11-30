from dataclasses import dataclass, field
import random
from typing import List, Dict

@dataclass
class AgentConfig:
    name: str
    Rself: float        # 自己責任の強度
    n: float            # 熟練度
    inertia: float      # 慣性（直前方向を引きずる度合い）
    theta: float        # 臨界閾値
    weights: Dict[str, float]  # 各項の重み

@dataclass
class TurnState:
    last_direction: List[float] = field(default_factory=lambda: [0.0, 0.0])
    last_msg: str = ""
    t: int = 0

def dot(a, b): return sum(x*y for x, y in zip(a, b))

def normalize(v):
    import math
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]

def extract_entropy_features(msg:str):
    # 超簡易：長さ・疑問符・感嘆符・抽象語を“外部圧E”と“新規性誘発”指標にする
    abstract = sum(msg.count(k) for k in ["意味","未来","可能性","なぜ","どうして"])
    E = 0.2*len(msg)/100 + 0.5*(msg.count("?")>0) + 0.3*(msg.count("!")>0) + 0.2*abstract
    return {"E": min(E, 2.0), "nov_bias": 0.1*abstract}

def gen_candidates(ctx:TurnState, incoming:str)->List[Dict]:
    # 返答候補を方向ベクトル付きで3案ほど（情報/共感/挑発 みたいな方向性）
    dirs = [[1,0], [0.7,0.7], [0,1]]  # 右・斜め・上 的な意味空間の仮想方向
    texts = [
        "要点を整理して次の実験手順を提案する。",
        "感情の行間を汲みつつ確認質問を返す。",
        "新しい視点を持ち込んで小さく挑発する。"
    ]
    return [{"dir": normalize(d), "text": txt} for d, txt in zip(dirs, texts)]

def score_candidate(cfg:AgentConfig, ctx:TurnState, cand:Dict, feats:Dict):
    w = cfg.weights
    Rdot = cfg.Rself * dot(cand["dir"], [1,0])  # 例：自己責任はx方向ベースに張る
    K    = cfg.inertia * dot(cand["dir"], ctx.last_direction)
    E    = feats["E"]
    Nuse = 0.7 if "提案" in cand["text"] else 0.5
    Novel= feats["nov_bias"] + (0.3 if "新しい" in cand["text"] else 0.1)
    eps  = random.gauss(0, 0.05)

    return (
        w["r"]*Rdot + w["n"]*cfg.n + w["k"]*K + w["e"]*E
        + w["u"]*Nuse + w["nov"]*Novel + eps
    )

def decide(cfg:AgentConfig, ctx:TurnState, incoming:str):
    feats = extract_entropy_features(incoming)
    cands = gen_candidates(ctx, incoming)
    scored = [(score_candidate(cfg, ctx, c, feats), c) for c in cands]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best = scored[0]
    decision = (best_score >= cfg.theta)
    return decision, best_score, best, scored

def update(ctx:TurnState, decision:bool, chosen:Dict):
    if decision:
        ctx.last_direction = chosen["dir"]
    else:
        # 返答保留＝Stillness延長（微弱に減衰）
        ctx.last_direction = [x*0.95 for x in ctx.last_direction]
    ctx.t += 1

# --- 使用例（2者文通ループのイメージ） ---
cfgA = AgentConfig(
    name="A", Rself=0.9, n=0.8, inertia=0.6, theta=1.6,
    weights={"r":0.7,"n":0.3,"k":0.5,"e":0.6,"u":0.6,"nov":0.4}
)
cfgB = AgentConfig(
    name="B", Rself=0.7, n=0.6, inertia=0.5, theta=1.4,
    weights={"r":0.6,"n":0.3,"k":0.4,"e":0.7,"u":0.5,"nov":0.5}
)

ctxA, ctxB = TurnState(), TurnState()
incoming_for_A = "意思発生を“臨界”として扱うなら、いつ越える？"
for _ in range(3):
    # Aが考える
    dA, sA, bestA, scoredA = decide(cfgA, ctxA, incoming_for_A)
    update(ctxA, dA, bestA if dA else {})
    outgoing_from_A = bestA["text"] if dA else "（保留：さらに内省中…）"

    # Bに届く → Bが考える
    dB, sB, bestB, scoredB = decide(cfgB, ctxB, outgoing_from_A)
    update(ctxB, dB, bestB if dB else {})
    incoming_for_A = bestB["text"] if dB else "（保留と思索を続ける…）"
