# conversation_engine.py
# しゃきしゃき流 4層アーキ（Genesys/Stillness/Motion/Coherence）+ 会話ループ
# - Genesys: ファジィ推論で「相手の状態」「話題意図」「温度感」を仮説生成
# - Stillness: 不確かさ×熱量でゲイン制御、待機/反射/即応を決定
# - Motion: 行動テンプレ（質問/共感/要約/提案/冗談/境界宣言）をパラ付き生成
# - Coherence: 返報（極簡易スコア）で方策を更新＆整合性チェック

import dataclasses
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


# -------------------------
# ユーティリティ
# -------------------------
def clamp(x, a, b): return max(a, min(b, x))
def sigmoid(x): return 1/(1+math.exp(-x))

# -------------------------
# モデル設定（YAML風の既定値）
# -------------------------
DEFAULT_YAML = """
persona:
  name: "しゃきアーキ"
  vibe: "フレンドリー/素早いユーモア/短文優先"
goals:
  - "相手の安心感を守る"
  - "会話の温度を適正化する"
  - "仮説→検証の速いループを回す"
genesys:
  fuzzy_sets:
    heat:     {low:[0,0,0.3], mid:[0.2,0.5,0.8], high:[0.6,1.0,1.0]}
    clarity:  {low:[0,0,0.4], mid:[0.3,0.6,0.9], high:[0.7,1.0,1.0]}
    affect:   {neg:[0,0,0.4], neutral:[0.3,0.5,0.7], pos:[0.6,1.0,1.0]}
  rules:
    - if: {heat: high, clarity: low}       # 盛り上がり強い×論点曖昧
      then: {intent: "narrow", confidence: 0.7}
    - if: {heat: mid, clarity: mid}
      then: {intent: "progress", confidence: 0.6}
    - if: {heat: low, clarity: low}
      then: {intent: "open", confidence: 0.6}
    - if: {affect: neg}
      then: {intent: "de-escalate", confidence: 0.8}
stillness:
  gain:
    base: 0.6         # 返答の攻め度合い
    wait_bias: 0.2    # 待つ方向の初期バイアス
    entropy_weight: 0.7
    heat_weight: 0.3
  wait_window_sec: [1.0, 3.0]  # 反射前に置く無応答の揺らぎ
motion:
  templates:
    open:        ["どの辺が気になってる？", "まずざっくり聞きたい：{echo}?"]
    narrow:      ["一番大事なのは{key}で合ってる？", "{key}に絞って深掘りしよ。"]
    progress:    ["今ので半分進んだ。次は{next}いこ。", "いい感じ。{next}やってみる？"]
    de-escalate: ["いったん呼吸合わせよ。無理せずで◎", "OK、ペース落とすね。何がしんどい？"]
    empathize:   ["それ、わかる。俺も似たとこ刺さる。", "共感ポイント：{echo}"]
    joke:        ["ちょい小ネタ：{quip}", "これは…ツッコミ待ち？笑"]
    boundary:    ["ここは線引いとこ。{rule}", "安全のため{rule}で進めよ。"]
coherence:
  reward:
    k_pos: 1.0
    k_neu: 0.2
    k_neg: -1.2
  consistency_penalty: 0.15
storage:
  path: "conv_state.json"
"""

# -------------------------
# 簡易 YAML パーサ（依存ゼロ）
# -------------------------
import re


def parse_yaml(s: str) -> Dict[str, Any]:
    # ざっくりJSON化（今回の既定値前提の超簡易）
    import yaml as _yy  # もしPyYAMLなければここを手書き変換に差し替え
    return _yy.safe_load(s)

try:
    import yaml  # type: ignore
except:
    # 最低限のフォールバック（環境にPyYAML無ければ小実装）
    def yaml_safe_load(s): return json.loads(json.dumps(parse_yaml(s)))  # 不使用
    pass

# -------------------------
# データ構造
# -------------------------
@dataclass
class Turn:
    user: str
    bot: str = ""
    heat: float = 0.5     # ヒート（主観）
    clarity: float = 0.5  # 明瞭さ
    affect: float = 0.5   # ポジ度（0=ネガ,1=ポジ）
    intent: str = "open"
    reward: float = 0.0

@dataclass
class Memory:
    turns: deque = field(default_factory=lambda: deque(maxlen=50))
    policy_scores: Dict[str, float] = field(default_factory=lambda: {
        "open":0.0,"narrow":0.0,"progress":0.0,"de-escalate":0.0,
        "empathize":0.0,"joke":0.0,"boundary":0.0
    })

# -------------------------
# Genesys（ファジィ推論）
# -------------------------
class Genesys:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fs = cfg["fuzzy_sets"]

    @staticmethod
    def tri(x,a,b,c):
        if x<=a or x>=c: return 0.0
        if x==b: return 1.0
        return (x-a)/(b-a) if x<b else (c-x)/(c-b)

    def fuzzify(self, x, setdef):
        # setdef: {"low":[0,0,0.3], "mid":[..], ...}
        mu = {}
        for name, tri in setdef.items():
            mu[name] = self.tri(x, tri[0], tri[1], tri[2])
        return mu

    def infer(self, text:str, estimates:Dict[str,float]) -> Tuple[str,float,Dict]:
        """ textは使わずに最小機能（キーワードで軽微補正） """
        heat = estimates.get("heat",0.5)
        clarity = estimates.get("clarity",0.5)
        affect = estimates.get("affect",0.5)

        # キーワードでヒューリスティック補正（例：恋/告白→heat↑）
        t = text.lower()
        if any(k in t for k in ["告白","アピール","好き","love"]):
            heat = clamp(heat+0.15,0,1)
        if any(k in t for k in ["不安","ムズい","無理"]):
            affect = clamp(affect-0.2,0,1)

        mu_heat    = self.fuzzify(heat,    self.fs["heat"])
        mu_clarity = self.fuzzify(clarity, self.fs["clarity"])
        mu_affect  = self.fuzzify(affect,  self.fs["affect"])

        best_intent, best_conf = "open", 0.5
        chosen_rule = None
        for rule in self.cfg["rules"]:
            cond = rule["if"]
            mlist=[]
            for k,v in cond.items():
                if k=="heat":    mlist.append(mu_heat[v])
                if k=="clarity": mlist.append(mu_clarity[v])
                if k=="affect":  mlist.append(mu_affect[v])
            fire = min(mlist) if mlist else 0.0
            conf = rule["then"]["confidence"] * fire
            if conf>best_conf:
                best_conf = conf
                best_intent = rule["then"]["intent"]
                chosen_rule = rule

        return best_intent, best_conf, {
            "heat":heat, "clarity":clarity, "affect":affect,
            "mu": {"heat":mu_heat,"clarity":mu_clarity,"affect":mu_affect},
            "rule": chosen_rule
        }

# -------------------------
# Stillness（ゲイン制御＋待機）
# -------------------------
class Stillness:
    def __init__(self, cfg):
        self.g = cfg["gain"]
        self.wait_rng = cfg["wait_window_sec"]

    def decide(self, intent:str, conf:float, est:Dict[str,float]) -> Dict[str,Any]:
        entropy = -sum(p*math.log(p+1e-9) for p in [
            est["heat"], est["clarity"], est["affect"]
        ])/math.log(3)  # 0..1 正規化

        gain = self.g["base"]
        gain -= self.g["entropy_weight"]*entropy
        gain += self.g["heat_weight"]*est["heat"]
        gain = clamp(gain,0,1)

        wait_prob = clamp(self.g["wait_bias"] + 0.6*(1-conf) + 0.3*entropy, 0, 1)
        will_wait = (random.random() < wait_prob)
        wait_sec = random.uniform(*self.wait_rng) if will_wait else 0.0

        return {"gain":gain, "wait":will_wait, "wait_sec":wait_sec,
                "entropy":entropy, "wait_prob":wait_prob}

# -------------------------
# Motion（行動テンプレ）
# -------------------------
class Motion:
    def __init__(self, cfg):
        self.templates = cfg["templates"]
        self.quips = ["それは秒で優勝では？", "脳内で拍手起きたわ", "それエグいw"]
        self.rules = [
            ("boundary", lambda ctx: ctx["gain"]<0.25),
            ("de-escalate", lambda ctx: ctx["entropy"]>0.7),
        ]

    def choose_policy(self, intent:str, ctx:Dict[str,Any]) -> str:
        # ルール優先で上書き
        for name,cond in self.rules:
            if cond(ctx): return name
        return intent

    def render(self, policy:str, vars:Dict[str,str]) -> str:
        bank = self.templates.get(policy, ["{echo}"])
        tmpl = random.choice(bank)
        return tmpl.format(**vars)

# -------------------------
# Coherence（報酬＋整合チェック）
# -------------------------
class Coherence:
    def __init__(self, cfg):
        self.k = cfg["reward"]
        self.penalty = cfg["consistency_penalty"]

    def score_turn(self, user_text:str, bot_text:str, est:Dict[str,float], intent:str) -> float:
        # ざっくり感情スコア：絵文字・肯定語で近似
        pos = sum(user_text.count(x) for x in ["👍","😊","助かる","いいね","なるほど","草"])
        neg = sum(user_text.count(x) for x in ["無理","最悪","は？","やめて","嫌"])
        base = self.k["k_pos"]*pos + self.k["k_neg"]*neg
        base += self.k["k_neu"]*(1 if pos==0 and neg==0 else 0)

        # 整合性：熱が低いのに冗談連打 等を少し罰
        if est["heat"]<0.35 and intent in ["joke","narrow"]:
            base -= self.penalty
        return base

# -------------------------
# エンジン
# -------------------------
class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gen = Genesys(cfg["genesys"])
        self.sti = Stillness(cfg["stillness"])
        self.mot = Motion(cfg["motion"])
        self.coh = Coherence(cfg["coherence"])
        self.mem = Memory()
        self.path = cfg["storage"]["path"]

    def estimate_from_text(self, text:str) -> Dict[str,float]:
        # 最小実装：長さ/疑問/感嘆/否定語で近似
        L = len(text)
        heat = clamp(0.25 + 0.02*text.count("！") + 0.01*text.count("!")+ 0.001*L, 0,1)
        clarity = clamp(0.6 - 0.15*text.count("？") - 0.05*text.count("?"), 0,1)
        affect = clamp(0.55 + 0.1*text.count("😊") - 0.12*sum(text.count(k) for k in ["嫌","無理","疲れ"]), 0,1)
        return {"heat":heat,"clarity":clarity,"affect":affect}

    def step(self, user_text:str) -> Tuple[str,Dict[str,Any]]:
        est = self.estimate_from_text(user_text)
        intent, conf, detail = self.gen.infer(user_text, est)
        sti = self.sti.decide(intent, conf, detail)

        # 待つ（会話UX：小さく間を置く/ここは実時間sleep許容）
        if sti["wait"]: time.sleep(sti["wait_sec"])

        # Motion
        policy = self.mot.choose_policy(intent, sti|detail)
        vars = {
            "echo": user_text[:32],
            "key": "論点",
            "next": "具体例",
            "quip": random.choice(self.mot.quips),
            "rule": "安全/敬意/境界を守る"
        }
        bot = self.mot.render(policy, vars)

        # Coherence
        reward = self.coh.score_turn(user_text, bot, detail, policy)
        t = Turn(user=user_text, bot=bot, heat=detail["heat"], clarity=detail["clarity"],
                 affect=detail["affect"], intent=policy, reward=reward)
        self.mem.turns.append(t)
        self.mem.policy_scores[policy] += reward

        debug = {"intent":intent,"policy":policy,"confidence":round(conf,3),
                 "gain":round(sti["gain"],3),"wait":sti["wait"],
                 "entropy":round(sti["entropy"],3),"reward":round(reward,3)}
        return bot, debug

    # 状態入出力
    def save(self):
        data = {
            "turns":[dataclasses.asdict(t) for t in self.mem.turns],
            "policy_scores":self.mem.policy_scores
        }
        with open(self.path,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)

    def load(self):
        if not os.path.exists(self.path): return
        with open(self.path,"r",encoding="utf-8") as f:
            data=json.load(f)
        dq=deque(maxlen=50)
        for d in data.get("turns",[]): dq.append(Turn(**d))
        self.mem.turns=dq
        self.mem.policy_scores=data.get("policy_scores",self.mem.policy_scores)

# -------------------------
# CLI ループ
# -------------------------
def main():
    cfg = parse_yaml(DEFAULT_YAML)
    eng = Engine(cfg)
    print("<< しゃき会話エンジン 起動 >>  /status /reset /save /load 使えます。")

    while True:
        try:
            s = input("\nあなた> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!"); break

        if not s: continue
        if s == "/reset":
            eng = Engine(cfg); print("状態リセット"); continue
        if s == "/save":
            eng.save(); print("保存OK"); continue
        if s == "/load":
            eng.load(); print("読込OK"); continue
        if s == "/status":
            ps = sorted(eng.mem.policy_scores.items(), key=lambda x:-x[1])
            print("方策スコア:", ps); 
            if eng.mem.turns:
                last = eng.mem.turns[-1]
                print(f"直近: intent={last.intent}, reward={round(last.reward,3)}, heat={round(last.heat,2)}")
            continue

        reply, dbg = eng.step(s)
        print(f"しゃき> {reply}")
        # デバッグ見たい時だけ↓コメント外す
        # print("dbg:", dbg)

if __name__ == "__main__":
    main()
