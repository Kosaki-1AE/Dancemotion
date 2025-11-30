# gsmc_runtime.py
# - YAML(core.yml)を読み込み、Genesys/Stillness/Motion/Coherence を離散時間で更新
# - Stillness に対して「いつでも干渉可能」な割り込みを実装（優先度/デバウンス/レート制限/サンドボックス）
# - 使い方例:
#   python gsmc_runtime.py --cfg /mnt/data/core.yml --steps 500 --dt 0.02 \
#       --inject "t=1.0,type=stillness.interrupt,payload={\"coherence_boost\":0.4,\"sigma_scale\":0.9}"
#
#   ログCSV: --log-csv out.csv  （標準出力にサマリも出る）

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# --------- 小ユーティリティ ---------
def clip(x, lo, hi):
    return max(lo, min(hi, x))

def now_ms():
    return int(time.time() * 1000)

# Ornstein-Uhlenbeck noise
class OU:
    def __init__(self, theta=2.0, sigma=0.1, mu=0.0, x0=0.0):
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.x = x0

    def step(self, dt: float) -> float:
        # x_{t+dt} = x + theta*(mu - x)*dt + sigma*sqrt(dt)*N(0,1)
        self.x += self.theta * (self.mu - self.x) * dt + self.sigma * math.sqrt(dt) * random.gauss(0,1)
        return self.x

# --------- イベントバス（優先度付きキュー＋ポリシー） ---------
@dataclass(order=True)
class Event:
    priority: int
    t: float
    type: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    enq_ms: int = field(compare=False, default_factory=now_ms)

class EventBus:
    def __init__(self, priorities: Dict[str,int], debounce_ms: int, rate_limit_ips: int, latency_budget_ms: int):
        self.priorities = priorities
        self.debounce_ms = debounce_ms
        self.rate_limit_ips = rate_limit_ips
        self.latency_budget_ms = latency_budget_ms
        self.queue: List[Event] = []
        self.last_emitted_ms: Dict[str,int] = {}
        self.counter_last_sec = 0
        self.counter_sec_epoch = int(time.time())

    def _priority_for(self, etype: str) -> int:
        # ワイルドカード sensor.* にも対応
        if etype in self.priorities:
            return self.priorities[etype]
        if etype.startswith("sensor.") and "sensor.*" in self.priorities:
            return self.priorities["sensor.*"]
        return 0

    def emit(self, sim_t: float, etype: str, payload: Dict[str,Any]):
        nowS = int(time.time())
        if nowS != self.counter_sec_epoch:
            self.counter_sec_epoch = nowS
            self.counter_last_sec = 0
        self.counter_last_sec += 1
        if self.counter_last_sec > self.rate_limit_ips:
            # 静かにドロップ（ポリシー違反）
            return

        last = self.last_emitted_ms.get(etype, -10**9)
        if now_ms() - last < self.debounce_ms:
            return
        self.last_emitted_ms[etype] = now_ms()

        pr = self._priority_for(etype)
        self.queue.append(Event(priority=-pr, t=sim_t, type=etype, payload=payload))  # heapにしなくてもOK: 高優先度先に処理
        # 優先度降順で並べ替え（priorityは負にしてるので昇順ソートでOK）
        self.queue.sort()

    def pop_all_ready(self) -> List[Event]:
        evs = self.queue[:]
        self.queue.clear()
        return evs

# --------- レイヤ状態 ---------
@dataclass
class Genesys:
    priors: Dict[str,Any] = field(default_factory=dict)
    constraints: List[Any] = field(default_factory=list)
    resources: List[Any] = field(default_factory=list)

@dataclass
class Stillness:
    coherence_time: float = 1.0     # s
    phase: str = "idle"             # idle/observing/gating/holding
    ou: OU = field(default_factory=lambda: OU(theta=2.0, sigma=0.1))
    sim_metric_stillness_in_motion: float = 0.0

@dataclass
class Motion:
    action_intent: Tuple[float,float,float] = (0.0, 0.0, 0.0)  # amplitude, tempo, risk
    executed: bool = False
    value: float = 0.0  # シンプルなスカラー運動状態

@dataclass
class Coherence:
    bandwidth: float = 1.0   # Hz
    score: float = 0.0

@dataclass
class Responsibility:
    vec: Tuple[float,float,float] = (0.0, 0.0, 0.0)  # self, other, system

@dataclass
class NoriEntropy:
    value: float = 0.0

# --------- メインシステム ---------
class GSMC:
    def __init__(self, cfg: Dict[str,Any]):
        self.cfg = cfg
        layers = cfg["state_space"]["layers"]

        # Stillness Policy
        st = layers["stillness"]
        st_state = st["state"]
        ou_model = st_state.get("micro_fluct", st_state.get("micro_fluc", {}))
        theta = float(ou_model.get("theta", 2.0))
        sigma = float(ou_model.get("sigma", 0.1))
        self.bounds = st["interrupt"]["safety"]["bounds"]
        self.latency_budget_ms = int(st["interrupt"]["policy"]["latency_budget_ms"])
        self.coh_time_bounds = self.bounds["coherence_time"]
        self.sigma_bounds = self.bounds["micro_fluct.sigma"]
        self.merge_spec = st["interrupt"]["merge_strategy"]
        self.sandbox_enabled = bool(st["interrupt"].get("sandbox",{}).get("enabled", True))
        self.accept_tests = st["interrupt"].get("sandbox",{}).get("acceptance",{}).get("tests", [])
        # damping
        self.motion_damping = float(cfg.get("defaults",{}).get("damping",{}).get("motion", 0.1))

        # Observables logging
        self.log_freq = int(cfg.get("observables",{}).get("logging",{}).get("frequency", 20))
        self.log_dt = 1.0 / max(self.log_freq, 1)
        self.next_log_t = 0.0

        # init states
        self.G = Genesys()
        self.S = Stillness(coherence_time=float(st_state["coherence_time"]["default"]), ou=OU(theta=theta, sigma=sigma))
        self.M = Motion()
        self.C = Coherence(bandwidth=float(layers["coherence"]["state"]["bandwidth"]["default"]))
        self.R = Responsibility()
        self.E = NoriEntropy()

        # Interrupt bus
        prio = layers["stillness"]["interrupt"]["priorities"]
        debounce_ms = layers["stillness"]["interrupt"]["policy"]["debounce_ms"]
        rate_ips = layers["stillness"]["interrupt"]["safety"]["rate_limits"]["interrupts_per_sec"]
        self.bus = EventBus(priorities=prio, debounce_ms=debounce_ms, rate_limit_ips=rate_ips, latency_budget_ms=self.latency_budget_ms)

        # thresholds
        self.thr_coh = float(cfg.get("defaults",{}).get("thresholds",{}).get("coherence", 0.6))
        self.thr_sim = float(cfg.get("defaults",{}).get("thresholds",{}).get("stillness_in_motion", 0.5))

        # bookkeeping
        self.t = 0.0
        self.logs: List[Dict[str,Any]] = []

    # ---- 外部からの割り込み注入（いつでもOK, プリエンプティブ） ----
    def inject(self, etype: str, payload: Dict[str,Any]):
        self.bus.emit(self.t, etype, payload)

    # ---- サンドボックス評価（超簡易版） ----
    def _sandbox_accept(self, predict_delta_metric: float, latency_ms: int) -> bool:
        ok = True
        if "coherence_gain>=0" in self.accept_tests:
            ok = ok and (predict_delta_metric >= 0)
        if "latency<=latency_budget_ms" in self.accept_tests:
            ok = ok and (latency_ms <= self.latency_budget_ms)
        if "stability" in self.accept_tests:
            # stillness_in_motion が[0,1]から大きく外れない前提を簡易チェック
            ok = ok and (0.0 <= self.S.sim_metric_stillness_in_motion <= 1.0)
        return ok

    # ---- Stillnessマージ（δS_ext ⊕ ノイズ） ----
    def _apply_stillness_delta(self, payload: Dict[str,Any]) -> None:
        # coherence_boost: coherence_timeを相対的に調整
        cboost = float(payload.get("coherence_boost", 0.0))
        sscale = float(payload.get("sigma_scale", 1.0))
        # 予測メトリクス変化（ダミー：coherence_time上昇→coherence_score上がると仮定）
        before = self.C.score
        # 反映（クリップ）
        self.S.coherence_time = clip(self.S.coherence_time + cboost, self.coh_time_bounds[0], self.coh_time_bounds[1])
        # sigma更新（OU）
        new_sigma = clip(self.S.ou.sigma * sscale, self.sigma_bounds[0], self.sigma_bounds[1])
        self.S.ou.sigma = new_sigma
        # 粗い予測
        after = min(1.0, max(0.0, before + 0.5 * cboost))
        self.C.score = after

    # ---- 1ステップ更新 ----
    def step(self, dt: float):
        # 1) イベントを即時処理（プリエンプト可能）
        evs = self.bus.pop_all_ready()
        for ev in evs:
            # レイテンシ測定
            latency = now_ms() - ev.enq_ms
            # サンドボックス: 予測ゲイン（簡易）
            predict_gain = 0.1  # ここは運用で学習していく想定
            if self.sandbox_enabled and not self._sandbox_accept(predict_gain, latency):
                continue  # 却下
            if ev.type == "stillness.interrupt" or ev.type == "stillness.tune" or ev.type.startswith("sensor.") or ev.type == "user.gesture":
                self._apply_stillness_delta(ev.payload)

        # 2) OUノイズで微ゆらぎ
        noise = self.S.ou.step(dt)

        # 3) Motion更新: dM/dt = u(t) - damping*M + coupling(C)
        #   ここでは簡易: u(t) = action_intent[0]、coupling = C.score
        amp, tempo, risk = self.M.action_intent
        u = amp
        self.M.value += dt * (u - self.motion_damping * self.M.value + 0.5 * self.C.score)

        # 4) Coherence更新（ここはダミー：StillnessとMotionの落差でスコアを調整）
        desired = 1.0 / (1.0 + abs(noise))   # ノイズが少ないほど高スコア
        # Stillnessのcoherence_timeが長め＆Motionが安定ならスコア↑
        smoothness = math.exp(-abs(self.M.value) * 0.05)
        self.C.score = clip(0.5 * desired + 0.5 * smoothness, 0.0, 1.0)

        # 5) Stillness-in-Motionメトリクス（シンプルな0-1スコア）
        velocity_like = abs(self.M.value)
        freeze_ratio = clip(1.0 - min(1.0, velocity_like*0.1), 0.0, 1.0)
        self.S.sim_metric_stillness_in_motion = clip( -velocity_like + freeze_ratio, 0.0, 1.0)

        # 6) Nori-Entropy 更新（仮説ルール）
        #   dE/dt = H(signal) - coherence_gain + responsibility_alignment_bonus
        #   超簡易: H≈|noise|, coherence_gain≈ΔC, alignment≈0（省略）
        dE = abs(noise) - (self.C.score - 0.5)
        self.E.value += dt * dE

        # 7) ロギング
        if self.t >= self.next_log_t:
            self.logs.append({
                "time": round(self.t, 4),
                "stillness.coherence_time": round(self.S.coherence_time, 4),
                "stillness.sigma": round(self.S.ou.sigma, 4),
                "sim.stillness_in_motion": round(self.S.sim_metric_stillness_in_motion, 4),
                "motion.value": round(self.M.value, 4),
                "coherence.score": round(self.C.score, 4),
                "nori_entropy": round(self.E.value, 4),
            })
            self.next_log_t += self.log_dt

        self.t += dt

    # ---- 走らせる ----
    def run(self, steps: int, dt: float, scheduled_injects: List[Dict[str,Any]]):
        inj_ix = 0
        scheduled_injects = sorted(scheduled_injects, key=lambda x: x["t"])
        for i in range(steps):
            # 時刻到達で割り込み投入
            while inj_ix < len(scheduled_injects) and scheduled_injects[inj_ix]["t"] <= self.t:
                inj = scheduled_injects[inj_ix]
                self.inject(inj["type"], inj.get("payload", {}))
                inj_ix += 1
            self.step(dt)

    def dump_csv(self, path: Optional[str]):
        if not path: 
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.logs[0].keys()))
            w.writeheader()
            w.writerows(self.logs)

# --------- CLI ---------
def parse_inject_arg(txt: str) -> Dict[str,Any]:
    # 例: t=1.0,type=stillness.interrupt,payload={"coherence_boost":0.4,"sigma_scale":0.9}
    parts = [p.strip() for p in txt.split(",")]
    got: Dict[str,Any] = {}
    for p in parts:
        if "=" not in p: 
            continue
        k,v = p.split("=",1)
        k = k.strip()
        v = v.strip()
        if k == "t":
            got["t"] = float(v)
        elif k == "type":
            got["type"] = v
        elif k == "payload":
            got["payload"] = json.loads(v)
    if "t" not in got or "type" not in got:
        raise ValueError("inject には t と type が必要です")
    return got

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="core.yml")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--inject", action="append", default=[], help='t=...,type=...,payload={"k":v}')
    ap.add_argument("--log-csv", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    sim = GSMC(cfg)

    injects = [parse_inject_arg(x) for x in args.inject]
    sim.run(steps=args.steps, dt=args.dt, scheduled_injects=injects)

    # 出力
    if sim.logs:
        # サマリ
        last = sim.logs[-1]
        print("[SUMMARY]")
        for k,v in last.items():
            print(f"{k}: {v}")
    else:
        print("no logs")

    sim.dump_csv(args.log_csv)
    if args.log_csv:
        print(f"[LOG] wrote {args.log_csv}")

if __name__ == "__main__":
    main()
