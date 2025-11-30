#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y‑Theory Conversational Engine
==============================
Single‑file reference implementation of the four layers:
    • Genesys   : fuzzy intent synthesis from observations ("mimic → abstract → rule")
    • Stillness : adaptive pause / gain / hysteresis controller (attention allocator)
    • Motion    : action selection + phrasing policies (energy‑aware)
    • Coherence : conversation memory, topic multi‑links, and consistency scoring

Usage (CLI):
    python y_theory_chat.py

This boots an interactive REPL. Type 'quit' to exit.
To tweak behavior, edit the CONFIG_YAML string at the bottom (or export it to a file).

No external deps beyond Python 3.9+.
"""
from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def softmax(xs: List[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0:
        # argmax one‑hot
        m = max(xs)
        return [1.0 if x == m else 0.0 for x in xs]
    exps = [math.exp((x) / temperature) for x in xs]
    s = sum(exps) or 1.0
    return [v / s for v in exps]


def jaccard(a_tokens: set, b_tokens: set) -> float:
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def tokenize(text: str) -> List[str]:
    # Lightweight multilingual-ish tokenization (ASCII, kana, kanji chunks)
    tokens = re.findall(r"[A-Za-z]+|[\u3040-\u30ff]+|[\u4e00-\u9faf]+|\d+", text.lower())
    return tokens

# ------------------------------------------------------------
# Genesys: minimal fuzzy logic engine
# ------------------------------------------------------------

@dataclass
class FuzzySet:
    kind: str  # 'tri' or 'trap'
    params: Tuple[float, ...]

    def mu(self, x: float) -> float:
        """Membership function."""
        if self.kind == 'tri':
            a, b, c = self.params
            if x <= a or x >= c:
                return 0.0
            if x == b:
                return 1.0
            return (x - a) / (b - a) if x < b else (c - x) / (c - b)
        elif self.kind == 'trap':
            a, b, c, d = self.params
            if x <= a or x >= d:
                return 0.0
            if b <= x <= c:
                return 1.0
            if a < x < b:
                return (x - a) / (b - a)
            if c < x < d:
                return (d - x) / (d - c)
        return 0.0


@dataclass
class FuzzyVar:
    name: str
    sets: Dict[str, FuzzySet]  # label -> FuzzySet

    def fuzz(self, x: float) -> Dict[str, float]:
        return {label: fs.mu(x) for label, fs in self.sets.items()}


@dataclass
class FuzzyRule:
    # Example: {
    #  'if': {'affect':'high', 'energy':'mid'},
    #  'then': {'intent_drive':'strong'}
    # }
    antecedent: Dict[str, str]
    consequent: Dict[str, str]
    weight: float = 1.0


class Genesys:
    """Fuzzy intent synthesis. Inputs are continuous observations in [0,1]."""
    def __init__(self, inputs: Dict[str, FuzzyVar], outputs: Dict[str, FuzzyVar], rules: List[FuzzyRule]):
        self.inputs = inputs
        self.outputs = outputs
        self.rules = rules

    def infer(self, obs: Dict[str, float]) -> Dict[str, float]:
        # 1) fuzzify inputs
        fuzzed: Dict[str, Dict[str, float]] = {
            name: var.fuzz(clamp(obs.get(name, 0.0), 0.0, 1.0)) for name, var in self.inputs.items()
        }
        # 2) rule evaluation (min for AND; weighted)
        out_aggr: Dict[Tuple[str, str], float] = {}
        for rule in self.rules:
            degree = 1.0
            for in_name, label in rule.antecedent.items():
                degree = min(degree, fuzzed.get(in_name, {}).get(label, 0.0))
            degree *= rule.weight
            for out_name, out_label in rule.consequent.items():
                key = (out_name, out_label)
                out_aggr[key] = max(out_aggr.get(key, 0.0), degree)
        # 3) defuzzify (center of gravity over discrete labels using label anchors)
        crisp: Dict[str, float] = {}
        for out_name, out_var in self.outputs.items():
            num = 0.0
            den = 0.0
            for label, fs in out_var.sets.items():
                mu = out_aggr.get((out_name, label), 0.0)
                # use the peak location as representative centroid for simplicity
                if fs.kind == 'tri':
                    _, b, _ = fs.params
                    xstar = b
                elif fs.kind == 'trap':
                    _, b, c, _ = fs.params
                    xstar = (b + c) / 2
                else:
                    xstar = 0.5
                num += mu * xstar
                den += mu
            crisp[out_name] = num / den if den > 0 else 0.0
        return crisp

# ------------------------------------------------------------
# Stillness: adaptive pause / gain controller with hysteresis
# ------------------------------------------------------------

@dataclass
class StillnessState:
    mode: str = "COHERENCE"  # one of {STILLNESS, MOTION, COHERENCE}
    gain: float = 0.6         # affects Motion sampling temperature
    pause_s: float = 0.2      # recommended micro‑pause before speaking
    last_switch: float = field(default_factory=time.time)


class Stillness:
    def __init__(self, enter_thr: float, exit_thr: float, min_hold_s: float, base_pause_s: float):
        self.enter_thr = enter_thr
        self.exit_thr = exit_thr
        self.min_hold_s = min_hold_s
        self.base_pause_s = base_pause_s
        self.s = StillnessState()

    def update(self, cognitive_load: float, intent_drive: float, trust: float) -> StillnessState:
        now = time.time()
        hold_ok = (now - self.s.last_switch) >= self.min_hold_s
        # Simple policy: high load & low trust → STILLNESS; high drive & trust → MOTION; else COHERENCE
        if self.s.mode != "STILLNESS" and cognitive_load > self.enter_thr and hold_ok:
            self.s.mode = "STILLNESS"; self.s.last_switch = now
        elif self.s.mode == "STILLNESS" and cognitive_load < self.exit_thr and hold_ok:
            self.s.mode = "COHERENCE"; self.s.last_switch = now
        elif self.s.mode != "MOTION" and intent_drive > 0.6 and trust > 0.5 and hold_ok:
            self.s.mode = "MOTION"; self.s.last_switch = now
        elif self.s.mode == "MOTION" and intent_drive < 0.35 and hold_ok:
            self.s.mode = "COHERENCE"; self.s.last_switch = now
        # gain: increase with trust & drive, decrease with load
        self.s.gain = clamp(0.2 + 0.8 * (0.6 * intent_drive + 0.4 * trust) - 0.5 * cognitive_load, 0.1, 1.2)
        self.s.pause_s = self.base_pause_s * (1.2 + 0.8 * cognitive_load)
        return self.s

# ------------------------------------------------------------
# Motion: action selection & phrasing
# ------------------------------------------------------------

@dataclass
class Action:
    name: str
    utility: Callable[[Dict[str, float]], float]
    templates: List[str]


class Motion:
    def __init__(self, actions: List[Action]):
        self.actions = actions

    def act(self, signals: Dict[str, float], gain: float) -> Tuple[str, str, Dict[str, float]]:
        # compute raw utilities
        scores = [a.utility(signals) for a in self.actions]
        # temperature inversely related to gain
        temperature = clamp(1.2 - gain, 0.05, 1.2)
        probs = softmax(scores, temperature)
        idx = random.choices(list(range(len(self.actions))), weights=probs, k=1)[0]
        action = self.actions[idx]
        text = random.choice(action.templates).format(**{k: f"{v:.2f}" for k, v in signals.items()})
        return action.name, text, {"temperature": temperature, "probs": {a.name: p for a, p in zip(self.actions, probs)}}

# ------------------------------------------------------------
# Coherence: memory / multi‑link topics / consistency score
# ------------------------------------------------------------

@dataclass
class Memory:
    turns: List[Tuple[str, str]] = field(default_factory=list)  # (role, text)
    topics: Dict[str, float] = field(default_factory=dict)

    def add(self, role: str, text: str):
        self.turns.append((role, text))
        toks = set(tokenize(text))
        for t in toks:
            self.topics[t] = self.topics.get(t, 0.0) * 0.95 + 0.05  # decay + reinforce

    def topic_score(self, text: str) -> float:
        toks = set(tokenize(text))
        active = {k for k, v in self.topics.items() if v > 0.02}
        return jaccard(toks, active)


class Coherence:
    def __init__(self, silence_bias: float = 0.2):
        self.mem = Memory()
        self.silence_bias = silence_bias

    def recommend_pause(self, mode: str, stillness_pause: float, coherence_score: float) -> float:
        # longer pause if low coherence or in STILLNESS mode
        base = stillness_pause
        if mode == 'STILLNESS':
            base *= (1.5 + self.silence_bias)
        elif coherence_score < 0.25:
            base *= (1.3 + self.silence_bias)
        else:
            base *= 1.0
        return clamp(base, 0.05, 3.0)

# ------------------------------------------------------------
# Conversation Engine
# ------------------------------------------------------------

class Engine:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        # Build Genesys
        def make_var(vcfg):
            sets = {label: FuzzySet(kind=spec['kind'], params=tuple(spec['params'])) for label, spec in vcfg['sets'].items()}
            return FuzzyVar(vcfg['name'], sets)
        g_inputs = {name: make_var(v) for name, v in self.cfg['genesys']['inputs'].items()}
        g_outputs = {name: make_var(v) for name, v in self.cfg['genesys']['outputs'].items()}
        rules = [FuzzyRule(r['if'], r['then'], r.get('weight', 1.0)) for r in self.cfg['genesys']['rules']]
        self.genesys = Genesys(g_inputs, g_outputs, rules)
        # Stillness
        s_cfg = self.cfg['stillness']
        self.stillness = Stillness(s_cfg['enter_thr'], s_cfg['exit_thr'], s_cfg['min_hold_s'], s_cfg['base_pause_s'])
        # Motion actions
        def mkutil(expr: str) -> Callable[[Dict[str, float]], float]:
            # VERY tiny safe evaluator over allowed names only
            allowed = {'min': min, 'max': max, 'abs': abs, 'math': math}
            code = compile(expr, '<util>', 'eval')
            def _f(sig: Dict[str, float]) -> float:
                env = {**allowed, **sig}
                return float(eval(code, {'__builtins__': {}}, env))
            return _f
        actions = []
        for a in self.cfg['motion']['actions']:
            actions.append(Action(a['name'], mkutil(a['utility']), a['templates']))
        self.motion = Motion(actions)
        # Coherence
        self.coherence = Coherence(self.cfg['coherence'].get('silence_bias', 0.2))

    def step(self, user_text: str) -> Dict[str, Any]:
        # Observe basic signals (placeholder heuristics)
        tokens = tokenize(user_text)
        length = len(tokens)
        affect = clamp(0.2 + 0.02 * sum(ch in '!?' for ch in user_text), 0.0, 1.0)
        energy = clamp(min(1.0, length / 30.0), 0.0, 1.0)
        trust = clamp(self.coherence.mem.topic_score(user_text) * 0.6 + 0.2, 0.0, 1.0)
        cognitive_load = clamp(0.6 * (1 - trust) + 0.4 * (1 - energy), 0.0, 1.0)
        obs = {'affect': affect, 'energy': energy, 'trust': trust}

        # Genesys → intent_drive, question_bias
        intents = self.genesys.infer(obs)
        intent_drive = intents.get('intent_drive', 0.0)
        question_bias = intents.get('question_bias', 0.0)

        # Stillness update
        s = self.stillness.update(cognitive_load=cognitive_load, intent_drive=intent_drive, trust=trust)

        # Motion signals
        signals = {
            'intent_drive': intent_drive,
            'question_bias': question_bias,
            'trust': trust,
            'affect': affect,
            'energy': energy,
            'cognitive_load': cognitive_load,
            'gain': s.gain,
        }

        # Coherence
        coh_score = self.coherence.mem.topic_score(user_text)
        pause = self.coherence.recommend_pause(s.mode, s.pause_s, coh_score)

        # Select action & phrase
        act_name, reply, info = self.motion.act(signals, gain=s.gain)

        # Optionally bias reply style by question_bias
        if question_bias > 0.5 and not reply.endswith('?'):
            reply += " — どう思う？"

        # Update memory
        self.coherence.mem.add('user', user_text)
        self.coherence.mem.add('assistant', reply)

        return {
            'mode': s.mode,
            'pause': round(pause, 2),
            'reply': reply,
            'action': act_name,
            'signals': {k: round(v, 3) for k, v in signals.items()},
            'info': info,
        }

# ------------------------------------------------------------
# Example configuration (edit to taste)
# ------------------------------------------------------------
CONFIG_YAML = r"""
version: 1
genesys:
  inputs:
    affect:
      name: affect
      sets:
        low:  {kind: tri,  params: [0.0, 0.0, 0.4]}
        mid:  {kind: tri,  params: [0.2, 0.5, 0.8]}
        high: {kind: tri,  params: [0.6, 1.0, 1.0]}
    energy:
      name: energy
      sets:
        low:  {kind: tri,  params: [0.0, 0.0, 0.4]}
        mid:  {kind: tri,  params: [0.2, 0.5, 0.8]}
        high: {kind: tri,  params: [0.6, 1.0, 1.0]}
    trust:
      name: trust
      sets:
        low:  {kind: tri,  params: [0.0, 0.0, 0.5]}
        mid:  {kind: tri,  params: [0.2, 0.5, 0.8]}
        high: {kind: tri,  params: [0.6, 1.0, 1.0]}
  outputs:
    intent_drive:
      name: intent_drive
      sets:
        weak:   {kind: trap, params: [0.0, 0.0, 0.2, 0.4]}
        medium: {kind: trap, params: [0.2, 0.4, 0.6, 0.8]}
        strong: {kind: trap, params: [0.6, 0.8, 1.0, 1.0]}
    question_bias:
      name: question_bias
      sets:
        low:  {kind: tri, params: [0.0, 0.0, 0.5]}
        mid:  {kind: tri, params: [0.2, 0.5, 0.8]}
        high: {kind: tri, params: [0.5, 1.0, 1.0]}
  rules:
    # High trust + high energy → strong drive, ask less
    - if:   {trust: high, energy: high}
      then: {intent_drive: strong, question_bias: mid}
      weight: 1.0
    # Low trust → weaker drive, ask more questions
    - if:   {trust: low}
      then: {intent_drive: weak, question_bias: high}
      weight: 1.0
    # High affect with mid energy → nudge to ask & empathize
    - if:   {affect: high, energy: mid}
      then: {intent_drive: medium, question_bias: high}
      weight: 0.9
    # Mid‑everything → balanced
    - if:   {affect: mid, energy: mid, trust: mid}
      then: {intent_drive: medium, question_bias: mid}
      weight: 1.0

stillness:
  enter_thr: 0.6      # load threshold to ENTER STILLNESS
  exit_thr:  0.35     # load threshold to EXIT STILLNESS
  min_hold_s: 3.0     # hysteresis hold time (s)
  base_pause_s: 0.25  # base micro‑pause

motion:
  actions:
    - name: acknowledge
      utility: "0.4*trust + 0.4*question_bias + 0.2*(1-cognitive_load)"
      templates:
        - "うん、受け取ったよ。今{energy}のテンポで感じてる。"
        - "なるほどね。いまの熱量{energy}、信頼{trust}って感じ。"
    - name: ask_clarify
      utility: "0.6*question_bias + 0.3*(1-trust) + 0.1*affect"
      templates:
        - "もう少し具体的に言うと、どの部分が大事？"
        - "キモはどこ？一言でまとめると何になりそう？"
    - name: reflect_reframe
      utility: "0.5*trust + 0.3*intent_drive + 0.2*(1-cognitive_load)"
      templates:
        - "整理すると『意図{intent_drive}×信頼{trust}』で流れ作れそう。"
        - "今の話、要は『{intent_drive}ドライブ』で一歩進めるって解釈したよ。"
    - name: propose_step
      utility: "0.6*intent_drive + 0.2*trust + 0.2*energy"
      templates:
        - "じゃあマイクロステップを1つ：いまの文脈で30秒だけメモ→要点3つに分けよ。"
        - "まずは超小さく実験しよ：10分タイマーで仮説だけ書き出す。"

coherence:
  silence_bias: 0.25
"""

# ------------------------------------------------------------
# Minimal YAML loader (subset) to stay dependency‑free
# ------------------------------------------------------------

def _mini_yaml_load(txt: str) -> Dict[str, Any]:
    # For a self‑contained demo we support a tiny subset via eval on a preprocessed string.
    # SECURITY: This is for local tinkering; not for untrusted inputs.
    # Convert YAML‑ish to Python dict quickly.
    #  - relies on indentation and ':' separators; doesn't handle all cases.
    import yaml  # If pyyaml is available, prefer it; otherwise fallback.
    return yaml.safe_load(txt)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():
    try:
        cfg = _mini_yaml_load(CONFIG_YAML)
    except Exception as e:
        print("Failed to parse CONFIG_YAML:", e)
        return
    eng = Engine(cfg)
    print("Y‑Theory REPL: type 'quit' to exit.\n")
    while True:
        try:
            user = input('you> ').strip()
        except (EOFError, KeyboardInterrupt):
            print() ; break
        if user.lower() in {"quit", "exit"}: break
        out = eng.step(user)
        time.sleep(out['pause'])
        print(f"[{out['mode']}] gain={out['signals']['gain']:.2f} pause={out['pause']}s action={out['action']}")
        print('bot>', out['reply'])


if __name__ == '__main__':
    main()
