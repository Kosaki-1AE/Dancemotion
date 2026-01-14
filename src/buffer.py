from __future__ import annotations

import math
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple


# ============================================================
# 1) 最小Buffer（1層 working のみ）
# ============================================================
@dataclass
class Item:
    payload: Any
    tags: List[str] = field(default_factory=list)

    # "脳っぽさ"最小指標（要らなければ使わなくてOK）
    arousal: float = 0.0      # 0..1
    novelty: float = 0.0      # 0..1
    confidence: float = 0.5   # 0..1

    t: float = field(default_factory=lambda: time.time())
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid, "t": self.t,
            "tags": self.tags,
            "arousal": self.arousal,
            "novelty": self.novelty,
            "confidence": self.confidence,
            "payload": self.payload,
        }


class Buffer:
    """
    1層バッファ：とにかく連結できるための箱。
    - push: 何でも入る
    - view: 最近n件を見る
    - select: 目的タグで軽く引く（統合ゲートの超ミニ版）
    """
    def __init__(self, cap: int = 64):
        self.q: Deque[Item] = deque(maxlen=cap)

    def push(self, payload: Any, tags: Optional[List[str]] = None,
             arousal: float = 0.0, novelty: float = 0.0, confidence: float = 0.5) -> Item:
        it = Item(payload=payload, tags=tags or [], arousal=arousal, novelty=novelty, confidence=confidence)
        self.q.append(it)
        return it

    def view(self, n: int = 10) -> List[Item]:
        return list(self.q)[-n:]

    def select(self, goal_tags: List[str], k: int = 8, now: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        goal_tagsに近いものを軽く上位抽出
        """
        now = now or time.time()

        def score(it: Item) -> float:
            # タグ一致
            tag_hit = 0.0
            if goal_tags and it.tags:
                hit = len(set(goal_tags) & set(it.tags))
                tag_hit = hit / max(1, len(goal_tags))

            # 最近性
            age = max(0.0, now - it.t)
            recency = math.exp(-age / 30.0)  # 30秒スケール（適当に調整）

            # 中核スコア
            core = 0.35 * it.novelty + 0.35 * it.arousal + 0.30 * it.confidence
            return 0.45 * core + 0.35 * tag_hit + 0.20 * recency

        scored = [(score(it), it) for it in self.q]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        return [{"score": s, "item": it.as_dict()} for (s, it) in top]


# ============================================================
# 2) MineLife（マインスイーパ×ライフゲーム混合）内蔵エンジン
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
    """
    tick() すると内部状態が進む。
    out をそのまま Buffer.push できる形で返す。
    """
    def __init__(self, cfg: MineLifeConfig):
        self.cfg = cfg
        if cfg.seed is not None:
            random.seed(cfg.seed)

        self.t = 0
        self.alive = [[1 if random.random() < 0.25 else 0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.mine = [[1 if random.random() < cfg.mine_ratio else 0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.revealed = [[0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self.risk = [[0.0 for _ in range(cfg.w)] for _ in range(cfg.h)]
        self._update_risk()

    def _n8(self, y: int, x: int) -> List[Tuple[int, int]]:
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.cfg.h and 0 <= nx < self.cfg.w:
                    out.append((ny, nx))
        return out

    def _count(self, grid: List[List[int]], y: int, x: int) -> int:
        return sum(grid[ny][nx] for (ny, nx) in self._n8(y, x))

    def _update_risk(self) -> None:
        base = [[0.0 for _ in range(self.cfg.w)] for _ in range(self.cfg.h)]
        for y in range(self.cfg.h):
            for x in range(self.cfg.w):
                m = self._count(self.mine, y, x)
                base[y][x] = min(1.0, m / 8.0)

        new_r = [[0.0 for _ in range(self.cfg.w)] for _ in range(self.cfg.h)]
        for y in range(self.cfg.h):
            for x in range(self.cfg.w):
                neigh = self._n8(y, x)
                avg = sum(base[ny][nx] for ny, nx in neigh) / max(1, len(neigh))
                new_r[y][x] = min(1.0, base[y][x] * (1.0 - self.cfg.danger_spread) + avg * self.cfg.danger_spread)
        self.risk = new_r

    def _life_step(self) -> None:
        nxt = [[0 for _ in range(self.cfg.w)] for _ in range(self.cfg.h)]
        for y in range(self.cfg.h):
            for x in range(self.cfg.w):
                n = self._count(self.alive, y, x)
                if self.alive[y][x] == 1:
                    nxt[y][x] = 1 if n in self.cfg.survive else 0
                else:
                    nxt[y][x] = 1 if n in self.cfg.birth else 0
        self.alive = nxt

    def _auto_reveal(self) -> List[Tuple[int, int, int, int]]:
        cand = [(y, x) for y in range(self.cfg.h) for x in range(self.cfg.w) if self.revealed[y][x] == 0]
        if not cand:
            return []
        cand.sort(key=lambda p: self.risk[p[0]][p[1]])
        picks = []
        for _ in range(min(self.cfg.reveal_per_tick, len(cand))):
            if random.random() < 0.7:
                y, x = cand.pop(0)
            else:
                y, x = cand.pop(-1)
            self.revealed[y][x] = 1
            mine_hit = self.mine[y][x]
            mine_n = self._count(self.mine, y, x)
            picks.append((y, x, mine_hit, mine_n))
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
        arousal = tension

        tags = ["MineLife"]
        if tension > 0.7:
            tags += ["high_tension"]
        elif tension < 0.35:
            tags += ["calm"]
        else:
            tags += ["mid_tension"]

        if alive_ratio > 0.55:
            tags += ["dense_motion"]
        elif alive_ratio < 0.25:
            tags += ["stillness"]
        else:
            tags += ["balanced"]

        return {
            "t": self.t,
            "alive_ratio": alive_ratio,
            "revealed_ratio": revealed_ratio,
            "risk_mean": risk_mean,
            "tension": tension,
            "reveals": reveals,
            "tags": tags,
            "novelty": novelty,
            "arousal": arousal,
        }


# ============================================================
# 3) BufferとMineLifeを連結する「だけ」のノリ
# ============================================================
class BrainBufferOneFile:
    """
    - Buffer が本体（1層）
    - MineLife は内蔵で、tick() でBufferに自動投入
    - それ以外も全部 buffer.push(...) で連結
    """
    def __init__(self, buffer_cap: int = 64, mine_cfg: Optional[MineLifeConfig] = None):
        self.buf = Buffer(cap=buffer_cap)
        self.mine = MineLife(mine_cfg or MineLifeConfig())

    def push(self, payload: Any, tags: Optional[List[str]] = None,
             arousal: float = 0.0, novelty: float = 0.0, confidence: float = 0.5) -> Item:
        return self.buf.push(payload, tags=tags, arousal=arousal, novelty=novelty, confidence=confidence)

    def tick(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        MineLifeをn回回して、その都度Bufferへpush
        """
        outs = []
        for _ in range(max(0, n)):
            s = self.mine.tick()
            self.buf.push(
                payload={"MineLife": {k: s[k] for k in ["t", "alive_ratio", "revealed_ratio", "risk_mean", "tension"]},
                         "reveals": s["reveals"]},
                tags=s["tags"],
                arousal=s["arousal"],
                novelty=s["novelty"],
                confidence=0.55,
            )
            outs.append(s)
        return outs

    def select(self, goal_tags: List[str], k: int = 8) -> List[Dict[str, Any]]:
        return self.buf.select(goal_tags=goal_tags, k=k)

    def view(self, n: int = 10) -> List[Dict[str, Any]]:
        return [it.as_dict() for it in self.buf.view(n)]


# ============================================================
# 4) 実行例
# ============================================================
if __name__ == "__main__":
    brain = BrainBufferOneFile(
        buffer_cap=96,
        mine_cfg=MineLifeConfig(w=20, h=20, mine_ratio=0.10, reveal_per_tick=3, seed=42),
    )

    # 何でも連結：文章、特徴、ログ、好きに入れてOK
    brain.push("Buffer開始。MineLifeが内側で回る想定。", tags=["L", "text", "Buffer"])
    brain.push({"beat": 120, "swing": 0.18}, tags=["R", "music", "features"], confidence=0.6)

    # 世界を回してバッファへ注入
    brain.tick(n=3)

    # 目的タグで軽く取り出し（超ミニ統合）
    top = brain.select(goal_tags=["MineLife", "Buffer"], k=6)
    print("=== TOP ===")
    for r in top:
        print(f"{r['score']:.3f}", r["item"]["tags"], r["item"]["payload"])

    print("\n=== RECENT ===")
    for it in brain.view(n=6):
        print(it["tags"], it["payload"])
