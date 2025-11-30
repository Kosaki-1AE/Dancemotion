# -*- coding: utf-8 -*-
"""
nori_memory.py
- フレーム単位の短期履歴（リングバッファ）
- 1秒ごとのCSV追記（軽い）
- セッション間の永続メモリ(JSON)で初期値ブートストラップ
"""

import csv, json, os, time
from collections import deque
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class Tick:
    t: float
    A: float
    T: float
    I: float
    pol: str
    state: str
    still: float
    fps: float
    f_motion: float
    f_eyedir: float
    f_blink: float
    f_mouth: float
    f_tong: float

class RingStats:
    def __init__(self, maxlen=300):
        self.buf = deque(maxlen=maxlen)

    def push(self, d: dict):
        self.buf.append(d)

    def mean(self, k, default=0.0):
        if not self.buf: return default
        return float(np.mean([x[k] for x in self.buf]))

    def std(self, k, default=0.0):
        if not self.buf: return default
        return float(np.std([x[k] for x in self.buf]))

    def trend(self, k, default=0.0):
        """直近と半分過去の差で超簡易トレンド"""
        n = len(self.buf)
        if n < 8: return default
        half = n // 2
        a = np.mean([x[k] for x in list(self.buf)[-half:]])
        b = np.mean([x[k] for x in list(self.buf)[:half]])
        return float(a - b)

class CsvLogger:
    def __init__(self, path="logs", basename="nori_run", every_sec=1.0, enabled=True):
        self.enabled = enabled
        self.path = path
        os.makedirs(path, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.fname = os.path.join(path, f"{basename}-{ts}.csv")
        self._last = 0.0
        self._every = every_sec
        self._fh = None
        self._writer = None

    def maybe_write(self, tick: Tick):
        if not self.enabled: return
        now = time.time()
        if self._fh is None:
            self._fh = open(self.fname, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._fh, fieldnames=list(asdict(tick).keys()))
            self._writer.writeheader()
        if now - self._last >= self._every:
            self._writer.writerow(asdict(tick))
            self._fh.flush()
            self._last = now

    def close(self):
        try:
            if self._fh: self._fh.close()
        except: pass

class SessionMemory:
    """
    セッション間の“癖”を保持。平均A/Iやしきい値補正を保存→次回初期化に反映。
    """
    def __init__(self, json_path="memory/nori_session.json"):
        self.path = json_path
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        self.data = dict(boot_A=0.5, boot_I=0.3, enter_scale=1.0, exit_scale=1.0,
                         eps_scale=1.0, deadzone_scale=1.0, seen=0)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data.update(json.load(f))
        return self.data

    def update_from_hist(self, hist: RingStats):
        # 直近平均で軽く補正を学習
        A_m = hist.mean("A", 0.5)
        I_m = hist.mean("I", 0.3)
        fps_m = hist.mean("fps", 20.0)

        # Aが常に高め→入りにくい/出やすい（集中寄り）
        enter_scale = np.clip(0.9 + 0.4*(1.0 - A_m), 0.7, 1.2)
        exit_scale  = np.clip(0.9 + 0.4*A_m,        0.7, 1.2)

        # 意思が強い→探索ノイズ縮小
        eps_scale   = np.clip(1.2 - 0.5*I_m, 0.6, 1.2)

        # FPSが低いときは微小ノイズ無視を強める
        deadzone_scale = np.clip(1.0 + (20.0 - fps_m)/60.0, 0.8, 1.3)

        self.data.update(dict(
            boot_A=A_m, boot_I=I_m, enter_scale=float(enter_scale),
            exit_scale=float(exit_scale), eps_scale=float(eps_scale),
            deadzone_scale=float(deadzone_scale),
            seen=int(self.data.get("seen",0))+1
        ))
        return self.data

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
