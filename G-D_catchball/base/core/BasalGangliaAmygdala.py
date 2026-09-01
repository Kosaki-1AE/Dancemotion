from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import numpy as np

try:
    from HippocampusPredictiveCortex import compute_deviation_state
except Exception:
    from .HippocampusPredictiveCortex import compute_deviation_state


@dataclass
class CellSpec:
    cell_id: str
    layer: str
    kind: str
    inputs: List[str]
    outputs: List[str]


@dataclass
class Cell:
    spec: CellSpec
    state: Dict[str, Any] = field(default_factory=dict)
    fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]] = None

    def run(self, bus: Dict[str, Any]):
        result = self.fn(bus, self.state) or {}
        bus.update(result)
        return result


class CellSystem:
    def __init__(self):
        self.cells = []
        self.bus = {}

    def add(self, cell: Cell):
        self.cells.append(cell)

    def run(self):
        for cell in self.cells:
            cell.run(self.bus)


def vector_fn(bus, state):
    v = bus.get("semantic.vector", np.zeros(3))
    return {"vector": np.asarray(v, dtype=float)}


def eval_fn(bus, state):
    s = bus.get("state", {})
    core = np.asarray(bus.get("self", np.zeros(3)), dtype=float)
    v = np.asarray(s.get("vector", np.zeros(3)), dtype=float)
    energy = float(s.get("energy", 0.0))

    align = np.dot(v, core) / ((np.linalg.norm(v) + 1e-6) * (np.linalg.norm(core) + 1e-6))

    # -----------------------------------------------------
    # 主客ズレモデル
    # 外部 v と 自我 core のズレを、粒度・意味・許容でフィルタする。
    # responsibility_arrow は「ズレが積分されて収束した方向」。
    # -----------------------------------------------------
    prev_R = state.get("responsibility_arrow", np.zeros(3))
    dev = compute_deviation_state(
        external_vector=v,
        self_vector=core,
        previous_responsibility=prev_R,
        granularity=min(1.0, energy),
    )
    state["responsibility_arrow"] = dev.responsibility_arrow

    # 既存scoreに、共鳴度と主客ギャップを少しだけ混ぜる。
    # resonance が高いほど通りやすく、gap が大きいほど違和感が強い。
    novelty = float(bus.get("novelty", 0.0))
    conflict = float(bus.get("conflict", 0.0))
    agreement = float(bus.get("agreement", 0.0))

    score = (
        0.40 * align
        + 0.20 * (1 - energy)
        + 0.15 * dev.resonance
        - 0.10 * dev.subject_object_gap
        + 0.20 * novelty          # ← 新しさで動く
        + 0.05 * conflict         # ← 違和感も少し加速
        + 0.05 * agreement        # ← 共鳴も少し評価
    )
    history = state.setdefault("score_history", [])
    history.append(float(score))

    deviation_history = state.setdefault("deviation_history", [])
    deviation_history.append({
        "prediction_error_norm": dev.prediction_error_norm,
        "granularity": dev.granularity,
        "meaning_alignment": dev.meaning_alignment,
        "tolerance": dev.tolerance,
        "subject_object_gap": dev.subject_object_gap,
        "resonance": dev.resonance,
        "responsibility_norm": dev.responsibility_norm,
    })

    return {
        "score": float(score),
        "score_history": history,
        "deviation": dev.to_dict(),
        "deviation_history": deviation_history,
        "responsibility_arrow": dev.responsibility_arrow,
    }