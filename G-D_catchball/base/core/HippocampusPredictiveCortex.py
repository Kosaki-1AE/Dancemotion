from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import numpy as np

EPS = 1e-9


def _vec(x: Any, dim: int = 3) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    return arr[:dim]


def _norm(x: Any) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def _cos(a: Any, b: Any) -> float:
    a = _vec(a)
    b = _vec(b)
    na = _norm(a)
    nb = _norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class DeviationState:
    # 外部・自我・予測
    external_vector: np.ndarray
    self_vector: np.ndarray
    predicted_vector: np.ndarray

    # 本体のズレ
    prediction_error: np.ndarray
    prediction_error_norm: float

    # 粒度・意味・許容
    granularity: float
    meaning_alignment: float
    tolerance: float

    # 主観・客観
    subjective_error: np.ndarray
    objective_error: np.ndarray
    subjective_error_norm: float
    objective_error_norm: float
    subject_object_gap: float
    resonance: float

    # 責任の矢
    responsibility_arrow: np.ndarray
    responsibility_norm: float

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.copy()
        return d

def update_current_state(bus, state):
    v = np.asarray(bus.get("vector", np.zeros(3)), dtype=float)
    prev = np.asarray(state.get("prev", np.zeros(3)), dtype=float)
    vel = v - prev
    energy = float(np.linalg.norm(vel))
    state["prev"] = v

    frames = state.setdefault("frames", [])
    frames.append({
        "vector": v.copy(),
        "velocity": vel.copy(),
        "energy": energy,
        "text": bus.get("text", ""),
    })

    return {
        "state": {
            "vector": v,
            "velocity": vel,
            "energy": energy,
        },
        "frames": frames,
    }


def update_self_anchor(bus, state):
    s = bus.get("state", {})
    v = np.asarray(s.get("vector", np.zeros(3)), dtype=float)

    if "core" not in state:
        state["core"] = v
    else:
        state["core"] = 0.9 * state["core"] + 0.1 * v

    return {"self": state["core"]}

def compute_prediction_error(
    external_vector: Any,
    self_vector: Any,
    previous_responsibility: Any | None = None,
    *,
    predicted_vector: Any | None = None,
    granularity: float | None = None,
    tolerance_scale: float = 1.0,
    responsibility_decay: float = 0.90,
) -> DeviationState:
    """
    主客ズレモデルの最小実装。

    external_vector: 外部入力 x
    self_vector: 自我ベクトル / 内部モデル
    predicted_vector: 予測 x_hat。未指定なら self_vector を予測として使う。

    流れ:
      D = x - x_hat
      D_subjective = tolerance * meaning_weight * granularity * D
      D_objective = D
      responsibility_arrow = EMA(D_subjective)
    """
    x = _vec(external_vector)
    self_v = _vec(self_vector)
    x_hat = _vec(self_v if predicted_vector is None else predicted_vector)

    D = x - x_hat
    D_norm = _norm(D)

    # 粒度: 未指定なら、動きが大きいほど細かく見る。ただし 0..1 に丸める。
    if granularity is None:
        granularity_value = float(np.clip(D_norm, 0.0, 1.0))
    else:
        granularity_value = float(np.clip(granularity, 0.0, 1.0))

    # 意味: 自我と外部の向きが合うほど正。-1..1 → 0..1 に変換して重みに使う。
    alignment = _cos(x, self_v)
    meaning_weight = float((alignment + 1.0) / 2.0)

    # 許容: ズレが大きいほど通りにくい。距離のフィルタ。
    tolerance = float(1.0 / (1.0 + tolerance_scale * D_norm))

    objective_error = D
    subjective_error = granularity_value * meaning_weight * tolerance * D

    prev_R = _vec(np.zeros(3) if previous_responsibility is None else previous_responsibility)
    responsibility_arrow = responsibility_decay * prev_R + (1.0 - responsibility_decay) * subjective_error

    subjective_error_norm = _norm(subjective_error)
    objective_error_norm = _norm(objective_error)
    subject_object_gap = _norm(objective_error - subjective_error)
    resonance = alignment * tolerance

    return DeviationState(
        external_vector=x,
        self_vector=self_v,
        predicted_vector=x_hat,
        prediction_error=D,
        prediction_error_norm=D_norm,
        granularity=granularity_value,
        meaning_alignment=alignment,
        tolerance=tolerance,
        subjective_error=subjective_error,
        objective_error=objective_error,
        subjective_error_norm=subjective_error_norm,
        objective_error_norm=objective_error_norm,
        subject_object_gap=subject_object_gap,
        resonance=float(resonance),
        responsibility_arrow=responsibility_arrow,
        responsibility_norm=_norm(responsibility_arrow),
    )
