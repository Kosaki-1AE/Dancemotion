# responsibility_allow/contrib.py
from typing import Tuple, Union

import numpy as np


def split_contrib(
    out_pos: np.ndarray,
    out_neg: np.ndarray,
    *,
    center: Union[float, str] = "auto",
    mode: str = "separate",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    "separate"=効き目/強さ区別, "strength"=両方強さ
    pos_part / neg_strength の解釈切り替え:
    - mode="separate": 正=効いた分, 負=強さ
    - mode="strength": 正=強さ, 負=強さ（対称的に扱う）

    center="auto" のとき、確率っぽければ 0.5、それ以外は 0.0 を基準に。
    """
    if center == "auto":
        is_probish = (
            out_pos.min() >= 0.0
            and out_pos.max() <= 1.0
            and out_neg.max() <= 0.0
        )
        center = 0.5 if is_probish else 0.0

    if center == 0.5:
        pos_raw = out_pos - 0.5
        neg_raw = (-out_neg) - 0.5
    else:
        pos_raw = out_pos
        neg_raw = -np.minimum(0.0, out_neg)

    if mode == "separate":
        pos_part = np.maximum(0.0, pos_raw)        # 正=効き目
        neg_strength = np.maximum(0.0, neg_raw)   # 負=強さ
    elif mode == "strength":
        pos_part = np.abs(pos_raw)                 # 正=強さ
        neg_strength = np.abs(neg_raw)             # 負=強さ
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return pos_part, neg_strength
