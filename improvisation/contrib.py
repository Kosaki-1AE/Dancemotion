# responsibility_allow/contrib.py
import numpy as np
from typing import Tuple, Union

def split_contrib(
    out_pos: np.ndarray,
    out_neg: np.ndarray,
    *,
    center: Union[float, str] = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    pos_part: 正の“効いた分”
    neg_strength: 負の“強さ”（絶対値相当）
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
        pos_part = np.maximum(0.0, out_pos - 0.5)
        neg_strength = np.maximum(0.0, (-out_neg) - 0.5)
    else:
        pos_part = np.maximum(0.0, out_pos)
        neg_strength = np.maximum(0.0, -np.minimum(0.0, out_neg))

    return pos_part, neg_strength
