from __future__ import annotations

import numpy as np
from core.HippocampusPredictiveCortex import compute_deviation_state


def main() -> None:
    external_vectors = [
        np.array([0.8, 0.2, 0.1]),
        np.array([0.7, 0.3, 0.2]),
        np.array([0.2, 0.8, 0.6]),
    ]

    self_vector = np.array([0.5, 0.2, 0.0])
    responsibility = np.zeros(3)

    for i, x in enumerate(external_vectors, start=1):
        dev = compute_deviation_state(
            external_vector=x,
            self_vector=self_vector,
            previous_responsibility=responsibility,
        )
        responsibility = dev.responsibility_arrow

        print(f"\n--- step {i} ---")
        print("external:", np.round(dev.external_vector, 4))
        print("self:", np.round(dev.self_vector, 4))
        print("prediction_error:", np.round(dev.prediction_error, 4))
        print("prediction_error_norm:", round(dev.prediction_error_norm, 6))
        print("granularity:", round(dev.granularity, 6))
        print("meaning_alignment:", round(dev.meaning_alignment, 6))
        print("tolerance:", round(dev.tolerance, 6))
        print("subject_object_gap:", round(dev.subject_object_gap, 6))
        print("resonance:", round(dev.resonance, 6))
        print("responsibility_arrow:", np.round(dev.responsibility_arrow, 4))

        # 簡易更新: 自我を責任の矢の方向へ少し動かす。
        self_vector = 0.92 * self_vector + 0.08 * (self_vector + responsibility)


if __name__ == "__main__":
    main()
