from __future__ import annotations

import numpy as np

from base.core.BasalGangliaAmygdala import (Cell, CellSpec, CellSystem,
                                            anchor_fn, eval_fn, state_fn,
                                            vector_fn)
from base.core.thalamus import MultiModalToCellBridge
from base.ui.multimodal_distance_ui import show_multimodal_distance_ui
from data.merge.multimodal_integrator import MultiModalIntegrator


def build_system() -> CellSystem:
    system = CellSystem()

    system.add(Cell(
        spec=CellSpec(
            cell_id="vector",
            layer="semantic",
            kind="aggregator",
            inputs=["semantic.vector"],
            outputs=["vector"],
        ),
        state={},
        fn=vector_fn,
    ))

    system.add(Cell(
        spec=CellSpec(
            cell_id="state",
            layer="state",
            kind="transformer",
            inputs=["vector"],
            outputs=["state", "frames"],
        ),
        state={},
        fn=state_fn,
    ))

    system.add(Cell(
        spec=CellSpec(
            cell_id="anchor",
            layer="subject",
            kind="self",
            inputs=["state"],
            outputs=["self"],
        ),
        state={},
        fn=anchor_fn,
    ))

    system.add(Cell(
        spec=CellSpec(
            cell_id="eval",
            layer="evaluation",
            kind="evaluator",
            inputs=["state", "self"],
            outputs=["score", "score_history"],
        ),
        state={},
        fn=eval_fn,
    ))

    return system


def main() -> None:
    system = build_system()

    integrator = MultiModalIntegrator(
        word_dir=".",
        music_dir=".",
        video_dir=".",
    )

    bridge = MultiModalToCellBridge(
        system=system,
        integrator=integrator,
    )

    result = bridge.step(label="initial")
    show_multimodal_distance_ui(result, bridge=bridge)

    print("\n=== MULTIMODAL → CELL RESULT ===")
    print("fused_vector:", np.round(result["fused_vector"], 4))
    print("self:", np.round(result["self"], 4))
    print("score:", round(float(result["score"]), 6))
    print("responsibility_arrow:", np.round(result["responsibility_arrow"], 4))

    dev = result["deviation"]
    if dev:
        print("\n--- deviation ---")
        print("prediction_error_norm:", round(float(dev.get("prediction_error_norm", 0.0)), 6))
        print("granularity:", round(float(dev.get("granularity", 0.0)), 6))
        print("meaning_alignment:", round(float(dev.get("meaning_alignment", 0.0)), 6))
        print("tolerance:", round(float(dev.get("tolerance", 0.0)), 6))
        print("subject_object_gap:", round(float(dev.get("subject_object_gap", 0.0)), 6))
        print("resonance:", round(float(dev.get("resonance", 0.0)), 6))

    fusion = result["fusion_result"]
    print("\n--- fusion ---")
    for line in fusion.status_lines:
        print(line)


if __name__ == "__main__":
    main()