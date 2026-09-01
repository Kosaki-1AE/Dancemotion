import numpy as np
from core.BasalGangliaAmygdala import (Cell, CellSpec, CellSystem, anchor_fn,
                                       eval_fn, state_fn, vector_fn)
from core.PrefrontalCortex import MeaningAdapter, build_attention_module

from data.realtime_3d_word import MeaningSpace


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
    engine = MeaningSpace()
    system = build_system()
    attention_module = build_attention_module(n_head=1, d_model=3, d_k=3, d_v=3)
    adapter = MeaningAdapter(engine=engine, system=system, attention_module=attention_module)

    print("Meaning Adapter Main")
    print("text -> MeaningSpace -> CellSystem -> attention/reaction")
    print("exit / quit で終了")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n終了")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            print("終了")
            break

        semantic_packet, state_packet, reaction, view_model = adapter.step(text)

        print("\n" + "=" * 72)
        print("text:", semantic_packet.text)
        print("tokens:", semantic_packet.tokens)
        print("semantic_vector:", np.round(semantic_packet.semantic_vector, 4))
        print("state_vector:", np.round(state_packet.vector, 4))
        print("self_vector:", np.round(state_packet.self_vector, 4))
        print("energy:", round(state_packet.energy, 6))
        print("score:", round(state_packet.score, 6))

        dev = system.bus.get("deviation", {})
        if dev:
            print("prediction_error_norm:", round(float(dev.get("prediction_error_norm", 0.0)), 6))
            print("granularity:", round(float(dev.get("granularity", 0.0)), 6))
            print("meaning_alignment:", round(float(dev.get("meaning_alignment", 0.0)), 6))
            print("tolerance:", round(float(dev.get("tolerance", 0.0)), 6))
            print("subject_object_gap:", round(float(dev.get("subject_object_gap", 0.0)), 6))
            print("resonance:", round(float(dev.get("resonance", 0.0)), 6))
            print("responsibility_arrow:", np.round(dev.get("responsibility_arrow", np.zeros(3)), 4))

        print("reaction_mode:", reaction.mode)
        print("reaction_target:", reaction.target_word)
        print("reaction_alignment:", round(reaction.alignment, 6))
        print("reaction_certainty:", round(reaction.certainty, 6))
        if reaction.attention_weights is not None:
            top_k = np.argsort(reaction.attention_weights)[::-1][:5]
            print("top_reactions:")
            memory_words = getattr(adapter.engine, "id_to_word", {})
            for idx in top_k:
                word = memory_words.get(int(idx), None)
                print(f"  {idx}: {word} -> {reaction.attention_weights[idx]:.4f}")
        print("status:")
        for line in view_model.status_lines:
            print(" ", line)


if __name__ == "__main__":
    import numpy as np
    main()
