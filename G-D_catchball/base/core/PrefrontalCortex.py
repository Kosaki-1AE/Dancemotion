from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
# from base.adapter import MeaningAdapter, build_attention_module
# from base.core.cell_base import (Cell, CellSpec, CellSystem, anchor_fn, eval_fn, state_fn, vector_fn)
from data.realtime_3d_word import MeaningSpace

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False


# =========================================================
# Small shared math helpers
# =========================================================
def safe_norm(x: Any) -> float:
    return float(np.linalg.norm(np.asarray(x, dtype=float)))


def cosine_sim(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =========================================================
# Bridge-side state containers
# =========================================================
@dataclass
class SemanticPacket:
    text: str
    tokens: List[str]
    semantic_vector: np.ndarray
    nearest_words: List[Tuple[str, float]] = field(default_factory=list)
    sentence_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatePacket:
    text: str
    vector: np.ndarray
    velocity: np.ndarray
    energy: float
    self_vector: np.ndarray
    score: float
    frames: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReactionPacket:
    mode: str
    target_word: Optional[str]
    target_vector: Optional[np.ndarray]
    attention_weights: Optional[np.ndarray]
    alignment: float
    certainty: float
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewModel:
    self_vector: np.ndarray
    latest_vector: np.ndarray
    trajectory: List[np.ndarray]
    reaction_target_word: Optional[str]
    reaction_target_vector: Optional[np.ndarray]
    reaction_score: float
    reaction_alignment: float
    reaction_certainty: float
    status_lines: List[str]


# =========================================================
# Main adapter
# =========================================================
class MeaningAdapter:
    """
    main.py の中間管理と、各モジュールの形式変換を担当する通訳。

    想定:
    - realtime_graph_3d.MeaningSpace を engine に渡す
    - cell_base / 独自 CellSystem を system に渡す
    - transformer.SubLayers.MultiHeadAttention を optional で渡す
    - genesys の select / verify は思想だけ軽量化してここで使う
    """

    def __init__(self, engine: Any, system: Any, attention_module: Optional[Any] = None):
        self.engine = engine
        self.system = system
        self.attention_module = attention_module

    # -----------------------------------------------------
    # Phase 1: realtime_graph_3d -> semantic packet
    # -----------------------------------------------------
    def text_to_semantic_packet(self, text: str) -> SemanticPacket:
        text = str(text).strip()
        if not text:
            return SemanticPacket(
                text="",
                tokens=[],
                semantic_vector=np.zeros(3, dtype=float),
                nearest_words=[],
                sentence_info={},
            )

        self.engine.process_text(text)
        tokens = self.engine.tokenize(text)
        vec = self.engine.sentence_vector(text)
        if vec is None:
            vec = np.zeros(3, dtype=float)
        else:
            vec = np.asarray(vec, dtype=float)

        nearest_words: List[Tuple[str, float]] = []
        if tokens:
            try:
                nearest_words = self.engine.nearest_words(tokens[-1])
            except Exception:
                nearest_words = []

        sentence_info = {}
        try:
            info = self.engine.sentence_to_color_and_sound(text)
            if info is not None:
                sentence_info = info
        except Exception:
            sentence_info = {}

        return SemanticPacket(
            text=text,
            tokens=tokens,
            semantic_vector=vec,
            nearest_words=nearest_words,
            sentence_info=sentence_info,
        )
    # -----------------------------------------------------
    # Phase 2: semantic packet -> cell/system state
    # -----------------------------------------------------
    def semantic_packet_to_state(self, packet: SemanticPacket) -> StatePacket:
        self.system.bus["text"] = packet.text
        self.system.bus["semantic.vector"] = np.asarray(packet.semantic_vector, dtype=float)
        self.system.run()

        state = self.system.bus.get("state", {})
        vector = np.asarray(state.get("vector", np.zeros(3)), dtype=float)
        velocity = np.asarray(state.get("velocity", np.zeros(3)), dtype=float)
        energy = float(state.get("energy", 0.0))
        self_vector = np.asarray(self.system.bus.get("self", np.zeros(3)), dtype=float)
        score = float(self.system.bus.get("score", 0.0))
        frames = list(self.system.bus.get("frames", []))

        return StatePacket(
            text=packet.text,
            vector=vector,
            velocity=velocity,
            energy=energy,
            self_vector=self_vector,
            score=score,
            frames=frames,
        )

    # -----------------------------------------------------
    # Phase 3: state -> attention inputs
    # -----------------------------------------------------
    def state_to_attention_inputs(self, state_packet: StatePacket) -> Dict[str, Any]:
        words: List[str] = []
        vecs: List[np.ndarray] = []

        for idx in getattr(self.engine, "id_to_word", {}):
            word = self.engine.id_to_word[idx]
            coord = np.asarray(self.engine.coords[idx], dtype=float)
            words.append(word)
            vecs.append(coord)

        if not vecs:
            memory = np.zeros((1, 3), dtype=float)
            words = [None]  # type: ignore[list-item]
        else:
            memory = np.asarray(vecs, dtype=float)

        query = np.asarray(state_packet.self_vector, dtype=float).reshape(1, 3)
        latest = np.asarray(state_packet.vector, dtype=float).reshape(1, 3)

        return {
            "query_self": query,
            "query_latest": latest,
            "memory_vectors": memory,
            "memory_words": words,
        }

    # -----------------------------------------------------
    # Phase 4: attention or fallback reaction
    # -----------------------------------------------------
    def attention_to_reaction(self, attn_inputs: Dict[str, Any], use_self_query: bool = True) -> ReactionPacket:
        query = attn_inputs["query_self"] if use_self_query else attn_inputs["query_latest"]
        memory = np.asarray(attn_inputs["memory_vectors"], dtype=float)
        memory_words = list(attn_inputs["memory_words"])

        if len(memory) == 0:
            return ReactionPacket(
                mode="empty",
                target_word=None,
                target_vector=None,
                attention_weights=None,
                alignment=0.0,
                certainty=0.0,
                score=0.0,
                meta={},
            )

        # ---- real transformer attention path ----
        if self.attention_module is not None and HAS_TORCH:
            try:
                q = torch.tensor(query[None, :, :], dtype=torch.float32)   # (1,1,3)
                k = torch.tensor(memory[None, :, :], dtype=torch.float32)  # (1,N,3)
                v = torch.tensor(memory[None, :, :], dtype=torch.float32)  # (1,N,3)

                out, attn = self.attention_module(q, k, v, mask=None)
                # MultiHeadAttention returns attn shape ~ (batch, head, len_q, len_k)
                if hasattr(attn, "detach"):
                    attn_weights = attn.detach().cpu().numpy()[0].mean(axis=0)[0]
                else:
                    attn_weights = np.ones(len(memory), dtype=float) / len(memory)

                idx = int(np.argmax(attn_weights))
                target_word = memory_words[idx]
                target_vector = memory[idx]
                alignment = cosine_sim(query[0], target_vector)
                certainty = float(np.max(attn_weights))
                score = 0.65 * alignment + 0.35 * certainty

                return ReactionPacket(
                    mode="attention",
                    target_word=target_word,
                    target_vector=np.asarray(target_vector, dtype=float),
                    attention_weights=np.asarray(attn_weights, dtype=float),
                    alignment=float(alignment),
                    certainty=float(certainty),
                    score=float(score),
                    meta={"selected_index": idx},
                )
            except Exception as e:
                # fallbackに落とす
                pass

        # ---- lightweight fallback path ----
        sims = np.asarray([cosine_sim(query[0], m) for m in memory], dtype=float)
        if len(sims) == 0:
            weights = np.asarray([1.0], dtype=float)
        else:
            exps = np.exp(sims - np.max(sims))
            weights = exps / (np.sum(exps) + 1e-9)

        idx = int(np.argmax(weights))
        target_word = memory_words[idx]
        target_vector = memory[idx]
        alignment = cosine_sim(query[0], target_vector)
        certainty = float(np.max(weights))
        score = 0.65 * alignment + 0.35 * certainty

        return ReactionPacket(
            mode="fallback_similarity",
            target_word=target_word,
            target_vector=np.asarray(target_vector, dtype=float),
            attention_weights=weights,
            alignment=float(alignment),
            certainty=float(certainty),
            score=float(score),
            meta={"selected_index": idx},
        )

    # -----------------------------------------------------
    # Phase 5: reaction -> UI model
    # -----------------------------------------------------
    def reaction_to_view_model(self, state_packet: StatePacket, reaction: ReactionPacket) -> ViewModel:
        trajectory: List[np.ndarray] = []
        for frame in state_packet.frames:
            v = np.asarray(frame.get("vector", np.zeros(3)), dtype=float)
            trajectory.append(v)

        status_lines = [
            f"states={len(state_packet.frames)}",
            f"energy={state_packet.energy:.4f}",
            f"score={state_packet.score:.4f}",
            f"reaction={reaction.mode}",
            f"target={reaction.target_word}",
            f"align={reaction.alignment:.4f}",
            f"certainty={reaction.certainty:.4f}",
        ]

        return ViewModel(
            self_vector=np.asarray(state_packet.self_vector, dtype=float),
            latest_vector=np.asarray(state_packet.vector, dtype=float),
            trajectory=trajectory,
            reaction_target_word=reaction.target_word,
            reaction_target_vector=(
                None if reaction.target_vector is None else np.asarray(reaction.target_vector, dtype=float)
            ),
            reaction_score=float(reaction.score),
            reaction_alignment=float(reaction.alignment),
            reaction_certainty=float(reaction.certainty),
            status_lines=status_lines,
        )

    # -----------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------
    def step(self, text: str, use_self_query: bool = True) -> Tuple[SemanticPacket, StatePacket, ReactionPacket, ViewModel]:
        semantic_packet = self.text_to_semantic_packet(text)
        state_packet = self.semantic_packet_to_state(semantic_packet)
        attn_inputs = self.state_to_attention_inputs(state_packet)
        reaction = self.attention_to_reaction(attn_inputs, use_self_query=use_self_query)
        view_model = self.reaction_to_view_model(state_packet, reaction)
        return semantic_packet, state_packet, reaction, view_model


# =========================================================
# Optional helper to build real transformer attention module
# =========================================================
def build_attention_module(n_head: int = 1, d_model: int = 3, d_k: int = 3, d_v: int = 3):
    """
    transformer/SubLayers.py の MultiHeadAttention をそのまま刺したいとき用。
    例:
        from SubLayers import MultiHeadAttention
        attn = build_attention_module(...)

    外から本物を渡す運用でもOK。
    """
    try:
        from SubLayers import MultiHeadAttention
        return MultiHeadAttention(n_head=n_head, d_model=d_model, d_k=d_k, d_v=d_v, dropout=0.0)
    except Exception:
        return None
