from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ModalityTables:
    modality: str
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    relationships: pd.DataFrame = field(default_factory=pd.DataFrame)
    transitions: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    layers: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class ModalityEvidence:
    modality: str
    node_count: int
    total_count: float
    relation_count: int
    transition_count: int
    density_score: float
    transition_score: float
    centrality_score: float
    rarity_score: float
    stability_score: float
    confidence: float
    centroid: np.ndarray
    spread: float
    dominant_nodes: List[Tuple[str, float]]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    modality_scores: Dict[str, float]
    normalized_weights: Dict[str, float]
    fused_vector: np.ndarray
    agreement_score: float
    conflict_score: float
    novelty_score: float
    dominant_modality: Optional[str]
    dominant_tokens: List[Tuple[str, float, str]]
    status_lines: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)

BASE_DIR = Path(__file__).resolve().parent.parent / "array"
OUT_DIR = Path(__file__).resolve().parent / "array"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _safe_read_csv(name: str) -> pd.DataFrame:
    path = BASE_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=float)))


def _entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    s = float(values.sum())
    if s <= 0.0:
        return 0.0
    p = values / s
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = _safe_norm(a)
    nb = _safe_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}
    keys = list(score_map.keys())
    arr = np.asarray([score_map[k] for k in keys], dtype=float)
    arr = np.maximum(arr, 0.0)
    s = float(arr.sum())
    if s <= 1e-12:
        n = len(keys)
        return {k: 1.0 / n for k in keys}
    return {k: float(v / s) for k, v in zip(keys, arr)}


def _top_items_from_summary(df: pd.DataFrame, top_k: int = 5) -> List[Tuple[str, float]]:
    if df.empty or "count" not in df.columns or "token" not in df.columns:
        return []
    work = df.copy()
    work["count"] = pd.to_numeric(work["count"], errors="coerce").fillna(0.0)
    work = work.sort_values("count", ascending=False).head(top_k)
    return [(str(row["token"]), float(row["count"])) for _, row in work.iterrows()]


class MultiModalIntegrator:
    """
    word / music / video のさらに一個上の統合層。
    既存CSVを読み、重みを再配分して、1つの統合結果にまとめる。
    """

    def __init__(self, word_dir: str = ".", music_dir: str = ".", video_dir: str = "."):
        self.word_dir = word_dir
        self.music_dir = music_dir
        self.video_dir = video_dir

    def load_all(self) -> Dict[str, ModalityTables]:
        word = ModalityTables(
            modality="word",
            summary=_safe_read_csv(os.path.join(self.word_dir, "word_token_summary.csv")),
            relationships=_safe_read_csv(os.path.join(self.word_dir, "word_token_relationships.csv")),
            transitions=_safe_read_csv(os.path.join(self.word_dir, "word_token_transitions.csv")),
            positions=_safe_read_csv(os.path.join(self.word_dir, "word_token_positions.csv")),
            layers=_safe_read_csv(os.path.join(self.word_dir, "word_token_layers.csv")),
        )

        music = ModalityTables(
            modality="music",
            summary=_safe_read_csv(os.path.join(self.music_dir, "token_summary.csv")),
            relationships=_safe_read_csv(os.path.join(self.music_dir, "token_relationships.csv")),
            transitions=_safe_read_csv(os.path.join(self.music_dir, "token_transitions.csv")),
            positions=_safe_read_csv(os.path.join(self.music_dir, "audio_token_positions.csv")),
            layers=_safe_read_csv(os.path.join(self.music_dir, "audio_token_layers.csv")),
        )

        video = ModalityTables(
            modality="video",
            summary=_safe_read_csv(os.path.join(self.video_dir, "video_token_summary.csv")),
            relationships=_safe_read_csv(os.path.join(self.video_dir, "video_token_relationships.csv")),
            transitions=_safe_read_csv(os.path.join(self.video_dir, "video_token_transitions.csv")),
            positions=_safe_read_csv(os.path.join(self.video_dir, "video_frame_metadata.csv")),
            layers=_safe_read_csv(os.path.join(self.video_dir, "video_token_layers.csv")),
        )

        return {"word": word, "music": music, "video": video}

    def compute_modality_evidence(self, tables: ModalityTables) -> ModalityEvidence:
        summary = tables.summary.copy()
        relationships = tables.relationships.copy()
        transitions = tables.transitions.copy()
        positions = tables.positions.copy()

        node_count = 0
        total_count = 0.0
        dominant_nodes: List[Tuple[str, float]] = []

        if not summary.empty and "token" in summary.columns:
            summary["count"] = pd.to_numeric(summary.get("count", 0), errors="coerce").fillna(0.0)
            node_count = int(len(summary))
            total_count = float(summary["count"].sum())
            dominant_nodes = _top_items_from_summary(summary, top_k=5)

        relation_count = int(len(relationships)) if not relationships.empty else 0
        transition_count = int(len(transitions)) if not transitions.empty else 0

        density_score = math.log1p(node_count + relation_count)
        transition_score = math.log1p(transition_count)

        centrality_score = 0.0
        if not relationships.empty and {"source", "target"}.issubset(relationships.columns):
            deg: Dict[str, int] = {}
            for _, row in relationships.iterrows():
                s = str(row["source"])
                t = str(row["target"])
                deg[s] = deg.get(s, 0) + 1
                deg[t] = deg.get(t, 0) + 1
            if deg:
                centrality_score = float(np.mean(list(deg.values())))

        rarity_score = 0.0
        if not summary.empty and "count" in summary.columns:
            counts = summary["count"].to_numpy(dtype=float)
            ent = _entropy(counts)
            rarity_score = ent / math.log(len(counts) + 1.0)

        centroid = np.zeros(3, dtype=float)
        spread = 0.0
        stability_score = 0.0

        if not positions.empty:
            work = positions.copy()
            rename_map = {}
            if "pc1" in work.columns:
                rename_map["pc1"] = "x"
            if "pc2" in work.columns:
                rename_map["pc2"] = "y"
            if "pc3" in work.columns:
                rename_map["pc3"] = "z"
            work = work.rename(columns=rename_map)

            if {"x", "y", "z"}.issubset(work.columns):
                xyz = work[["x", "y", "z"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
                if len(xyz) > 0:
                    centroid = xyz.mean(axis=0)
                    spread = float(np.mean(np.linalg.norm(xyz - centroid, axis=1)))
                    stability_score = 1.0 / (1.0 + spread)

        confidence = (
            0.22 * density_score
            + 0.18 * transition_score
            + 0.18 * centrality_score
            + 0.16 * rarity_score
            + 0.26 * stability_score
        )

        return ModalityEvidence(
            modality=tables.modality,
            node_count=node_count,
            total_count=total_count,
            relation_count=relation_count,
            transition_count=transition_count,
            density_score=float(density_score),
            transition_score=float(transition_score),
            centrality_score=float(centrality_score),
            rarity_score=float(rarity_score),
            stability_score=float(stability_score),
            confidence=float(confidence),
            centroid=centroid,
            spread=float(spread),
            dominant_nodes=dominant_nodes,
            meta={
                "has_layers": not tables.layers.empty,
                "has_positions": not tables.positions.empty,
            },
        )

    def fuse(self, evidences: Dict[str, ModalityEvidence]) -> FusionResult:
        if not evidences:
            return FusionResult(
                modality_scores={},
                normalized_weights={},
                fused_vector=np.zeros(3, dtype=float),
                agreement_score=0.0,
                conflict_score=0.0,
                novelty_score=0.0,
                dominant_modality=None,
                dominant_tokens=[],
                status_lines=["no evidence"],
                meta={},
            )

        modality_scores = {name: ev.confidence for name, ev in evidences.items()}
        weights = _normalize_scores(modality_scores)

        fused_vector = np.zeros(3, dtype=float)
        for name, ev in evidences.items():
            fused_vector += weights.get(name, 0.0) * ev.centroid

        modalities = list(evidences.keys())
        pair_sims: List[float] = []
        pair_dists: List[float] = []
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                a = evidences[modalities[i]].centroid
                b = evidences[modalities[j]].centroid
                pair_sims.append(_cosine(a, b))
                pair_dists.append(_safe_norm(a - b))

        agreement_score = float(np.mean(pair_sims)) if pair_sims else 0.0
        conflict_score = float(np.mean(pair_dists)) if pair_dists else 0.0
        novelty_score = float(
            0.55 * np.mean([ev.rarity_score for ev in evidences.values()]) +
            0.45 * min(1.0, conflict_score)
        )

        dominant_modality = max(weights.items(), key=lambda x: x[1])[0] if weights else None

        dominant_tokens: List[Tuple[str, float, str]] = []
        for name, ev in evidences.items():
            if ev.dominant_nodes:
                token, score = ev.dominant_nodes[0]
                dominant_tokens.append((token, score, name))
        dominant_tokens.sort(key=lambda x: x[1], reverse=True)

        status_lines = [
            f"weights={{{', '.join(f'{k}:{weights[k]:.3f}' for k in weights)}}}",
            f"agreement={agreement_score:.4f}",
            f"conflict={conflict_score:.4f}",
            f"novelty={novelty_score:.4f}",
            f"dominant_modality={dominant_modality}",
        ]
        for name, ev in evidences.items():
            status_lines.append(
                f"{name}: conf={ev.confidence:.4f}, spread={ev.spread:.4f}, nodes={ev.node_count}"
            )

        return FusionResult(
            modality_scores=modality_scores,
            normalized_weights=weights,
            fused_vector=fused_vector,
            agreement_score=agreement_score,
            conflict_score=conflict_score,
            novelty_score=novelty_score,
            dominant_modality=dominant_modality,
            dominant_tokens=dominant_tokens[:5],
            status_lines=status_lines,
            meta={
                "pair_similarities": pair_sims,
                "pair_distances": pair_dists,
            },
        )

    def export_fusion_csv(
        self,
        evidences: Dict[str, ModalityEvidence],
        result: FusionResult,
        prefix: str = "fusion",
    ) -> None:
        rows = []
        for name, ev in evidences.items():
            rows.append({
                "modality": name,
                "node_count": ev.node_count,
                "total_count": ev.total_count,
                "relation_count": ev.relation_count,
                "transition_count": ev.transition_count,
                "density_score": ev.density_score,
                "transition_score": ev.transition_score,
                "centrality_score": ev.centrality_score,
                "rarity_score": ev.rarity_score,
                "stability_score": ev.stability_score,
                "confidence": ev.confidence,
                "fusion_weight": result.normalized_weights.get(name, 0.0),
                "centroid_x": ev.centroid[0],
                "centroid_y": ev.centroid[1],
                "centroid_z": ev.centroid[2],
                "spread": ev.spread,
            })
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{prefix}_modality_scores.csv", index=False, encoding="utf-8")
        pd.DataFrame([{
            "fused_x": result.fused_vector[0],
            "fused_y": result.fused_vector[1],
            "fused_z": result.fused_vector[2],
            "agreement_score": result.agreement_score,
            "conflict_score": result.conflict_score,
            "novelty_score": result.novelty_score,
            "dominant_modality": result.dominant_modality,
        }]).to_csv(OUT_DIR / f"{prefix}_result.csv", index=False, encoding="utf-8")

        dom_rows = [{
            "token": token,
            "score": score,
            "modality": modality,
        } for token, score, modality in result.dominant_tokens]
        pd.DataFrame(dom_rows).to_csv(OUT_DIR / f"{prefix}_dominant_tokens.csv", index=False, encoding="utf-8")

    def run(self, export_prefix: str = "fusion") -> Tuple[Dict[str, ModalityEvidence], FusionResult]:
        tables = self.load_all()
        evidences = {name: self.compute_modality_evidence(tbl) for name, tbl in tables.items()}
        result = self.fuse(evidences)
        self.export_fusion_csv(evidences, result, prefix=export_prefix)
        return evidences, result


def main() -> None:
    integrator = MultiModalIntegrator(word_dir=".", music_dir=".", video_dir=".")
    evidences, result = integrator.run(export_prefix="fusion")

    print("\n--- modality evidence ---")
    for name, ev in evidences.items():
        print(
            f"{name}: confidence={ev.confidence:.4f}, "
            f"nodes={ev.node_count}, "
            f"relations={ev.relation_count}, "
            f"transitions={ev.transition_count}, "
            f"spread={ev.spread:.4f}"
        )
        if ev.dominant_nodes:
            print("  dominant:", ev.dominant_nodes[:3])

    print("\n--- fusion result ---")
    for line in result.status_lines:
        print(line)

    print("\n出力ファイル:")
    print(" - fusion_modality_scores.csv")
    print(" - fusion_result.csv")
    print(" - fusion_dominant_tokens.csv")


if __name__ == "__main__":
    main()
