from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent / "array"
OUT_DIR = Path(__file__).resolve().parent / "array"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_csv_if_exists(name: str) -> pd.DataFrame:
    path = BASE_DIR / name
    if not path.exists():
        print(f"[skip] not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize_summary(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["modality"] = modality
    out = out.rename(columns={"token": "node"})
    if "count" not in out.columns:
        out["count"] = 1
    return out[["modality", "node", "count"]]


def normalize_relationships(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["modality"] = modality
    out = out.rename(columns={"source": "src", "target": "dst"})
    if "distance" not in out.columns:
        out["distance"] = None
    return out[["modality", "src", "dst", "distance"]]


def normalize_positions(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["modality"] = modality

    if "token" in out.columns:
        out = out.rename(columns={"token": "node"})
    elif "word" in out.columns:
        out = out.rename(columns={"word": "node"})

    rename_map = {}
    if "x0" in out.columns:
        rename_map["x0"] = "x"
    if "x1" in out.columns:
        rename_map["x1"] = "y"
    if "x2" in out.columns:
        rename_map["x2"] = "z"

    out = out.rename(columns=rename_map)

    needed = ["modality", "node", "x", "y", "z"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        print(f"[skip] missing columns for {modality}: {missing}")
        return pd.DataFrame()

    return out[needed]


def main():
    # word
    word_summary = load_csv_if_exists("word_token_summary.csv")
    word_rel = load_csv_if_exists("word_token_relationships.csv")
    word_pos = load_csv_if_exists("word_token_positions.csv")

    # music
    music_summary = load_csv_if_exists("token_summary.csv")
    music_rel = load_csv_if_exists("token_relationships.csv")
    music_trans = load_csv_if_exists("token_transitions.csv")

    # video
    video_summary = load_csv_if_exists("video_token_summary.csv")
    video_rel = load_csv_if_exists("video_token_relationships.csv")
    video_trans = load_csv_if_exists("video_token_transitions.csv")
    video_frame = load_csv_if_exists("video_frame_metadata.csv")

    # 統合
    summary_all = pd.concat([
        normalize_summary(word_summary, "word"),
        normalize_summary(music_summary, "music"),
        normalize_summary(video_summary, "video"),
    ], ignore_index=True)

    relationships_all = pd.concat([
        normalize_relationships(word_rel, "word"),
        normalize_relationships(music_rel, "music"),
        normalize_relationships(video_rel, "video"),
    ], ignore_index=True)

    transitions_all = []
    if not music_trans.empty:
        t = music_trans.copy()
        t["modality"] = "music"
        transitions_all.append(t[["modality", "source", "target", "count"]])
    if not video_trans.empty:
        t = video_trans.copy()
        t["modality"] = "video"
        transitions_all.append(t[["modality", "source", "target", "count"]])

    transitions_all_df = (
        pd.concat(transitions_all, ignore_index=True)
        if transitions_all else pd.DataFrame(columns=["modality", "source", "target", "count"])
    )

    positions_all = pd.concat([
        normalize_positions(word_pos, "word"),
        normalize_positions(video_frame.rename(columns={
            "token": "node", "pc1": "x", "pc2": "y", "pc3": "z"
        }), "video_raw"),
    ], ignore_index=True)

    summary_all.to_csv(OUT_DIR / "merged_token_summary.csv", index=False, encoding="utf-8")
    relationships_all.to_csv(OUT_DIR / "merged_token_relationships.csv", index=False, encoding="utf-8")
    transitions_all_df.to_csv(OUT_DIR / "merged_token_transitions.csv", index=False, encoding="utf-8")
    positions_all.to_csv(OUT_DIR / "merged_positions.csv", index=False, encoding="utf-8")

    print("統合CSVを出力しました")
    print(" - merged_token_summary.csv")
    print(" - merged_token_relationships.csv")
    print(" - merged_token_transitions.csv")
    print(" - merged_positions.csv")


if __name__ == "__main__":
    main()