from __future__ import annotations

"""
layer追加用の別ファイル。
今は“足場”だけまとめてある。

目的:
- word: 和語 / 漢語 / 外来語
- music: 音色 / リズム / 構造
- video: 見た目 / 動き / 構造

を token ごとに score 化して *_token_layers.csv を出せる形へ持っていく。

このファイルはまだ既存スクリプトを書き換えずに、
layerの考え方を1箇所にまとめるための土台。
"""

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


# =========================================================
# Word layer
# =========================================================
def guess_japanese_word_layer(token: str) -> Dict[str, float]:
    """
    超簡易ヒューリスティック版。
    後で辞書ベースに差し替える前提。
    """
    token = str(token)

    has_katakana = any("ァ" <= ch <= "ヶ" or ch == "ー" for ch in token)
    has_kanji = any("\u4e00" <= ch <= "\u9fff" for ch in token)
    has_hiragana = any("ぁ" <= ch <= "ゖ" for ch in token)

    wago = 0.0
    kango = 0.0
    gairaigo = 0.0

    if has_katakana:
        gairaigo += 1.0
    if has_kanji and not has_hiragana:
        kango += 1.0
    if has_hiragana:
        wago += 1.0
    if has_kanji and has_hiragana:
        wago += 0.5
        kango += 0.5

    total = wago + kango + gairaigo
    if total <= 0:
        return {"wago": 1/3, "kango": 1/3, "gairaigo": 1/3, "dominant_layer": "unknown"}

    wago /= total
    kango /= total
    gairaigo /= total

    dominant = max(
        [("wago", wago), ("kango", kango), ("gairaigo", gairaigo)],
        key=lambda x: x[1]
    )[0]

    return {
        "wago": wago,
        "kango": kango,
        "gairaigo": gairaigo,
        "dominant_layer": dominant,
    }


def build_word_layers_csv(
    summary_csv: str = "word_token_summary.csv",
    out_csv: str = "word_token_layers.csv"
) -> None:
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(summary_csv)

    df = pd.read_csv(summary_csv)
    rows: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        token = str(row["token"])
        info = guess_japanese_word_layer(token)
        rows.append({
            "token": token,
            "wago_score": info["wago"],
            "kango_score": info["kango"],
            "gairaigo_score": info["gairaigo"],
            "dominant_layer": info["dominant_layer"],
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


# =========================================================
# Music layer
# =========================================================
def build_music_layers_csv(
    summary_csv: str = "token_summary.csv",
    transitions_csv: str = "token_transitions.csv",
    relationships_csv: str = "token_relationships.csv",
    out_csv: str = "audio_token_layers.csv",
) -> None:
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(summary_csv)

    summary = pd.read_csv(summary_csv)
    transitions = pd.read_csv(transitions_csv) if os.path.exists(transitions_csv) else pd.DataFrame()
    relationships = pd.read_csv(relationships_csv) if os.path.exists(relationships_csv) else pd.DataFrame()

    rows: List[Dict[str, object]] = []

    for _, row in summary.iterrows():
        token = str(row["token"])
        count = float(row.get("count", 0.0))

        transition_count = 0.0
        if not transitions.empty:
            transition_count = float(
                transitions.loc[
                    (transitions["source"] == token) | (transitions["target"] == token),
                    "count"
                ].sum()
            )

        relation_count = 0.0
        if not relationships.empty:
            relation_count = float(
                ((relationships["source"] == token) | (relationships["target"] == token)).sum()
            )

        # 暫定スコア:
        timbre_score = min(1.0, 0.15 + 0.08 * relation_count)
        rhythm_score = min(1.0, 0.15 + 0.05 * transition_count)
        structure_score = min(1.0, 0.15 + 0.03 * count)

        dominant = max(
            [("timbre", timbre_score), ("rhythm", rhythm_score), ("structure", structure_score)],
            key=lambda x: x[1]
        )[0]

        rows.append({
            "token": token,
            "timbre_score": timbre_score,
            "rhythm_score": rhythm_score,
            "structure_score": structure_score,
            "dominant_layer": dominant,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


# =========================================================
# Video layer
# =========================================================
def build_video_layers_csv(
    summary_csv: str = "video_token_summary.csv",
    frame_csv: str = "video_frame_metadata.csv",
    transitions_csv: str = "video_token_transitions.csv",
    out_csv: str = "video_token_layers.csv",
) -> None:
    if not os.path.exists(summary_csv):
        raise FileNotFoundError(summary_csv)

    summary = pd.read_csv(summary_csv)
    frame_df = pd.read_csv(frame_csv) if os.path.exists(frame_csv) else pd.DataFrame()
    transitions = pd.read_csv(transitions_csv) if os.path.exists(transitions_csv) else pd.DataFrame()

    rows: List[Dict[str, object]] = []

    for _, row in summary.iterrows():
        token = str(row["token"])
        count = float(row.get("count", 0.0))

        appearance_score = min(1.0, 0.1 + 0.02 * count)

        motion_score = 0.2
        if not frame_df.empty and "token" in frame_df.columns and {"pc1", "pc2", "pc3"}.issubset(frame_df.columns):
            sub = frame_df[frame_df["token"] == token]
            if len(sub) > 1:
                xyz = sub[["pc1", "pc2", "pc3"]].to_numpy(dtype=float)
                diffs = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
                motion_score = min(1.0, 0.15 + float(np.mean(diffs)))

        structure_score = 0.2
        if not transitions.empty:
            structure_score = min(
                1.0,
                0.15 + 0.04 * float(
                    transitions.loc[
                        (transitions["source"] == token) | (transitions["target"] == token),
                        "count"
                    ].sum()
                )
            )

        dominant = max(
            [("appearance", appearance_score), ("motion", motion_score), ("structure", structure_score)],
            key=lambda x: x[1]
        )[0]

        rows.append({
            "token": token,
            "appearance_score": appearance_score,
            "motion_score": motion_score,
            "structure_score": structure_score,
            "dominant_layer": dominant,
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


def main() -> None:
    print("layer scaffold fileです。")
    print("必要に応じて以下を実行してください:")
    print(" - build_word_layers_csv()")
    print(" - build_music_layers_csv()")
    print(" - build_video_layers_csv()")


if __name__ == "__main__":
    main()
