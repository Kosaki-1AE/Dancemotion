#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal MNIST-style extractor + multi-clustering analyzer

概要:
- 入力: zip / csv / tsv / json / jsonl / parquet / npy / npz / xlsx / ディレクトリ
- 出力:
    1. 任意データを数値特徴へ変換
    2. 各サンプルを 28x28 の MNIST 風ベクトルへ統一
    3. 5〜10 個のクラスタリングを並列実行
    4. 各クラスタの「特徴の山 (feature peaks)」を抽出
    5. 解釈テキストを自動生成
    6. 代表サンプル画像や CSV / JSON レポートを保存

使い方例:
    python universal_mnist_cluster.py --input ./data.zip --output ./out --clusters 8 --runs 8
    python universal_mnist_cluster.py --input ./my_folder --output ./out --clusters 6 --runs 5

注意:
- 基本は「数値化できるデータ」を対象にしている。
- 文字列列は自動で factorize / hash 特徴化。
- 高品質な意味解釈は元データの質に依存するので、ここでは汎用ヒューリスティクスで解釈を付与する。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -----------------------------
# Utility
# -----------------------------

SUPPORTED_EXTS = {
    ".csv", ".tsv", ".txt", ".json", ".jsonl", ".parquet",
    ".npy", ".npz", ".xlsx", ".xls"
}


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def is_probably_numeric_series(s: pd.Series) -> bool:
    try:
        converted = pd.to_numeric(s.dropna(), errors="coerce")
        if len(converted) == 0:
            return False
        return converted.notna().mean() >= 0.8
    except Exception:
        return False


def factorize_with_unknown(s: pd.Series) -> pd.Series:
    values, _ = pd.factorize(s.astype(str).fillna("__nan__"))
    return pd.Series(values, index=s.index)


def hash_text_to_vector(text: str, dim: int = 16) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+", str(text).lower())
    if not tokens:
        return vec
    for tok in tokens:
        idx = stable_hash(tok) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def resize_1d_to_784(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros(784, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=arr.size)
    x_new = np.linspace(0.0, 1.0, num=784)
    out = np.interp(x_new, x_old, arr).astype(np.float32)
    return out


def vector_to_image28(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    if vec.size != 784:
        vec = resize_1d_to_784(vec)
    img = vec.reshape(28, 28)
    mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
    if not np.isfinite(mn) or not np.isfinite(mx) or math.isclose(mx, mn):
        return np.zeros((28, 28), dtype=np.float32)
    img = (img - mn) / (mx - mn)
    return img.astype(np.float32)


def save_image_grid(images: List[np.ndarray], path: Path, title: str = "") -> None:
    if plt is None or len(images) == 0:
        return
    n = len(images)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))
    if title:
        fig.suptitle(title)
    for i, img in enumerate(images, start=1):
        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Data loading
# -----------------------------

def extract_zip_to_temp(zip_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="universal_mnist_cluster_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)
    return temp_dir


def list_candidate_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".zip":
            extracted = extract_zip_to_temp(input_path)
            return [p for p in extracted.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
        return [input_path]
    if input_path.is_dir():
        return [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    raise FileNotFoundError(f"入力が見つからない: {input_path}")


def load_dataframe_from_file(path: Path) -> Optional[pd.DataFrame]:
    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        if ext == ".txt":
            # txt は csv っぽさを推定して読み込む。難しければ 1 列として扱う
            try:
                return pd.read_csv(path, sep=None, engine="python")
            except Exception:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [line.rstrip("\n") for line in f]
                return pd.DataFrame({"text": lines})
        if ext == ".json":
            try:
                return pd.read_json(path)
            except ValueError:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    return pd.json_normalize(obj)
                if isinstance(obj, dict):
                    return pd.json_normalize(obj)
                return pd.DataFrame({"value": [str(obj)]})
        if ext == ".jsonl":
            return pd.read_json(path, lines=True)
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".xlsx" or ext == ".xls":
            sheets = pd.read_excel(path, sheet_name=None)
            frames = []
            for sheet_name, df in sheets.items():
                df = df.copy()
                df["__sheet__"] = sheet_name
                frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else None
        if ext == ".npy":
            arr = np.load(path, allow_pickle=True)
            if arr.ndim == 1:
                return pd.DataFrame({"value": arr})
            return pd.DataFrame(arr)
        if ext == ".npz":
            npz = np.load(path, allow_pickle=True)
            frames = []
            for key in npz.files:
                arr = npz[key]
                if arr.ndim == 1:
                    df = pd.DataFrame({f"{key}_value": arr})
                else:
                    df = pd.DataFrame(arr)
                    df.columns = [f"{key}_{c}" for c in df.columns]
                frames.append(df.reset_index(drop=True))
            if not frames:
                return None
            max_len = max(len(df) for df in frames)
            aligned = []
            for df in frames:
                dfa = df.reindex(range(max_len))
                aligned.append(dfa)
            return pd.concat(aligned, axis=1)
    except Exception as e:
        log(f"[WARN] 読み込み失敗: {path} ({e})")
        return None
    return None


def build_master_dataframe(files: List[Path]) -> pd.DataFrame:
    frames = []
    for fp in files:
        df = load_dataframe_from_file(fp)
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        df["__source_file__"] = str(fp)
        df["__row_id__"] = np.arange(len(df))
        frames.append(df)
    if not frames:
        raise ValueError("有効なデータを読み込めなかった。CSV / JSON / NPY などを確認してほしい。")
    return pd.concat(frames, ignore_index=True)


# -----------------------------
# Feature engineering
# -----------------------------

@dataclass
class FeatureBundle:
    X_original: np.ndarray
    X_scaled: np.ndarray
    X_mnist: np.ndarray
    feature_names: List[str]
    metadata: pd.DataFrame
    semantic_df: pd.DataFrame


def dataframe_to_numeric_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    meta_cols = [c for c in df.columns if c.startswith("__")]
    meta = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    work_cols = [c for c in df.columns if c not in meta_cols]
    work = df[work_cols].copy()

    # -----------------------------
    # まず汎用数値化
    # -----------------------------
    numeric_parts = []
    numeric_feature_names: List[str] = []

    for col in work.columns:
        s = work[col]

        if pd.api.types.is_numeric_dtype(s) or is_probably_numeric_series(s):
            num = pd.to_numeric(s, errors="coerce").astype(np.float32)
            numeric_parts.append(num.to_frame(name=col))
            numeric_feature_names.append(col)
            continue

        lowered = str(col).lower()
        if "date" in lowered or "time" in lowered:
            dt = pd.to_datetime(s, errors="coerce")
            dt_df = pd.DataFrame({
                f"{col}__year": dt.dt.year,
                f"{col}__month": dt.dt.month,
                f"{col}__day": dt.dt.day,
                f"{col}__weekday": dt.dt.weekday,
                f"{col}__hour": dt.dt.hour,
            }, index=s.index)
            numeric_parts.append(dt_df.astype(np.float32))
            numeric_feature_names.extend(list(dt_df.columns))
            continue

        nunique = s.astype(str).nunique(dropna=False)
        if nunique <= 32:
            cat = factorize_with_unknown(s).astype(np.float32)
            numeric_parts.append(cat.to_frame(name=f"{col}__cat"))
            numeric_feature_names.append(f"{col}__cat")
            continue

        text_vecs = np.vstack([hash_text_to_vector(v, dim=16) for v in s.astype(str).fillna("")])
        text_cols = [f"{col}__hash_{i:02d}" for i in range(text_vecs.shape[1])]
        numeric_parts.append(pd.DataFrame(text_vecs, columns=text_cols, index=s.index))
        numeric_feature_names.extend(text_cols)

    if not numeric_parts:
        raise ValueError("数値特徴を作れなかった。少なくとも何らかの表形式データが必要。")

    numeric_df = pd.concat(numeric_parts, axis=1)
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy="median")
    X_numeric = imputer.fit_transform(numeric_df).astype(np.float32)
    numeric_df = pd.DataFrame(X_numeric, columns=numeric_df.columns, index=numeric_df.index)

    # -----------------------------
    # 軸推定
    # -----------------------------
    all_cols = list(numeric_df.columns)

    time_keywords = [
        "time", "frame", "duration", "start", "end", "seq", "tempo", "beat",
        "timestamp", "tick", "phase", "latency", "delay"
    ]
    space_keywords = [
        "x", "y", "z", "pos", "coord", "joint", "angle", "pose", "center",
        "bbox", "height", "width", "depth", "left", "right", "top", "bottom"
    ]
    force_keywords = [
        "vel", "velocity", "acc", "accel", "jerk", "momentum", "force", "energy",
        "speed", "impulse", "power"
    ]

    def pick_cols_by_keywords(columns: List[str], keywords: List[str]) -> List[str]:
        picked = []
        for c in columns:
            lc = c.lower()
            if any(k in lc for k in keywords):
                picked.append(c)
        return picked

    time_cols = pick_cols_by_keywords(all_cols, time_keywords)
    space_cols = pick_cols_by_keywords(all_cols, space_keywords)
    force_cols = pick_cols_by_keywords(all_cols, force_keywords)

    # force系がない場合、space系から差分近似で作る
    if len(force_cols) == 0 and len(space_cols) > 0:
        approx_force = {}
        for c in space_cols[: min(len(space_cols), 32)]:
            vals = numeric_df[c].values.astype(np.float32)
            v = np.diff(vals, prepend=vals[0])
            a = np.diff(v, prepend=v[0])
            approx_force[f"{c}__vel_proxy"] = np.abs(v)
            approx_force[f"{c}__acc_proxy"] = np.abs(a)
        approx_force_df = pd.DataFrame(approx_force, index=numeric_df.index)
        numeric_df = pd.concat([numeric_df, approx_force_df], axis=1)
        force_cols = list(approx_force_df.columns)

    def safe_group_stats(df_num: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
        if len(cols) == 0:
            return pd.DataFrame({
                f"{prefix}_mean": np.zeros(len(df_num), dtype=np.float32),
                f"{prefix}_std": np.zeros(len(df_num), dtype=np.float32),
                f"{prefix}_l2": np.zeros(len(df_num), dtype=np.float32),
                f"{prefix}_maxabs": np.zeros(len(df_num), dtype=np.float32),
            }, index=df_num.index)

        sub = df_num[cols].astype(np.float32)
        return pd.DataFrame({
            f"{prefix}_mean": sub.mean(axis=1),
            f"{prefix}_std": sub.std(axis=1),
            f"{prefix}_l2": np.sqrt((sub ** 2).sum(axis=1)),
            f"{prefix}_maxabs": sub.abs().max(axis=1),
        }, index=df_num.index)

    time_stats = safe_group_stats(numeric_df, time_cols, "time")
    space_stats = safe_group_stats(numeric_df, space_cols, "space")
    force_stats = safe_group_stats(numeric_df, force_cols, "force")

    semantic_df = pd.concat([time_stats, space_stats, force_stats], axis=1)

    # -----------------------------
    # 偏差化（平均との差）
    # -----------------------------
    for axis in ["time", "space", "force"]:
        base_col = f"{axis}_mean"
        semantic_df[f"{axis}_deviation"] = semantic_df[base_col] - float(semantic_df[base_col].mean())

    # -----------------------------
    # 関係偏差
    # -----------------------------
    semantic_df["time_space_relation"] = (
        semantic_df["time_deviation"] * semantic_df["space_deviation"]
    )
    semantic_df["space_force_relation"] = (
        semantic_df["space_deviation"] * semantic_df["force_deviation"]
    )
    semantic_df["time_force_relation"] = (
        semantic_df["time_deviation"] * semantic_df["force_deviation"]
    )

    semantic_df["total_deviation_l1"] = (
        semantic_df["time_deviation"].abs()
        + semantic_df["space_deviation"].abs()
        + semantic_df["force_deviation"].abs()
    )

    semantic_df["total_deviation_l2"] = np.sqrt(
        semantic_df["time_deviation"] ** 2
        + semantic_df["space_deviation"] ** 2
        + semantic_df["force_deviation"] ** 2
    )

    semantic_df["balance_t_minus_s"] = semantic_df["time_deviation"] - semantic_df["space_deviation"]
    semantic_df["balance_s_minus_f"] = semantic_df["space_deviation"] - semantic_df["force_deviation"]
    semantic_df["balance_t_minus_f"] = semantic_df["time_deviation"] - semantic_df["force_deviation"]

    semantic_df = semantic_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    # -----------------------------
    # クラスタリングに使う特徴
    # semantic主軸 + 補助的に非支配な元特徴を少し混ぜる
    # -----------------------------
    dominant_raw_cols = set(time_cols[:])  # 時間列は生のまま支配しやすいので直接使いすぎない
    assist_cols = [c for c in numeric_df.columns if c not in dominant_raw_cols][:32]

    final_df = pd.concat([semantic_df, numeric_df[assist_cols]], axis=1)
    final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    X = final_df.values.astype(np.float32)
    feature_names = list(final_df.columns)

    return X, feature_names, meta, semantic_df


def convert_to_mnist_style(X_scaled: np.ndarray) -> np.ndarray:
    # 各サンプルを 784 次元へ補間 → 28x28 に整形可能なベクトルとして保持
    mnist = np.vstack([resize_1d_to_784(row) for row in X_scaled]).astype(np.float32)

    # 全体で 0-1 正規化
    scaler = MinMaxScaler()
    mnist = scaler.fit_transform(mnist).astype(np.float32)
    return mnist


def build_feature_bundle(df: pd.DataFrame) -> FeatureBundle:
    X_orig, feature_names, meta, semantic_df = dataframe_to_numeric_features(df)
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X_orig).astype(np.float32)
    X_mnist = convert_to_mnist_style(X_scaled)
    return FeatureBundle(
        X_original=X_orig,
        X_scaled=X_scaled,
        X_mnist=X_mnist,
        feature_names=feature_names,
        metadata=meta,
        semantic_df=semantic_df,
    )


# -----------------------------
# Clustering
# -----------------------------

@dataclass
class ClusteringResult:
    name: str
    labels: np.ndarray
    n_clusters_found: int
    silhouette: Optional[float]
    calinski_harabasz: Optional[float]
    davies_bouldin: Optional[float]
    ari_vs_kmeans: Optional[float]
    interpretation: str
    cluster_summaries: List[Dict[str, Any]]


def safe_score(func, X, labels) -> Optional[float]:
    unique = np.unique(labels)
    if len(unique) < 2:
        return None
    try:
        return float(func(X, labels))
    except Exception:
        return None


def pick_representatives(X: np.ndarray, labels: np.ndarray, max_per_cluster: int = 5) -> Dict[int, List[int]]:
    reps: Dict[int, List[int]] = {}
    for clu in sorted(set(labels.tolist())):
        idx = np.where(labels == clu)[0]
        if len(idx) == 0:
            reps[clu] = []
            continue
        center = X[idx].mean(axis=0, keepdims=True)
        dists = np.linalg.norm(X[idx] - center, axis=1)
        order = idx[np.argsort(dists)[:max_per_cluster]]
        reps[clu] = order.tolist()
    return reps


def summarize_clusters(
    X_original: np.ndarray,
    X_scaled: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    global_mean = X_original.mean(axis=0)
    summaries = []
    time_idx = [i for i, f in enumerate(feature_names) if "time_" in f]
    space_idx = [i for i, f in enumerate(feature_names) if "space_" in f]
    force_idx = [i for i, f in enumerate(feature_names) if "force_" in f]
    rel_idx = [
        i for i, f in enumerate(feature_names)
        if "relation" in f or "balance_" in f or "total_deviation" in f
    ]

    def mean_abs_delta_from_idx(delta_vec: np.ndarray, idxs: List[int]) -> float:
        if len(idxs) == 0:
            return 0.0
        return float(np.mean(np.abs(delta_vec[idxs])))
    
    for clu in sorted(set(labels.tolist())):
        idx = np.where(labels == clu)[0]
        Xc = X_original[idx]
        mean = Xc.mean(axis=0)
        delta = mean - global_mean

        top_idx = np.argsort(np.abs(delta))[::-1][:10]
        peaks = []
        for j in top_idx:
            peaks.append({
                "feature": feature_names[j] if j < len(feature_names) else f"f{j}",
                "cluster_mean": float(mean[j]),
                "global_mean": float(global_mean[j]),
                "delta": float(delta[j]),
            })

        spread = float(np.mean(np.std(Xc, axis=0)))
        density = float(1.0 / (spread + 1e-8))

        summaries.append({
            "cluster_id": int(clu),
            "size": int(len(idx)),
            "ratio": float(len(idx) / len(labels)),
            "spread": spread,
            "density_like": density,
            "time_deviation_strength": mean_abs_delta_from_idx(delta, time_idx),
            "space_deviation_strength": mean_abs_delta_from_idx(delta, space_idx),
            "force_deviation_strength": mean_abs_delta_from_idx(delta, force_idx),
            "relation_deviation_strength": mean_abs_delta_from_idx(delta, rel_idx),
            "feature_peaks": peaks,
        })
    return summaries


def auto_interpret_cluster_summary(cluster_summaries: List[Dict[str, Any]]) -> str:
    if not cluster_summaries:
        return "解釈不能: クラスタ要約が空。"

    sizes = np.array([c["size"] for c in cluster_summaries], dtype=np.float32)
    spreads = np.array([c["spread"] for c in cluster_summaries], dtype=np.float32)

    largest_id = int(cluster_summaries[int(np.argmax(sizes))]["cluster_id"])
    tightest_id = int(cluster_summaries[int(np.argmin(spreads))]["cluster_id"])
    loosest_id = int(cluster_summaries[int(np.argmax(spreads))]["cluster_id"])

    def dominant_axis(summary: Dict[str, Any]) -> str:
        axis_scores = {
            "時間偏差": summary.get("time_deviation_strength", 0.0),
            "空間偏差": summary.get("space_deviation_strength", 0.0),
            "力学偏差": summary.get("force_deviation_strength", 0.0),
            "関係偏差": summary.get("relation_deviation_strength", 0.0),
        }
        return max(axis_scores, key=axis_scores.get)

    axis_texts = []
    for c in cluster_summaries[:3]:
        axis_texts.append(f"クラスタ{c['cluster_id']}は{dominant_axis(c)}優勢")

    top_words = []
    for c in cluster_summaries[:3]:
        peaks = c["feature_peaks"][:3]
        for p in peaks:
            feat = p["feature"]
            direction = "高い" if p["delta"] > 0 else "低い"
            top_words.append(f"{feat}が{direction}")

    joined = " / ".join(top_words) if top_words else "顕著特徴なし"
    axis_joined = " / ".join(axis_texts)

    return (
        f"全体解釈: 最大勢力クラスタは {largest_id}。"
        f"最もまとまりが強いクラスタは {tightest_id}、"
        f"最もばらつくクラスタは {loosest_id}。"
        f"{axis_joined}。"
        f"主要な特徴の山として {joined} が見られる。"
    )


def run_single_clustering(
    method_name: str,
    X_for_cluster: np.ndarray,
    X_original: np.ndarray,
    feature_names: List[str],
    n_clusters: int,
    random_state: int = 42,
) -> ClusteringResult:
    if method_name == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
        labels = model.fit_predict(X_for_cluster)

    elif method_name == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        labels = model.fit_predict(X_for_cluster)

    elif method_name == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X_for_cluster)

    elif method_name == "birch":
        model = Birch(n_clusters=n_clusters)
        labels = model.fit_predict(X_for_cluster)

    elif method_name == "spectral":
        # サンプル数が多いと重いので簡易近傍化
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=random_state,
            assign_labels="kmeans",
            affinity="nearest_neighbors",
            n_neighbors=min(10, max(2, len(X_for_cluster) - 1)),
        )
        labels = model.fit_predict(X_for_cluster)

    else:
        raise ValueError(f"未知の手法: {method_name}")

    labels = np.asarray(labels, dtype=int)
    n_found = len(np.unique(labels))

    sil = safe_score(silhouette_score, X_for_cluster, labels)
    chs = safe_score(calinski_harabasz_score, X_for_cluster, labels)
    dbs = safe_score(davies_bouldin_score, X_for_cluster, labels)

    km_baseline = KMeans(n_clusters=n_clusters, random_state=999, n_init=10).fit_predict(X_for_cluster)
    ari = float(adjusted_rand_score(km_baseline, labels)) if len(np.unique(labels)) >= 2 else None

    summaries = summarize_clusters(X_original, X_for_cluster, labels, feature_names)
    interpretation = auto_interpret_cluster_summary(summaries)

    return ClusteringResult(
        name=method_name,
        labels=labels,
        n_clusters_found=n_found,
        silhouette=sil,
        calinski_harabasz=chs,
        davies_bouldin=dbs,
        ari_vs_kmeans=ari,
        interpretation=interpretation,
        cluster_summaries=summaries,
    )


# -----------------------------
# Consensus / peak analysis
# -----------------------------

def build_consensus_matrix(label_runs: List[np.ndarray]) -> np.ndarray:
    n = len(label_runs[0])
    mat = np.zeros((n, n), dtype=np.float32)
    for labels in label_runs:
        same = (labels[:, None] == labels[None, :]).astype(np.float32)
        mat += same
    mat /= max(len(label_runs), 1)
    return mat


def derive_consensus_labels(consensus: np.ndarray, n_clusters: int) -> np.ndarray:
    # 合意行列をクラスタリング
    km = KMeans(n_clusters=n_clusters, random_state=123, n_init=20)
    return km.fit_predict(consensus)


def cluster_peak_frequency(cluster_summaries_list: List[List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for run_id, summaries in enumerate(cluster_summaries_list):
        for s in summaries:
            cid = s["cluster_id"]
            for peak in s["feature_peaks"]:
                rows.append({
                    "run_id": run_id,
                    "cluster_id": cid,
                    "feature": peak["feature"],
                    "delta": peak["delta"],
                    "abs_delta": abs(peak["delta"]),
                })
    if not rows:
        return pd.DataFrame(columns=["feature", "count", "mean_abs_delta"])
    df = pd.DataFrame(rows)
    out = (
        df.groupby("feature")
        .agg(count=("feature", "count"), mean_abs_delta=("abs_delta", "mean"))
        .sort_values(["count", "mean_abs_delta"], ascending=[False, False])
        .reset_index()
    )
    return out


def make_global_interpretation(
    result_table: pd.DataFrame,
    peak_freq_df: pd.DataFrame,
    consensus_summaries: List[Dict[str, Any]],
) -> str:
    best_row = result_table.sort_values(
        by=["silhouette", "calinski_harabasz", "davies_bouldin"],
        ascending=[False, False, True],
        na_position="last",
    ).iloc[0]

    top_features = peak_freq_df.head(8)["feature"].tolist()
    feat_phrase = " / ".join(top_features) if top_features else "特徴語なし"

    cluster_bits = []
    for c in consensus_summaries[:5]:
        t = c.get("time_deviation_strength", 0.0)
        s = c.get("space_deviation_strength", 0.0)
        f = c.get("force_deviation_strength", 0.0)
        r = c.get("relation_deviation_strength", 0.0)

        axis_scores = {
            "時間偏差": t,
            "空間偏差": s,
            "力学偏差": f,
            "関係偏差": r,
        }
        dominant = max(axis_scores, key=axis_scores.get)

        cluster_bits.append(
            f"クラスタ{c['cluster_id']}={dominant}"
            f"(T={t:.3f}, S={s:.3f}, F={f:.3f}, R={r:.3f})"
        )
    cluster_phrase = " | ".join(cluster_bits)

    return (
        f"総合解釈: 最良スコア傾向の手法は {best_row['method']}。"
        f"全ランで頻出した特徴の山は {feat_phrase}。"
        f"合意クラスタの要点は {cluster_phrase}。"
        f"つまり、このデータ群は時間・空間・力学の偏差と、"
        f"それらの関係偏差を軸にいくつかの安定した型へ分かれていると解釈できる。"
    )


# -----------------------------
# Main pipeline
# -----------------------------

def save_npz_mnist(X_mnist: np.ndarray, path: Path) -> None:
    np.savez_compressed(path, X_mnist=X_mnist)


def save_representatives(
    X_mnist: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    prefix: str,
    max_per_cluster: int = 5,
) -> None:
    reps = pick_representatives(X_mnist, labels, max_per_cluster=max_per_cluster)
    for clu, idxs in reps.items():
        imgs = [vector_to_image28(X_mnist[i]) for i in idxs]
        save_image_grid(imgs, output_dir / f"{prefix}_cluster_{clu}_representatives.png", title=f"{prefix} cluster {clu}")


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    n_clusters: int,
    n_runs: int,
    workers: int,
    use_pca_dim: int,
) -> None:
    safe_mkdir(output_dir)

    log("[1/6] 入力ファイル探索中...")
    files = list_candidate_files(input_path)
    if not files:
        raise ValueError("対応形式のファイルが見つからなかった。")
    log(f"  -> 対象ファイル数: {len(files)}")

    log("[2/6] データ読込中...")
    master_df = build_master_dataframe(files)
    master_df.to_csv(output_dir / "merged_raw_preview.csv", index=False)
    log(f"  -> 行数: {len(master_df)}, 列数: {len(master_df.columns)}")

    log("[3/6] 数値特徴化 + MNIST化中...")
    bundle = build_feature_bundle(master_df)
    bundle.semantic_df.to_csv(output_dir / "semantic_deviation_features.csv", index=False)
    save_npz_mnist(bundle.X_mnist, output_dir / "mnist_style_dataset.npz")

    # PCA 圧縮 (クラスタリングを安定させる用)
    pca_dim = min(use_pca_dim, bundle.X_mnist.shape[0], bundle.X_mnist.shape[1])
    pca = PCA(n_components=max(2, pca_dim), random_state=42)
    X_cluster = pca.fit_transform(bundle.X_mnist)
    pd.DataFrame(X_cluster).to_csv(output_dir / "cluster_features_pca.csv", index=False)

    if plt is not None:
        preview_imgs = [vector_to_image28(bundle.X_mnist[i]) for i in range(min(16, len(bundle.X_mnist)))]
        save_image_grid(preview_imgs, output_dir / "mnist_preview.png", title="MNIST-style preview")

    log("[4/6] 多重クラスタリング中...")
    base_methods = ["kmeans", "gmm", "agglomerative", "birch", "spectral"]

    # 5〜10 回という指定に合わせて、手法+seed違いで埋める
    jobs = []
    for i in range(n_runs):
        method = base_methods[i % len(base_methods)]
        seed = 42 + i * 13
        jobs.append((method, seed))

    results: List[ClusteringResult] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_job = {
            ex.submit(
                run_single_clustering,
                method_name=method,
                X_for_cluster=X_cluster,
                X_original=bundle.X_original,
                feature_names=bundle.feature_names,
                n_clusters=n_clusters,
                random_state=seed,
            ): (method, seed)
            for method, seed in jobs
        }
        for fut in as_completed(future_to_job):
            method, seed = future_to_job[fut]
            try:
                res = fut.result()
                results.append(res)
                log(f"  -> 完了: {method} (seed={seed})")
            except Exception as e:
                log(f"[WARN] {method} (seed={seed}) 失敗: {e}")

    if not results:
        raise RuntimeError("クラスタリングがすべて失敗した。")

    log("[5/6] 特徴の山と合意クラスタ生成中...")
    result_rows = []
    label_runs = []
    all_summaries = []

    for res in results:
        label_runs.append(res.labels)
        all_summaries.append(res.cluster_summaries)
        result_rows.append({
            "method": res.name,
            "n_clusters_found": res.n_clusters_found,
            "silhouette": res.silhouette,
            "calinski_harabasz": res.calinski_harabasz,
            "davies_bouldin": res.davies_bouldin,
            "ari_vs_kmeans": res.ari_vs_kmeans,
            "interpretation": res.interpretation,
        })

        pd.DataFrame({"label": res.labels}).to_csv(output_dir / f"labels_{res.name}.csv", index=False)
        res_dict = asdict(res)
        res_dict.pop("labels", None)  # ←ここが本質

        with open(output_dir / f"summary_{res.name}.json", "w", encoding="utf-8") as f:
            json.dump([res_dict], f, ensure_ascii=False, indent=2)

        save_representatives(bundle.X_mnist, res.labels, output_dir, prefix=res.name)

    result_table = pd.DataFrame(result_rows)
    result_table.to_csv(output_dir / "clustering_results.csv", index=False)

    consensus = build_consensus_matrix(label_runs)
    np.save(output_dir / "consensus_matrix.npy", consensus)

    consensus_labels = derive_consensus_labels(consensus, n_clusters=n_clusters)
    pd.DataFrame({"consensus_label": consensus_labels}).to_csv(output_dir / "consensus_labels.csv", index=False)

    consensus_summaries = summarize_clusters(
        bundle.X_original,
        X_cluster,
        consensus_labels,
        bundle.feature_names
    )
    with open(output_dir / "consensus_summary.json", "w", encoding="utf-8") as f:
        json.dump(consensus_summaries, f, ensure_ascii=False, indent=2)

    peak_freq_df = cluster_peak_frequency(all_summaries)
    peak_freq_df.to_csv(output_dir / "feature_peak_frequency.csv", index=False)

    save_representatives(bundle.X_mnist, consensus_labels, output_dir, prefix="consensus")

    # 異常度も一応出す
    iso = IsolationForest(random_state=42, contamination="auto")
    anomaly_score = -iso.fit_score(bundle.X_mnist) if hasattr(iso, "fit_score") else -iso.fit(bundle.X_mnist).score_samples(bundle.X_mnist)
    pd.DataFrame({"anomaly_score": anomaly_score}).to_csv(output_dir / "anomaly_scores.csv", index=False)

    global_text = make_global_interpretation(result_table, peak_freq_df, consensus_summaries)

    report = {
        "input_path": str(input_path),
        "n_files": len(files),
        "n_samples": int(bundle.X_mnist.shape[0]),
        "n_original_features": int(bundle.X_original.shape[1]),
        "n_clusters": int(n_clusters),
        "n_runs": int(len(results)),
        "global_interpretation": global_text,
        "best_methods_preview": result_table.sort_values(
            by=["silhouette", "calinski_harabasz", "davies_bouldin"],
            ascending=[False, False, True],
            na_position="last"
        ).head(5).to_dict(orient="records"),
        "top_feature_peaks": peak_freq_df.head(20).to_dict(orient="records"),
        "consensus_summary": consensus_summaries,
    }

    with open(output_dir / "final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(output_dir / "final_report.txt", "w", encoding="utf-8") as f:
        f.write(global_text + "\n\n")
        f.write("=== Top feature peaks ===\n")
        for _, row in peak_freq_df.head(20).iterrows():
            f.write(f"- {row['feature']}: count={row['count']}, mean_abs_delta={row['mean_abs_delta']:.4f}\n")
        f.write("\n=== Consensus clusters ===\n")
        for c in consensus_summaries:
            f.write(f"Cluster {c['cluster_id']} size={c['size']} ratio={c['ratio']:.3f} spread={c['spread']:.4f}\n")
            for peak in c["feature_peaks"][:5]:
                f.write(
                    f"  * {peak['feature']} | delta={peak['delta']:.4f} "
                    f"(cluster_mean={peak['cluster_mean']:.4f}, global_mean={peak['global_mean']:.4f})\n"
                )
            f.write("\n")

    meta_out = bundle.metadata.copy()
    meta_out["consensus_label"] = consensus_labels
    if len(anomaly_score) == len(meta_out):
        meta_out["anomaly_score"] = anomaly_score
    meta_out.to_csv(output_dir / "sample_metadata_with_labels.csv", index=False)

    log("[6/6] 完了")
    log(f"出力先: {output_dir.resolve()}")
    log("主要ファイル:")
    log("  - mnist_style_dataset.npz")
    log("  - clustering_results.csv")
    log("  - consensus_labels.csv")
    log("  - feature_peak_frequency.csv")
    log("  - final_report.json / final_report.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universal MNIST-style multi-clustering analyzer")
    p.add_argument("--input", required=True, help="入力ファイル or ディレクトリ or zip")
    p.add_argument("--output", required=True, help="出力ディレクトリ")
    p.add_argument("--clusters", type=int, default=8, help="想定クラスタ数")
    p.add_argument("--runs", type=int, default=8, help="クラスタリング実行数 (5〜10 推奨)")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 2), help="並列数")
    p.add_argument("--pca-dim", type=int, default=32, help="クラスタリング前の PCA 次元")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.runs < 1:
        raise ValueError("--runs は 1 以上")
    if args.clusters < 2:
        raise ValueError("--clusters は 2 以上")

    run_pipeline(
        input_path=input_path,
        output_dir=output_dir,
        n_clusters=args.clusters,
        n_runs=args.runs,
        workers=args.workers,
        use_pca_dim=args.pca_dim,
    )

def interactive_mode():
    print("=== MNIST汎用クラスタリング (対話モード) ===")

    # 入力パス
    input_path = input("データのパスを入力してください（例: ./data.zip or ./folder）: ").strip()
    if input_path == "":
        print("入力が空です")
        return

    # 出力先
    output_path = input("出力先フォルダ（Enterで ./out）: ").strip()
    if output_path == "":
        output_path = "./out"

    # クラスタ数
    try:
        clusters = input("クラスタ数（Enterで8）: ").strip()
        clusters = int(clusters) if clusters else 8
    except:
        clusters = 8

    # 実行回数
    try:
        runs = input("クラスタリング回数（Enterで8）: ").strip()
        runs = int(runs) if runs else 8
    except:
        runs = 8

    print("\n🚀 実行開始...\n")

    run_pipeline(
        input_path=Path(input_path),
        output_dir=Path(output_path),
        n_clusters=clusters,
        n_runs=runs,
        workers=max(1, os.cpu_count() // 2 if os.cpu_count() else 2),
        use_pca_dim=32,
    )

    print("\n✅ 完了！")

    # ここが本質👇
    while True:
        print("\n=== 次にやりたいこと ===")
        print("1: 結果概要を見る")
        print("2: 特徴の山を見る")
        print("3: 偏差特徴を見る")
        print("4: 別設定で再実行")
        print("0: 終了")

        choice = input("選択: ").strip()

        if choice == "1":
            try:
                with open(Path(output_path) / "final_report.txt", "r", encoding="utf-8") as f:
                    print("\n--- 概要 ---\n")
                    print(f.read())
            except:
                print("読み込み失敗")

        elif choice == "2":
            try:
                df = pd.read_csv(Path(output_path) / "feature_peak_frequency.csv")
                print("\n--- 特徴の山 TOP10 ---\n")
                print(df.head(10))
            except:
                print("読み込み失敗")

        elif choice == "3":
            try:
                df = pd.read_csv(Path(output_path) / "semantic_deviation_features.csv")
                print("\n--- 偏差特徴 TOP10 ---\n")
                print(df.head(10))
            except:
                print("読み込み失敗")
                
        elif choice == "4":
            print("\n再実行します\n")
            return interactive_mode()

        elif choice == "0":
            print("終了")
            break

        else:
            print("不明な選択")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()
