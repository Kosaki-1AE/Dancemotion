from __future__ import annotations

import csv
import math
import os
import random
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class MeaningSpace:
    def __init__(self, dim: int = 3, window: int = 3, lr: float = 0.05, seed: int = 42):
        self.dim = dim
        self.window = window
        self.lr = lr

        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.coords: Dict[int, List[float]] = {}
        self.word_count: Dict[str, int] = {}
        self.neighbor_variety: Dict[str, set[str]] = {}

        random.seed(seed)

    def get_id(self, word: str) -> int:
        if word not in self.word_to_id:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
            self.coords[idx] = [random.uniform(-0.1, 0.1) for _ in range(self.dim)]
        return self.word_to_id[word]

    def distance(self, id1: int, id2: int) -> float:
        v1 = self.coords[id1]
        v2 = self.coords[id2]
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))

    def expected_distance(self, token_distance: int) -> float:
        return 1.0 / token_distance

    def update_pair(self, id1: int, id2: int, token_distance: int) -> None:
        v1 = self.coords[id1]
        v2 = self.coords[id2]

        real_d = self.distance(id1, id2)
        exp_d = self.expected_distance(token_distance)

        if real_d < 1e-12:
            for k in range(self.dim):
                v2[k] += random.uniform(-0.001, 0.001)
            real_d = self.distance(id1, id2)

        if real_d < 1e-12:
            return

        error = real_d - exp_d
        strength = 1.0 / token_distance

        word1 = self.id_to_word[id1]
        word2 = self.id_to_word[id2]

        self.neighbor_variety.setdefault(word1, set()).add(word2)
        self.neighbor_variety.setdefault(word2, set()).add(word1)

        var1 = len(self.neighbor_variety[word1])
        var2 = len(self.neighbor_variety[word2])
        var_weight = 1.0 / math.sqrt(max(1, var1 * var2))

        freq1 = self.word_count.get(word1, 1)
        freq2 = self.word_count.get(word2, 1)
        freq_weight = 1.0 / math.sqrt(max(1, freq1 * freq2))

        step = self.lr * strength * error * freq_weight * var_weight
        direction = [(b - a) / real_d for a, b in zip(v1, v2)]

        for k in range(self.dim):
            delta = step * direction[k]
            v1[k] += delta
            v2[k] -= delta

    def process_tokens(self, tokens: Sequence[str]) -> None:
        if not tokens:
            return

        for tok in tokens:
            self.word_count[tok] = self.word_count.get(tok, 0) + 1

        token_ids = [self.get_id(tok) for tok in tokens]

        for i in range(len(token_ids)):
            for j in range(i + 1, min(len(token_ids), i + 1 + self.window)):
                id1 = token_ids[i]
                id2 = token_ids[j]

                if id1 == id2:
                    continue

                td = j - i
                self.update_pair(id1, id2, td)

        self.center_all()
        self.normalize_all()

    def normalize_all(self) -> None:
        for idx in self.coords:
            v = self.coords[idx]
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 0:
                self.coords[idx] = [x / norm for x in v]

    def center_all(self) -> None:
        n = len(self.coords)
        if n == 0:
            return

        mean = [0.0] * self.dim
        for v in self.coords.values():
            for i in range(self.dim):
                mean[i] += v[i]
        mean = [x / n for x in mean]

        for idx in self.coords:
            self.coords[idx] = [v - m for v, m in zip(self.coords[idx], mean)]

    def nearest_words(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if word not in self.word_to_id:
            return []

        id1 = self.word_to_id[word]
        results = []
        for id2 in self.coords:
            if id2 == id1:
                continue
            d = self.distance(id1, id2)
            results.append((self.id_to_word[id2], d))

        results.sort(key=lambda x: x[1])
        return results[:top_k]


def load_video_frames(
    path: str,
    sample_every_n_frames: int = 3,
    resize_to: Tuple[int, int] = (160, 90),
) -> Tuple[List[np.ndarray], np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise ValueError("対応形式は .mp4 / .avi / .mov / .mkv のみです")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frames: List[np.ndarray] = []
    times: List[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize_to)
            frames.append(frame)
            times.append(frame_idx / fps)

        frame_idx += 1

    cap.release()

    if not frames:
        raise RuntimeError("フレームを取得できませんでした")

    return frames, np.asarray(times, dtype=float)


def frame_to_feature(
    frame_rgb: np.ndarray,
    prev_gray: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    # 色ヒストグラム
    hist_r = cv2.calcHist([frame_rgb], [0], None, [16], [0, 256]).flatten()
    hist_g = cv2.calcHist([frame_rgb], [1], None, [16], [0, 256]).flatten()
    hist_b = cv2.calcHist([frame_rgb], [2], None, [16], [0, 256]).flatten()

    hist_r /= max(hist_r.sum(), 1e-8)
    hist_g /= max(hist_g.sum(), 1e-8)
    hist_b /= max(hist_b.sum(), 1e-8)

    # 輝度統計
    gray_f = gray.astype(np.float32)
    brightness_mean = np.array([gray_f.mean()], dtype=np.float32)
    brightness_std = np.array([gray_f.std()], dtype=np.float32)

    # エッジ量
    edges = cv2.Canny(gray, 80, 160)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)

    # フレーム差分（動きの粗い量）
    if prev_gray is None:
        motion = np.array([0.0], dtype=np.float32)
    else:
        diff = cv2.absdiff(gray, prev_gray)
        motion = np.array([diff.mean() / 255.0], dtype=np.float32)

    # Laplacian分散（テクスチャ/シャープさ）
    lap_var = np.array([cv2.Laplacian(gray, cv2.CV_64F).var()], dtype=np.float32)

    feature = np.concatenate([
        hist_r, hist_g, hist_b,
        brightness_mean,
        brightness_std,
        edge_density,
        motion,
        lap_var,
    ]).astype(np.float32)

    return feature, gray


def extract_video_features(frames: Sequence[np.ndarray]) -> np.ndarray:
    features: List[np.ndarray] = []
    prev_gray: np.ndarray | None = None

    for frame in frames:
        feat, prev_gray = frame_to_feature(frame, prev_gray)
        features.append(feat)

    return np.vstack(features)


def standardize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


def reduce_to_3d(features: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(features)
    return coords, pca


def build_video_tokens(
    features_std: np.ndarray,
    n_clusters: int = 8,
) -> Tuple[List[str], np.ndarray, KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_std)
    tokens = [f"video_tok_{int(label)}" for label in labels]
    return tokens, labels, kmeans


def compress_consecutive_tokens(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []

    out = [tokens[0]]
    for tok in tokens[1:]:
        if tok != out[-1]:
            out.append(tok)
    return out


def build_meaning_space_from_video_tokens(
    tokens: Sequence[str],
    window: int = 4,
    lr: float = 0.06,
) -> MeaningSpace:
    engine = MeaningSpace(dim=3, window=window, lr=lr, seed=42)
    engine.process_tokens(tokens)
    return engine


def meaning_dataframe(engine: MeaningSpace) -> pd.DataFrame:
    rows = []
    for idx, word in engine.id_to_word.items():
        coord = engine.coords[idx]
        rows.append({
            "token": word,
            "x": coord[0],
            "y": coord[1],
            "z": coord[2],
            "count": engine.word_count.get(word, 1),
            "variety": len(engine.neighbor_variety.get(word, set())),
        })
    return pd.DataFrame(rows)


def token_transition_edges(tokens: Sequence[str]) -> Dict[Tuple[str, str], int]:
    edges: Dict[Tuple[str, str], int] = {}
    for a, b in zip(tokens[:-1], tokens[1:]):
        if a == b:
            continue
        key = (a, b)
        edges[key] = edges.get(key, 0) + 1
    return edges


def save_token_summary(labels: np.ndarray, prefix: str = "video") -> None:
    uniq, cnt = np.unique(labels, return_counts=True)

    with open(f"{prefix}_token_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count"])
        for u, c in zip(uniq, cnt):
            writer.writerow([f"video_tok_{int(u)}", int(c)])


def save_token_relationships(engine: MeaningSpace, prefix: str = "video") -> None:
    with open(f"{prefix}_token_relationships.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "distance"])

        for token in engine.word_to_id.keys():
            neighbors = engine.nearest_words(token, top_k=3)
            for target, dist in neighbors:
                writer.writerow([token, target, float(dist)])


def save_transitions(tokens: Sequence[str], prefix: str = "video") -> None:
    transitions: Dict[Tuple[str, str], int] = {}

    for a, b in zip(tokens[:-1], tokens[1:]):
        if a == b:
            continue
        key = (a, b)
        transitions[key] = transitions.get(key, 0) + 1

    with open(f"{prefix}_token_transitions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "count"])

        for (a, b), c in transitions.items():
            writer.writerow([a, b, c])


def save_frame_metadata_csv(
    times: np.ndarray,
    labels: np.ndarray,
    raw_coords: np.ndarray,
    prefix: str = "video",
) -> None:
    df = pd.DataFrame({
        "frame_index": np.arange(len(times)),
        "time_sec": times,
        "token": [f"video_tok_{int(x)}" for x in labels],
        "pc1": raw_coords[:, 0],
        "pc2": raw_coords[:, 1],
        "pc3": raw_coords[:, 2],
    })
    df.to_csv(f"{prefix}_frame_metadata.csv", index=False, encoding="utf-8")


def plot_results(
    raw_coords: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    engine: MeaningSpace,
    compressed_tokens: Sequence[str],
    path: str,
) -> None:
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    # 左: raw video trajectory
    ax1.scatter(
        raw_coords[:, 0],
        raw_coords[:, 1],
        raw_coords[:, 2],
        c=labels,
        s=10,
        alpha=0.8,
    )
    ax1.plot(
        raw_coords[:, 0],
        raw_coords[:, 1],
        raw_coords[:, 2],
        alpha=0.25,
        linewidth=0.8,
    )
    ax1.set_title("Raw Video Trajectory")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")

    # 右: token meaning space
    df = meaning_dataframe(engine)
    if not df.empty:
        sizes = 100 + 60 * np.log1p(df["count"].to_numpy())
        ax2.scatter(
            df["x"],
            df["y"],
            df["z"],
            s=sizes,
            alpha=0.88,
        )

        for _, row in df.iterrows():
            ax2.text(row["x"], row["y"], row["z"], row["token"], fontsize=9)

        edges = token_transition_edges(compressed_tokens)
        token_to_xyz = {
            row["token"]: (row["x"], row["y"], row["z"])
            for _, row in df.iterrows()
        }

        for (a, b), w in edges.items():
            if a not in token_to_xyz or b not in token_to_xyz:
                continue
            xa, ya, za = token_to_xyz[a]
            xb, yb, zb = token_to_xyz[b]

            ax2.plot(
                [xa, xb],
                [ya, yb],
                [za, zb],
                alpha=min(0.9, 0.20 + 0.08 * w),
                linewidth=min(4.0, 0.8 + 0.25 * w),
            )

    ax2.set_title("Video Token Meaning Space")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    fig.suptitle(f"Video Meaning Evolution: {os.path.basename(path)}", fontsize=14)
    plt.tight_layout()
    plt.show()


def print_summary(
    path: str,
    features: np.ndarray,
    labels: np.ndarray,
    tokens: Sequence[str],
    compressed_tokens: Sequence[str],
    engine: MeaningSpace,
    pca: PCA,
) -> None:
    print("\n--- summary ---")
    print(f"file: {path}")
    print(f"frames: {len(features)}")
    print(f"feature_dim: {features.shape[1]}")
    print(f"raw token length: {len(tokens)}")
    print(f"compressed token length: {len(compressed_tokens)}")
    print(f"unique video tokens: {len(set(tokens))}")

    ratio = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {[round(float(x), 4) for x in ratio]}")

    print("\n--- token counts ---")
    uniq, cnt = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"video_tok_{int(u)}: {int(c)}")

    print("\n--- nearest tokens in meaning space ---")
    for token in sorted(engine.word_to_id.keys()):
        near = engine.nearest_words(token, top_k=3)
        print(f"{token} -> {near}")


def main() -> None:
    print("mp4 / avi / mov / mkv を入力してください")
    path = input(">>> ").strip()

    frames, times = load_video_frames(
        path,
        sample_every_n_frames=3,
        resize_to=(160, 90),
    )
    print("動画読み込み完了")
    print(f"取得フレーム数: {len(frames)}")

    features = extract_video_features(frames)
    print("特徴抽出完了:", features.shape)

    features_std = standardize_features(features)

    raw_coords, pca = reduce_to_3d(features_std)
    print("raw 3D変換完了")

    tokens, labels, _ = build_video_tokens(features_std, n_clusters=8)
    print("動画トークン化完了")

    compressed_tokens = compress_consecutive_tokens(tokens)
    print("連続重複圧縮完了")

    engine = build_meaning_space_from_video_tokens(
        compressed_tokens,
        window=4,
        lr=0.06,
    )
    print("意味空間生成完了")

    # CSV出力
    save_token_summary(labels, prefix="video")
    save_token_relationships(engine, prefix="video")
    save_transitions(compressed_tokens, prefix="video")
    save_frame_metadata_csv(times, labels, raw_coords, prefix="video")
    print("CSV出力完了")

    print_summary(
        path=path,
        features=features,
        labels=labels,
        tokens=tokens,
        compressed_tokens=compressed_tokens,
        engine=engine,
        pca=pca,
    )

    plot_results(
        raw_coords=raw_coords,
        labels=labels,
        times=times,
        engine=engine,
        compressed_tokens=compressed_tokens,
        path=path,
    )


if __name__ == "__main__":
    main()