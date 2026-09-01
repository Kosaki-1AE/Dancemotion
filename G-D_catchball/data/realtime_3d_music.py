from __future__ import annotations

import csv
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def save_token_summary(labels):
    uniq, cnt = np.unique(labels, return_counts=True)

    with open("token_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count"])

        for u, c in zip(uniq, cnt):
            writer.writerow([f"audio_tok_{int(u)}", int(c)])


def save_token_relationships(engine):
    with open("token_relationships.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "distance"])

        for token in engine.word_to_id.keys():
            neighbors = engine.nearest_words(token, top_k=3)
            for target, dist in neighbors:
                writer.writerow([token, target, dist])


def save_transitions(tokens):
    transitions = {}

    for a, b in zip(tokens[:-1], tokens[1:]):
        if a == b:
            continue
        key = (a, b)
        transitions[key] = transitions.get(key, 0) + 1

    with open("token_transitions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "count"])

        for (a, b), c in transitions.items():
            writer.writerow([a, b, c])

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

        # 完全一致 or 極小距離を安全に処理
        if real_d < 1e-12:
            for k in range(self.dim):
                v2[k] += random.uniform(-0.001, 0.001)
            real_d = self.distance(id1, id2)

        # それでも近すぎるならこの更新は捨てる
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


def load_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in {".wav", ".mp3"}:
        raise ValueError("対応形式は .wav / .mp3 のみです")

    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def extract_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_mfcc: int = 13,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    # time方向に揃える
    min_frames = min(
        mfcc.shape[1],
        centroid.shape[1],
        bandwidth.shape[1],
        rolloff.shape[1],
        zcr.shape[1],
        rms.shape[1],
        chroma.shape[1],
        contrast.shape[1],
    )

    mfcc = mfcc[:, :min_frames]
    centroid = centroid[:, :min_frames]
    bandwidth = bandwidth[:, :min_frames]
    rolloff = rolloff[:, :min_frames]
    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]
    chroma = chroma[:, :min_frames]
    contrast = contrast[:, :min_frames]

    features = np.vstack([
        mfcc,
        centroid,
        bandwidth,
        rolloff,
        zcr,
        rms,
        chroma,
        contrast,
    ]).T

    times = librosa.frames_to_time(np.arange(min_frames), sr=sr, hop_length=hop_length)

    aux = {
        "times": times,
        "rms": rms.flatten(),
        "centroid": centroid.flatten(),
        "zcr": zcr.flatten(),
    }
    return features, aux


def standardize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


def reduce_to_3d(features: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(features)
    return coords, pca


def build_audio_tokens(features_std: np.ndarray, n_clusters: int = 8) -> Tuple[List[str], np.ndarray, KMeans]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_std)
    tokens = [f"audio_tok_{int(label)}" for label in labels]
    return tokens, labels, kmeans


def compress_consecutive_tokens(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []

    out = [tokens[0]]
    for tok in tokens[1:]:
        if tok != out[-1]:
            out.append(tok)
    return out


def build_meaning_space_from_audio_tokens(
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

    # 左: raw trajectory
    scatter = ax1.scatter(
        raw_coords[:, 0],
        raw_coords[:, 1],
        raw_coords[:, 2],
        c=labels,
        s=8,
        alpha=0.75,
    )
    ax1.plot(
        raw_coords[:, 0],
        raw_coords[:, 1],
        raw_coords[:, 2],
        alpha=0.25,
        linewidth=0.8,
    )
    ax1.set_title("Raw Audio Trajectory")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")

    # 右: token meaning space
    df = meaning_dataframe(engine)

    if not df.empty:
        sizes = 80 + 50 * np.log1p(df["count"].to_numpy())
        ax2.scatter(
            df["x"],
            df["y"],
            df["z"],
            s=sizes,
            alpha=0.85,
        )

        for _, row in df.iterrows():
            ax2.text(
                row["x"],
                row["y"],
                row["z"],
                row["token"],
                fontsize=9,
            )

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

    ax2.set_title("Audio Token Meaning Space")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    fig.suptitle(f"Music Meaning Evolution: {os.path.basename(path)}", fontsize=14)
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
    print(f"unique audio tokens: {len(set(tokens))}")

    ratio = pca.explained_variance_ratio_
    print(f"PCA explained variance ratio: {[round(float(x), 4) for x in ratio]}")

    print("\n--- token counts ---")
    uniq, cnt = np.unique(labels, return_counts=True)
    for u, c in zip(uniq, cnt):
        print(f"audio_tok_{int(u)}: {int(c)}")

    print("\n--- nearest tokens in meaning space ---")
    for token in sorted(engine.word_to_id.keys()):
        near = engine.nearest_words(token, top_k=3)
        print(f"{token} -> {near}")


def main() -> None:
    print("mp3 / wav を入力してください")
    path = input(">>> ").strip()

    y, sr = load_audio(path, sr=22050)
    print("音声読み込み完了")

    features, aux = extract_features(y, sr, hop_length=512, n_mfcc=13)
    print("特徴抽出完了:", features.shape)

    features_std = standardize_features(features)

    raw_coords, pca = reduce_to_3d(features_std)
    print("raw 3D変換完了")

    tokens, labels, _ = build_audio_tokens(features_std, n_clusters=8)
    print("音トークン化完了")

    compressed_tokens = compress_consecutive_tokens(tokens)
    print("連続重複圧縮完了")

    engine = build_meaning_space_from_audio_tokens(
        compressed_tokens,
        window=4,
        lr=0.06,
    )
    print("意味空間生成完了")
    # CSV出力
    save_token_summary(labels)
    save_token_relationships(engine)
    save_transitions(compressed_tokens)

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
        times=aux["times"],
        engine=engine,
        compressed_tokens=compressed_tokens,
        path=path,
    )


if __name__ == "__main__":
    main()