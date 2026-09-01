from __future__ import annotations

import csv
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.cluster import KMeans

try:
    import MeCab
except Exception:
    MeCab = None

plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


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
        self.tagger = None
        if MeCab is not None:
            try:
                self.tagger = MeCab.Tagger("-r /etc/mecabrc")
            except Exception:
                self.tagger = None

        self.cluster_color_names = [
            "red", "blue", "green", "yellow", "purple",
            "cyan", "orange", "magenta", "lime", "pink"
        ]

    def tokenize(self, text: str) -> List[str]:
        text = str(text).strip()
        if not text:
            return []

        if self.tagger is not None:
            parsed = self.tagger.parse(text)
            if parsed is not None:
                tokens: List[str] = []
                for line in parsed.splitlines():
                    if line == "EOS" or not line:
                        continue
                    surface = line.split("\t")[0].strip()
                    if surface:
                        tokens.append(surface)
                if tokens:
                    return tokens

        if " " in text:
            return [tok for tok in text.split() if tok]
        return [ch for ch in text if ch.strip()]

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

    def process_text(self, text: str) -> None:
        tokens = self.tokenize(text)
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

    def coord_to_rgb(self, coord: Sequence[float]) -> Tuple[int, int, int]:
        padded = list(coord[:3]) + [0.0] * max(0, 3 - len(coord[:3]))
        rgb = []
        for x in padded[:3]:
            v = int((x + 1.0) / 2.0 * 255)
            v = max(0, min(255, v))
            rgb.append(v)
        return tuple(rgb)  # type: ignore[return-value]

    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])

    def coord_to_freqs(self, coord: Sequence[float], base_freq: float = 440.0) -> List[float]:
        freqs = []
        for x in coord:
            freq = base_freq * (2 ** x)
            freqs.append(round(freq, 2))
        return freqs

    def cluster_words(self, k: int = 3) -> List[Dict[str, Any]]:
        coords = [self.coords[idx] for idx in sorted(self.id_to_word.keys())]
        words = [self.id_to_word[idx] for idx in sorted(self.id_to_word.keys())]

        if len(coords) < k:
            return []

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        results: List[Dict[str, Any]] = []
        for word, label in zip(words, labels):
            idx = self.word_to_id[word]
            coord = self.coords[idx]
            rgb = self.coord_to_rgb(coord)
            hex_color = self.rgb_to_hex(rgb)
            freqs = self.coord_to_freqs(coord)

            cluster_color_name = self.cluster_color_names[label % len(self.cluster_color_names)]

            results.append({
                "word": word,
                "cluster": int(label),
                "cluster_color_name": cluster_color_name,
                "rgb": rgb,
                "hex": hex_color,
                "freqs": freqs,
            })

        return results

    def save_to_csv(self, filename: str = "coords_full.csv") -> None:
        existing_words = set()

        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row:
                        existing_words.add(row[0])

        file_exists = os.path.exists(filename)

        clusters = {}
        cluster_data = self.cluster_words(k=3)
        for item in cluster_data:
            clusters[item["word"]] = item

        with open(filename, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            if not file_exists:
                header = (
                    ["word"]
                    + [f"x{i}" for i in range(self.dim)]
                    + ["R", "G", "B", "HEX"]
                    + [f"freq{i}" for i in range(self.dim)]
                    + ["cluster", "cluster_color"]
                )
                writer.writerow(header)

            for idx in self.coords:
                word = self.id_to_word[idx]
                if word in existing_words:
                    continue

                coord = self.coords[idx]
                rgb = self.coord_to_rgb(coord)
                hex_color = self.rgb_to_hex(rgb)
                freqs = self.coord_to_freqs(coord)

                cluster = clusters.get(word, {})
                cluster_id = cluster.get("cluster", -1)
                cluster_color = cluster.get("cluster_color_name", "none")

                row = (
                    [word]
                    + coord
                    + list(rgb)
                    + [hex_color]
                    + freqs
                    + [cluster_id, cluster_color]
                )
                writer.writerow(row)

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

    def nearest_words_from_vector(self, vector: Sequence[float], top_k: int = 5) -> List[Tuple[str, float]]:
        vec = np.asarray(vector, dtype=float)
        results: List[Tuple[str, float]] = []
        for idx, word in self.id_to_word.items():
            coord = np.asarray(self.coords[idx], dtype=float)
            d = float(np.linalg.norm(vec - coord))
            results.append((word, d))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def word_to_color_and_sound(self, word: str) -> Optional[Dict[str, Any]]:
        if word not in self.word_to_id:
            return None

        idx = self.word_to_id[word]
        coord = self.coords[idx]
        rgb = self.coord_to_rgb(coord)
        hex_color = self.rgb_to_hex(rgb)
        freqs = self.coord_to_freqs(coord)

        return {
            "word": word,
            "coord": [round(x, 4) for x in coord],
            "rgb": rgb,
            "hex": hex_color,
            "freqs": freqs,
        }

    def sentence_vector(self, text: str) -> Optional[List[float]]:
        tokens = self.tokenize(text)
        vecs = []

        for tok in tokens:
            if tok in self.word_to_id:
                idx = self.word_to_id[tok]
                vecs.append(self.coords[idx])

        if not vecs:
            return None

        return [sum(values) / len(values) for values in zip(*vecs)]

    def sentence_to_color_and_sound(self, text: str) -> Optional[Dict[str, Any]]:
        vec = self.sentence_vector(text)
        if vec is None:
            return None

        rgb = self.coord_to_rgb(vec)
        hex_color = self.rgb_to_hex(rgb)
        freqs = self.coord_to_freqs(vec)

        return {
            "text": text,
            "vector": [round(x, 4) for x in vec],
            "rgb": rgb,
            "hex": hex_color,
            "freqs": freqs,
        }

    def print_state(self) -> None:
        print("\n--- 座標 / 色 / 音 ---")
        for idx in self.id_to_word:
            word = self.id_to_word[idx]
            coord = self.coords[idx]
            rgb = self.coord_to_rgb(coord)
            hex_color = self.rgb_to_hex(rgb)
            freqs = self.coord_to_freqs(coord)

            print(
                f"{word} | "
                f"coord={[round(x, 3) for x in coord]} | "
                f"rgb={rgb} | hex={hex_color} | "
                f"freqs={freqs}"
            )

def save_word_token_summary(engine, filename="word_token_summary.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "count"])
        for token, count in sorted(engine.word_count.items()):
            writer.writerow([token, count])


def save_word_token_relationships(engine, top_k=3, filename="word_token_relationships.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "distance"])
        for token in sorted(engine.word_to_id.keys()):
            neighbors = engine.nearest_words(token, top_k=top_k)
            for target, dist in neighbors:
                writer.writerow([token, target, float(dist)])


def save_word_token_positions(engine, filename="word_token_positions.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token", "x", "y", "z"])
        for idx, token in engine.id_to_word.items():
            x, y, z = engine.coords[idx]
            writer.writerow([token, float(x), float(y), float(z)])

def base_color_from_signs(row: pd.Series) -> str:
    pos_count = sum([row["x0"] >= 0, row["x1"] >= 0, row["x2"] >= 0])
    return "tomato" if pos_count >= 2 else "royalblue"


def and_strength(row: pd.Series) -> float:
    pos_count = sum([row["x0"] >= 0, row["x1"] >= 0, row["x2"] >= 0])
    neg_count = 3 - pos_count
    return max(pos_count, neg_count) / 3.0


def distance3(a: pd.Series, b: pd.Series) -> float:
    return ((a["x0"] - b["x0"]) ** 2 + (a["x1"] - b["x1"]) ** 2 + (a["x2"] - b["x2"]) ** 2) ** 0.5


def build_knn_edges(dataframe: pd.DataFrame, k: int = 2):
    pts = dataframe.reset_index(drop=True)
    edges = set()
    for i, row_i in pts.iterrows():
        dists = []
        for j, row_j in pts.iterrows():
            if i == j:
                continue
            dists.append((distance3(row_i, row_j), j))
        dists.sort(key=lambda x: x[0])
        for _, j in dists[:k]:
            edges.add(tuple(sorted((i, j))))
    return list(edges), pts


def draw_3d_hull(ax, sub_df: pd.DataFrame, color_name: str) -> None:
    if len(sub_df) < 4:
        return

    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return

    pts = sub_df[["x0", "x1", "x2"]].to_numpy()
    try:
        hull = ConvexHull(pts)
    except Exception:
        return

    faces = [pts[simplex] for simplex in hull.simplices]
    poly = Poly3DCollection(
        faces,
        facecolor=color_name,
        edgecolor=color_name,
        linewidths=1.0,
        alpha=max(0.08, min(0.22, sub_df["strength"].mean() * 0.22)),
    )
    ax.add_collection3d(poly)


def build_dataframe_from_engine(engine: MeaningSpace) -> pd.DataFrame:
    rows = []
    for idx, word in engine.id_to_word.items():
        coord = engine.coords[idx]
        rows.append({
            "word": word,
            "x0": coord[0],
            "x1": coord[1],
            "x2": coord[2],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["base_color"] = df.apply(base_color_from_signs, axis=1)
    df["strength"] = df.apply(and_strength, axis=1)
    return df


def update_plot(ax, engine: MeaningSpace, view_model: Optional[Any] = None) -> None:
    ax.cla()
    df = build_dataframe_from_engine(engine)

    ax.set_title("Meaning Graph Realtime")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_zlabel("x2")

    if df.empty:
        plt.draw()
        plt.pause(0.01)
        return

    edges, pts = build_knn_edges(df, k=2)

    warm_group = df[df["base_color"] == "tomato"]
    cool_group = df[df["base_color"] == "royalblue"]

    draw_3d_hull(ax, warm_group, "tomato")
    draw_3d_hull(ax, cool_group, "royalblue")

    for i, j in edges:
        xi, yi, zi = pts.loc[i, ["x0", "x1", "x2"]]
        xj, yj, zj = pts.loc[j, ["x0", "x1", "x2"]]
        ax.plot([xi, xj], [yi, yj], [zi, zj], linewidth=0.8, alpha=0.35, color="gray")

    for _, row in df.iterrows():
        alpha_value = 0.25 + 0.75 * row["strength"]
        size_value = 80 + 140 * row["strength"]
        ax.scatter(
            row["x0"], row["x1"], row["x2"],
            s=size_value,
            c=row["base_color"],
            alpha=alpha_value,
            edgecolors="black",
            linewidths=0.4,
        )

    for _, row in df.iterrows():
        ax.text(row["x0"], row["x1"], row["x2"], str(row["word"]), fontsize=8)

    if view_model is not None:
        self_vec = np.asarray(getattr(view_model, "self_vector", np.zeros(3)), dtype=float)
        latest_vec = np.asarray(getattr(view_model, "latest_vector", np.zeros(3)), dtype=float)
        trajectory = [np.asarray(v, dtype=float) for v in getattr(view_model, "trajectory", [])]
        target_word = getattr(view_model, "reaction_target_word", None)
        target_vec = getattr(view_model, "reaction_target_vector", None)

        ax.scatter(self_vec[0], self_vec[1], self_vec[2], s=260, c="none", edgecolors="crimson", linewidths=2.0, marker="P")
        ax.scatter(self_vec[0], self_vec[1], self_vec[2], s=90, c="crimson", alpha=0.65, marker="P")
        ax.text(self_vec[0], self_vec[1], self_vec[2], "SELF", color="crimson", fontsize=10)

        ax.scatter(latest_vec[0], latest_vec[1], latest_vec[2], s=140, c="white", edgecolors="black", linewidths=1.2, marker="x")
        ax.text(latest_vec[0], latest_vec[1], latest_vec[2], "LATEST", color="black", fontsize=9)

        if len(trajectory) >= 2:
            xs = [v[0] for v in trajectory]
            ys = [v[1] for v in trajectory]
            zs = [v[2] for v in trajectory]
            ax.plot(xs, ys, zs, linewidth=1.2, alpha=0.55, color="black")

        if target_vec is not None:
            target_vec = np.asarray(target_vec, dtype=float)
            ax.scatter(target_vec[0], target_vec[1], target_vec[2], s=180, c="gold", edgecolors="black", linewidths=0.9, marker="*")
            if target_word is not None:
                ax.text(target_vec[0], target_vec[1], target_vec[2], f"TARGET:{target_word}", color="darkgoldenrod", fontsize=9)
            ax.plot(
                [self_vec[0], target_vec[0]],
                [self_vec[1], target_vec[1]],
                [self_vec[2], target_vec[2]],
                linewidth=1.5,
                alpha=0.85,
                color="crimson",
            )

    try:
        xr = df["x0"].max() - df["x0"].min()
        yr = df["x1"].max() - df["x1"].min()
        zr = df["x2"].max() - df["x2"].min()
        ax.set_box_aspect((max(xr, 1e-6), max(yr, 1e-6), max(zr, 1e-6)))
    except Exception:
        pass

    plt.draw()
    plt.pause(0.01)


class RealtimeGraphUI:
    def __init__(self, title: str = "Meaning Graph Realtime"):
        plt.ion()
        self.fig = plt.figure(figsize=(13, 9))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.command_box_ax = self.fig.add_axes([0.60, 0.05, 0.24, 0.05])
        self.command_box = TextBox(self.command_box_ax, "cmd ", initial="")
        self.run_button_ax = self.fig.add_axes([0.86, 0.05, 0.10, 0.05])
        self.run_button = Button(self.run_button_ax, "RUN")
        self.status_ax = self.fig.add_axes([0.74, 0.72, 0.22, 0.18])
        self.status_ax.axis("off")
        self._callback = None

        self.fig.text(
            0.02, 0.97,
            "Meaning Graph Realtime\n"
            "Warm/Cool: sign dominant, SELF: red, target: gold, latest: black",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )
        self.ax.set_title(title)

        self.run_button.on_clicked(self._on_run_clicked)
        self.command_box.on_submit(self._on_submit)

    def bind(self, callback) -> None:
        self._callback = callback

    def _on_run_clicked(self, _event) -> None:
        if self._callback is not None:
            self._callback(self.command_box.text)

    def _on_submit(self, text: str) -> None:
        if self._callback is not None:
            self._callback(text)

    def set_status(self, lines: List[str]) -> None:
        self.status_ax.cla()
        self.status_ax.axis("off")
        y = 0.95
        for line in lines:
            self.status_ax.text(0.0, y, line, fontsize=9, va="top")
            y -= 0.16

    def refresh(self, engine: MeaningSpace, view_model: Optional[Any] = None) -> None:
        update_plot(self.ax, engine, view_model=view_model)
        if view_model is not None and hasattr(view_model, "status_lines"):
            self.set_status(list(view_model.status_lines))

    def clear_command(self) -> None:
        self.command_box.set_val("")


def main() -> None:
    engine = MeaningSpace()

    plt.ion()
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    fig.text(
        0.02, 0.97,
        "Warm set: + sign dominant\n"
        "Cool set: - sign dominant\n"
        "Color intensity / size: AND consistency\n"
        "Edges: 3D k-nearest neighbor graph",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    update_plot(ax, engine)

    print("文章入力（exit / quit で終了）")
    print("入力のたびに3D集合グラフをリアルタイム更新します。")

    try:
        while True:
            text = input(">>> ").strip()
            if not text:
                continue
            if text.lower() in {"exit", "quit"}:
                break

            engine.process_text(text)
            engine.save_to_csv()
            save_word_token_summary(engine)
            save_word_token_relationships(engine, top_k=3)
            save_word_token_positions(engine)
            update_plot(ax, engine)

            engine.print_state()

            tokens = engine.tokenize(text)
            if tokens:
                target = tokens[-1]
                print("\n近い単語:", engine.nearest_words(target))
                print("\n対象単語の色と音:", engine.word_to_color_and_sound(target))

            print("\nクラスタ:")
            for item in engine.cluster_words():
                print(item)

            sentence_info = engine.sentence_to_color_and_sound(text)
            print("\n文全体の色と音:", sentence_info)
            print("-" * 60)

    except KeyboardInterrupt:
        print("\n終了")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
