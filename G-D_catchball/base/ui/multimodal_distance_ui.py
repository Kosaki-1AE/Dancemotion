from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox


def _vec3(v):
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size < 3:
        v = np.pad(v, (0, 3 - v.size))
    return v[:3]


def distance(a, b) -> float:
    return float(np.linalg.norm(_vec3(a) - _vec3(b)))


def build_vectors_from_result(result: Dict[str, Any]) -> Dict[str, np.ndarray]:
    fusion = result["fusion_result"]
    evidences = result["evidences"]

    vectors = {
        "fused": fusion.fused_vector,
        "self": result.get("self", np.zeros(3)),
        "state": result.get("state", {}).get("vector", np.zeros(3)),
        "responsibility": result.get("responsibility_arrow", np.zeros(3)),
    }

    for name, ev in evidences.items():
        vectors[name] = ev.centroid

    return vectors


def draw_3d_vectors(ax, vectors: Dict[str, np.ndarray]):
    ax.cla()
    ax.set_title("Multimodal Meaning Space")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    markers = {
        "word": "o",
        "music": "^",
        "video": "s",
        "fused": "*",
        "self": "P",
        "state": "x",
        "responsibility": "D",
    }

    sizes = {
        "fused": 260,
        "self": 220,
        "responsibility": 180,
        "state": 160,
        "word": 140,
        "music": 140,
        "video": 140,
    }

    for name, vec in vectors.items():
        v = _vec3(vec)
        ax.scatter(
            v[0], v[1], v[2],
            s=sizes.get(name, 120),
            marker=markers.get(name, "o"),
            edgecolors="black",
            linewidths=1.0,
            alpha=0.9,
        )
        ax.text(v[0], v[1], v[2], name.upper(), fontsize=9)

    # fused中心で線を引く
    if "fused" in vectors:
        f = _vec3(vectors["fused"])
        for name, vec in vectors.items():
            if name == "fused":
                continue
            v = _vec3(vec)
            d = distance(f, v)
            ax.plot(
                [f[0], v[0]],
                [f[1], v[1]],
                [f[2], v[2]],
                linewidth=max(0.6, 2.5 / (1.0 + d)),
                alpha=0.55,
            )
            mid = (f + v) / 2.0
            ax.text(mid[0], mid[1], mid[2], f"{d:.2f}", fontsize=8)

    try:
        pts = np.array([_vec3(v) for v in vectors.values()])
        max_range = np.ptp(pts, axis=0).max()
        center = pts.mean(axis=0)
        r = max(max_range / 2, 0.5)
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_zlim(center[2] - r, center[2] + r)
    except Exception:
        pass


def build_distance_lines(vectors: Dict[str, np.ndarray]) -> list[str]:
    lines = []

    key_pairs = [
        ("fused", "self"),
        ("fused", "word"),
        ("fused", "music"),
        ("fused", "video"),
        ("self", "word"),
        ("self", "music"),
        ("self", "video"),
        ("word", "music"),
        ("word", "video"),
        ("music", "video"),
        ("responsibility", "fused"),
        ("responsibility", "self"),
    ]

    for a, b in key_pairs:
        if a in vectors and b in vectors:
            lines.append(f"{a:14s} ↔ {b:14s}: {distance(vectors[a], vectors[b]):.4f}")

    return lines


def draw_distance_panel(ax, vectors: Dict[str, np.ndarray], extra_lines=None):
    ax.cla()
    ax.axis("off")
    ax.set_title("Distance / Status")

    lines = build_distance_lines(vectors)

    if extra_lines:
        lines += [""]
        lines += list(extra_lines)

    y = 0.96
    for line in lines:
        ax.text(0.02, y, line, fontsize=9, family="monospace", va="top")
        y -= 0.055


def show_multimodal_distance_ui(result: Dict[str, Any], bridge=None):
    fig = plt.figure(figsize=(16, 9))
    ax3d = fig.add_subplot(121, projection="3d")
    axpanel = fig.add_subplot(122)

    textbox_ax = fig.add_axes([0.12, 0.03, 0.55, 0.04])
    textbox = TextBox(textbox_ax, "text ", initial="")

    button_ax = fig.add_axes([0.70, 0.03, 0.10, 0.04])
    button = Button(button_ax, "RUN")

    current_result = {"value": result}

    def redraw(res):
        vectors = build_vectors_from_result(res)
        fusion = res["fusion_result"]

        extra_lines = ["--- fusion ---"]
        extra_lines.extend(fusion.status_lines)

        draw_3d_vectors(ax3d, vectors)
        draw_distance_panel(axpanel, vectors, extra_lines=extra_lines)
        fig.canvas.draw_idle()

    def on_run(_event=None):
        if bridge is None:
            return
        text = textbox.text.strip() or "multimodal_input"
        new_result = bridge.step(label=text)
        current_result["value"] = new_result
        redraw(new_result)

    button.on_clicked(on_run)
    textbox.on_submit(lambda _text: on_run())

    redraw(result)

    plt.tight_layout()
    plt.show()