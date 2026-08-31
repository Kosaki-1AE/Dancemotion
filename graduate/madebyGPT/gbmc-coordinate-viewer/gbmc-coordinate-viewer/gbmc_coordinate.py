import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_DATA = {'axes': {'x': 'G/D: Interest ↔ Attractor', 'y': 'B/S: Candidate Distribution ↔ Resonance', 'z': 'M/Q: Credit/Predictability ↔ Exploration', 'c': 'C/C: Trust/Recursion ↔ Freedom'}, 'nodes': [{'id': 'Dance', 'type': 'Reference', 'gd': 0.95, 'bs': 0.88, 'mq': 0.55, 'cc': 0.82, 'desc': 'represent / root reference'}, {'id': 'Groove', 'type': 'StreetDance', 'gd': 0.72, 'bs': 0.96, 'mq': 0.48, 'cc': 0.78, 'desc': 'resonance-heavy body/rhythm relation'}, {'id': 'Isolation', 'type': 'StreetDance', 'gd': 0.65, 'bs': 0.35, 'mq': 0.76, 'cc': 0.62, 'desc': 'module separation'}, {'id': 'Represent', 'type': 'StreetDance', 'gd': 0.92, 'bs': 0.52, 'mq': 0.78, 'cc': 0.85, 'desc': 'reference frame'}, {'id': 'Attention', 'type': 'Transformer', 'gd': 0.48, 'bs': 0.82, 'mq': 0.58, 'cc': 0.46, 'desc': 'information mediation'}, {'id': 'Query-Key Matching', 'type': 'Transformer', 'gd': 0.42, 'bs': 0.88, 'mq': 0.52, 'cc': 0.4, 'desc': 'matching relation'}, {'id': 'Embedding', 'type': 'Transformer', 'gd': 0.5, 'bs': 0.42, 'mq': 0.7, 'cc': 0.45, 'desc': 'modular representation'}, {'id': 'Pipe', 'type': 'Unix', 'gd': 0.38, 'bs': 0.72, 'mq': 0.84, 'cc': 0.52, 'desc': 'standard mediation surface'}, {'id': 'Small Tool', 'type': 'Unix', 'gd': 0.36, 'bs': 0.3, 'mq': 0.92, 'cc': 0.58, 'desc': 'isolated module'}, {'id': 'Observer', 'type': 'ControlTheory', 'gd': 0.44, 'bs': 0.44, 'mq': 0.86, 'cc': 0.36, 'desc': 'state translator'}, {'id': 'Feedback', 'type': 'ControlTheory', 'gd': 0.46, 'bs': 0.58, 'mq': 0.8, 'cc': 0.62, 'desc': 'responsibility correction'}, {'id': 'Servo Limit', 'type': 'HumanPoweredAircraft', 'gd': 0.55, 'bs': 0.3, 'mq': 0.92, 'cc': 0.35, 'desc': 'physical responsibility range'}, {'id': 'AngleToPWM', 'type': 'HumanPoweredAircraft', 'gd': 0.52, 'bs': 0.64, 'mq': 0.88, 'cc': 0.44, 'desc': 'translator from intention to actuator signal'}, {'id': 'Pico2W', 'type': 'HumanPoweredAircraft', 'gd': 0.58, 'bs': 0.45, 'mq': 0.82, 'cc': 0.5, 'desc': 'candidate generator / planner'}, {'id': 'ESP32', 'type': 'HumanPoweredAircraft', 'gd': 0.5, 'bs': 0.4, 'mq': 0.9, 'cc': 0.43, 'desc': 'execution filter / translator'}, {'id': 'Research', 'type': 'Self', 'gd': 0.86, 'bs': 0.62, 'mq': 0.72, 'cc': 0.63, 'desc': 'dance-derived structure investigation'}, {'id': 'Linux', 'type': 'Self', 'gd': 0.6, 'bs': 0.5, 'mq': 0.88, 'cc': 0.57, 'desc': 'modular system philosophy'}, {'id': 'Job Hunting', 'type': 'Self', 'gd': 0.35, 'bs': 0.7, 'mq': 0.42, 'cc': 0.28, 'desc': 'candidate distribution explosion'}, {'id': 'Human-Powered Aircraft', 'type': 'Self', 'gd': 0.74, 'bs': 0.46, 'mq': 0.86, 'cc': 0.48, 'desc': 'physical GBMC implementation field'}, {'id': 'VR', 'type': 'Self', 'gd': 0.7, 'bs': 0.75, 'mq': 0.55, 'cc': 0.66, 'desc': 'embodied visualization field'}]}


def load_data(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_DATA
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_node(nodes: list[dict], query: str) -> dict | None:
    q = query.lower()
    aliases = {
        "servolimit": "servo limit",
        "hpa": "human-powered aircraft",
        "humanpoweredaircraft": "human-powered aircraft",
        "job": "job hunting",
        "qk": "query-key matching",
    }
    q = aliases.get(q, q)

    for n in nodes:
        if n["id"].lower() == q:
            return n

    for n in nodes:
        name = n["id"].lower()
        if q in name or name in q:
            return n

    return None


def coord(n: dict) -> tuple[float, float, float, float]:
    return n["gd"], n["bs"], n["mq"], n["cc"]


def distance(a: dict, b: dict) -> float:
    ax, ay, az, ac = coord(a)
    bx, by, bz, bc = coord(b)
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2 + (ac - bc) ** 2)


def print_position(data: dict, query: str) -> None:
    nodes = data["nodes"]
    node = find_node(nodes, query)

    print("=== GBMC Coordinate Viewer ===")
    print(f"Query: {query}")

    if node is None:
        print("No position found.")
        print("Try: Dance, Groove, Attention, Pipe, Research, Linux, Job Hunting, Human-Powered Aircraft")
        return

    print("\nCurrent Position")
    print(f"  node: {node['id']}")
    print(f"  type: {node['type']}")
    print(f"  desc: {node.get('desc', '')}")
    print("\nCoordinates")
    print(f"  G/D: {node['gd']:.2f}  Interest ↔ Attractor")
    print(f"  B/S: {node['bs']:.2f}  Candidate Distribution ↔ Resonance")
    print(f"  M/Q: {node['mq']:.2f}  Credit/Predictability ↔ Exploration")
    print(f"  C/C: {node['cc']:.2f}  Trust/Recursion ↔ Freedom")

    print("\nNearest Concepts")
    ranked = sorted(
        [(distance(node, other), other) for other in nodes if other["id"] != node["id"]],
        key=lambda x: x[0],
    )
    for d, other in ranked[:6]:
        print(f"  {other['id']:<24} dist={d:.3f}  type={other['type']}")

    dance = find_node(nodes, "Dance")
    if dance and node["id"] != "Dance":
        print(f"\nDistance from Represent:Dance")
        print(f"  Δx = {distance(node, dance):.3f}")

    print("\nInterpretation")
    interpret(node)


def interpret(node: dict) -> None:
    gd, bs, mq, cc = coord(node)

    if gd > 0.75:
        print("  - Strongly connected to interest/root/attractor.")
    elif gd < 0.45:
        print("  - Far from root-interest; may feel external or obligation-like.")

    if bs > 0.75:
        print("  - Resonance/candidate activity is high.")
    elif bs < 0.40:
        print("  - More isolated/module-like than resonance-like.")

    if mq > 0.75:
        print("  - High predictability/exploration-control axis; good for engineering/system design.")
    elif mq < 0.45:
        print("  - Low predictability; meaning wall risk can increase.")

    if cc > 0.70:
        print("  - High freedom/trust-recursion feeling.")
    elif cc < 0.40:
        print("  - Low freedom/trust; can feel heavy or constrained.")


def draw(data: dict, focus: str | None, output: Path) -> None:
    nodes = data["nodes"]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    types = sorted(set(n["type"] for n in nodes))

    for t in types:
        group = [n for n in nodes if n["type"] == t]
        xs = [n["gd"] for n in group]
        ys = [n["bs"] for n in group]
        zs = [n["mq"] for n in group]
        cs = [n["cc"] for n in group]

        sc = ax.scatter(xs, ys, zs, s=80, alpha=0.9, label=t, c=cs, cmap="viridis", vmin=0, vmax=1)

        for n in group:
            ax.text(n["gd"], n["bs"], n["mq"], n["id"], fontsize=8)

    focus_node = find_node(nodes, focus) if focus else None
    if focus_node:
        ax.scatter(
            [focus_node["gd"]],
            [focus_node["bs"]],
            [focus_node["mq"]],
            s=260,
            edgecolors="black",
            linewidths=2.5,
            c=[focus_node["cc"]],
            cmap="viridis",
            vmin=0,
            vmax=1,
        )

    ax.set_title("GBMC Coordinate Viewer")
    ax.set_xlabel("G/D: Interest ↔ Attractor")
    ax.set_ylabel("B/S: Candidate ↔ Resonance")
    ax.set_zlabel("M/Q: Credit ↔ Exploration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65)
    cbar.set_label("C/C: Trust/Recursion ↔ Freedom")

    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    print(f"Saved coordinate image: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GBMC Coordinate Viewer")
    parser.add_argument("query", nargs="?", help="concept to locate")
    parser.add_argument("--data", type=Path, help="custom coordinate json")
    parser.add_argument("--draw", action="store_true", help="draw 3D coordinate image")
    parser.add_argument("--output", type=Path, default=Path("gbmc_coordinates.png"))
    args = parser.parse_args()

    data = load_data(args.data)

    if args.query:
        print_position(data, args.query)
    else:
        print("GBMC Coordinate Viewer")
        print("Usage:")
        print("  python gbmc_coordinate.py Groove")
        print("  python gbmc_coordinate.py Dance --draw")
        print("  python gbmc_coordinate.py --draw")

    if args.draw:
        draw(data, args.query, args.output)


if __name__ == "__main__":
    main()
