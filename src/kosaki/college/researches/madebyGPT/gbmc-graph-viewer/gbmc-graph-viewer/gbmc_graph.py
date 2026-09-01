import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


DEFAULT_DATA = {'nodes': [{'id': 'Module', 'type': 'GBMC', 'desc': 'isolated part'}, {'id': 'Port', 'type': 'GBMC', 'desc': 'mediation surface'}, {'id': 'Translator', 'type': 'GBMC', 'desc': 'type/concept converter'}, {'id': 'ResponsibilityRange', 'type': 'GBMC', 'desc': 'guarantee boundary'}, {'id': 'Resonance', 'type': 'GBMC', 'desc': 'continuable connection'}, {'id': 'ReferenceFrame', 'type': 'GBMC', 'desc': 'represent/origin'}, {'id': 'Flow', 'type': 'GBMC', 'desc': 'transition order'}, {'id': 'MeaningWall', 'type': 'GBMC', 'desc': 'undefined boundary'}, {'id': 'ModuleState', 'type': 'GBMC', 'desc': 'current condition'}, {'id': 'Groove', 'type': 'StreetDance', 'desc': 'body/rhythm/space continuity'}, {'id': 'Isolation', 'type': 'StreetDance', 'desc': 'separate body part control'}, {'id': 'Represent', 'type': 'StreetDance', 'desc': 'reference identity'}, {'id': 'Attention', 'type': 'Transformer', 'desc': 'information mediation'}, {'id': 'Query-Key Matching', 'type': 'Transformer', 'desc': 'matching relation'}, {'id': 'Embedding', 'type': 'Transformer', 'desc': 'representation unit'}, {'id': 'Pipe', 'type': 'Unix', 'desc': 'command mediation'}, {'id': 'stdin/stdout', 'type': 'Unix', 'desc': 'standard IO surface'}, {'id': 'Small Tool', 'type': 'Unix', 'desc': 'isolated command'}, {'id': 'Observer', 'type': 'ControlTheory', 'desc': 'state estimator'}, {'id': 'Feedback', 'type': 'ControlTheory', 'desc': 'corrective loop'}, {'id': 'State', 'type': 'ControlTheory', 'desc': 'system condition'}, {'id': 'Servo Limit', 'type': 'HumanPoweredAircraft', 'desc': 'physical guarantee'}, {'id': 'AngleToPWM', 'type': 'HumanPoweredAircraft', 'desc': 'signal converter'}, {'id': 'Pico2W', 'type': 'HumanPoweredAircraft', 'desc': 'candidate generator'}, {'id': 'ESP32', 'type': 'HumanPoweredAircraft', 'desc': 'execution filter'}, {'id': 'Servo', 'type': 'HumanPoweredAircraft', 'desc': 'physical output'}], 'edges': [{'from': 'Module', 'to': 'Port', 'label': 'exposes'}, {'from': 'Port', 'to': 'Translator', 'label': 'requires when incompatible'}, {'from': 'Translator', 'to': 'Flow', 'label': 'allows continuation'}, {'from': 'Port', 'to': 'Resonance', 'label': 'enables'}, {'from': 'Resonance', 'to': 'Flow', 'label': 'stabilizes'}, {'from': 'ResponsibilityRange', 'to': 'MeaningWall', 'label': 'defines boundary'}, {'from': 'ReferenceFrame', 'to': 'ResponsibilityRange', 'label': 'anchors'}, {'from': 'ReferenceFrame', 'to': 'MeaningWall', 'label': 'reduces'}, {'from': 'ModuleState', 'to': 'Flow', 'label': 'moves through'}, {'from': 'Translator', 'to': 'MeaningWall', 'label': 'resolves'}, {'from': 'Groove', 'to': 'Resonance', 'label': 'maps to'}, {'from': 'Isolation', 'to': 'Module', 'label': 'maps to'}, {'from': 'Represent', 'to': 'ReferenceFrame', 'label': 'maps to'}, {'from': 'Attention', 'to': 'Port', 'label': 'maps to'}, {'from': 'Query-Key Matching', 'to': 'Resonance', 'label': 'maps to'}, {'from': 'Embedding', 'to': 'Module', 'label': 'maps to'}, {'from': 'Pipe', 'to': 'Port', 'label': 'maps to'}, {'from': 'stdin/stdout', 'to': 'Port', 'label': 'maps to'}, {'from': 'Small Tool', 'to': 'Module', 'label': 'maps to'}, {'from': 'Observer', 'to': 'Translator', 'label': 'maps to'}, {'from': 'Feedback', 'to': 'ResponsibilityRange', 'label': 'maps to'}, {'from': 'State', 'to': 'ModuleState', 'label': 'maps to'}, {'from': 'Servo Limit', 'to': 'ResponsibilityRange', 'label': 'maps to'}, {'from': 'AngleToPWM', 'to': 'Translator', 'label': 'maps to'}, {'from': 'Pico2W', 'to': 'Module', 'label': 'is'}, {'from': 'ESP32', 'to': 'Translator', 'label': 'acts as'}, {'from': 'Servo', 'to': 'Module', 'label': 'is'}]}


def load_data(path: Path | None) -> dict:
    if path is None:
        return DEFAULT_DATA
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(data: dict) -> nx.Graph:
    graph = nx.Graph()
    for node in data["nodes"]:
        graph.add_node(node["id"], **node)
    for edge in data["edges"]:
        graph.add_edge(edge["from"], edge["to"], label=edge.get("label", ""))
    return graph


def find_node(graph: nx.Graph, query: str) -> str | None:
    q = query.lower()
    aliases = {
        "servolimit": "Servo Limit",
        "servo_limit": "Servo Limit",
        "stdin": "stdin/stdout",
        "stdout": "stdin/stdout",
        "qk": "Query-Key Matching",
        "query-key": "Query-Key Matching",
    }
    if q in aliases:
        return aliases[q]
    for node in graph.nodes:
        if node.lower() == q:
            return node
    for node in graph.nodes:
        if q in node.lower() or node.lower() in q:
            return node
    return None


def print_position(graph: nx.Graph, query: str) -> None:
    node = find_node(graph, query)
    print("=== GBMC Graph Viewer ===")
    print(f"Query: {query}")

    if node is None:
        print("No node found.")
        print("Try: Groove, Attention, Pipe, Observer, Servo Limit, Port, Resonance")
        return

    attrs = graph.nodes[node]
    print("\nCurrent Position")
    print(f"  node : {node}")
    print(f"  type : {attrs.get('type', '')}")
    print(f"  desc : {attrs.get('desc', '')}")

    print("\nNearby Nodes")
    for neighbor in graph.neighbors(node):
        label = graph.edges[node, neighbor].get("label", "")
        ntype = graph.nodes[neighbor].get("type", "")
        print(f"  {node} --[{label}]-- {neighbor} ({ntype})")

    print("\nDistance to GBMC Core")
    core_nodes = [
        "Module",
        "Port",
        "Translator",
        "ResponsibilityRange",
        "Resonance",
        "ReferenceFrame",
        "Flow",
        "MeaningWall",
    ]
    distances = []
    for core in core_nodes:
        try:
            d = nx.shortest_path_length(graph, node, core)
            distances.append((d, core))
        except nx.NetworkXNoPath:
            pass
    for d, core in sorted(distances)[:5]:
        print(f"  {core}: {d}")


def draw_graph(graph: nx.Graph, focus: str | None, output: Path) -> None:
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(graph, seed=42, k=0.65)

    node_types = nx.get_node_attributes(graph, "type")
    groups = sorted(set(node_types.values()))

    for group in groups:
        nodes = [n for n, t in node_types.items() if t == group]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_size=1100 if group == "GBMC" else 750,
            alpha=0.9,
            label=group,
        )

    nx.draw_networkx_edges(graph, pos, alpha=0.35, width=1.5)
    nx.draw_networkx_labels(graph, pos, font_size=9)

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    if focus:
        node = find_node(graph, focus)
        if node:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[node],
                node_size=1700,
                linewidths=3,
                edgecolors="black",
            )

    plt.title("GBMC Concept Graph")
    plt.axis("off")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    print(f"Saved graph image: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GBMC Graph Viewer")
    parser.add_argument("query", nargs="?", help="concept to locate")
    parser.add_argument("--data", type=Path, help="custom graph json")
    parser.add_argument("--draw", action="store_true", help="draw graph image")
    parser.add_argument("--output", type=Path, default=Path("gbmc_graph.png"))
    args = parser.parse_args()

    data = load_data(args.data)
    graph = build_graph(data)

    if args.query:
        print_position(graph, args.query)
    else:
        print("GBMC Graph Viewer")
        print("Usage:")
        print("  python gbmc_graph.py Groove")
        print("  python gbmc_graph.py Attention --draw")
        print("  python gbmc_graph.py --draw")

    if args.draw:
        draw_graph(graph, args.query, args.output)


if __name__ == "__main__":
    main()
