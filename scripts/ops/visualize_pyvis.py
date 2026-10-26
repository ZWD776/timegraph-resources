from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
from pyvis.network import Network

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.temporal.pathing import resolve_repo_input_path


STATIC_NODE_COLOR = "#4B5563"
DYNAMIC_NODE_COLOR = "#38BDF8"
STATE_EDGE_COLOR = "#22C55E"
RELATION_EDGE_COLOR = "#94A3B8"
TRANSITION_NODE_COLOR = "#F59E0B"
TRANSITION_EDGE_COLOR = "#FCD34D"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a temporal graph JSON document into a local PyVis HTML preview."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to temporal_graph.siglip.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("temporal_graph_35_frames.html"),
        help="HTML output path.",
    )
    parser.add_argument(
        "--include-transition-events",
        action="store_true",
        help="Render transition events as explicit temporal nodes.",
    )
    return parser


def load_graph_document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_nx_graph(
    document: dict,
    *,
    include_transition_events: bool = False,
) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()

    for entity in document["entities"]:
        is_static = bool(entity.get("is_static", False))
        lifecycle = f"[{entity['first_seen_frame']}, {entity['last_seen_frame']}]"
        title = (
            f"<b>{entity['category']}</b><br>"
            f"entity_id: {entity['entity_id']}<br>"
            f"type: {'static' if is_static else 'dynamic'}<br>"
            f"lifecycle: {lifecycle}<br>"
            f"observations: {entity['observation_count']}"
        )
        graph.add_node(
            entity["entity_id"],
            label=entity["category"],
            color=STATIC_NODE_COLOR if is_static else DYNAMIC_NODE_COLOR,
            title=title,
            is_static=is_static,
            observation_count=entity["observation_count"],
            first_seen_frame=entity["first_seen_frame"],
            last_seen_frame=entity["last_seen_frame"],
        )

    for episode in document["relation_episodes"]:
        edge_label = (
            f"{episode['predicate']} "
            f"[{episode['start_frame']}-{episode['end_frame']}]"
        )
        edge_title = (
            f"<b>{episode['predicate']}</b><br>"
            f"frames: [{episode['start_frame']}, {episode['end_frame']}]<br>"
            f"support_frames: {episode['support_frames']}<br>"
            f"time: [{episode['start_ts']:.3f}, {episode['end_ts']:.3f}]"
        )
        graph.add_edge(
            episode["subject_entity_id"],
            episode["object_entity_id"],
            key=episode["relation_episode_id"],
            label=edge_label,
            title=edge_title,
            color=RELATION_EDGE_COLOR,
            dashes=False,
            relation_type="relation_episode",
        )

    for episode in document["state_episodes"]:
        edge_label = (
            f"{episode['attr_value']} "
            f"[{episode['start_frame']}-{episode['end_frame']}]"
        )
        edge_title = (
            f"<b>{episode['attr_key']}</b><br>"
            f"value: {episode['attr_value']}<br>"
            f"frames: [{episode['start_frame']}, {episode['end_frame']}]<br>"
            f"support_frames: {episode['support_frames']}<br>"
            f"time: [{episode['start_ts']:.3f}, {episode['end_ts']:.3f}]"
        )
        graph.add_edge(
            episode["entity_id"],
            episode["entity_id"],
            key=episode["state_episode_id"],
            label=edge_label,
            title=edge_title,
            color=STATE_EDGE_COLOR,
            dashes=True,
            relation_type="state_episode",
        )

    if include_transition_events:
        for event in document.get("transition_events", []):
            event_label = f"{event['event_type']} @{event['frame_index']}"
            event_title = (
                f"<b>{event['event_type']}</b><br>"
                f"frame: {event['frame_index']} (rank {event['frame_rank']})<br>"
                f"time: {event['timestamp_sec']:.3f}"
            )
            if event.get("predicate"):
                event_title += f"<br>predicate: {event['predicate']}"
            if event.get("current_state_value"):
                event_title += f"<br>state: {event['current_state_value']}"
            if event.get("previous_state_value"):
                event_title += f"<br>previous: {event['previous_state_value']}"

            graph.add_node(
                event["event_id"],
                label=event_label,
                color=TRANSITION_NODE_COLOR,
                title=event_title,
                is_static=False,
                is_transition=True,
            )

            if event.get("entity_id"):
                graph.add_edge(
                    event["entity_id"],
                    event["event_id"],
                    key=f"{event['event_id']}:entity",
                    label=event["event_type"],
                    title=event_title,
                    color=TRANSITION_EDGE_COLOR,
                    dashes=True,
                    relation_type="transition_event",
                )
            if event.get("subject_entity_id"):
                graph.add_edge(
                    event["subject_entity_id"],
                    event["event_id"],
                    key=f"{event['event_id']}:subject",
                    label=f"subject:{event['event_type']}",
                    title=event_title,
                    color=TRANSITION_EDGE_COLOR,
                    dashes=True,
                    relation_type="transition_event",
                )
            if event.get("object_entity_id"):
                graph.add_edge(
                    event["event_id"],
                    event["object_entity_id"],
                    key=f"{event['event_id']}:object",
                    label="object",
                    title=event_title,
                    color=TRANSITION_EDGE_COLOR,
                    dashes=True,
                    relation_type="transition_event",
                )

    return graph


def render_pyvis(graph: nx.MultiDiGraph, output_path: Path) -> None:
    network = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#111827",
        font_color="#F9FAFB",
    )
    network.barnes_hut()

    for node_id, attrs in graph.nodes(data=True):
        is_transition = bool(attrs.get("is_transition", False))
        network.add_node(
            node_id,
            label=attrs["label"],
            title=attrs["title"],
            color=attrs["color"],
            shape="diamond" if is_transition else "dot",
            size=20 if is_transition else (28 if attrs["is_static"] else 34),
        )

    for source, target, attrs in graph.edges(data=True):
        edge_kwargs = {
            "title": attrs["title"],
            "label": attrs["label"],
            "color": attrs["color"],
            "dashes": attrs["dashes"],
            "font": {"align": "top", "size": 14},
            "arrows": "to",
        }
        if source == target:
            edge_kwargs["smooth"] = {"type": "curvedCW", "roundness": 0.25}
        network.add_edge(source, target, **edge_kwargs)

    network.set_options(
        """
        const options = {
          "physics": {
            "enabled": true,
            "barnesHut": {
              "gravitationalConstant": -2200,
              "springLength": 180,
              "springConstant": 0.04
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "dynamic"
            }
          }
        }
        """
    )
    network.write_html(str(output_path), notebook=False)


def main() -> None:
    args = build_parser().parse_args()
    document = load_graph_document(resolve_repo_input_path(args.input_json))
    graph = build_nx_graph(
        document,
        include_transition_events=args.include_transition_events,
    )
    render_pyvis(graph, args.output)
    print(f"HTML graph written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
