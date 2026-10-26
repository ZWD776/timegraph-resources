from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.temporal.pathing import resolve_repo_input_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate where a canonical temporal memory graph can be compacted "
            "without changing the graph. This is a diagnostics-only script."
        )
    )
    parser.add_argument("canonical_graph_json", type=Path)
    parser.add_argument(
        "--temporal-graph-json",
        type=Path,
        default=None,
        help="Optional source temporal_graph.json for scene_base / scene_delta counts.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional diagnostics JSON path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional markdown report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph_path = resolve_repo_input_path(args.canonical_graph_json)
    graph = CanonicalTemporalMemoryGraph.model_validate_json(
        graph_path.read_text(encoding="utf-8")
    )
    temporal_payload = None
    temporal_path = None
    if args.temporal_graph_json is not None:
        temporal_path = resolve_repo_input_path(args.temporal_graph_json)
        temporal_payload = json.loads(temporal_path.read_text(encoding="utf-8"))

    payload = build_diagnostics(
        graph,
        graph_path=graph_path,
        temporal_payload=temporal_payload,
        temporal_path=temporal_path,
    )
    print_summary(payload)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"output_json={args.output_json.resolve()}")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(build_markdown(payload), encoding="utf-8")
        print(f"output_md={args.output_md.resolve()}")


def build_diagnostics(
    graph: CanonicalTemporalMemoryGraph,
    *,
    graph_path: Path,
    temporal_payload: dict[str, Any] | None,
    temporal_path: Path | None,
) -> dict[str, Any]:
    node_counts = {
        "scene_nodes": len(graph.scene_nodes),
        "entity_nodes": len(graph.entity_nodes),
        "stable_identity_nodes": len(graph.stable_identity_nodes),
        "state_value_nodes": len(graph.state_value_nodes),
        "interaction_type_nodes": len(graph.interaction_type_nodes),
        "episode_nodes": len(graph.episode_nodes),
        "transition_event_nodes": len(graph.transition_event_nodes),
        "relation_observed_samples": len(graph.relation_observed_samples),
    }
    edge_counts = Counter(edge.edge_type for edge in graph.edges)
    episode_counts = Counter(node.episode_kind for node in graph.episode_nodes)
    entity_label_counts = Counter(_norm(node.label) for node in graph.entity_nodes)
    entity_role_counts = Counter(node.node_role for node in graph.entity_nodes)

    stable_groups = _stable_identity_groups(graph)
    scene_label_groups = _scene_label_groups(graph)
    relation_signature_groups = _relation_signature_groups(graph)
    entity_segment_signature_groups = _entity_segment_signature_groups(graph)

    temporal_counts = {}
    if temporal_payload is not None:
        temporal_counts = {
            key: len(temporal_payload.get(key, []))
            for key in (
                "entities",
                "entity_segments",
                "relation_episodes",
                "state_episodes",
                "scene_base_nodes",
                "scene_base_edges",
                "scene_delta_nodes",
                "scene_delta_observations",
                "scene_delta_edges",
                "transition_events",
            )
        }

    opportunities = {
        "entity_to_stable_identity": _group_opportunity(
            total=len(graph.entity_nodes),
            groups=stable_groups,
        ),
        "entity_to_scene_label_anchor": _group_opportunity(
            total=len(graph.entity_nodes),
            groups=scene_label_groups,
        ),
        "relation_episode_to_signature": _group_opportunity(
            total=episode_counts.get("RELATION", 0),
            groups=relation_signature_groups,
        ),
        "entity_segment_to_identity_scene": _group_opportunity(
            total=episode_counts.get("ENTITY_SEGMENT", 0),
            groups=entity_segment_signature_groups,
        ),
    }

    return {
        "diagnostic": "canonical_compaction_opportunities",
        "source_graph": str(graph_path),
        "source_temporal_graph": str(temporal_path) if temporal_path is not None else None,
        "video_id": graph.video_id,
        "node_total": graph.node_count,
        "edge_total": graph.edge_count,
        "node_counts": node_counts,
        "edge_counts": dict(edge_counts.most_common()),
        "episode_counts": dict(episode_counts.most_common()),
        "entity_role_counts": dict(entity_role_counts.most_common()),
        "top_entity_labels": dict(entity_label_counts.most_common(30)),
        "temporal_graph_counts": temporal_counts,
        "opportunities": opportunities,
        "top_groups": {
            "stable_identity": _top_groups(stable_groups, limit=20),
            "scene_label_anchor": _top_groups(scene_label_groups, limit=20),
            "relation_signature": _top_groups(relation_signature_groups, limit=20),
            "entity_segment_identity_scene": _top_groups(
                entity_segment_signature_groups,
                limit=20,
            ),
        },
        "recommendation": _recommendation(opportunities),
    }


def _stable_identity_groups(
    graph: CanonicalTemporalMemoryGraph,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for node in graph.entity_nodes:
        key = node.stable_identity_id or f"entity:{node.entity_id}"
        groups[key].append(node.entity_id)
    return dict(groups)


def _scene_label_groups(
    graph: CanonicalTemporalMemoryGraph,
) -> dict[str, list[str]]:
    scene_by_entity: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.edge_type != "IN_SCENE":
            continue
        if not edge.source_node_id.startswith("entity:"):
            continue
        entity_id = edge.source_node_id.removeprefix("entity:")
        scene_by_entity[entity_id].add(edge.target_node_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for node in graph.entity_nodes:
        scenes = sorted(scene_by_entity.get(node.entity_id) or ["scene:unknown"])
        for scene_id in scenes:
            key = f"{scene_id}|{_norm(node.label)}|{node.node_role}"
            groups[key].append(node.entity_id)
    return dict(groups)


def _relation_signature_groups(
    graph: CanonicalTemporalMemoryGraph,
) -> dict[str, list[str]]:
    stable_by_entity = {
        node.entity_id: node.stable_identity_id or node.entity_id
        for node in graph.entity_nodes
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for node in graph.episode_nodes:
        if node.episode_kind != "RELATION":
            continue
        subject = stable_by_entity.get(node.subject_entity_id or "", node.subject_entity_id or "")
        obj = stable_by_entity.get(node.object_entity_id or "", node.object_entity_id or "")
        key = "|".join(
            [
                node.scene_id or "scene:unknown",
                subject,
                _norm(node.predicate),
                obj,
            ]
        )
        groups[key].append(node.episode_id)
    return dict(groups)


def _entity_segment_signature_groups(
    graph: CanonicalTemporalMemoryGraph,
) -> dict[str, list[str]]:
    stable_by_entity = {
        node.entity_id: node.stable_identity_id or node.entity_id
        for node in graph.entity_nodes
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for node in graph.episode_nodes:
        if node.episode_kind != "ENTITY_SEGMENT":
            continue
        identity = stable_by_entity.get(node.primary_entity_id or "", node.primary_entity_id or "")
        key = "|".join(
            [
                node.scene_id or "scene:unknown",
                identity,
                str(node.attributes.get("segment_type") or ""),
            ]
        )
        groups[key].append(node.episode_id)
    return dict(groups)


def _group_opportunity(total: int, groups: dict[str, list[str]]) -> dict[str, Any]:
    compact_count = len(groups)
    multi_groups = [items for items in groups.values() if len(items) > 1]
    removable = max(0, total - compact_count)
    return {
        "original_count": total,
        "compact_group_count": compact_count,
        "estimated_removed": removable,
        "estimated_remaining_ratio": round(compact_count / total, 4) if total else 0.0,
        "multi_member_group_count": len(multi_groups),
        "max_group_size": max((len(items) for items in groups.values()), default=0),
        "avg_multi_group_size": (
            round(sum(len(items) for items in multi_groups) / len(multi_groups), 2)
            if multi_groups
            else 0.0
        ),
    }


def _top_groups(groups: dict[str, list[str]], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for key, values in groups.items():
        if len(values) <= 1:
            continue
        rows.append(
            {
                "group": key,
                "count": len(values),
                "examples": values[:8],
            }
        )
    rows.sort(key=lambda item: (-item["count"], item["group"]))
    return rows[:limit]


def _recommendation(opportunities: dict[str, Any]) -> list[str]:
    recs = [
        "Keep the full canonical graph as an audit layer.",
        "Build a compact QA graph as an optional derived artifact, not a replacement.",
    ]
    relation = opportunities["relation_episode_to_signature"]
    entity_segment = opportunities["entity_segment_to_identity_scene"]
    scene_anchor = opportunities["entity_to_scene_label_anchor"]
    if relation["estimated_remaining_ratio"] < 0.8:
        recs.append("Aggregate repeated relation episodes by scene + stable subject + predicate + stable object.")
    if entity_segment["estimated_remaining_ratio"] < 0.8:
        recs.append("Aggregate entity segment episodes by scene + stable identity/entity + segment type.")
    if scene_anchor["estimated_remaining_ratio"] < 0.8:
        recs.append("Use scene-label anchors for high-frequency static/dependent labels.")
    return recs


def print_summary(payload: dict[str, Any]) -> None:
    print(f"video_id={payload['video_id']}")
    print(f"node_total={payload['node_total']}")
    print(f"edge_total={payload['edge_total']}")
    for name, item in payload["opportunities"].items():
        print(
            f"{name}: original={item['original_count']} "
            f"compact={item['compact_group_count']} "
            f"removed~={item['estimated_removed']} "
            f"remaining_ratio={item['estimated_remaining_ratio']}"
        )


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Canonical Compaction Opportunity Report",
        "",
        f"- Source graph: `{payload['source_graph']}`",
        f"- Source temporal graph: `{payload.get('source_temporal_graph') or '(not provided)'}`",
        f"- Video id: `{payload['video_id']}`",
        f"- Nodes: {payload['node_total']}",
        f"- Edges: {payload['edge_total']}",
        "",
        "## Opportunities",
        "",
        "| area | original | compact groups | estimated removed | remaining ratio | max group |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in payload["opportunities"].items():
        lines.append(
            f"| `{name}` | {item['original_count']} | {item['compact_group_count']} | "
            f"{item['estimated_removed']} | {item['estimated_remaining_ratio']} | "
            f"{item['max_group_size']} |"
        )
    lines.extend(["", "## Top Entity Labels", ""])
    for label, count in list(payload["top_entity_labels"].items())[:20]:
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in payload["recommendation"])
    lines.append("")
    return "\n".join(lines)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


if __name__ == "__main__":
    main()
