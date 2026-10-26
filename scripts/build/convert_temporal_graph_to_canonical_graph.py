from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.models.temporal_graph import TemporalGraphDocument
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.transformers import (
    SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID,
    SECOND_MAINLINE_PROFILE_ID,
    build_canonical_temporal_memory_graph,
    build_time_fact_core_graph,
)


FIRST_MAINLINE_PROFILE_ID = "canonical"
SECOND_MAINLINE_PROFILE_ALIASES = (
    SECOND_MAINLINE_PROFILE_ID,
    SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID,
)


def normalize_graph_profile(value: str) -> str:
    if value in SECOND_MAINLINE_PROFILE_ALIASES:
        return SECOND_MAINLINE_PROFILE_ID
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a temporal graph JSON document into a canonical temporal memory graph."
    )
    parser.add_argument(
        "input_graph_json",
        type=Path,
        help="Path to temporal_graph.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <input_graph_json>/../canonical_temporal_memory_graph.json.",
    )
    parser.add_argument(
        "--graph-profile",
        choices=(FIRST_MAINLINE_PROFILE_ID, *SECOND_MAINLINE_PROFILE_ALIASES),
        default=FIRST_MAINLINE_PROFILE_ID,
        help=(
            "Canonical graph build profile. 'canonical' is the first-mainline "
            "profile. 'time_fact_core' is the second-mainline direct graph "
            "profile; 'topology_pruned_after_scene_fact' is accepted as a legacy alias."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help=(
            "Optional profile report path. Useful for "
            "--graph-profile time_fact_core."
        ),
    )
    parser.add_argument(
        "--no-stable-identities",
        action="store_true",
        help="Skip the stable identity layer.",
    )
    parser.add_argument(
        "--stable-identity-merge-mode",
        choices=("semantic", "none"),
        default="semantic",
        help=(
            "How stable identity nodes are formed. 'semantic' keeps the default "
            "cross-track semantic merge; 'none' keeps one stable identity per "
            "temporal entity."
        ),
    )
    parser.add_argument(
        "--stable-identity-include-fragments",
        action="store_true",
        help="Include fragment entities when building stable identity nodes.",
    )
    parser.add_argument(
        "--stable-identity-include-noise",
        action="store_true",
        help="Include noise entities when building stable identity nodes.",
    )
    parser.add_argument(
        "--compact-relation-observed-events",
        action="store_true",
        help="Keep RELATION_OBSERVED evidence as compact samples instead of event nodes.",
    )
    parser.add_argument(
        "--materialize-derived-temporal-edges",
        action="store_true",
        help="Materialize derivable OVERLAPS_WITH/TEMPORALLY_NEAR edges.",
    )
    parser.add_argument(
        "--materialize-previous-episode-edges",
        action="store_true",
        help="Materialize derivable PREVIOUS_EPISODE edges.",
    )
    parser.add_argument(
        "--compact-scene-label-entities",
        action="store_true",
        help=(
            "Aggregate low-value same-scene same-label entity fragments into "
            "compact scene-label entity anchors."
        ),
    )
    parser.add_argument(
        "--no-compact-scene-label-entities",
        action="store_true",
        help=(
            "Disable scene-label entity anchor compaction. For time_fact_core, "
            "this overrides the profile default."
        ),
    )
    parser.add_argument(
        "--compact-scene-label-min-members",
        type=int,
        default=4,
        help="Minimum group size for --compact-scene-label-entities.",
    )
    parser.add_argument(
        "--no-compact-scene-label-entity-segments",
        action="store_true",
        help=(
            "When compacting scene-label entities, keep original entity segment "
            "episodes instead of compacting them by scene/entity anchor."
        ),
    )
    parser.add_argument(
        "--no-stable-core-scene-label-compaction",
        action="store_true",
        help=(
            "For time_fact_core, do not compact stable-core entities by same "
            "scene/label/role when no stable identity is present."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph_profile = normalize_graph_profile(args.graph_profile)
    input_graph_json = resolve_repo_input_path(args.input_graph_json)
    document = TemporalGraphDocument.model_validate_json(
        input_graph_json.read_text(encoding="utf-8")
    )
    if args.compact_scene_label_entities and args.no_compact_scene_label_entities:
        raise SystemExit(
            "--compact-scene-label-entities and "
            "--no-compact-scene-label-entities cannot be used together"
        )
    report: dict[str, object] | None = None
    if graph_profile == FIRST_MAINLINE_PROFILE_ID:
        graph = build_canonical_temporal_memory_graph(
            document,
            source_graph_path=input_graph_json,
            include_stable_identities=not args.no_stable_identities,
            compact_relation_observed_events=args.compact_relation_observed_events,
            compact_derived_temporal_edges=not args.materialize_derived_temporal_edges,
            compact_previous_episode_edges=not args.materialize_previous_episode_edges,
            compact_scene_label_entities=(
                args.compact_scene_label_entities
                and not args.no_compact_scene_label_entities
            ),
            compact_scene_label_min_members=args.compact_scene_label_min_members,
            compact_scene_label_entity_segments=(
                not args.no_compact_scene_label_entity_segments
            ),
            stable_identity_merge_mode=args.stable_identity_merge_mode,
            stable_identity_include_fragments=args.stable_identity_include_fragments,
            stable_identity_include_noise=args.stable_identity_include_noise,
        )
    else:
        if args.no_stable_identities:
            raise SystemExit(
                "--graph-profile time_fact_core requires stable identities"
            )
        if args.compact_relation_observed_events:
            raise SystemExit(
                "--compact-relation-observed-events is not part of the "
                "second-mainline time_fact_core profile"
            )
        if args.materialize_derived_temporal_edges or args.materialize_previous_episode_edges:
            raise SystemExit(
                "materialized derived/previous episode edges are not part of the "
                "second-mainline time_fact_core profile"
            )
        result = build_time_fact_core_graph(
            document,
            source_graph_path=input_graph_json,
            compact_scene_label_entities=not args.no_compact_scene_label_entities,
            compact_scene_label_min_members=args.compact_scene_label_min_members,
            compact_scene_label_entity_segments=(
                not args.no_compact_scene_label_entity_segments
            ),
            stable_identity_merge_mode=args.stable_identity_merge_mode,
            stable_identity_include_fragments=args.stable_identity_include_fragments,
            stable_identity_include_noise=args.stable_identity_include_noise,
            stable_core_compact_scene_label_entities=(
                not args.no_stable_core_scene_label_compaction
            ),
        )
        graph = result.graph
        report = result.report

    output_path = args.output or (
        input_graph_json.parent / "canonical_temporal_memory_graph.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        graph.model_dump_json(indent=2),
        encoding="utf-8",
    )
    if args.report_json is not None:
        if report is None:
            report = {
                "contract": "canonical_graph_profile_report_v1",
                "graph_profile": graph_profile,
                "requested_graph_profile": args.graph_profile,
                "source_graph_path": str(input_graph_json),
                "output": str(output_path),
                "nodes": graph.node_count,
                "edges": graph.edge_count,
            }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"output={output_path.resolve()}")
    print(f"graph_profile={graph_profile}")
    if args.graph_profile != graph_profile:
        print(f"requested_graph_profile={args.graph_profile}")
    print(f"stable_identity_merge_mode={args.stable_identity_merge_mode}")
    print(f"stable_identity_include_fragments={args.stable_identity_include_fragments}")
    print(f"stable_identity_include_noise={args.stable_identity_include_noise}")
    print(f"schema_version={graph.schema_version}")
    print(f"nodes={graph.node_count}")
    print(f"edges={graph.edge_count}")
    print(f"stable_identity_nodes={len(graph.stable_identity_nodes)}")
    print(f"scene_nodes={len(graph.scene_nodes)}")
    print(f"entity_nodes={len(graph.entity_nodes)}")
    print(f"episode_nodes={len(graph.episode_nodes)}")
    print(f"transition_event_nodes={len(graph.transition_event_nodes)}")
    print(f"relation_observed_samples={len(graph.relation_observed_samples)}")
    if args.report_json is not None:
        print(f"report_json={args.report_json.resolve()}")


if __name__ == "__main__":
    main()
