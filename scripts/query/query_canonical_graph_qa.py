from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.qa.canonical_graph_qa_context import (
    answer_canonical_graph_query,
    retrieve_canonical_graph_context,
)
from vlkg.datasets.navqa.qa.graph_qa_pipeline import load_document
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.transformers import build_canonical_temporal_memory_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query the canonical temporal memory graph built from a temporal graph JSON document."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a temporal graph JSON document.",
    )
    subparsers = parser.add_subparsers(dest="query_type", required=True)

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Run canonical graph retrieval and inspect the graph-backed evidence bundle.",
    )
    _add_query_arguments(retrieve_parser)

    answer_parser = subparsers.add_parser(
        "answer",
        help="Run canonical graph retrieval and then render the answer through graph_only/mock/live mode.",
    )
    _add_query_arguments(answer_parser)
    _add_question_input_group(answer_parser)
    answer_parser.add_argument(
        "--answer-mode",
        choices=("graph_only", "mock", "live"),
        default="graph_only",
        help="graph_only uses only the canonical graph bundle, mock is deterministic offline, live calls the configured OpenAI-compatible LLM endpoint.",
    )
    answer_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional domain config path used to load graph_qa settings for live mode.",
    )

    return parser


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-mode",
        choices=("temporal", "entity"),
        default="temporal",
        help="temporal prioritizes episodes/events, entity prioritizes entity-centered lifecycle context.",
    )
    parser.add_argument(
        "--query-text",
        type=str,
        default=None,
        help="Optional retrieval text. In answer mode, the question text is used as fallback when omitted.",
    )
    parser.add_argument(
        "--entity-id",
        action="append",
        default=[],
        help="Optional entity id filter. Repeat this flag to supply multiple entity ids.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Optional scene id filter. Repeat this flag to supply multiple scene ids.",
    )
    parser.add_argument("--start-ts", type=float, default=None)
    parser.add_argument("--end-ts", type=float, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--compact-relation-observed-events",
        action="store_true",
        help="Keep RELATION_OBSERVED evidence as compact samples instead of canonical transition event nodes.",
    )
    parser.add_argument(
        "--compact-derived-temporal-edges",
        action="store_true",
        default=True,
        help="Do not materialize derivable OVERLAPS_WITH/TEMPORALLY_NEAR edges in the canonical graph.",
    )
    parser.add_argument(
        "--materialize-derived-temporal-edges",
        dest="compact_derived_temporal_edges",
        action="store_false",
        help="Opt out of the default compact main-graph mode and physically store OVERLAPS_WITH/TEMPORALLY_NEAR edges.",
    )
    parser.add_argument(
        "--compact-previous-episode-edges",
        dest="compact_previous_episode_edges",
        action="store_true",
        default=True,
        help="Do not materialize derivable PREVIOUS_EPISODE edges in the canonical graph. This is the default mainline behavior.",
    )
    parser.add_argument(
        "--materialize-previous-episode-edges",
        dest="compact_previous_episode_edges",
        action="store_false",
        help="Opt out of the default compact main-graph mode and physically store PREVIOUS_EPISODE edges.",
    )
    parser.add_argument(
        "--include-base-delta-projection",
        action="store_true",
        help=(
            "Opt in to virtual scene-base / scene-delta evidence records during "
            "canonical graph retrieval. The canonical graph itself is unchanged."
        ),
    )


def _add_question_input_group(parser: argparse.ArgumentParser) -> None:
    question_group = parser.add_mutually_exclusive_group(required=True)
    question_group.add_argument(
        "--question",
        type=str,
        help="Inline natural-language question.",
    )
    question_group.add_argument(
        "--question-file",
        type=Path,
        help="UTF-8 text file containing one question.",
    )


def _read_question(args: argparse.Namespace) -> str:
    question = (
        args.question
        if args.question is not None
        else args.question_file.read_text(encoding="utf-8-sig").strip()
    )
    return str(question).strip()


def _build_query(args: argparse.Namespace) -> MemoryQuery:
    return MemoryQuery(
        query_text=args.query_text,
        entity_ids=list(args.entity_id or []),
        scene_ids=list(args.scene_id or []),
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        limit=args.limit,
    )


def _canonical_summary(graph) -> dict[str, object]:
    return {
        "schema_version": graph.schema_version,
        "video_id": graph.video_id,
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "scene_nodes": len(graph.scene_nodes),
        "entity_nodes": len(graph.entity_nodes),
        "stable_identity_nodes": len(graph.stable_identity_nodes),
        "state_value_nodes": len(graph.state_value_nodes),
        "interaction_type_nodes": len(graph.interaction_type_nodes),
        "episode_nodes": len(graph.episode_nodes),
        "transition_event_nodes": len(graph.transition_event_nodes),
        "relation_observed_samples": len(graph.relation_observed_samples),
        "episode_kind_counts": dict(
            Counter(node.episode_kind for node in graph.episode_nodes)
        ),
        "edge_type_counts": dict(Counter(edge.edge_type for edge in graph.edges)),
    }


def main() -> None:
    args = build_parser().parse_args()
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    document = load_document(resolve_repo_input_path(args.input_json))
    canonical_graph = build_canonical_temporal_memory_graph(
        document,
        compact_relation_observed_events=args.compact_relation_observed_events,
        compact_derived_temporal_edges=args.compact_derived_temporal_edges,
        compact_previous_episode_edges=args.compact_previous_episode_edges,
    )
    query = _build_query(args)

    if args.query_type == "retrieve":
        payload = retrieve_canonical_graph_context(
            canonical_graph,
            query,
            retrieval_mode=args.retrieval_mode,
            limit=args.limit,
            temporal_document=document,
            include_base_delta_projection=args.include_base_delta_projection,
        )
    elif args.query_type == "answer":
        question = _read_question(args)
        payload = answer_canonical_graph_query(
            canonical_graph,
            query,
            question=question,
            retrieval_mode=args.retrieval_mode,
            answer_mode=args.answer_mode,
            config_path=args.config,
            limit=args.limit,
            temporal_document=document,
            include_base_delta_projection=args.include_base_delta_projection,
        )
    else:  # pragma: no cover - argparse guarantees subcommand
        raise SystemExit(f"Unsupported query type: {args.query_type}")

    output = {
        "video_id": document.video_id,
        "query_type": args.query_type,
        "canonical_graph_summary": _canonical_summary(canonical_graph),
        "result": payload,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
