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

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.navqa.qa.canonical_graph_qa_context import (
    answer_canonical_graph_query,
    retrieve_canonical_graph_context,
)
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.temporal.pathing import resolve_repo_input_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query a saved canonical temporal memory graph JSON document."
    )
    parser.add_argument(
        "canonical_graph_json",
        type=Path,
        help="Path to canonical_temporal_memory_graph.json.",
    )
    subparsers = parser.add_subparsers(dest="query_type", required=True)

    retrieve_parser = subparsers.add_parser("retrieve")
    _add_query_arguments(retrieve_parser)

    answer_parser = subparsers.add_parser("answer")
    _add_query_arguments(answer_parser)
    question_group = answer_parser.add_mutually_exclusive_group(required=True)
    question_group.add_argument("--question", type=str)
    question_group.add_argument("--question-file", type=Path)
    answer_parser.add_argument(
        "--answer-mode",
        choices=("graph_only", "mock", "live"),
        default="graph_only",
    )
    answer_parser.add_argument("--config", type=str, default=None)
    return parser


def _add_query_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--retrieval-mode",
        choices=("temporal", "entity"),
        default="temporal",
    )
    parser.add_argument("--query-text", type=str, default=None)
    parser.add_argument("--entity-id", action="append", default=[])
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--start-ts", type=float, default=None)
    parser.add_argument("--end-ts", type=float, default=None)
    parser.add_argument("--limit", type=int, default=8)


def _read_question(args: argparse.Namespace) -> str:
    if args.question is not None:
        return args.question.strip()
    return args.question_file.read_text(encoding="utf-8-sig").strip()


def _build_query(args: argparse.Namespace) -> MemoryQuery:
    return MemoryQuery(
        query_text=args.query_text,
        entity_ids=list(args.entity_id or []),
        scene_ids=list(args.scene_id or []),
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        limit=args.limit,
    )


def _canonical_summary(graph: CanonicalTemporalMemoryGraph) -> dict[str, object]:
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

    input_path = resolve_repo_input_path(args.canonical_graph_json)
    graph = CanonicalTemporalMemoryGraph.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    query = _build_query(args)

    if args.query_type == "retrieve":
        payload = retrieve_canonical_graph_context(
            graph,
            query,
            retrieval_mode=args.retrieval_mode,
            limit=args.limit,
        )
    elif args.query_type == "answer":
        payload = answer_canonical_graph_query(
            graph,
            query,
            question=_read_question(args),
            retrieval_mode=args.retrieval_mode,
            answer_mode=args.answer_mode,
            config_path=args.config,
            limit=args.limit,
        )
    else:  # pragma: no cover
        raise SystemExit(f"Unsupported query type: {args.query_type}")

    output = {
        "video_id": graph.video_id,
        "query_type": args.query_type,
        "canonical_graph_summary": _canonical_summary(graph),
        "result": payload,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
