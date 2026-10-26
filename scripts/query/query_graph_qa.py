from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.qa.graph_qa_pipeline import (
    answer_question,
    execute_parsed_query,
    load_document,
    parsed_query_payload,
)
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.graph_retrieval.query_engine import TemporalGraphQueryEngine
from vlkg.core.graph_retrieval.query_parser import parse_question, repair_question_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query local temporal-graph evidence for entity/state/relation questions (secondary path)."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a temporal graph JSON document.",
    )
    subparsers = parser.add_subparsers(dest="query_type", required=True)

    entity_parser = subparsers.add_parser(
        "entity",
        help="Query entity timeline / appear-disappear evidence.",
    )
    entity_parser.add_argument("--entity-id", type=str, default=None)
    entity_parser.add_argument("--label", type=str, default=None)
    entity_parser.add_argument("--text", type=str, default=None)
    entity_parser.add_argument("--limit", type=int, default=20)

    state_parser = subparsers.add_parser(
        "state",
        help="Query entity state intervals and evidence.",
    )
    state_parser.add_argument("--entity-id", type=str, default=None)
    state_parser.add_argument("--label", type=str, default=None)
    state_parser.add_argument("--text", type=str, default=None)
    state_parser.add_argument("--state-text", type=str, default=None)
    state_parser.add_argument("--limit", type=int, default=50)

    relation_parser = subparsers.add_parser(
        "relation",
        help="Query relation intervals and evidence.",
    )
    relation_parser.add_argument("--subject-entity-id", type=str, default=None)
    relation_parser.add_argument("--subject-label", type=str, default=None)
    relation_parser.add_argument("--subject-text", type=str, default=None)
    relation_parser.add_argument("--object-entity-id", type=str, default=None)
    relation_parser.add_argument("--object-label", type=str, default=None)
    relation_parser.add_argument("--object-text", type=str, default=None)
    relation_parser.add_argument("--predicate", type=str, default=None)
    relation_parser.add_argument("--limit", type=int, default=50)

    scene_parser = subparsers.add_parser(
        "scene",
        help="Query scene container summaries and scene-level evidence.",
    )
    scene_parser.add_argument("--scene-id", type=str, default=None)
    scene_parser.add_argument("--frame-index", type=int, default=None)
    scene_parser.add_argument("--entity-id", type=str, default=None)
    scene_parser.add_argument("--label", type=str, default=None)
    scene_parser.add_argument("--text", type=str, default=None)
    scene_parser.add_argument("--limit", type=int, default=20)

    frame_parser = subparsers.add_parser(
        "frame",
        help="Query frame-level static/dynamic snapshot evidence.",
    )
    frame_parser.add_argument("--frame-index", type=int, required=True)
    frame_parser.add_argument("--entity-id", type=str, default=None)
    frame_parser.add_argument("--label", type=str, default=None)
    frame_parser.add_argument("--text", type=str, default=None)
    frame_parser.add_argument("--limit", type=int, default=20)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Parse a simple natural-language question into a structured graph query.",
    )
    _add_question_input_group(ask_parser)

    answer_parser = subparsers.add_parser(
        "answer",
        help="Retrieve graph evidence first, then optionally render the answer through an LLM layer.",
    )
    _add_question_input_group(answer_parser)
    answer_parser.add_argument(
        "--answer-mode",
        choices=("graph_only", "mock", "live"),
        default="graph_only",
        help="graph_only uses only the local graph bundle, mock is deterministic offline, live calls the DeepSeek-compatible API over the local temporal-graph path.",
    )
    answer_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional domain config path used to load graph_qa settings for live mode.",
    )

    return parser


def _add_question_input_group(parser: argparse.ArgumentParser) -> None:
    ask_input_group = parser.add_mutually_exclusive_group(required=True)
    ask_input_group.add_argument(
        "--question",
        type=str,
        help='Inline natural-language question, for example: "robot when appear".',
    )
    ask_input_group.add_argument(
        "--question-file",
        type=Path,
        help=(
            "UTF-8 text file containing one question. "
            "Recommended when terminal encoding is unreliable."
        ),
    )


def _read_question(args: argparse.Namespace) -> str:
    question = (
        args.question
        if args.question is not None
        else args.question_file.read_text(encoding="utf-8-sig").strip()
    )
    return repair_question_text(question)


def main() -> None:
    args = build_parser().parse_args()
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    document = load_document(resolve_repo_input_path(args.input_json))
    engine = TemporalGraphQueryEngine(document)

    if args.query_type == "entity":
        payload = engine.answer_entity_timeline(
            entity_id=args.entity_id,
            label=args.label,
            text=args.text,
            limit=args.limit,
        )
    elif args.query_type == "scene":
        payload = engine.answer_scene_summary(
            scene_id=args.scene_id,
            frame_index=args.frame_index,
            entity_id=args.entity_id,
            label=args.label,
            text=args.text,
            limit=args.limit,
        )
    elif args.query_type == "frame":
        payload = engine.answer_frame_snapshot(
            frame_index=args.frame_index,
            entity_id=args.entity_id,
            label=args.label,
            text=args.text,
            limit=args.limit,
        )
    elif args.query_type == "state":
        payload = engine.answer_state_history(
            entity_id=args.entity_id,
            label=args.label,
            text=args.text,
            state_text=args.state_text,
            limit=args.limit,
        )
    elif args.query_type == "relation":
        payload = engine.answer_relation_history(
            subject_entity_id=args.subject_entity_id,
            subject_label=args.subject_label,
            subject_text=args.subject_text,
            object_entity_id=args.object_entity_id,
            object_label=args.object_label,
            object_text=args.object_text,
            predicate=args.predicate,
            limit=args.limit,
        )
    elif args.query_type == "ask":
        question = _read_question(args)
        parsed = parse_question(question)
        payload = {
            "question": question,
            "parsed_query": parsed_query_payload(parsed),
            "answer": execute_parsed_query(engine, parsed),
        }
    elif args.query_type == "answer":
        question = _read_question(args)
        payload = answer_question(
            engine,
            question,
            answer_mode=args.answer_mode,
            config_path=args.config,
        )
    else:  # pragma: no cover - argparse guarantees subcommand
        raise SystemExit(f"Unsupported query type: {args.query_type}")

    output = {
        "video_id": document.video_id,
        "query_type": args.query_type,
        "result": payload,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
