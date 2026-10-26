from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.qa.canonical_graph_qa_context import answer_canonical_graph_query
from vlkg.datasets.navqa.qa.graph_qa_pipeline import load_document
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.transformers import build_canonical_temporal_memory_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start an interactive canonical temporal memory graph QA chat session."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a temporal graph JSON document.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("temporal", "entity"),
        default="temporal",
        help="temporal prioritizes episodes/events, entity prioritizes entity-centered lifecycle context.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=("graph_only", "mock", "live"),
        default="graph_only",
        help="graph_only uses only canonical graph evidence, mock is deterministic offline, live calls the configured OpenAI-compatible LLM endpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional domain config path used to load graph_qa settings for live mode.",
    )
    parser.add_argument(
        "--entity-id",
        action="append",
        default=[],
        help="Optional fixed entity id filter. Repeat this flag to supply multiple entity ids.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Optional fixed scene id filter. Repeat this flag to supply multiple scene ids.",
    )
    parser.add_argument("--start-ts", type=float, default=None)
    parser.add_argument("--end-ts", type=float, default=None)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument(
        "--include-base-delta-projection",
        action="store_true",
        help=(
            "Opt in to virtual scene-base / scene-delta evidence records during "
            "canonical graph retrieval. The canonical graph itself is unchanged."
        ),
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print retrieval mode, focus frames, and evidence line count after each answer.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON for /last.",
    )
    return parser


def _print_help() -> None:
    print("Commands:")
    print("  /help    Show this help")
    print("  /history Show recent turns")
    print("  /last    Show the last raw JSON payload")
    print("  /clear   Clear chat memory")
    print("  /quit    Exit the chat")


def _print_history(history: list[dict[str, Any]]) -> None:
    if not history:
        print("History is empty.")
        return
    for index, item in enumerate(history, start=1):
        print(f"[{index}] Q: {item.get('question', '')}")
        print(f"    A: {item.get('answer_text', 'No answer.')}")


def _build_query(args: argparse.Namespace, question: str) -> MemoryQuery:
    return MemoryQuery(
        query_text=question,
        entity_ids=list(args.entity_id or []),
        scene_ids=list(args.scene_id or []),
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        limit=args.limit,
    )


def _print_response(payload: dict[str, Any], *, answer_mode: str) -> None:
    llm_answer = payload.get("llm_answer", {})
    answer_text = str(llm_answer.get("answer_text", "")).strip() or "No answer."
    print(f"Assistant> {answer_text}")

    if answer_mode == "graph_only":
        uncertainty = list(
            (llm_answer.get("structured_answer") or {}).get("uncertainty", [])
        )
        if uncertainty:
            print("Notes> " + "; ".join(str(item) for item in uncertainty))


def _print_metadata(payload: dict[str, Any]) -> None:
    graph_answer = payload.get("graph_answer", {})
    parsed_query = payload.get("parsed_query", {})
    llm_answer = payload.get("llm_answer", {})
    structured_answer = llm_answer.get("structured_answer", {})
    print(
        "Meta> "
        f"retrieval_mode={parsed_query.get('retrieval_mode', 'unknown')} "
        f"focus_frames={','.join(str(item) for item in structured_answer.get('used_focus_frames', [])) or '-'} "
        f"evidence_lines={len(structured_answer.get('used_evidence_lines', []))} "
        f"matches={len(graph_answer.get('matches', []))}"
    )


def main() -> None:
    args = build_parser().parse_args()
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    document = load_document(resolve_repo_input_path(args.input_json))
    canonical_graph = build_canonical_temporal_memory_graph(document)
    history: list[dict[str, Any]] = []
    last_payload: dict[str, Any] | None = None

    print(f"Canonical graph chat ready for video: {document.video_id}")
    print(f"Retrieval mode: {args.retrieval_mode}")
    print(f"Answer mode: {args.answer_mode}")
    _print_help()

    while True:
        try:
            raw_question = input("\nYou> ").strip()
        except EOFError:
            print("\nBye.")
            break
        except KeyboardInterrupt:
            print("\nBye.")
            break

        if not raw_question:
            continue
        if raw_question in {"/quit", "/exit"}:
            print("Bye.")
            break
        if raw_question == "/help":
            _print_help()
            continue
        if raw_question == "/history":
            _print_history(history)
            continue
        if raw_question == "/clear":
            history.clear()
            last_payload = None
            print("Chat memory cleared.")
            continue
        if raw_question == "/last":
            if last_payload is None:
                print("No previous payload.")
            else:
                print(json.dumps(last_payload, ensure_ascii=False, indent=2 if args.pretty else None))
            continue

        try:
            payload = answer_canonical_graph_query(
                canonical_graph,
                _build_query(args, raw_question),
                question=raw_question,
                retrieval_mode=args.retrieval_mode,
                answer_mode=args.answer_mode,
                config_path=args.config,
                limit=args.limit,
                temporal_document=document,
                include_base_delta_projection=args.include_base_delta_projection,
            )
        except KeyboardInterrupt:
            print("\nRequest interrupted.")
            continue
        except Exception as exc:  # pragma: no cover - defensive CLI behavior
            print(f"Error: {type(exc).__name__}: {exc}")
            continue

        last_payload = payload
        answer_text = str((payload.get("llm_answer") or {}).get("answer_text") or "").strip() or "No answer."
        history.append(
            {
                "question": raw_question,
                "answer_text": answer_text,
            }
        )

        _print_response(payload, answer_mode=args.answer_mode)
        if args.show_metadata:
            _print_metadata(payload)


if __name__ == "__main__":
    main()
