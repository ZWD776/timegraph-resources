from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.qa.chat_session import GraphChatSession
from vlkg.datasets.navqa.qa.graph_qa_pipeline import load_document
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.graph_retrieval.query_engine import TemporalGraphQueryEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start an interactive multi-turn local temporal-graph QA chat session in the terminal (secondary path)."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a temporal graph JSON document.",
    )
    parser.add_argument(
        "--answer-mode",
        choices=("graph_only", "mock", "live"),
        default="graph_only",
        help="graph_only uses only local graph evidence, mock is deterministic offline, live uses the DeepSeek-compatible API over the local temporal-graph pipeline.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional domain config path used to load graph_qa settings for live mode.",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=8,
        help="How many recent turns are kept for follow-up question resolution.",
    )
    parser.add_argument(
        "--show-metadata",
        action="store_true",
        help="Print parsed query type, focus frames, and evidence line count after each answer.",
    )
    return parser


def _print_help() -> None:
    print("Commands:")
    print("  /help    Show this help")
    print("  /history Show recent turns")
    print("  /clear   Clear chat memory")
    print("  /quit    Exit the chat")


def _print_history(session: GraphChatSession) -> None:
    if not session.turns:
        print("History is empty.")
        return
    for index, turn in enumerate(session.turns, start=1):
        print(f"[{index}] Q: {turn.question}")
        if turn.resolved_question != turn.question:
            print(f"    resolved: {turn.resolved_question}")
        print(f"    type: {turn.parsed_query_type or 'unknown'}")
        if turn.used_focus_frames:
            print(
                "    focus_frames: "
                + ", ".join(str(item) for item in turn.used_focus_frames)
            )
        print(f"    A: {turn.answer_text}")


def main() -> None:
    args = build_parser().parse_args()
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    document = load_document(resolve_repo_input_path(args.input_json))
    engine = TemporalGraphQueryEngine(document)
    session = GraphChatSession(
        engine,
        answer_mode=args.answer_mode,
        config_path=args.config,
        max_history_turns=args.max_history_turns,
    )

    print(f"Graph chat ready for video: {document.video_id}")
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
        if raw_question == "/clear":
            session.clear()
            print("Chat memory cleared.")
            continue
        if raw_question == "/history":
            _print_history(session)
            continue

        try:
            payload = session.ask(raw_question)
        except KeyboardInterrupt:
            print("\nRequest interrupted.")
            continue
        except Exception as exc:  # pragma: no cover - defensive CLI behavior
            print(f"Error: {type(exc).__name__}: {exc}")
            continue

        llm_answer = payload.get("llm_answer", {})
        parsed_query = payload.get("parsed_query", {})
        structured_answer = llm_answer.get("structured_answer", {})
        resolved_question = (
            payload.get("chat_context", {}) or {}
        ).get("resolved_question", raw_question)
        answer_text = str(llm_answer.get("answer_text", "")).strip() or "No answer."
        print(f"Assistant> {answer_text}")

        if resolved_question != raw_question:
            print(f"Resolved> {resolved_question}")

        if args.show_metadata:
            used_focus_frames = [
                str(item)
                for item in structured_answer.get("used_focus_frames", [])
                if isinstance(item, int)
            ]
            used_evidence_lines = list(structured_answer.get("used_evidence_lines", []))
            print(
                "Meta> "
                f"type={parsed_query.get('query_type', 'unknown')} "
                f"focus_frames={','.join(used_focus_frames) or '-'} "
                f"evidence_lines={len(used_evidence_lines)}"
            )


if __name__ == "__main__":
    main()
