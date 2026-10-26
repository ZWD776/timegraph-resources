from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.models.temporal_graph import TemporalGraphDocument
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.transformers import build_memory_ir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a temporal graph JSON document into a lightweight Memory IR document."
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
        help=(
            "Output JSON path. Defaults to `memory_ir.json` for the standard "
            "profile and `memory_ir.compact.json` for the compact profile."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("standard", "compact"),
        default="standard",
        help=(
            "standard keeps the fuller Memory IR coverage. compact removes "
            "fragment-backed memory records by default."
        ),
    )
    parser.add_argument(
        "--exclude-fragments",
        action="store_true",
        help="Exclude dynamic fragment entities from entity identities, state updates, and interaction events.",
    )
    parser.add_argument(
        "--include-unmapped-relations",
        action="store_true",
        help="Keep relation episodes whose predicate is UNMAPPED_RELATION.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_graph_json = resolve_repo_input_path(args.input_graph_json)
    is_compact_profile = args.profile == "compact"

    document = TemporalGraphDocument.model_validate_json(
        input_graph_json.read_text(encoding="utf-8")
    )
    memory_ir = build_memory_ir(
        document,
        source_graph_path=input_graph_json,
        include_fragments=not (args.exclude_fragments or is_compact_profile),
        include_unmapped_relations=args.include_unmapped_relations,
    )

    default_name = "memory_ir.compact.json" if is_compact_profile else "memory_ir.json"
    output_path = args.output or (input_graph_json.parent / default_name)
    output_path.write_text(
        memory_ir.model_dump_json(indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
