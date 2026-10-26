"""Build a fixed NAVQA route-strategy composite without per-question selection."""

from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
EXPECTED_QUESTION_COUNT = 139
PREDICTION_FILENAME = "qa_non_position_live_with_question_ir.json"

# These fields are produced before visual evidence fusion. A route overlay must
# not silently change the frozen graph retrieval input or graph evidence.
FROZEN_GRAPH_FIELDS = (
    "retrieval_query",
    "question_retrieval_window",
    "query_start_ts",
    "query_end_ts",
    "query_plan",
    "global_scope_score",
    "query_family",
    "core_terms",
    "retrieval_budget",
    "candidate_strategy",
    "candidate_count",
    "scene_seed_count",
    "seed_limit",
    "max_hops",
    "evidence_lines",
    "evidence_aggregation_lines",
    "scene_lines",
    "temporal_span_candidate_lines",
    "temporal_span_candidates",
    "semantic_event_candidates",
)


def _load_rows(path: Path, name: str) -> list[dict[str, Any]]:
    resolved = path.resolve()
    files: list[Path]
    if resolved.is_dir():
        files = sorted(resolved.rglob(PREDICTION_FILENAME))
        if not files:
            raise ValueError(f"{name} directory has no {PREDICTION_FILENAME}")
    else:
        files = [resolved]

    rows: list[dict[str, Any]] = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
            values = payload["results"]
        else:
            raise ValueError(f"{name} must contain a results list: {file_path}")
        if not all(isinstance(row, dict) for row in values):
            raise ValueError(f"{name} contains a non-object prediction: {file_path}")
        rows.extend(values)
    return [dict(row) for row in rows]


def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        uuid = str(row.get("uuid") or "").strip()
        if not uuid or uuid in indexed:
            raise ValueError(f"{name} has an empty or duplicate uuid: {uuid!r}")
        indexed[uuid] = row
    return indexed


def _route_id(row: dict[str, Any]) -> str:
    direct = str(row.get("question_route_id") or "").strip()
    if direct:
        return direct
    route = row.get("question_route")
    if isinstance(route, dict):
        return str(route.get("route_id") or "").strip()
    return ""


def _uses_gold(row: dict[str, Any]) -> bool:
    if row.get("uses_gold_answer") is True or row.get("uses_official_type") is True:
        return True
    route_backbone = row.get("question_route_hybrid_backbone")
    return isinstance(route_backbone, dict) and (
        route_backbone.get("uses_gold_answers") is True
        or route_backbone.get("uses_official_type") is True
    )


def _frozen_graph_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in FROZEN_GRAPH_FIELDS}


def _model_identity(rows: dict[str, dict[str, Any]], name: str) -> tuple[str, str]:
    identities = {
        (
            str(row.get("provider") or "").strip(),
            str(row.get("model") or "").strip(),
        )
        for row in rows.values()
        if str(row.get("provider") or "").strip()
        or str(row.get("model") or "").strip()
    }
    if len(identities) != 1:
        raise ValueError(f"{name} has mixed or missing provider/model identities")
    return next(iter(identities))


def _parse_overlay(value: str) -> tuple[str, Path]:
    route_id, separator, path_text = value.partition("=")
    route_id = route_id.strip()
    path_text = path_text.strip()
    if not separator or not route_id or not path_text:
        raise ValueError("overlay must use ROUTE_ID=PATH syntax")
    return route_id, Path(path_text)


def _load_official_module() -> Any:
    path = REPO_ROOT / "scripts/datasets/navqa/qa/run_navqa_official_qa_on_canonical_graph.py"
    spec = importlib.util.spec_from_file_location("navqa_official_qa_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge fixed NAVQA route strategies by question_route_id and uuid. "
            "This never selects the better answer per question."
        )
    )
    parser.add_argument("--base", type=Path, required=True, help="Base full-run JSON or run root.")
    parser.add_argument(
        "--overlay",
        action="append",
        required=True,
        metavar="ROUTE_ID=PATH",
        help="Route-wide replacement full-run JSON/run root; may be repeated.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_QUESTION_COUNT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base_rows = _load_rows(args.base, "base predictions")
    base = _index(base_rows, "base predictions")
    expected_count = int(args.expected_count)
    if len(base) != expected_count:
        raise ValueError(f"base predictions cover {len(base)} rows, expected {expected_count}")
    if any(_uses_gold(row) for row in base.values()):
        raise ValueError("base prediction purity failed: gold/official type controls a route")
    base_model_identity = _model_identity(base, "base predictions")

    base_routes = {uuid: _route_id(row) for uuid, row in base.items()}
    if any(not route_id for route_id in base_routes.values()):
        raise ValueError("base predictions contain a row without question_route_id")

    overlays: dict[str, Path] = {}
    for raw_overlay in args.overlay:
        route_id, path = _parse_overlay(raw_overlay)
        if route_id in overlays:
            raise ValueError(f"duplicate overlay route: {route_id}")
        overlays[route_id] = path

    merged = {uuid: deepcopy(row) for uuid, row in base.items()}
    provenance: list[dict[str, Any]] = []
    overlay_counts: dict[str, int] = {}
    for route_id, path in overlays.items():
        target_ids = {uuid for uuid, value in base_routes.items() if value == route_id}
        if not target_ids:
            raise ValueError(f"base predictions contain no questions for route {route_id!r}")
        candidate = _index(_load_rows(path, f"overlay {route_id}"), f"overlay {route_id}")
        if _model_identity(candidate, f"overlay {route_id}") != base_model_identity:
            raise ValueError(
                f"overlay {route_id!r} uses a different provider/model than the base run"
            )
        candidate_ids = set(candidate)
        if candidate_ids not in (target_ids, set(base)):
            raise ValueError(
                f"overlay {route_id!r} must cover exactly its route ({len(target_ids)}) "
                f"or the full set ({len(base)}); got {len(candidate_ids)}"
            )
        selected = {uuid: candidate[uuid] for uuid in target_ids}
        for uuid, row in selected.items():
            if _route_id(row) != route_id:
                raise ValueError(f"overlay route mismatch for {uuid}: expected {route_id}")
            if _uses_gold(row):
                raise ValueError(f"overlay purity failed for {uuid}")
            if _frozen_graph_projection(row) != _frozen_graph_projection(base[uuid]):
                raise ValueError(f"overlay changed frozen graph evidence for {uuid}")
            merged[uuid] = deepcopy(row)
        overlay_counts[route_id] = len(selected)

    for uuid in sorted(merged):
        provenance.append(
            {
                "uuid": uuid,
                "question_route_id": base_routes[uuid],
                "prediction_source": (
                    "route_overlay" if base_routes[uuid] in overlays else "base"
                ),
                "selects_better_answer_per_question": False,
            }
        )

    ordered_rows = [merged[uuid] for uuid in sorted(merged)]
    official = _load_official_module()
    metric = official.navqa_official_style_accuracy_summary(ordered_rows)
    payload = {
        "contract": "navqa_route_strategy_composite_v1",
        "question_count": len(ordered_rows),
        "base_predictions": str(args.base.resolve()),
        "route_overlays": {route: str(path.resolve()) for route, path in overlays.items()},
        "overlay_question_counts": overlay_counts,
        "selects_better_answer_per_question": False,
        "uses_gold_answer_for_selection": False,
        "uses_official_type_for_selection": False,
        "frozen_graph_projection_required": True,
        "provenance": provenance,
        "results": ordered_rows,
        "navqa_official_style_accuracy": metric,
    }
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_md:
        output_md = args.output_md.resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        scored_by_uuid = {
            str(item.get("uuid") or ""): item
            for item in metric.get("per_question", [])
            if isinstance(item, dict)
        }
        route_scores: dict[str, tuple[int, int]] = {}
        for uuid, route_id in base_routes.items():
            previous_correct, previous_total = route_scores.get(route_id, (0, 0))
            route_scores[route_id] = (
                previous_correct + int(bool(scored_by_uuid.get(uuid, {}).get("matched"))),
                previous_total + 1,
            )
        lines = [
            "# NAVQA Route Strategy Composite",
            "",
            f"- question_count: `{len(ordered_rows)}`",
            f"- overall: `{metric.get('correct', 0)}/{metric.get('total', 0)} = {metric.get('accuracy_percent', 0.0):.2f}%`",
            "- selects_better_answer_per_question: `false`",
            "- uses_gold_answer_for_selection: `false`",
            "- frozen_graph_projection_required: `true`",
            "",
            "| route | overlay questions | correct | total | accuracy |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for route_id, count in sorted(overlay_counts.items()):
            route_correct, route_total = route_scores.get(route_id, (0, 0))
            route_accuracy = route_correct * 100.0 / route_total if route_total else 0.0
            lines.append(
                f"| `{route_id}` | {count} | {route_correct} | "
                f"{route_total} | {route_accuracy:.2f}% |"
            )
        output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"output_json={output_json}")
    print(
        "overall="
        f"{metric.get('correct', 0)}/{metric.get('total', 0)} "
        f"= {float(metric.get('accuracy_percent', 0.0)):.2f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
