from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark_canonical_graph_qa import default_case_file, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical graph QA benchmark across multiple answer modes and summarize the comparison."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=default_case_file(),
        help="Path to a canonical graph QA benchmark case JSON file.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("graph_only", "mock", "live"),
        default=["graph_only", "mock", "live"],
        help="Answer modes to compare.",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("temporal", "entity"),
        default=None,
        help="Optional retrieval mode override applied to every case.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path, mainly relevant for live mode.",
    )
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
        "--include-improvement-metrics",
        action="store_true",
        help=(
            "Include read-only maturity metrics for the original time-axis, "
            "base/delta, stable-identity, and canonical-graph improvement goals."
        ),
    )
    parser.add_argument(
        "--include-base-delta-projection",
        action="store_true",
        help=(
            "Opt in to virtual scene-base / scene-delta evidence records during "
            "canonical graph retrieval. The canonical graph itself is unchanged."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the matrix JSON output.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to write a markdown summary report.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON payload.",
    )
    return parser


def _case_stub(case_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case_result.get("case_id"),
        "dataset": case_result.get("dataset"),
        "input_json": case_result.get("input_json"),
        "retrieval_mode": case_result.get("retrieval_mode"),
        "coverage_tags": case_result.get("coverage_tags", []),
        "include_base_delta_projection": case_result.get("include_base_delta_projection", False),
        "question": case_result.get("question"),
        "query": case_result.get("query"),
    }


def _mode_projection(case_result: dict[str, Any]) -> dict[str, Any]:
    failed_checks = [
        check
        for check in case_result.get("checks", [])
        if not check.get("passed")
    ]
    return {
        "passed": case_result.get("passed"),
        "graph_rag_status": case_result.get("graph_rag_status"),
        "graph_query_type": case_result.get("graph_query_type"),
        "match_count": case_result.get("match_count"),
        "focus_frame_count": case_result.get("focus_frame_count"),
        "scene_count": case_result.get("scene_count"),
        "used_focus_frames": case_result.get("used_focus_frames", []),
        "used_evidence_line_count": len(case_result.get("used_evidence_lines", [])),
        "top_match_source_record_ids": case_result.get("top_match_source_record_ids", []),
        "answer_text": case_result.get("answer_text"),
        "failed_checks": failed_checks,
    }


def build_matrix_payload(
    *,
    cases_path: Path,
    modes: list[str],
    retrieval_mode_override: str | None = None,
    config_path: str | None = None,
    compact_relation_observed_events: bool = False,
    compact_derived_temporal_edges: bool = True,
    compact_previous_episode_edges: bool = True,
    include_improvement_metrics: bool = False,
    include_base_delta_projection: bool = False,
) -> dict[str, Any]:
    mode_payloads: dict[str, dict[str, Any]] = {}
    case_matrix: dict[str, dict[str, Any]] = {}

    for mode in modes:
        payload = run_benchmark(
            cases_path=cases_path,
            answer_mode=mode,
            retrieval_mode_override=retrieval_mode_override,
            config_path=config_path,
            compact_relation_observed_events=compact_relation_observed_events,
            compact_derived_temporal_edges=compact_derived_temporal_edges,
            compact_previous_episode_edges=compact_previous_episode_edges,
            include_improvement_metrics=include_improvement_metrics,
            include_base_delta_projection=include_base_delta_projection,
        )
        mode_payloads[mode] = payload

        for case_result in payload.get("cases", []):
            case_id = str(case_result.get("case_id"))
            entry = case_matrix.setdefault(case_id, _case_stub(case_result))
            entry.setdefault("mode_results", {})
            entry["mode_results"][mode] = _mode_projection(case_result)

    ordered_case_results = list(case_matrix.values())
    mode_summaries = {
        mode: payload.get("summary", {})
        for mode, payload in mode_payloads.items()
    }
    return {
        "benchmark": "canonical_graph_qa_matrix",
        "case_file": str(cases_path),
        "modes": modes,
        "compact_relation_observed_events": compact_relation_observed_events,
        "compact_derived_temporal_edges": compact_derived_temporal_edges,
        "compact_previous_episode_edges": compact_previous_episode_edges,
        "include_improvement_metrics": include_improvement_metrics,
        "include_base_delta_projection": include_base_delta_projection,
        "graph_metrics_by_input": next(
            (
                payload.get("graph_metrics_by_input", [])
                for payload in mode_payloads.values()
                if payload.get("graph_metrics_by_input")
            ),
            [],
        ),
        "identity_diagnostics_by_input": next(
            (
                payload.get("identity_diagnostics_by_input", [])
                for payload in mode_payloads.values()
                if payload.get("identity_diagnostics_by_input")
            ),
            [],
        ),
        "improvement_metrics_by_input": next(
            (
                payload.get("improvement_metrics_by_input", [])
                for payload in mode_payloads.values()
                if payload.get("improvement_metrics_by_input")
            ),
            [],
        ),
        "mode_summaries": mode_summaries,
        "case_results": ordered_case_results,
        "mode_payloads": mode_payloads,
    }


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Canonical Graph QA Matrix")
    lines.append("")
    lines.append(f"- Case file: `{payload.get('case_file')}`")
    lines.append(f"- Modes: {', '.join(payload.get('modes', []))}")
    lines.append(
        f"- Compact RELATION_OBSERVED events: {payload.get('compact_relation_observed_events', False)}"
    )
    lines.append(
        f"- Compact derived temporal edges: {payload.get('compact_derived_temporal_edges', False)}"
    )
    lines.append(
        f"- Compact PREVIOUS_EPISODE edges: {payload.get('compact_previous_episode_edges', False)}"
    )
    lines.append(
        f"- Include improvement metrics: {payload.get('include_improvement_metrics', False)}"
    )
    lines.append(
        f"- Include base/delta projection: {payload.get('include_base_delta_projection', False)}"
    )
    graph_metrics_by_input = payload.get("graph_metrics_by_input", [])
    if graph_metrics_by_input:
        lines.append("")
        lines.append("## Graph Metrics")
        lines.append("")
        for item in graph_metrics_by_input:
            metrics = item.get("graph_metrics", {})
            lines.append(f"### {item.get('input_json')}")
            lines.append("")
            lines.append(
                f"- canonical graph: {metrics.get('node_count', 0)} nodes / "
                f"{metrics.get('edge_count', 0)} edges"
            )
            lines.append(
                f"- node breakdown: scenes={metrics.get('scene_node_count', 0)}, "
                f"entities={metrics.get('entity_node_count', 0)}, "
                f"stable_identities={metrics.get('stable_identity_node_count', 0)}, "
                f"episodes={metrics.get('episode_node_count', 0)}, "
                f"events={metrics.get('transition_event_node_count', 0)}"
            )
            top_edge_counts = sorted(
                metrics.get("edge_type_counts", {}).items(),
                key=lambda pair: (-int(pair[1]), str(pair[0])),
            )[:5]
            if top_edge_counts:
                edge_text = ", ".join(
                    f"{edge_type}={count}"
                    for edge_type, count in top_edge_counts
                )
                lines.append(f"- top edge types: {edge_text}")
            lines.append("")
    identity_diagnostics_by_input = payload.get("identity_diagnostics_by_input", [])
    if identity_diagnostics_by_input:
        lines.append("")
        lines.append("## Stable Identity Diagnostics")
        lines.append("")
        for item in identity_diagnostics_by_input:
            identity = item.get("entity_identity", {})
            diagnostics = identity.get("quality_diagnostics", {})
            lines.append(f"### {item.get('input_json')}")
            lines.append("")
            lines.append(
                f"- identities: {identity.get('stable_identity_count', 0)}, "
                f"singleton_ratio={identity.get('singleton_identity_ratio', 0.0)}, "
                f"member_count_distribution={identity.get('member_count_distribution', {})}"
            )
            lines.append(
                f"- weak human singletons: "
                f"{diagnostics.get('weak_human_singleton_count', 0)}"
            )
            lines.append(
                f"- sparse key objects: "
                f"{diagnostics.get('sparse_key_object_count', 0)}"
            )
            lines.append(
                f"- review merge suggestions: "
                f"{diagnostics.get('review_merge_suggestion_count', 0)}"
            )
            weak_singletons = diagnostics.get("weak_human_singletons", [])
            if weak_singletons:
                weak_text = ", ".join(
                    f"{item.get('short_id')}({item.get('category')})"
                    for item in weak_singletons[:5]
                )
                lines.append(f"- top weak human singletons: {weak_text}")
            sparse_objects = diagnostics.get("sparse_key_objects", [])
            if sparse_objects:
                sparse_text = ", ".join(
                    f"{item.get('short_id')}({item.get('category')})"
                    for item in sparse_objects[:5]
                )
                lines.append(f"- sparse key object examples: {sparse_text}")
            lines.append("")
    metrics_by_input = payload.get("improvement_metrics_by_input", [])
    if metrics_by_input:
        lines.append("")
        lines.append("## Improvement Metrics")
        lines.append("")
        for item in metrics_by_input:
            metrics = item.get("improvement_metrics", {})
            time_axis = metrics.get("time_axis", {})
            base_delta = metrics.get("base_delta", {})
            identity = metrics.get("entity_identity", {})
            canonical = metrics.get("canonical_graph", {}) or {}
            lines.append(f"### {item.get('input_json')}")
            lines.append("")
            lines.append(
                f"- canonical graph: {canonical.get('node_count', 0)} nodes / "
                f"{canonical.get('edge_count', 0)} edges"
            )
            lines.append(
                f"- time axis: entity_segment_coverage_ratio="
                f"{time_axis.get('entity_segment_coverage_ratio', 0.0)}"
            )
            lines.append(
                f"- base/delta: delta_to_observation_ratio="
                f"{base_delta.get('delta_to_observation_ratio', 0.0)}"
            )
            lines.append(
                f"- stable identity: singleton_identity_ratio="
                f"{identity.get('singleton_identity_ratio', 0.0)}"
            )
            lines.append("")
    lines.append("")
    lines.append("## Mode Summary")
    lines.append("")
    for mode in payload.get("modes", []):
        summary = payload.get("mode_summaries", {}).get(mode, {})
        lines.append(
            f"- `{mode}`: {summary.get('passed_case_count', 0)}/{summary.get('case_count', 0)} passed"
        )
        failed_ids = summary.get("failed_case_ids", [])
        if failed_ids:
            lines.append(f"  failed: {', '.join(str(item) for item in failed_ids)}")

    first_mode_summary = next(
        (
            payload.get("mode_summaries", {}).get(mode, {})
            for mode in payload.get("modes", [])
            if payload.get("mode_summaries", {}).get(mode, {})
        ),
        {},
    )
    coverage_tag_counts = first_mode_summary.get("coverage_tag_counts", {})
    if coverage_tag_counts:
        lines.append("")
        lines.append("## Coverage Summary")
        lines.append("")
        for tag in sorted(coverage_tag_counts):
            lines.append(f"- `{tag}`: {coverage_tag_counts[tag]} cases")
        missing_tags = first_mode_summary.get("cases_without_coverage_tags", [])
        if missing_tags:
            lines.append(
                f"- cases without coverage tags: {', '.join(str(item) for item in missing_tags)}"
            )
    lines.append("")
    lines.append("## Case Matrix")
    lines.append("")
    for case_result in payload.get("case_results", []):
        lines.append(f"### {case_result.get('case_id')}")
        lines.append("")
        lines.append(f"- Dataset: `{case_result.get('dataset')}`")
        lines.append(f"- Retrieval mode: `{case_result.get('retrieval_mode')}`")
        coverage_tags = [
            str(item)
            for item in case_result.get("coverage_tags", [])
            if str(item).strip()
        ]
        if coverage_tags:
            lines.append(f"- Coverage tags: {', '.join(coverage_tags)}")
        if case_result.get("include_base_delta_projection"):
            lines.append("- Base/delta projection: case-level opt-in")
        lines.append(f"- Question: {case_result.get('question')}")
        lines.append("")
        for mode in payload.get("modes", []):
            mode_result = case_result.get("mode_results", {}).get(mode, {})
            status = "PASS" if mode_result.get("passed") else "FAIL"
            lines.append(
                f"- `{mode}`: {status}, graph_rag_status={mode_result.get('graph_rag_status')}, "
                f"matches={mode_result.get('match_count')}, "
                f"focus_frames={mode_result.get('focus_frame_count')}, "
                f"used_evidence_lines={mode_result.get('used_evidence_line_count')}"
            )
            answer_text = str(mode_result.get("answer_text") or "").strip()
            if answer_text:
                lines.append(f"  answer: {answer_text}")
            failed_checks = mode_result.get("failed_checks", [])
            if failed_checks:
                failed_names = ", ".join(str(item.get("name")) for item in failed_checks)
                lines.append(f"  failed_checks: {failed_names}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = build_parser().parse_args()
    for stream_name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    payload = build_matrix_payload(
        cases_path=args.cases,
        modes=list(args.modes),
        retrieval_mode_override=args.retrieval_mode,
        config_path=args.config,
        compact_relation_observed_events=args.compact_relation_observed_events,
        compact_derived_temporal_edges=args.compact_derived_temporal_edges,
        compact_previous_episode_edges=args.compact_previous_episode_edges,
        include_improvement_metrics=args.include_improvement_metrics,
        include_base_delta_projection=args.include_base_delta_projection,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None),
            encoding="utf-8",
        )

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(
            build_markdown_report(payload),
            encoding="utf-8",
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
