from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.config.config import GraphQAConfig, load_domain_config
from vlkg.datasets.navqa.runtime.paths import default_navqa_sequence_layout
from vlkg.core.transport.openai_compatible_payload import apply_reasoning_options
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.graph_retrieval.question_compiler import compile_question
from vlkg.datasets.navqa.retrieval.question_router import route_question


FIELD_SEQ_ID = "Seq ID"
FIELD_UUID = "UUID"
FIELD_QUESTION = "Question"
FIELD_TYPE = "Type \n(binary, position, time, text)"
FIELD_CATEGORY = "Question\nCategory"
FIELD_TEXT_ANSWER = "Text answer"
FIELD_PARSABLE_ANSWER = "Parsable answer"

NON_POSITION_QA_TYPES = ("binary", "text", "time", "duration")

ALLOWED_ANSWER_TYPES = {
    "binary",
    "position",
    "time",
    "duration",
    "text",
    "count",
    "location_text",
}
ALLOWED_OPERATIONS = {
    "attribute_query",
    "count_query",
    "descriptive_query",
    "existence_query",
    "functional_query",
    "identity_query",
    "location_query",
    "onset",
    "offset",
    "interval",
    "current_state",
    "occurrence",
    "spatial_relation",
    "state_query",
    "world_knowledge_query",
    "navigation_direction",
    "position_query",
    "unsupported",
    "unknown",
}
ALLOWED_RENDERING = {
    "yes_no",
    "minutes_ago",
    "duration_minutes",
    "location_phrase",
    "number_or_short_phrase",
    "color_only",
    "short_phrase",
    "free_text",
    "position_coordinates",
    "unsupported",
    "unknown",
}
ALLOWED_QA_FOCUS = {
    "descriptive",
    "time_duration",
    "binary",
    "spatial_geometry",
    "navigation_direction",
    "position",
    "unsupported",
    "unknown",
}
ALLOWED_EVIDENCE_QUESTION_TYPES = {
    "temporal.point",
    "temporal.interval",
    "non_temporal.entity.existence",
    "non_temporal.entity.identity",
    "non_temporal.entity.attribute",
    "non_temporal.entity.count",
    "non_temporal.entity.relation",
    "non_temporal.event.existence",
    "non_temporal.event.participant",
    "non_temporal.event.outcome",
    "non_temporal.state.scene",
    "non_temporal.state.object",
    "non_temporal.state.traversability",
    "non_temporal.ego.motion",
    "non_temporal.ego.direction",
    "non_temporal.ego.interaction",
    "non_temporal.symbolic.ocr_logo",
    "non_temporal.spatial.localization",
    "non_temporal.spatial.relation",
    "non_temporal.functional.affordance",
    "non_temporal.world_knowledge.contextual",
    "non_temporal.open_descriptive",
    "unknown",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experimentally compile NaVQA questions into LLM-generated semantic IR "
            "without providing official answers or graph evidence."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Official NaVQA CSV.")
    parser.add_argument("--seq-id", default="0")
    parser.add_argument("--cam", default="0")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--layout-root", type=Path, default=Path("datasets/navqa/source/navqa-coda-every10-v1/payload/navqa"))
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--include-types", nargs="+", default=None)
    parser.add_argument("--exclude-types", nargs="+", default=None)
    parser.add_argument(
        "--non-position-only",
        action="store_true",
        help="Shortcut for --include-types binary text time duration.",
    )
    parser.add_argument("--domain-profile", default="generic")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=("live", "mock"),
        default="live",
        help="mock uses the local rule compiler and does not call an LLM.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
    )
    return parser


def load_questions(csv_path: Path, seq_id: str) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if str(row.get(FIELD_SEQ_ID, "")).strip() == str(seq_id)
        ]


def filter_questions_by_type(
    rows: list[dict[str, str]],
    *,
    include_types: list[str] | tuple[str, ...] | None = None,
    exclude_types: list[str] | tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    includes = _normalized_type_set(include_types)
    excludes = _normalized_type_set(exclude_types) or set()
    selected: list[dict[str, str]] = []
    for row in rows:
        qtype = _row_type(row)
        if includes is not None and qtype not in includes:
            continue
        if qtype in excludes:
            continue
        selected.append(row)
    return selected


def compile_questions_with_llm(
    rows: list[dict[str, str]],
    *,
    config: GraphQAConfig,
    batch_size: int,
    max_tokens: int,
    temperature: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    api_key = os.getenv(config.api_key_env_var)
    if not api_key:
        raise RuntimeError(f"Missing environment variable: {config.api_key_env_var}")

    items: list[dict[str, Any]] = []
    batch_logs: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(_batched(rows, max(1, batch_size)), start=1):
        payload = _build_llm_payload(
            batch,
            config=config,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_response = _post_payload(payload, api_key=api_key, config=config)
        content = _extract_content(raw_response)
        parsed = _parse_json_value(content or "")
        raw_items, extraction_log = _extract_ir_items(parsed)
        aligned_items, alignment_log = _align_ir_items_to_rows(raw_items, batch)
        items.extend(aligned_items)
        batch_logs.append(
            {
                "batch_index": batch_index,
                "question_count": len(batch),
                "response_item_count": len(raw_items),
                "raw_content_chars": len(content or ""),
                "raw_content": _truncate_text(content or "", max_chars=20000),
                "parsed_shape": _json_shape(parsed),
                "extraction": extraction_log,
                "alignment": alignment_log,
                "response_metadata": _response_metadata(raw_response),
            }
        )
    return items, batch_logs


def compile_questions_mock(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        plan = compile_question(
            _row_question(row),
            answer_type=_row_type(row),
            category=str(row.get(FIELD_CATEGORY, "")).strip() or None,
            domain_profile="generic",
        )
        operation = _operation_from_rule_plan(plan.to_dict())
        items.append(
            {
                "uuid": row.get(FIELD_UUID, ""),
                "answer_type": plan.answer_type,
                "operation": operation,
                "target_state": "",
                "target_entities": list(plan.target_entities),
                "attribute_dimension": (list(plan.attributes) or [""])[0],
                "event_constraints": [],
                "spatial_constraints": list(plan.spatial_constraints),
                "hard_constraints": [],
                "aggregation": "single_best",
                "rendering": plan.answer_rendering or "free_text",
                "qa_focus": _qa_focus_from_rule_plan(plan.to_dict()),
                "evidence_requirements": [],
                "graph_information_needs": [],
                "unsupported_reason": "",
                "confidence": 0.5,
                "rationale": "mock item from local rule compiler",
            }
        )
    return items


def build_records(
    rows: list[dict[str, str]],
    raw_ir_items: list[dict[str, Any]],
    *,
    domain_profile: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        raw_ir = raw_ir_items[index - 1] if index - 1 < len(raw_ir_items) else {}
        compiler_plan = compile_question(
            _row_question(row),
            answer_type=_row_type(row),
            category=str(row.get(FIELD_CATEGORY, "")).strip() or None,
            domain_profile=domain_profile,
        )
        rule_plan = compiler_plan.to_dict()
        normalized_ir = normalize_ir_item(raw_ir, row=row)
        question_route = route_question(
            _row_question(row),
            official_question_type=_row_type(row) or None,
            question_ir=normalized_ir,
            compiler_plan=compiler_plan,
            domain_profile=domain_profile,
            category=str(row.get(FIELD_CATEGORY, "")).strip() or None,
        ).to_dict()
        evidence_question_type = str(
            normalized_ir.get("evidence_question_type") or ""
        ).strip()
        if not evidence_question_type or evidence_question_type == "unknown":
            evidence_question_type = str(
                question_route.get("taxonomy_path") or "unknown"
            ).strip()
        normalized_ir["evidence_question_type"] = evidence_question_type
        normalized_ir["question_route_id"] = question_route.get("route_id") or ""
        normalized_ir["question_route"] = question_route
        validation = validate_ir_item(normalized_ir, row=row)
        records.append(
            {
                "index": index,
                "uuid": row.get(FIELD_UUID, ""),
                "question": _row_question(row),
                "question_type": _row_type(row),
                "question_category": str(row.get(FIELD_CATEGORY, "")).strip(),
                "llm_question_ir": normalized_ir,
                "validation": validation,
                "rule_question_plan": rule_plan,
            }
        )
    return records


def normalize_ir_item(raw: dict[str, Any], *, row: dict[str, str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    answer_type = _normalize_enum(
        raw.get("answer_type"),
        allowed=ALLOWED_ANSWER_TYPES,
        default=_row_type(row) or "text",
    )
    operation = _normalize_enum(
        raw.get("operation"),
        allowed=ALLOWED_OPERATIONS,
        default="unknown",
    )
    rendering = _normalize_enum(
        raw.get("rendering"),
        allowed=ALLOWED_RENDERING,
        default=_default_rendering(answer_type),
    )
    qa_focus = _normalize_enum(
        raw.get("qa_focus"),
        allowed=ALLOWED_QA_FOCUS,
        default=_default_focus(answer_type),
    )
    evidence_question_type = _normalize_evidence_question_type(
        raw.get("evidence_question_type") or raw.get("question_type_taxonomy")
    )
    if not evidence_question_type:
        evidence_question_type = _infer_evidence_question_type_from_ir(
            answer_type=answer_type,
            operation=operation,
            qa_focus=qa_focus,
            attribute_dimension=_string_or_empty(raw.get("attribute_dimension")),
        )
    return {
        "uuid": str(raw.get("uuid") or row.get(FIELD_UUID, "")).strip(),
        "answer_type": answer_type,
        "operation": operation,
        "evidence_question_type": evidence_question_type,
        "target_state": _string_or_empty(raw.get("target_state")),
        "target_entities": _string_list(raw.get("target_entities")),
        "attribute_dimension": _string_or_empty(raw.get("attribute_dimension")),
        "temporal_relation": _string_or_empty(raw.get("temporal_relation")),
        "event_constraints": _object_list(raw.get("event_constraints")),
        "spatial_constraints": _object_list(raw.get("spatial_constraints")),
        "hard_constraints": _object_list(raw.get("hard_constraints")),
        "aggregation": _string_or_empty(raw.get("aggregation")) or "unknown",
        "rendering": rendering,
        "qa_focus": qa_focus,
        "evidence_requirements": _string_list(raw.get("evidence_requirements")),
        "graph_information_needs": _string_list(raw.get("graph_information_needs")),
        "unsupported_reason": _string_or_empty(raw.get("unsupported_reason")),
        "confidence": _bounded_float(raw.get("confidence"), default=0.0),
        "rationale": _string_or_empty(raw.get("rationale")),
        "parse_error": _string_or_empty(raw.get("parse_error")),
    }


def validate_ir_item(ir: dict[str, Any], *, row: dict[str, str]) -> dict[str, Any]:
    qtype = _row_type(row)
    question = _row_question(row).lower()
    issues: list[str] = []
    warnings: list[str] = []

    if ir.get("parse_error"):
        issues.append(f"parse_error:{ir['parse_error']}")
    if ir["answer_type"] != qtype and qtype:
        warnings.append(f"answer_type_differs_from_official_type:{ir['answer_type']}!={qtype}")
    evidence_question_type = str(ir.get("evidence_question_type") or "").strip()
    if not evidence_question_type or evidence_question_type == "unknown":
        warnings.append("missing_evidence_question_type")

    operation = str(ir.get("operation") or "")
    rendering = str(ir.get("rendering") or "")
    focus = str(ir.get("qa_focus") or "")
    attribute = str(ir.get("attribute_dimension") or "").lower()

    if qtype == "time":
        if operation not in {"onset", "offset", "occurrence", "current_state"}:
            issues.append(f"time_operation_unexpected:{operation}")
        if rendering != "minutes_ago":
            issues.append(f"time_rendering_unexpected:{rendering}")
    elif qtype == "duration":
        if operation not in {"interval", "current_state"}:
            issues.append(f"duration_operation_unexpected:{operation}")
        if rendering != "duration_minutes":
            issues.append(f"duration_rendering_unexpected:{rendering}")
    elif qtype == "binary":
        if rendering != "yes_no":
            issues.append(f"binary_rendering_unexpected:{rendering}")
    elif qtype == "position":
        if operation != "position_query":
            warnings.append(f"position_operation_unexpected:{operation}")
        if rendering != "position_coordinates":
            warnings.append(f"position_rendering_unexpected:{rendering}")

    if re.search(r"\bcolou?r|colored|colors\b", question):
        if operation != "attribute_query":
            issues.append(f"color_question_not_attribute_query:{operation}")
        if attribute != "color":
            issues.append(f"color_question_attribute_unexpected:{attribute or 'empty'}")
        if rendering != "color_only":
            issues.append(f"color_question_rendering_unexpected:{rendering}")
        if focus != "descriptive":
            warnings.append(f"color_question_focus_unexpected:{focus}")

    if re.search(r"\bwhat\s+side\b|\bwhich\s+side\b", question):
        if focus != "spatial_geometry":
            warnings.append(f"side_question_focus_should_be_spatial_geometry:{focus}")
        if operation != "spatial_relation":
            warnings.append(f"side_question_operation_should_be_spatial_relation:{operation}")

    if re.search(r"\bturn(?:ed)?\b|\bdirection\b", question):
        if focus != "navigation_direction":
            warnings.append(f"turn_question_focus_should_be_navigation_direction:{focus}")
        if operation != "navigation_direction":
            warnings.append(f"turn_question_operation_should_be_navigation_direction:{operation}")

    if not ir.get("target_entities") and qtype not in {"binary"}:
        warnings.append("empty_target_entities")

    if _question_has_constraint_clause(question):
        has_structured_constraints = bool(
            ir.get("event_constraints")
            or ir.get("spatial_constraints")
            or ir.get("hard_constraints")
            or str(ir.get("temporal_relation") or "").strip()
        )
        if not has_structured_constraints:
            warnings.append("possible_missing_hard_constraints")

    return {
        "auto_reasonable": not issues,
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "issues": issues,
        "warnings": warnings,
    }


def _question_has_constraint_clause(question: str) -> bool:
    return bool(
        re.search(
            (
                r"\b(that|which|who|whose|where|when|near|nearby|around|after|"
                r"before|during|while|with|beside|by|inside|outside|enter(?:ed|ing)?|"
                r"leave|leaving|left|exit(?:ed|ing)?|open(?:ed|ing)?|pass(?:ed|ing)?)\b"
            ),
            question,
        )
    )


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    by_operation: dict[str, int] = {}
    by_focus: dict[str, int] = {}
    by_evidence_question_type: dict[str, int] = {}
    by_question_route: dict[str, int] = {}
    reasonable = 0
    for record in records:
        qtype = str(record.get("question_type") or "unknown")
        validation = record.get("validation", {})
        ir = record.get("llm_question_ir", {})
        bucket = by_type.setdefault(qtype, {"total": 0, "auto_reasonable": 0})
        bucket["total"] += 1
        if validation.get("auto_reasonable"):
            bucket["auto_reasonable"] += 1
            reasonable += 1
        operation = str(ir.get("operation") or "unknown")
        focus = str(ir.get("qa_focus") or "unknown")
        evidence_question_type = str(ir.get("evidence_question_type") or "unknown")
        route_id = str((ir.get("question_route") or {}).get("route_id") or "unknown")
        by_operation[operation] = by_operation.get(operation, 0) + 1
        by_focus[focus] = by_focus.get(focus, 0) + 1
        by_evidence_question_type[evidence_question_type] = (
            by_evidence_question_type.get(evidence_question_type, 0) + 1
        )
        by_question_route[route_id] = by_question_route.get(route_id, 0) + 1
    for bucket in by_type.values():
        bucket["auto_reasonable_percent"] = round(
            (bucket["auto_reasonable"] / bucket["total"] * 100.0)
            if bucket["total"]
            else 0.0,
            2,
        )
    total = len(records)
    return {
        "total": total,
        "auto_reasonable": reasonable,
        "auto_reasonable_percent": round(
            (reasonable / total * 100.0) if total else 0.0,
            2,
        ),
        "by_type": by_type,
        "by_operation": by_operation,
        "by_focus": by_focus,
        "by_evidence_question_type": dict(sorted(by_evidence_question_type.items())),
        "by_question_route": dict(sorted(by_question_route.items())),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    lines = [
        "# NaVQA LLM Question IR Experiment",
        "",
        f"- Questions: {summary.get('total', 0)}",
        (
            f"- Auto-reasonable IR: {summary.get('auto_reasonable', 0)}/"
            f"{summary.get('total', 0)} "
            f"({summary.get('auto_reasonable_percent', 0)}%)"
        ),
        f"- Prompt excludes official answers: {payload.get('prompt_excludes_official_answers')}",
        f"- Prompt excludes graph evidence: {payload.get('prompt_excludes_graph_evidence')}",
        "",
        "## By Type",
        "",
    ]
    for qtype, bucket in sorted(summary.get("by_type", {}).items()):
        lines.append(
            f"- {qtype}: {bucket.get('auto_reasonable', 0)}/"
            f"{bucket.get('total', 0)} ({bucket.get('auto_reasonable_percent', 0)}%)"
        )
    lines.extend(["", "## By Evidence Question Type", ""])
    for qtype, count in sorted(summary.get("by_evidence_question_type", {}).items()):
        lines.append(f"- {qtype}: {count}")
    lines.extend(["", "## By Question Route", ""])
    for route_id, count in sorted(summary.get("by_question_route", {}).items()):
        lines.append(f"- {route_id}: {count}")
    lines.extend(["", "## Questions", ""])
    for record in records:
        ir = record.get("llm_question_ir", {})
        validation = record.get("validation", {})
        status = "OK" if validation.get("auto_reasonable") else "CHECK"
        lines.extend(
            [
                f"### {record.get('index')}. {status} {record.get('question')}",
                "",
                f"- Type: `{record.get('question_type')}`",
                f"- Evidence type: `{ir.get('evidence_question_type')}`",
                f"- Question route: `{(ir.get('question_route') or {}).get('route_id')}`",
                f"- Operation: `{ir.get('operation')}`",
                f"- Focus: `{ir.get('qa_focus')}`",
                f"- Target state: `{ir.get('target_state')}`",
                f"- Targets: `{', '.join(ir.get('target_entities') or [])}`",
                f"- Attribute: `{ir.get('attribute_dimension')}`",
                f"- Hard constraints: `{json.dumps(ir.get('hard_constraints') or [], ensure_ascii=False)}`",
                f"- Rendering: `{ir.get('rendering')}`",
                f"- Confidence: `{ir.get('confidence')}`",
                f"- Rationale: {ir.get('rationale') or ''}",
            ]
        )
        if validation.get("issues"):
            lines.append(f"- Issues: `{'; '.join(validation['issues'])}`")
        if validation.get("warnings"):
            lines.append(f"- Warnings: `{'; '.join(validation['warnings'])}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_llm_payload(
    rows: list[dict[str, str]],
    *,
    config: GraphQAConfig,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    prompt_payload = {
        "task": "compile_video_navigation_questions_to_semantic_ir",
        "dataset_context": {
            "viewpoint": "first-person robot/ego-camera navigation video",
            "pronoun_policy": (
                "In these questions, 'I', 'you', and 'we' refer to the robot/ego "
                "camera, not an external human observer."
            ),
            "observation_scope": (
                "Questions ask about what the robot saw, where the robot was, "
                "or how the robot moved through the environment."
            ),
            "context_is_not_evidence": (
                "This background only disambiguates viewpoint and pronouns; it "
                "must not be treated as an answer or as graph evidence."
            ),
        },
        "do_not_answer_questions": True,
        "use_only_question_wording_and_type_hints": True,
        "official_answers_are_not_provided": True,
        "graph_evidence_is_not_provided": True,
        "output_contract": {
            "top_level_object": "Return exactly one JSON object with the single key 'items'.",
            "item_alignment": (
                "Return exactly one item for each input question, in the same order, "
                "and copy each uuid exactly."
            ),
            "no_extra_text": "Do not include markdown, explanations outside JSON, or hidden chain-of-thought.",
            "answer_type_hint": (
                "Treat official_question_type_hint as a strong output contract. "
                "Only deviate if the wording makes the hint impossible, and explain that briefly in rationale."
            ),
        },
        "evidence_question_type_taxonomy": [
            "temporal.point",
            "temporal.interval",
            "non_temporal.entity.existence",
            "non_temporal.entity.identity",
            "non_temporal.entity.attribute",
            "non_temporal.entity.count",
            "non_temporal.entity.relation",
            "non_temporal.event.existence",
            "non_temporal.event.participant",
            "non_temporal.event.outcome",
            "non_temporal.state.scene",
            "non_temporal.state.object",
            "non_temporal.state.traversability",
            "non_temporal.ego.motion",
            "non_temporal.ego.direction",
            "non_temporal.ego.interaction",
            "non_temporal.symbolic.ocr_logo",
            "non_temporal.spatial.localization",
            "non_temporal.spatial.relation",
            "non_temporal.functional.affordance",
            "non_temporal.world_knowledge.contextual",
            "non_temporal.open_descriptive",
            "unknown",
        ],
        "schema": {
            "items": [
                {
                    "uuid": "string",
                    "answer_type": "binary|position|time|duration|text|count|location_text",
                    "operation": (
                        "attribute_query|existence_query|onset|offset|interval|"
                        "current_state|occurrence|spatial_relation|navigation_direction|"
                        "position_query|identity_query|count_query|location_query|"
                        "state_query|functional_query|world_knowledge_query|"
                        "descriptive_query|unsupported|unknown"
                    ),
                    "evidence_question_type": (
                        "one value from evidence_question_type_taxonomy; this is the "
                        "dataset-independent evidence-need type used by downstream routing"
                    ),
                    "target_state": "open vocabulary state, or empty string",
                    "target_entities": ["open vocabulary entity names"],
                    "attribute_dimension": (
                        "color|clothing|material|appearance|type|state|side|"
                        "direction|busy|empty string"
                    ),
                    "temporal_relation": "first|latest|current|before|after|during|empty",
                    "event_constraints": [
                        {
                            "relation": "near|after|before|during|where|when",
                            "operation": "same operation enum when applicable",
                            "target_state": "open vocabulary",
                            "target_entities": ["open vocabulary entity names"],
                        }
                    ],
                    "spatial_constraints": [
                        {"relation": "near|on|in|beside|side_of", "target": "string"}
                    ],
                    "hard_constraints": [
                        {
                            "kind": "entity|attribute|event|spatial|temporal|relation|state",
                            "text": "meaningful qualifier copied or normalized from the question",
                            "role": (
                                "target_modifier|event_condition|spatial_condition|"
                                "temporal_condition|relation_condition"
                            ),
                            "target_entities": ["entities affected by this qualifier"],
                            "must_bind_to": [
                                "target|answer_entity|same_scene|same_time_window"
                            ],
                        }
                    ],
                    "aggregation": "single_best|multiple|all|interval|yes_no|unknown",
                    "rendering": (
                        "yes_no|minutes_ago|duration_minutes|number_or_short_phrase|"
                        "location_phrase|color_only|short_phrase|free_text|"
                        "position_coordinates|unsupported|unknown"
                    ),
                    "qa_focus": (
                        "descriptive|time_duration|binary|spatial_geometry|"
                        "navigation_direction|position|unsupported|unknown"
                    ),
                    "evidence_requirements": ["what evidence would be needed"],
                    "graph_information_needs": [
                        "information a graph must contain to answer, e.g. ego trajectory"
                    ],
                    "unsupported_reason": "why this is hard/unsupported, or empty",
                    "confidence": "number between 0 and 1",
                    "rationale": "one short sentence explaining the parse",
                }
            ]
        },
        "decision_rules": {
            "general": [
                "Compile the question's information need; do not answer it.",
                "Assign evidence_question_type according to the evidence form needed to answer the question, not according to dataset id or answer content.",
                "Use temporal.point for a requested time anchor and temporal.interval for a requested duration/span.",
                "Use non_temporal.entity.existence for yes/no object-presence questions, non_temporal.entity.attribute for direct object attributes, non_temporal.entity.count for set counting, and non_temporal.spatial.relation for relative layout.",
                "Use non_temporal.entity.identity when the answer is an object identity or class name.",
                "Use non_temporal.spatial.localization when the question asks where an object or room is and expects a location phrase.",
                "Use non_temporal.functional.affordance for questions about what an object or room can be used for.",
                "Use non_temporal.world_knowledge.contextual when commonsense must be grounded in scene context.",
                "Use non_temporal.symbolic.ocr_logo when the answer depends on visible text, signs, logos, letters, or printed words.",
                "Use non_temporal.ego.motion/direction/interaction when the evidence is about the camera/robot's own movement, side, turn, or interaction.",
                "Interpret first-person pronouns ('I', 'you', 'we') as the robot/ego camera.",
                "Treat 'see/saw' as visual observation by the robot camera.",
                "Treat 'right now/currently' as the robot's state at the question time, not as wall-clock now.",
                "Prefer open-vocabulary target_entities and target_state over dataset-specific labels.",
                "Keep target_entities short and retrieval-friendly: use the core object/person/place names only.",
                "Keep entities separate from attributes: e.g. a red object -> entity object, attribute_dimension color, target_state or constraints mention red.",
                "Do not put relative clauses, event clauses, spatial clauses, temporal clauses, counts, or descriptive modifiers inside target_entities; put them in event_constraints, spatial_constraints, temporal_relation, attribute_dimension, or hard_constraints.",
                "For phrases shaped like 'target that/which/who ...', keep only the target entity in target_entities and preserve the clause as a hard_constraint or event_constraint.",
                "For phrases shaped like 'target near/next to/in front of/behind ...', keep only the target entity in target_entities and preserve the location phrase as a spatial_constraint and hard_constraint.",
                "For attribute phrases, keep only the object type in target_entities and put the attribute in attribute_dimension, target_state, or hard_constraints.",
                "Use hard_constraints to preserve question factors that must be considered downstream: relative clauses, event clauses, spatial clauses, temporal clauses, and prepositional phrases must be represented in event_constraints, spatial_constraints, temporal_relation, or hard_constraints.",
                "hard_constraints preserve question wording and retrieval intent; they are not an instruction to reject reasonable graph-supported inference when direct one-line evidence is absent.",
                "Do not drop qualifiers such as relative clauses, action clauses, entry/exit location clauses, object-with-attribute clauses, or clauses that bind an entity to an event; represent them as generic question factors without answering the question.",
                "Do not mark a question unsupported just because it requires hard evidence; classify the semantic operation and list graph_information_needs.",
            ],
            "time_duration": [
                "Use onset for the start of a state or event: first enter, start being inside/on/at.",
                "Use offset for the end of a state: leave, exit, get off, stop being inside/on/at.",
                "Use interval only for explicit duration wording: how long, for how long, how much time, duration.",
                "For official_question_type_hint=time, rendering should normally be minutes_ago, not duration_minutes.",
                "For 'When were you in/at/on X?' with a time hint, classify as occurrence or current_state over the state of being at X; do not use interval unless the wording asks how long.",
                "Use occurrence for event/process timing when the question asks when something happened but does not ask for a start or end.",
                "Use current_state for right now/currently questions about the present state.",
            ],
            "descriptive": [
                "Use attribute_query for color, clothing, material, appearance, object type, occupancy, busy/empty state, or other visual descriptions.",
                "Set qa_focus=descriptive for ordinary visual attribute questions.",
                "Use color_only rendering only when the requested answer is specifically a color; otherwise use short_phrase or free_text.",
                "For questions that describe an entity through an event, bind the event as an event_constraint and query the entity attribute.",
            ],
            "binary": [
                "Use existence_query for did you see/was there/are there yes-no existence questions.",
                "Use current_state for right now/currently yes-no state questions.",
                "Binary questions should render as yes_no even when they involve attributes.",
            ],
            "spatial_navigation": [
                "Use spatial_relation for side/left/right/relative-location questions.",
                "Use navigation_direction for turn or movement-direction questions.",
                "Do not force side or turn questions into descriptive just because their official type is text.",
            ],
            "position": [
                "Use position_query when the answer is a location or coordinates of an object/place.",
                "If a position question includes attributes such as nearest, red, empty, or in front of, keep operation=position_query and express those as constraints or attribute_dimension.",
            ],
        },
        "generic_examples": [
            {
                "question": "When did you first enter the referenced place?",
                "type_hint": "time",
                "expected": {
                    "answer_type": "time",
                    "operation": "onset",
                    "rendering": "minutes_ago",
                    "qa_focus": "time_duration",
                },
            },
            {
                "question": "How long were you in the referenced place?",
                "type_hint": "duration",
                "expected": {
                    "answer_type": "duration",
                    "operation": "interval",
                    "rendering": "duration_minutes",
                    "qa_focus": "time_duration",
                },
            },
            {
                "question": "When were you near the referenced place with the described objects?",
                "type_hint": "time",
                "expected": {
                    "answer_type": "time",
                    "operation": "occurrence",
                    "rendering": "minutes_ago",
                    "qa_focus": "time_duration",
                },
            },
            {
                "question": "What color was the target object associated with the described entity?",
                "type_hint": "text",
                "expected": {
                    "answer_type": "text",
                    "operation": "attribute_query",
                    "attribute_dimension": "color",
                    "rendering": "color_only",
                    "qa_focus": "descriptive",
                    "hard_constraints": [
                        {
                            "kind": "event",
                            "text": "described entity associated with the referenced object",
                            "role": "event_condition",
                            "target_entities": ["described entity", "referenced object"],
                            "must_bind_to": ["answer_entity", "same_time_window"],
                        }
                    ],
                },
            },
            {
                "question": "What side of the referenced path are you moving on?",
                "type_hint": "text",
                "expected": {
                    "answer_type": "text",
                    "operation": "spatial_relation",
                    "attribute_dimension": "side",
                    "rendering": "short_phrase",
                    "qa_focus": "spatial_geometry",
                },
            },
        ],
        "questions": [
            {
                "uuid": row.get(FIELD_UUID, ""),
                "question": _row_question(row),
                "official_question_type_hint": _row_type(row),
                "question_category_hint": str(row.get(FIELD_CATEGORY, "")).strip(),
            }
            for row in rows
        ],
    }
    payload: dict[str, Any] = {
        "model": config.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a semantic compiler for first-person robot navigation video QA. "
                    "In the dataset, 'I', 'you', and 'we' refer to the robot/ego camera. "
                    "Return exactly one valid JSON object with key 'items'. Do not "
                    "answer the questions. Do not use hidden labels, official "
                    "answers, or graph evidence."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            },
        ],
    }
    apply_reasoning_options(
        payload,
        reasoning_effort=config.reasoning_effort,
        thinking_enabled=config.thinking_enabled,
    )
    return payload


def _post_payload(
    payload: dict[str, Any],
    *,
    api_key: str,
    config: GraphQAConfig,
) -> dict[str, Any]:
    endpoint = urljoin(
        config.api_base_url.rstrip("/") + "/",
        config.endpoint_path.lstrip("/"),
    )
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=config.timeout_sec) as response:
        raw_response = json.loads(response.read().decode("utf-8"))
    if not isinstance(raw_response, dict):
        raise json.JSONDecodeError("Response JSON is not an object", str(raw_response), 0)
    return raw_response


def _extract_content(raw_response: dict[str, Any]) -> str | None:
    choices = raw_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        merged = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict)
        ).strip()
        return merged or None
    return None


def _parse_json_object(content: str) -> dict[str, Any] | None:
    parsed = _parse_json_value(content)
    return parsed if isinstance(parsed, dict) else None


def _parse_json_value(content: str) -> Any | None:
    stripped = _strip_code_fences(content)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        candidate = _extract_first_json_value(stripped)
        if candidate is None:
            return None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return parsed


def _extract_ir_items(parsed: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[tuple[str, Any]] = []
    if isinstance(parsed, list):
        candidates.append(("root", parsed))
    elif isinstance(parsed, dict):
        for key in (
            "items",
            "questions",
            "results",
            "ir_items",
            "question_ir",
            "question_irs",
            "outputs",
            "data",
        ):
            if key in parsed:
                candidates.append((key, parsed.get(key)))
        for outer_key in ("response", "result", "payload", "answer"):
            nested = parsed.get(outer_key) if isinstance(parsed, dict) else None
            if not isinstance(nested, dict):
                continue
            for key in (
                "items",
                "questions",
                "results",
                "ir_items",
                "question_ir",
                "question_irs",
                "outputs",
                "data",
            ):
                if key in nested:
                    candidates.append((f"{outer_key}.{key}", nested.get(key)))

    for source, value in candidates:
        items = _coerce_ir_item_list(value)
        if items:
            return items, {
                "source": source,
                "candidate_count": len(candidates),
                "item_count": len(items),
            }
    return [], {
        "source": "",
        "candidate_count": len(candidates),
        "item_count": 0,
    }


def _coerce_ir_item_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        nested = value.get("items")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
        if any(key in value for key in ("operation", "answer_type", "qa_focus", "uuid")):
            return [value]
    return []


def _align_ir_items_to_rows(
    raw_items: list[dict[str, Any]],
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_uuid = {
        str(item.get("uuid") or "").strip(): item
        for item in raw_items
        if str(item.get("uuid") or "").strip()
    }
    if by_uuid:
        aligned: list[dict[str, Any]] = []
        missing = 0
        for row in rows:
            uuid = str(row.get(FIELD_UUID, "")).strip()
            item = by_uuid.get(uuid)
            if item is None:
                missing += 1
                item = {"uuid": uuid, "parse_error": "missing_item"}
            aligned.append(item)
        return aligned, {
            "method": "uuid",
            "uuid_item_count": len(by_uuid),
            "missing_count": missing,
        }

    if len(raw_items) == len(rows):
        aligned = []
        for row, item in zip(rows, raw_items):
            normalized = dict(item)
            normalized.setdefault("uuid", str(row.get(FIELD_UUID, "")).strip())
            aligned.append(normalized)
        return aligned, {
            "method": "position",
            "uuid_item_count": 0,
            "missing_count": 0,
        }

    return [
        {"uuid": str(row.get(FIELD_UUID, "")).strip(), "parse_error": "missing_item"}
        for row in rows
    ], {
        "method": "none",
        "uuid_item_count": 0,
        "missing_count": len(rows),
    }


def _response_metadata(raw_response: dict[str, Any]) -> dict[str, Any]:
    choices = raw_response.get("choices")
    choice: dict[str, Any] = {}
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
    return {
        "id": raw_response.get("id"),
        "model": raw_response.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "usage": raw_response.get("usage"),
    }


def _strip_code_fences(content: str) -> str:
    stripped = str(content or "").strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_first_json_value(content: str) -> str | None:
    object_start = content.find("{")
    array_start = content.find("[")
    starts = [index for index in (object_start, array_start) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    opening = content[start]
    closing = "}" if opening == "{" else "]"
    stack: list[str] = []
    if start < 0:
        return None
    in_string = False
    escaped = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if not stack:
                return content[start : index + 1]
    return None


def _truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n...[truncated]"


def _json_shape(value: Any) -> str:
    if isinstance(value, dict):
        keys = ",".join(sorted(str(key) for key in value.keys())[:12])
        return f"object:{keys}"
    if isinstance(value, list):
        return f"array:{len(value)}"
    if value is None:
        return "none"
    return type(value).__name__


def _operation_from_rule_plan(plan: dict[str, Any]) -> str:
    answer_type = str(plan.get("answer_type") or "").lower()
    events = set(plan.get("events") or [])
    if answer_type == "time":
        if "enter" in events:
            return "onset"
        if "leave" in events or "get_off" in events:
            return "offset"
        return "occurrence"
    if answer_type == "duration":
        return "interval"
    if answer_type == "binary":
        return "existence_query"
    if answer_type == "position":
        return "position_query"
    if plan.get("attributes"):
        return "attribute_query"
    return "unknown"


def _qa_focus_from_rule_plan(plan: dict[str, Any]) -> str:
    answer_type = str(plan.get("answer_type") or "").lower()
    if answer_type in {"time", "duration"}:
        return "time_duration"
    if answer_type == "binary":
        return "binary"
    if answer_type == "position":
        return "position"
    if plan.get("attributes"):
        return "descriptive"
    return "unknown"


def _normalized_type_set(values: list[str] | tuple[str, ...] | None) -> set[str] | None:
    if values is None:
        return None
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }


def _row_type(row: dict[str, str]) -> str:
    return str(row.get(FIELD_TYPE, "")).strip().lower()


def _row_question(row: dict[str, str]) -> str:
    return str(row.get(FIELD_QUESTION, "")).strip()


def _official_answer(row: dict[str, str]) -> str:
    text = str(row.get(FIELD_TEXT_ANSWER, "")).strip()
    parsable = str(row.get(FIELD_PARSABLE_ANSWER, "")).strip()
    if text and parsable:
        return f"{text} (parsable: {parsable})"
    return text or parsable


def _normalize_enum(value: Any, *, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in allowed else default


def _normalize_evidence_question_type(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    normalized = normalized.replace("/", ".")
    aliases = {
        "temporal_point": "temporal.point",
        "temporal.interval.duration": "temporal.interval",
        "temporal_interval": "temporal.interval",
        "binary_existence": "non_temporal.entity.existence",
        "entity_existence": "non_temporal.entity.existence",
        "entity_attribute": "non_temporal.entity.attribute",
        "entity_count": "non_temporal.entity.count",
        "count_set": "non_temporal.entity.count",
        "entity_relation_spatial": "non_temporal.spatial.relation",
        "spatial_relation": "non_temporal.spatial.relation",
        "ocr_logo_text": "non_temporal.symbolic.ocr_logo",
        "ego_motion": "non_temporal.ego.motion",
        "ego_direction": "non_temporal.ego.direction",
        "ego_motion_side_turn": "non_temporal.ego.direction",
        "open_descriptive": "non_temporal.open_descriptive",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in ALLOWED_EVIDENCE_QUESTION_TYPES else ""


def _infer_evidence_question_type_from_ir(
    *,
    answer_type: str,
    operation: str,
    qa_focus: str,
    attribute_dimension: str,
) -> str:
    attribute = attribute_dimension.strip().lower()
    if answer_type == "duration" or operation == "interval":
        return "temporal.interval"
    if answer_type == "time" or operation in {"onset", "offset"}:
        return "temporal.point"
    if answer_type == "binary" or operation in {"existence_query", "current_state"}:
        if operation == "current_state" or attribute == "state":
            return "non_temporal.state.object"
        return "non_temporal.entity.existence"
    if answer_type == "count" or operation == "count_query":
        return "non_temporal.entity.count"
    if answer_type == "location_text" or operation == "location_query":
        return "non_temporal.spatial.localization"
    if operation == "identity_query":
        return "non_temporal.entity.identity"
    if operation == "functional_query":
        return "non_temporal.functional.affordance"
    if operation == "world_knowledge_query":
        return "non_temporal.world_knowledge.contextual"
    if operation == "state_query":
        return "non_temporal.state.object"
    if operation == "attribute_query" or attribute:
        return "non_temporal.entity.attribute"
    if operation == "spatial_relation" or qa_focus == "spatial_geometry":
        return "non_temporal.spatial.relation"
    if operation == "navigation_direction" or qa_focus == "navigation_direction":
        return "non_temporal.ego.direction"
    if operation == "position_query" or answer_type == "position":
        return "non_temporal.spatial.relation"
    return "unknown"


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value).strip()
    return json.dumps(value, ensure_ascii=False)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif str(item).strip():
            result.append({"value": str(item).strip()})
    return result


def _bounded_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _default_rendering(answer_type: str) -> str:
    if answer_type == "binary":
        return "yes_no"
    if answer_type == "time":
        return "minutes_ago"
    if answer_type == "duration":
        return "duration_minutes"
    if answer_type == "count":
        return "number_or_short_phrase"
    if answer_type == "location_text":
        return "location_phrase"
    if answer_type == "position":
        return "position_coordinates"
    return "free_text"


def _default_focus(answer_type: str) -> str:
    if answer_type in {"time", "duration"}:
        return "time_duration"
    if answer_type == "count":
        return "descriptive"
    if answer_type == "location_text":
        return "spatial_geometry"
    if answer_type == "binary":
        return "binary"
    if answer_type == "position":
        return "position"
    return "unknown"


def _batched(items: list[dict[str, str]], batch_size: int) -> list[list[dict[str, str]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def main() -> None:
    args = build_parser().parse_args()
    layout = default_navqa_sequence_layout(
        seq_id=args.seq_id,
        cam=args.cam,
        stride=args.stride,
        layout_root=args.layout_root,
    )
    output_json = args.output_json or (
        layout.sample_root / "llm_question_ir_experiment.json"
    )
    output_md = args.output_md or (
        layout.sample_root / "llm_question_ir_experiment.md"
    )
    csv_path = resolve_repo_input_path(args.csv)
    rows = load_questions(csv_path, args.seq_id)
    include_types = args.include_types
    if args.non_position_only:
        include_types = list(NON_POSITION_QA_TYPES)
    selected = filter_questions_by_type(
        rows,
        include_types=include_types,
        exclude_types=args.exclude_types,
    )
    selected = selected[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit(f"No questions selected for seq_id={args.seq_id!r}")

    domain_config = load_domain_config(args.config)
    graph_qa_config = domain_config.graph_qa.model_copy(
        update={
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            **(
                {"timeout_sec": args.timeout_sec}
                if args.timeout_sec is not None
                else {}
            ),
        }
    )

    print(f"csv={csv_path}", flush=True)
    print(f"seq_id={args.seq_id}", flush=True)
    print(f"questions={len(selected)}", flush=True)
    print(f"mode={args.mode}", flush=True)
    print(f"model={graph_qa_config.model}", flush=True)

    batch_logs: list[dict[str, Any]] = []
    try:
        if args.mode == "mock":
            raw_ir_items = compile_questions_mock(selected)
        else:
            raw_ir_items, batch_logs = compile_questions_with_llm(
                selected,
                config=graph_qa_config,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
    except (RuntimeError, HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"LLM question IR experiment failed: {exc}") from exc

    records = build_records(
        selected,
        raw_ir_items,
        domain_profile=args.domain_profile,
    )
    payload = {
        "contract": "navqa_llm_question_ir_experiment_v1",
        "mode": args.mode,
        "provider": graph_qa_config.provider,
        "model": graph_qa_config.model,
        "seq_id": args.seq_id,
        "source_csv": str(csv_path),
        "question_count": len(records),
        "prompt_excludes_official_answers": True,
        "prompt_excludes_graph_evidence": True,
        "summary": summarize_records(records),
        "batch_logs": batch_logs,
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_md, payload)
    print(f"output_json={output_json.resolve()}", flush=True)
    print(f"output_md={output_md.resolve()}", flush=True)


if __name__ == "__main__":
    main()
