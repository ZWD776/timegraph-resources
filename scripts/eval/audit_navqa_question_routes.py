from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.retrieval.question_router import (  # noqa: E402
    QUESTION_EVIDENCE_TAXONOMY_CONTRACT,
    QUESTION_ROUTE_CONTRACT,
    route_question,
)


FIELD_SEQ_ID = "Seq ID"
FIELD_UUID = "UUID"
FIELD_QUESTION = "Question"
FIELD_TYPE = "Type \n(binary, position, time, text)"
FIELD_CATEGORY = "Question\nCategory"
NON_POSITION_QA_TYPES = ("binary", "text", "time", "duration")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a QuestionRoute audit table for NaVQA questions without "
            "loading graphs, FAISS indexes, or calling an LLM."
        )
    )
    parser.add_argument("--csv", type=Path, default=Path(".scratch/navqa_official_data.csv"))
    parser.add_argument("--question-ir-json", type=Path, default=None)
    parser.add_argument(
        "--sequence-manifest",
        type=Path,
        default=None,
        help=(
            "Optional sequence manifest whose per-sequence question_ir_json paths "
            "are merged before routing. This is the preferred CG6 reuse mode."
        ),
    )
    parser.add_argument("--seq-id", nargs="+", default=None)
    parser.add_argument("--include-types", nargs="+", default=list(NON_POSITION_QA_TYPES))
    parser.add_argument("--exclude-types", nargs="+", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--domain-profile", default="generic")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(".scratch/navqa_question_route_audit/question_routes.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(".scratch/navqa_question_route_audit/question_routes.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(".scratch/navqa_question_route_audit/question_routes.md"),
    )
    return parser


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_question_ir_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = _resolve_repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    records = payload.get("records", []) if isinstance(payload, dict) else payload
    by_uuid: dict[str, dict[str, Any]] = {}
    if isinstance(records, dict):
        for uuid, value in records.items():
            if isinstance(value, dict):
                ir = value.get("llm_question_ir") if "llm_question_ir" in value else value
                if isinstance(ir, dict):
                    by_uuid[str(uuid).strip()] = ir
        return by_uuid
    if not isinstance(records, list):
        raise ValueError(f"question IR records must be a list or mapping: {resolved}")
    for record in records:
        if not isinstance(record, dict):
            continue
        uuid = str(record.get("uuid") or "").strip()
        ir = record.get("llm_question_ir")
        if uuid and isinstance(ir, dict):
            by_uuid[uuid] = ir
    return by_uuid


def load_question_ir_records_from_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resolved = _resolve_repo_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    sequences = payload.get("sequences", []) if isinstance(payload, dict) else []
    if not isinstance(sequences, list):
        raise ValueError(f"manifest sequences must be a list: {resolved}")
    merged: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for sequence in sequences:
        if not isinstance(sequence, dict):
            continue
        ir_path_text = str(sequence.get("question_ir_json") or "").strip()
        if not ir_path_text:
            continue
        ir_path = _resolve_repo_path(Path(ir_path_text))
        if not ir_path.exists():
            missing.append(str(ir_path))
            continue
        merged.update(load_question_ir_records(ir_path))
    if missing:
        print(
            f"warning: missing {len(missing)} question IR files from manifest",
            file=sys.stderr,
        )
    return merged


def filter_rows(
    rows: list[dict[str, str]],
    *,
    seq_ids: list[str] | None,
    include_types: list[str] | tuple[str, ...] | None,
    exclude_types: list[str] | tuple[str, ...] | None,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict[str, str]]:
    seq_set = {str(value).strip() for value in (seq_ids or []) if str(value).strip()}
    includes = _normalized_set(include_types)
    excludes = _normalized_set(exclude_types) or set()
    selected: list[dict[str, str]] = []
    for row in rows:
        seq_id = str(row.get(FIELD_SEQ_ID, "")).strip()
        qtype = str(row.get(FIELD_TYPE, "")).strip().lower()
        if seq_set and seq_id not in seq_set:
            continue
        if includes is not None and qtype not in includes:
            continue
        if qtype in excludes:
            continue
        selected.append(row)
    start = max(0, int(offset))
    end = None if limit is None else start + max(0, int(limit))
    return selected[start:end]


def audit_question_routes(
    rows: list[dict[str, str]],
    *,
    question_ir_records: dict[str, dict[str, Any]] | None = None,
    domain_profile: str | None = "generic",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ir_by_uuid = question_ir_records or {}
    for row in rows:
        uuid = str(row.get(FIELD_UUID, "")).strip()
        question = str(row.get(FIELD_QUESTION, "")).strip()
        official_type = str(row.get(FIELD_TYPE, "")).strip().lower()
        category = str(row.get(FIELD_CATEGORY, "")).strip()
        question_ir = ir_by_uuid.get(uuid)
        route = route_question(
            question,
            official_question_type=official_type or None,
            question_ir=question_ir,
            domain_profile=domain_profile,
            category=category or None,
        )
        records.append(
            {
                "uuid": uuid,
                "seq_id": str(row.get(FIELD_SEQ_ID, "")).strip(),
                "question": question,
                "official_question_type": official_type,
                "question_category": category,
                "has_llm_question_ir": isinstance(question_ir, dict),
                "ir_operation": str((question_ir or {}).get("operation") or "").strip(),
                "ir_answer_type": str((question_ir or {}).get("answer_type") or "").strip(),
                "ir_aggregation": str((question_ir or {}).get("aggregation") or "").strip(),
                "compiler_family": route.compiler_family,
                "compiler_answer_type": route.compiler_answer_type,
                "question_route_id": route.route_id,
                "taxonomy_path": route.taxonomy_path,
                "taxonomy_root": route.taxonomy_root,
                "taxonomy_domain": route.taxonomy_domain,
                "taxonomy_task": route.taxonomy_task,
                "evidence_family": route.evidence_family,
                "answer_shape": route.answer_shape,
                "confidence": route.confidence,
                "source": route.source,
                "fallback_route": route.fallback_route,
                "conflicts": list(route.conflicts),
                "route_signals": list(route.route_signals),
                "retrieval_policy": route.retrieval_policy,
                "evidence_policy": route.evidence_policy,
                "primary_target_terms": list(route.primary_target_terms),
                "attribute_terms": list(route.attribute_terms),
                "event_constraints": list(route.event_constraints),
                "spatial_constraints": list(route.spatial_constraints),
                "time_requirements": route.time_requirements,
            }
        )
    return records


def write_outputs(
    records: list[dict[str, Any]],
    *,
    csv_path: Path,
    question_ir_json: Path | None,
    sequence_manifest: Path | None = None,
    output_json: Path,
    output_csv: Path,
    output_md: Path,
) -> None:
    route_counts = Counter(str(record.get("question_route_id") or "") for record in records)
    taxonomy_counts = Counter(str(record.get("taxonomy_path") or "") for record in records)
    payload = {
        "contract": QUESTION_ROUTE_CONTRACT,
        "taxonomy_contract": QUESTION_EVIDENCE_TAXONOMY_CONTRACT,
        "task": "navqa_question_route_audit",
        "source_csv": str(csv_path),
        "question_ir_json": str(question_ir_json) if question_ir_json else "",
        "sequence_manifest": str(sequence_manifest) if sequence_manifest else "",
        "question_count": len(records),
        "route_distribution": dict(sorted(route_counts.items())),
        "taxonomy_distribution": dict(sorted(taxonomy_counts.items())),
        "conflict_count": sum(1 for record in records if record.get("conflicts")),
        "records": records,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "uuid",
        "seq_id",
        "official_question_type",
        "question_category",
        "question_route_id",
        "taxonomy_path",
        "taxonomy_root",
        "taxonomy_domain",
        "taxonomy_task",
        "evidence_family",
        "answer_shape",
        "confidence",
        "source",
        "compiler_family",
        "ir_operation",
        "ir_answer_type",
        "ir_aggregation",
        "conflicts",
        "route_signals",
        "primary_target_terms",
        "attribute_terms",
        "event_constraints",
        "spatial_constraints",
        "retrieval_policy",
        "evidence_policy",
        "question",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                key: _csv_value(record.get(key))
                for key in fieldnames
            }
            writer.writerow(row)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown_summary(payload), encoding="utf-8")


def _markdown_summary(payload: dict[str, Any]) -> str:
    records = payload.get("records", [])
    lines = [
        "# NaVQA QuestionRoute Audit",
        "",
        f"- contract: `{payload.get('contract')}`",
        f"- taxonomy_contract: `{payload.get('taxonomy_contract')}`",
        f"- question_count: {payload.get('question_count')}",
        f"- conflict_count: {payload.get('conflict_count')}",
        "",
        "## Route Distribution",
        "",
        "| route | count |",
        "| --- | ---: |",
    ]
    for route_id, count in (payload.get("route_distribution") or {}).items():
        lines.append(f"| `{route_id}` | {count} |")
    lines.extend(
        [
            "",
            "## Taxonomy Distribution",
            "",
            "| taxonomy_path | count |",
            "| --- | ---: |",
        ]
    )
    for taxonomy_path, count in (payload.get("taxonomy_distribution") or {}).items():
        lines.append(f"| `{taxonomy_path}` | {count} |")
    lines.extend(
        [
            "",
            "## Per Question",
            "",
            "| uuid | seq | type | taxonomy | route | confidence | source | conflicts | question |",
            "| --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for record in records:
        conflicts = ", ".join(record.get("conflicts") or [])
        question = str(record.get("question") or "").replace("|", "\\|")
        lines.append(
            "| {uuid} | {seq} | {qtype} | `{taxonomy}` | `{route}` | {conf:.2f} | {source} | {conflicts} | {question} |".format(
                uuid=str(record.get("uuid") or ""),
                seq=str(record.get("seq_id") or ""),
                qtype=str(record.get("official_question_type") or ""),
                taxonomy=str(record.get("taxonomy_path") or ""),
                route=str(record.get("question_route_id") or ""),
                conf=float(record.get("confidence") or 0.0),
                source=str(record.get("source") or ""),
                conflicts=conflicts.replace("|", "\\|"),
                question=question,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _normalized_set(values: list[str] | tuple[str, ...] | None) -> set[str] | None:
    if values is None:
        return None
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def main() -> None:
    args = build_parser().parse_args()
    rows = filter_rows(
        load_rows(args.csv),
        seq_ids=args.seq_id,
        include_types=args.include_types,
        exclude_types=args.exclude_types,
        offset=args.offset,
        limit=args.limit,
    )
    question_ir_records = load_question_ir_records_from_manifest(args.sequence_manifest)
    question_ir_records.update(load_question_ir_records(args.question_ir_json))
    records = audit_question_routes(
        rows,
        question_ir_records=question_ir_records,
        domain_profile=args.domain_profile,
    )
    write_outputs(
        records,
        csv_path=args.csv,
        question_ir_json=args.question_ir_json,
        sequence_manifest=args.sequence_manifest,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    print(f"question_count={len(records)}", flush=True)
    print(f"output_json={args.output_json.resolve()}", flush=True)
    print(f"output_csv={args.output_csv.resolve()}", flush=True)
    print(f"output_md={args.output_md.resolve()}", flush=True)


if __name__ == "__main__":
    main()
