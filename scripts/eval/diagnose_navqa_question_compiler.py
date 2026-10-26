from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.qa.position_reference import (
    FIELD_CATEGORY,
    FIELD_PARSABLE_ANSWER,
    FIELD_QUESTION,
    FIELD_SEQ_ID,
    FIELD_TEXT_ANSWER,
    FIELD_TIMESTAMP,
    FIELD_TYPE,
    FIELD_UUID,
)
from vlkg.core.config.domain_lexicon import DEFAULT_DOMAIN_PROFILE, available_domain_profiles
from vlkg.core.temporal.pathing import resolve_repo_input_path
from vlkg.core.graph_retrieval.question_compiler import compile_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile NaVQA questions into structured retrieval plans for diagnosis."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Official NaVQA data.csv.")
    parser.add_argument("--seq-id", default="0")
    parser.add_argument(
        "--domain-profile",
        choices=available_domain_profiles(),
        default=DEFAULT_DOMAIN_PROFILE,
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-csv",
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
            and str(row.get(FIELD_QUESTION, "")).strip()
        ]


def compile_rows(rows: list[dict[str, str]], *, domain_profile: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        question = str(row.get(FIELD_QUESTION, "")).strip()
        plan = compile_question(
            question,
            answer_type=str(row.get(FIELD_TYPE, "")).strip(),
            category=str(row.get(FIELD_CATEGORY, "")).strip(),
            domain_profile=domain_profile,
        )
        results.append(
            {
                "index": index,
                "uuid": row.get(FIELD_UUID, ""),
                "seq_id": row.get(FIELD_SEQ_ID, ""),
                "question": question,
                "official_type": row.get(FIELD_TYPE, ""),
                "official_category": row.get(FIELD_CATEGORY, ""),
                "official_timestamp": row.get(FIELD_TIMESTAMP, ""),
                "official_text_answer": row.get(FIELD_TEXT_ANSWER, ""),
                "official_parsable_answer": row.get(FIELD_PARSABLE_ANSWER, ""),
                "compiled_plan": plan.to_dict(),
            }
        )
    return results


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "uuid",
        "question",
        "official_type",
        "official_category",
        "answer_type",
        "query_family",
        "target_entities",
        "attributes",
        "events",
        "temporal_constraints",
        "spatial_constraints",
        "current_window_required",
        "ranking_policy",
        "retrieval_strategy",
        "preferred_source_kinds",
        "answer_rendering",
        "risk_notes",
        "normalized_query_terms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            plan = item["compiled_plan"]
            row = {
                "index": item["index"],
                "uuid": item["uuid"],
                "question": item["question"],
                "official_type": item["official_type"],
                "official_category": item["official_category"],
                "answer_type": plan["answer_type"],
                "query_family": plan["query_family"],
                "target_entities": json.dumps(plan["target_entities"], ensure_ascii=False),
                "attributes": json.dumps(plan["attributes"], ensure_ascii=False),
                "events": json.dumps(plan["events"], ensure_ascii=False),
                "temporal_constraints": json.dumps(
                    plan["temporal_constraints"],
                    ensure_ascii=False,
                ),
                "spatial_constraints": json.dumps(
                    plan["spatial_constraints"],
                    ensure_ascii=False,
                ),
                "current_window_required": plan["current_window_required"],
                "ranking_policy": json.dumps(plan["ranking_policy"], ensure_ascii=False),
                "retrieval_strategy": plan["retrieval_strategy"],
                "preferred_source_kinds": json.dumps(
                    plan["preferred_source_kinds"],
                    ensure_ascii=False,
                ),
                "answer_rendering": plan["answer_rendering"],
                "risk_notes": json.dumps(plan["risk_notes"], ensure_ascii=False),
                "normalized_query_terms": json.dumps(
                    plan["normalized_query_terms"],
                    ensure_ascii=False,
                ),
            }
            writer.writerow(row)


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# NaVQA Question Compiler Diagnosis",
        "",
        f"- questions: {len(results)}",
        "- scope: question processing only; no graph retrieval and no LLM answering",
        "",
    ]
    for item in results:
        plan = item["compiled_plan"]
        lines.extend(
            [
                f"## {item['index']:02d}. {item['question']}",
                "",
                f"- official: `{item['official_type']}` / `{item['official_category']}`",
                f"- family: `{plan['query_family']}`",
                f"- answer_type: `{plan['answer_type']}` -> `{plan['answer_rendering']}`",
                f"- target_entities: `{plan['target_entities']}`",
                f"- attributes: `{plan['attributes']}`",
                f"- events: `{plan['events']}`",
                f"- temporal_constraints: `{plan['temporal_constraints']}`",
                f"- spatial_constraints: `{plan['spatial_constraints']}`",
                f"- current_window_required: `{plan['current_window_required']}`",
                f"- ranking_policy: `{plan['ranking_policy']}`",
                f"- retrieval_strategy: `{plan['retrieval_strategy']}`",
                f"- preferred_source_kinds: `{plan['preferred_source_kinds']}`",
                f"- normalized_query_terms: `{plan['normalized_query_terms']}`",
            ]
        )
        if plan["risk_notes"]:
            lines.append(f"- risk_notes: `{plan['risk_notes']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    diagnostics_dir = Path(".scratch/navqa") / f"seq{args.seq_id}" / "diagnostics"
    if args.output_json is None:
        args.output_json = diagnostics_dir / "question_compiler_diagnosis.json"
    if args.output_csv is None:
        args.output_csv = diagnostics_dir / "question_compiler_diagnosis.csv"
    if args.output_md is None:
        args.output_md = diagnostics_dir / "question_compiler_diagnosis.md"
    csv_path = resolve_repo_input_path(args.csv)
    rows = load_questions(csv_path, args.seq_id)
    results = compile_rows(rows, domain_profile=args.domain_profile)
    payload = {
        "contract": "navqa_question_compiler_diagnosis_v1",
        "source_csv": str(csv_path),
        "seq_id": args.seq_id,
        "domain_profile": args.domain_profile,
        "question_count": len(results),
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_csv, results)
    write_markdown(args.output_md, results)
    print(f"questions={len(results)}")
    print(f"output_json={args.output_json.resolve()}")
    print(f"output_csv={args.output_csv.resolve()}")
    print(f"output_md={args.output_md.resolve()}")


if __name__ == "__main__":
    main()
