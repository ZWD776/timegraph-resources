from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.temporal.pathing import resolve_repo_input_path  # noqa: E402
from vlkg.core.temporal.processed_attributes import (  # noqa: E402
    OpenAICompatibleAttributeStructurer,
    PassthroughAttributeStructurer,
    ProcessedAttributesBuildConfig,
    SimpleAttributeStructurer,
    build_processed_attributes,
    collect_attribute_inputs,
    deduplicate_attribute_inputs,
    filter_priority_attribute_records,
    summarize_attribute_inputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build persistent HSGG processed_attributes.json sidecars. The sidecar "
            "is stored next to HSGG attributes.json and should be read downstream "
            "instead of reparsing raw natural-language attributes inside CG/IR-QA."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--hsgg-input-dir",
        type=Path,
        help="One HSGG directory containing objects.json and attributes.json.",
    )
    source.add_argument(
        "--sequence-manifest",
        type=Path,
        help="Manifest with sequences[*].hsgg_input_dir entries.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Output path for single --hsgg-input-dir mode. Defaults to "
            "<hsgg_input_dir>/processed_attributes.json."
        ),
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help=(
            "Output filename under each HSGG input dir. Useful for manifest "
            "mode when building an alternate sidecar such as "
            "processed_attributes_llm_priority_v1.json."
        ),
    )
    parser.add_argument("--seq-id", action="append", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--priority-filter",
        choices=("text-number",),
        default=None,
        help=(
            "Build only a question-independent priority subset. text-number "
            "selects raw HSGG attributes with explicit text/logo/sign or "
            "number/count cues."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=("llm", "simple", "passthrough"),
        default="llm",
        help=(
            "llm builds structured attributes with an OpenAI-compatible model. "
            "simple builds conservative local structured attributes without API "
            "calls. passthrough is only for smoke tests and schema checks; it "
            "preserves raw text without parsing."
        ),
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--endpoint-path", default="/chat/completions")
    parser.add_argument("--api-key-env-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a progress checkpoint when it exists.",
    )
    parser.add_argument(
        "--progress-path",
        type=Path,
        default=None,
        help=(
            "Optional progress checkpoint path for single input mode. Defaults to "
            "<output>.progress.json."
        ),
    )
    parser.add_argument("--checkpoint-every-batches", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-sec", type=float, default=5.0)
    parser.add_argument(
        "--continue-on-batch-failure",
        action="store_true",
        help=(
            "Keep processing later batches after a failed batch and save the "
            "failed batch list in the progress checkpoint. Final output is not "
            "written until all batches succeed."
        ),
    )
    parser.add_argument(
        "--thinking-enabled",
        choices=("true", "false"),
        default=None,
        help="Optional OpenAI-compatible thinking flag when the provider supports it.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count selected attribute records; do not call LLM or write output.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    hsgg_dirs = selected_hsgg_dirs(args)
    if args.output_path is not None and len(hsgg_dirs) != 1:
        raise SystemExit("--output-path is only supported with a single HSGG input dir")
    if args.output_path is not None and args.output_filename is not None:
        raise SystemExit("--output-path and --output-filename are mutually exclusive")
    if args.progress_path is not None and len(hsgg_dirs) != 1:
        raise SystemExit("--progress-path is only supported with a single HSGG input dir")

    if args.dry_run:
        rows = []
        for hsgg_dir in hsgg_dirs:
            records = collect_attribute_inputs(
                hsgg_dir,
                max_frames=args.max_frames,
                max_records=args.max_records,
            )
            summary = summarize_attribute_inputs(records)
            if args.priority_filter:
                unique_records, record_to_attribute_key = deduplicate_attribute_inputs(records)
                priority_unique = filter_priority_attribute_records(
                    unique_records,
                    args.priority_filter,
                )
                priority_keys = set(priority_unique)
                priority_record_count = sum(
                    1
                    for attribute_key in record_to_attribute_key.values()
                    if attribute_key in priority_keys
                )
                summary = {
                    **summary,
                    "priority_filter": args.priority_filter,
                    "priority_record_count": priority_record_count,
                    "priority_unique_attribute_count": len(priority_unique),
                }
            rows.append(
                {
                    "hsgg_input_dir": repo_relative(resolve_repo_input_path(hsgg_dir)),
                    **summary,
                    "first_record_id": records[0].record_id if records else None,
                }
            )
        print(json.dumps({"dry_run": True, "rows": rows}, ensure_ascii=False, indent=2))
        return

    structurer = build_structurer(args)
    summaries = []
    for hsgg_dir in hsgg_dirs:
        if args.output_path is not None:
            output_path = args.output_path
        elif args.output_filename:
            output_path = resolve_repo_input_path(hsgg_dir) / args.output_filename
        else:
            output_path = None
        config = ProcessedAttributesBuildConfig(
            hsgg_input_dir=hsgg_dir,
            output_path=output_path,
            max_frames=args.max_frames,
            max_records=args.max_records,
            batch_size=args.batch_size,
            priority_filter=args.priority_filter,
            force=bool(args.force),
            generator_name="build_hsgg_processed_attributes.py",
            generator_model=args.model if args.mode == "llm" else None,
            generator_mode=args.mode,
            resume=bool(args.resume),
            progress_path=args.progress_path,
            checkpoint_every_batches=args.checkpoint_every_batches,
            max_retries=args.max_retries,
            retry_sleep_sec=args.retry_sleep_sec,
            continue_on_batch_failure=bool(args.continue_on_batch_failure),
        )
        summary = build_processed_attributes(config, structurer=structurer)
        summaries.append(summary)
        print(
            f"{summary['hsgg_input_dir']}: records={summary['record_count']} "
            f"unique_attributes={summary['unique_attribute_count']} "
            f"output={summary['output_path']}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "contract": "hsgg_processed_attributes_cli_summary_v1",
                "mode": args.mode,
                "sequence_count": len(summaries),
                "record_count": sum(int(item["record_count"]) for item in summaries),
                "unique_attribute_count": sum(
                    int(item["unique_attribute_count"]) for item in summaries
                ),
                "outputs": summaries,
                "structurer_stats": structurer.stats(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def selected_hsgg_dirs(args: argparse.Namespace) -> list[Path]:
    if args.hsgg_input_dir is not None:
        return [resolve_repo_input_path(args.hsgg_input_dir)]

    manifest_path = resolve_repo_input_path(args.sequence_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    seq_filter = {str(seq_id) for seq_id in args.seq_id or []}
    dirs: list[Path] = []
    for item in manifest.get("sequences", []):
        if not isinstance(item, dict):
            continue
        seq_id = str(item.get("seq_id", ""))
        if seq_filter and seq_id not in seq_filter:
            continue
        hsgg_input_dir = item.get("hsgg_input_dir")
        if not hsgg_input_dir:
            continue
        dirs.append(resolve_repo_input_path(hsgg_input_dir))
    if not dirs:
        raise SystemExit(f"No HSGG input dirs selected from {manifest_path}")
    return dirs


def build_structurer(args: argparse.Namespace):
    if args.mode == "passthrough":
        return PassthroughAttributeStructurer()
    if args.mode == "simple":
        return SimpleAttributeStructurer()
    thinking_enabled = None
    if args.thinking_enabled == "true":
        thinking_enabled = True
    elif args.thinking_enabled == "false":
        thinking_enabled = False
    return OpenAICompatibleAttributeStructurer(
        model=args.model,
        api_base_url=args.api_base_url,
        endpoint_path=args.endpoint_path,
        api_key_env_var=args.api_key_env_var,
        timeout_sec=args.timeout_sec,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        thinking_enabled=thinking_enabled,
    )


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
