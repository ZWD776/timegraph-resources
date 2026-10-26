from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.config.config import load_domain_config  # noqa: E402
from vlkg.datasets.navqa.runtime.paths import normalize_seq_id  # noqa: E402
from vlkg.core.temporal.pathing import resolve_repo_input_path  # noqa: E402
from vlkg.core.transformers import JSONToGraphTransformer  # noqa: E402


DEFAULT_MANIFEST = Path("configs/navqa_clean_mainline_sequences.json")
DEFAULT_OUTPUT_ROOT = Path(".scratch/navqa_llm_fallback_audits")
STANDARD_HSGG_JSON_FILES = (
    "objects.json",
    "attributes.json",
    "relations.json",
    "hierarchy_mapping.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit NaVQA LLM fallback trigger counts during HSGG JSON -> temporal "
            "graph construction. By default this does not call the LLM."
        )
    )
    parser.add_argument("--sequence-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seq-id", nargs="+", default=None)
    parser.add_argument("--config", type=Path, default=Path("default.yaml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--max-requests-per-run",
        type=int,
        default=0,
        help=(
            "Maximum live LLM requests per sequence. The default 0 audits trigger "
            "counts only and spends no LLM tokens."
        ),
    )
    parser.add_argument(
        "--estimated-tokens-per-call",
        type=float,
        default=1300.0,
        help=(
            "Rough cost estimate for trigger-only audits. Historical seq22 logs "
            "were about 1300 tokens per call."
        ),
    )
    parser.add_argument(
        "--enable-static-fallback",
        action="store_true",
        help="Also allow fallback for observations classified as static.",
    )
    parser.add_argument(
        "--enable-noise-fallback",
        action="store_true",
        help="Also allow fallback for observations classified as noise.",
    )
    parser.add_argument(
        "--enable-candidate-context-fallback",
        action="store_true",
        help="Allow candidate-context fallback even without embedding ranked matches.",
    )
    parser.add_argument("--run", action="store_true")
    return parser


def repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def has_standard_hsgg_json(input_dir: Path) -> bool:
    json_dir = input_dir / "json" if (input_dir / "json").exists() else input_dir
    return all((json_dir / file_name).exists() for file_name in STANDARD_HSGG_JSON_FILES)


def resolve_hsgg_input_dir(sequence: dict[str, Any]) -> Path:
    sample_root = repo_path(sequence.get("sample_root"))
    if sample_root is None:
        raise ValueError("sequence item is missing sample_root")
    local_paths = sample_root / "hsgg_local_paths"
    if has_standard_hsgg_json(local_paths):
        return local_paths
    if has_standard_hsgg_json(sample_root):
        return sample_root
    raise FileNotFoundError(
        "Cannot find standard HSGG JSON files under "
        f"{local_paths} or {sample_root}"
    )


def load_manifest(path: Path, seq_ids: set[str] | None) -> list[dict[str, Any]]:
    manifest_path = resolve_repo_input_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    sequences = payload.get("sequences", [])
    if not isinstance(sequences, list):
        raise ValueError(f"manifest sequences must be a list: {manifest_path}")
    if seq_ids is None:
        return [item for item in sequences if isinstance(item, dict)]
    return [
        item
        for item in sequences
        if isinstance(item, dict)
        and normalize_seq_id(item.get("seq_id", "")) in seq_ids
    ]


def audit_sequence(
    *,
    sequence: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    max_requests_per_run: int,
    estimated_tokens_per_call: float,
    enable_static_fallback: bool,
    enable_noise_fallback: bool,
    enable_candidate_context_fallback: bool,
) -> dict[str, Any]:
    seq_id = normalize_seq_id(sequence.get("seq_id", ""))
    input_dir = resolve_hsgg_input_dir(sequence)
    frame_time_index = repo_path(sequence.get("frame_time_index_json"))
    decision_log_path = output_dir / f"seq{seq_id}.decisions.jsonl"

    config = load_domain_config(config_path, config_dir=REPO_ROOT / "configs")
    config = config.model_copy(
        update={
            "llm_fallback": config.llm_fallback.model_copy(
                update={
                    "enabled": True,
                    "max_requests_per_run": max(0, max_requests_per_run),
                    "decision_log_path": str(decision_log_path),
                    "enable_static_fallback": enable_static_fallback,
                    "enable_noise_fallback": enable_noise_fallback,
                    "enable_candidate_context_fallback": (
                        enable_candidate_context_fallback
                    ),
                }
            )
        }
    )

    started_at = time.perf_counter()
    transformer = JSONToGraphTransformer(
        input_dir=input_dir,
        config=config,
        frame_time_index_path=frame_time_index,
    )
    document = transformer.transform()
    runtime_stats = transformer.get_runtime_stats()
    elapsed_sec = time.perf_counter() - started_at

    association_policy = runtime_stats.get("association_policy", {})
    fallback_requests = int(association_policy.get("fallback_requests_built", 0))
    llm_fallback = runtime_stats.get("llm_fallback", {})
    token_usage = llm_fallback.get("token_usage", {})
    estimated_tokens = int(round(fallback_requests * estimated_tokens_per_call))
    return {
        "seq_id": seq_id,
        "input_dir": str(input_dir),
        "frame_time_index_json": str(frame_time_index) if frame_time_index else None,
        "frame_count": document.frame_count,
        "entity_count_before_gc": len(document.entities),
        "runtime_sec": round(elapsed_sec, 3),
        "alignment": runtime_stats.get("alignment", {}),
        "association_policy": association_policy,
        "llm_fallback": llm_fallback,
        "decision_log_path": str(decision_log_path),
        "actual_total_tokens": token_usage.get("total_tokens", 0),
        "estimated_total_tokens_if_all_called": estimated_tokens,
        "estimated_tokens_per_call": estimated_tokens_per_call,
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_fallback_requests = 0
    total_api_calls = 0
    total_actual_tokens = 0
    total_estimated_tokens = 0
    reason_counts: dict[str, int] = {}
    decision_counts: dict[str, int] = {}
    for result in results:
        policy = result.get("association_policy", {})
        llm_fallback = result.get("llm_fallback", {})
        total_fallback_requests += int(policy.get("fallback_requests_built", 0))
        total_api_calls += int(llm_fallback.get("api_calls_attempted", 0))
        total_actual_tokens += int(result.get("actual_total_tokens", 0) or 0)
        total_estimated_tokens += int(
            result.get("estimated_total_tokens_if_all_called", 0) or 0
        )
        for key, value in (policy.get("fallback_request_reason_counts", {}) or {}).items():
            reason_counts[str(key)] = reason_counts.get(str(key), 0) + int(value)
        for key, value in (llm_fallback.get("decision_path_counts", {}) or {}).items():
            decision_counts[str(key)] = decision_counts.get(str(key), 0) + int(value)
    return {
        "sequence_count": len(results),
        "total_fallback_requests": total_fallback_requests,
        "total_api_calls_attempted": total_api_calls,
        "total_actual_tokens": total_actual_tokens,
        "total_estimated_tokens_if_all_called": total_estimated_tokens,
        "fallback_request_reason_counts": reason_counts,
        "llm_decision_path_counts": decision_counts,
    }


def main() -> None:
    args = build_parser().parse_args()
    if not args.run:
        raise SystemExit("Refusing to run without --run.")

    seq_ids = (
        {normalize_seq_id(item) for item in args.seq_id}
        if args.seq_id is not None
        else None
    )
    run_name = args.run_name or time.strftime("audit_%Y%m%d_%H%M%S")
    output_dir = repo_path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sequences = load_manifest(args.sequence_manifest, seq_ids)
    results: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for sequence in sequences:
        seq_id = str(sequence.get("seq_id", "unknown"))
        print(f"auditing seq{seq_id}...", flush=True)
        try:
            result = audit_sequence(
                sequence=sequence,
                config_path=args.config,
                output_dir=output_dir,
                max_requests_per_run=args.max_requests_per_run,
                estimated_tokens_per_call=args.estimated_tokens_per_call,
                enable_static_fallback=args.enable_static_fallback,
                enable_noise_fallback=args.enable_noise_fallback,
                enable_candidate_context_fallback=(
                    args.enable_candidate_context_fallback
                ),
            )
            results.append(result)
            policy = result.get("association_policy", {})
            llm_fallback = result.get("llm_fallback", {})
            print(
                "  "
                f"fallback_requests={policy.get('fallback_requests_built', 0)} "
                f"api_calls={llm_fallback.get('api_calls_attempted', 0)} "
                f"estimated_tokens={result.get('estimated_total_tokens_if_all_called', 0)} "
                f"runtime_sec={result.get('runtime_sec')}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - diagnostic entrypoint
            failed.append({"seq_id": seq_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)

    summary = {
        "contract": "navqa_llm_fallback_audit_v1",
        "sequence_manifest": str(resolve_repo_input_path(args.sequence_manifest)),
        "config": str(args.config),
        "output_dir": str(output_dir),
        "max_requests_per_run": args.max_requests_per_run,
        "live_llm_calls_enabled": args.max_requests_per_run > 0,
        "results": results,
        "failed": failed,
        "summary": summarize_results(results),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"summary_json={summary_path}", flush=True)
    print(
        "summary="
        + json.dumps(summary["summary"], ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
