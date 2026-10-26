from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.runtime.paths import (  # noqa: E402
    NavQASequenceLayout,
    default_navqa_sequence_layout,
)


STAGE_ORDER = (
    "metadata",
    "download-images",
    "rewrite-hsgg-paths",
    "frame-time-index",
    "preflight-temporal-graph",
    "temporal-graph",
    "canonical-graph",
    "qa",
)


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str] | None = None
    copy_source: Path | None = None
    copy_target: Path | None = None
    note: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and run the mainline NaVQA pipeline for one CODa sequence "
            "using the unified .scratch/navqa/seq*/cam*/every*/ layout."
        )
    )
    parser.add_argument("--seq-id", required=True)
    parser.add_argument("--cam", default="0")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--layout-root", type=Path, default=Path(".scratch/navqa"))
    parser.add_argument(
        "--hsgg-output-dir",
        type=Path,
        required=True,
        help="Upstream HSGG output directory for this sampled video.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(".scratch/navqa_official_data.csv"),
        help="Official NaVQA data.csv.",
    )
    parser.add_argument(
        "--qa-unfilled-json",
        type=Path,
        default=None,
        help=(
            "Optional source qa_unfilled.json. When provided, it is copied into "
            "the sequence layout before downstream stages."
        ),
    )
    parser.add_argument(
        "--timestamps",
        type=Path,
        default=None,
        help=(
            "Optional source CODa timestamps/<seq>.txt. When omitted, the metadata "
            "stage downloads timestamps/<seq-id>.txt from the CODa sequence ZIP."
        ),
    )
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--answer-mode", choices=("graph_only", "mock", "live"), default="live")
    parser.add_argument("--retrieval-limit", type=int, default=10)
    parser.add_argument("--domain-profile", default="generic")
    parser.add_argument("--question-ir-json", type=Path, default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--include-manifest-images", action="store_true")
    parser.add_argument("--force-rewrite", action="store_true")
    parser.add_argument(
        "--use-raw-hsgg-paths",
        action="store_true",
        help=(
            "Use --hsgg-output-dir directly for graph construction. By default, "
            "temporal graph construction uses an existing hsgg_local_paths "
            "directory when present, or the rewrite stage output when selected."
        ),
    )
    parser.add_argument(
        "--compact-scene-label-entities",
        action="store_true",
        help="Pass through to canonical graph conversion.",
    )
    parser.add_argument(
        "--graph-profile",
        choices=("canonical", "time_fact_core", "topology_pruned_after_scene_fact"),
        default="canonical",
        help=(
            "Canonical graph profile. The default is the first-mainline canonical "
            "graph; time_fact_core is the second-mainline direct graph. "
            "topology_pruned_after_scene_fact is accepted as a legacy alias."
        ),
    )
    parser.add_argument(
        "--canonical-graph-json",
        type=Path,
        default=None,
        help=(
            "Optional canonical graph output path. Defaults to the unified layout "
            "canonical_temporal_memory_graph.json."
        ),
    )
    parser.add_argument(
        "--canonical-report-json",
        type=Path,
        default=None,
        help="Optional report path for profile-specific canonical graph conversion.",
    )
    parser.add_argument(
        "--include-position",
        action="store_true",
        help="Do not add --non-position-only to the QA stage.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(*STAGE_ORDER, "all"),
        default=["all"],
        help="Stages to print or run, in mainline order.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the planned steps. Without this flag, only print the plan.",
    )
    return parser


def selected_stages(values: list[str]) -> set[str]:
    if "all" in values:
        return set(STAGE_ORDER)
    return set(values)


def command_line(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def python_cmd(script: str, *args: str | Path) -> list[str]:
    return [sys.executable, str(REPO_ROOT / script), *(str(arg) for arg in args)]


def has_standard_hsgg_json(input_dir: Path) -> bool:
    json_dir = input_dir / "json" if (input_dir / "json").exists() else input_dir
    return all(
        (json_dir / name).exists()
        for name in (
            "objects.json",
            "attributes.json",
            "relations.json",
            "hierarchy_mapping.json",
        )
    )


def resolve_graph_input_dir(
    args: argparse.Namespace,
    layout: NavQASequenceLayout,
    wanted: set[str],
) -> Path:
    if args.use_raw_hsgg_paths:
        return args.hsgg_output_dir
    if "rewrite-hsgg-paths" in wanted:
        return layout.hsgg_local_paths_dir
    if has_standard_hsgg_json(layout.hsgg_local_paths_dir):
        return layout.hsgg_local_paths_dir
    return args.hsgg_output_dir


def build_steps(args: argparse.Namespace, layout: NavQASequenceLayout) -> list[PipelineStep]:
    wanted = selected_stages(args.stages)
    hsgg_output_dir = args.hsgg_output_dir
    canonical_graph_json = (
        getattr(args, "canonical_graph_json", None) or layout.canonical_graph_json
    )
    steps: list[PipelineStep] = [
        PipelineStep(
            name="layout",
            note=(
                f"sequence_root={layout.sequence_root}; "
                f"sample_root={layout.sample_root}"
            ),
        )
    ]

    if "metadata" in wanted:
        if args.qa_unfilled_json is not None:
            steps.append(
                PipelineStep(
                    name="metadata:qa-unfilled",
                    copy_source=args.qa_unfilled_json,
                    copy_target=layout.qa_unfilled_json,
                )
            )
        else:
            steps.append(
                PipelineStep(
                    name="metadata:qa-unfilled",
                    note=(
                        f"no source supplied; expecting existing file at "
                        f"{layout.qa_unfilled_json}"
                    ),
                )
            )

        if args.timestamps is not None:
            steps.append(
                PipelineStep(
                    name="metadata:timestamps",
                    copy_source=args.timestamps,
                    copy_target=layout.timestamps_path,
                )
            )
        else:
            steps.append(
                PipelineStep(
                    name="metadata:timestamps",
                    command=python_cmd(
                        "scripts/download_coda_zip_member.py",
                        f"timestamps/{layout.seq_id}.txt",
                        "--seq-id",
                        layout.seq_id,
                        "--out",
                        layout.timestamps_path,
                    ),
                )
            )

    if "download-images" in wanted:
        command = python_cmd(
            "scripts/download_navqa_images_from_hsgg_output.py",
            hsgg_output_dir,
            "--seq-id",
            layout.seq_id,
            "--cam",
            layout.cam,
            "--stride",
            str(layout.stride or ""),
            "--layout-root",
            args.layout_root,
            "--out",
            layout.images_dir,
            "--workers",
            str(args.workers),
        )
        if args.include_manifest_images:
            command.append("--include-manifest")
        steps.append(PipelineStep(name="download-images", command=command))

    graph_input_dir = resolve_graph_input_dir(args, layout, wanted)
    steps.append(
        PipelineStep(
            name="graph-input",
            note=f"temporal_graph_input={graph_input_dir}",
        )
    )
    if "rewrite-hsgg-paths" in wanted:
        graph_input_dir = layout.hsgg_local_paths_dir
        command = python_cmd(
            "scripts/rewrite_hsgg_image_paths.py",
            hsgg_output_dir,
            "--images-dir",
            layout.images_dir,
            "--output-dir",
            layout.hsgg_local_paths_dir,
        )
        if args.force_rewrite:
            command.append("--force")
        steps.append(PipelineStep(name="rewrite-hsgg-paths", command=command))

    if "frame-time-index" in wanted:
        steps.append(
            PipelineStep(
                name="frame-time-index",
                command=python_cmd(
                    "scripts/build_navqa_frame_time_index.py",
                    "--seq-id",
                    layout.seq_id,
                    "--cam",
                    layout.cam,
                    "--stride",
                    str(layout.stride or ""),
                    "--layout-root",
                    args.layout_root,
                    "--image-dir",
                    layout.images_dir,
                    "--timestamps",
                    layout.timestamps_path,
                    "--qa-unfilled",
                    layout.qa_unfilled_json,
                    "--out-json",
                    layout.frame_time_index_json,
                    "--out-csv",
                    layout.frame_time_index_csv,
                ),
            )
        )

    if "preflight-temporal-graph" in wanted:
        command = python_cmd(
            "scripts/build/preflight_temporal_graph_input.py",
            graph_input_dir,
            "--frame-time-index",
            layout.frame_time_index_json,
        )
        steps.append(PipelineStep(name="preflight-temporal-graph", command=command))

    if "temporal-graph" in wanted:
        command = python_cmd(
            "scripts/build/convert_json_to_temporal_graph.py",
            graph_input_dir,
            "--output",
            layout.temporal_graph_json,
            "--config",
            args.config,
            "--frame-time-index",
            layout.frame_time_index_json,
        )
        if args.video_id:
            command.extend(["--video-id", args.video_id])
        steps.append(PipelineStep(name="temporal-graph", command=command))

    if "canonical-graph" in wanted:
        command = python_cmd(
            "scripts/build/convert_temporal_graph_to_canonical_graph.py",
            layout.temporal_graph_json,
            "--output",
            canonical_graph_json,
            "--graph-profile",
            args.graph_profile,
        )
        if args.canonical_report_json is not None:
            command.extend(["--report-json", args.canonical_report_json])
        if args.compact_scene_label_entities:
            command.append("--compact-scene-label-entities")
        steps.append(PipelineStep(name="canonical-graph", command=command))

    if "qa" in wanted:
        command = python_cmd(
            "scripts/datasets/navqa/qa/run_navqa_official_qa_on_canonical_graph.py",
            canonical_graph_json,
            "--csv",
            args.csv,
            "--seq-id",
            layout.seq_id,
            "--cam",
            layout.cam,
            "--stride",
            str(layout.stride or ""),
            "--layout-root",
            args.layout_root,
            "--answer-mode",
            args.answer_mode,
            "--config",
            args.config,
            "--retrieval-limit",
            str(args.retrieval_limit),
            "--qa-unfilled-json",
            layout.qa_unfilled_json,
            "--frame-time-index",
            layout.frame_time_index_json,
            "--projection-cache",
            layout.projection_cache,
            "--domain-profile",
            args.domain_profile,
            "--output-json",
            layout.qa_results_json,
            "--output-csv",
            layout.qa_results_csv,
            "--output-md",
            layout.qa_results_md,
        )
        if args.question_ir_json is not None:
            command.extend(["--question-ir-json", str(args.question_ir_json)])
        if not args.include_position:
            command.append("--non-position-only")
        steps.append(PipelineStep(name="qa", command=command))

    return steps


def run_step(step: PipelineStep) -> None:
    if step.copy_source is not None and step.copy_target is not None:
        source = step.copy_source
        target = step.copy_target
        if not source.exists():
            raise SystemExit(f"Missing input for {step.name}: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return
    if step.command is not None:
        subprocess.run(step.command, check=True, cwd=REPO_ROOT)


def print_step(index: int, step: PipelineStep) -> None:
    prefix = f"[{index}] {step.name}"
    if step.note:
        print(f"{prefix}: {step.note}")
    elif step.copy_source is not None and step.copy_target is not None:
        print(f"{prefix}: copy {step.copy_source} -> {step.copy_target}")
    elif step.command is not None:
        print(f"{prefix}: {command_line(step.command)}")
    else:
        print(prefix)


def main() -> None:
    args = build_parser().parse_args()
    layout = default_navqa_sequence_layout(
        seq_id=args.seq_id,
        cam=args.cam,
        stride=args.stride,
        layout_root=args.layout_root,
    )
    steps = build_steps(args, layout)
    print(f"mode={'run' if args.run else 'dry-run'}")
    for index, step in enumerate(steps, start=1):
        print_step(index, step)
    if not args.run:
        print("Add --run to execute these steps.")
        return
    for step in steps:
        if step.command is None and step.copy_source is None:
            continue
        run_step(step)


if __name__ == "__main__":
    main()
