from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_navqa_sequence.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_navqa_sequence_test", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "seq_id": "3",
        "cam": "0",
        "stride": 10,
        "layout_root": Path(".scratch/navqa"),
        "hsgg_output_dir": Path(".scratch/hsgg"),
        "csv": Path(".scratch/navqa_official_data.csv"),
        "qa_unfilled_json": None,
        "timestamps": None,
        "config": "default.yaml",
        "answer_mode": "live",
        "retrieval_limit": 10,
        "domain_profile": "generic",
        "question_ir_json": None,
        "video_id": None,
        "workers": 24,
        "include_manifest_images": False,
        "force_rewrite": False,
        "use_raw_hsgg_paths": False,
        "compact_scene_label_entities": False,
        "graph_profile": "canonical",
        "canonical_graph_json": None,
        "canonical_report_json": None,
        "include_position": False,
        "stages": ["canonical-graph"],
        "run": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_prepare_defaults_to_first_mainline_canonical_profile() -> None:
    module = _load_module()
    layout = module.default_navqa_sequence_layout(seq_id="3", cam="0", stride=10)

    steps = module.build_steps(_args(), layout)
    command = next(step.command for step in steps if step.name == "canonical-graph")

    assert "--graph-profile" in command
    assert command[command.index("--graph-profile") + 1] == "canonical"
    assert command[command.index("--output") + 1] == str(layout.canonical_graph_json)


def test_prepare_can_route_canonical_stage_to_second_mainline_profile() -> None:
    module = _load_module()
    layout = module.default_navqa_sequence_layout(seq_id="3", cam="0", stride=10)
    output = Path(".scratch/navqa_clean_second_mainline/time_fact_core/seq3/canonical_temporal_memory_graph.json")
    report = Path(".scratch/navqa_clean_second_mainline/time_fact_core/seq3/time_fact_core_report.json")

    steps = module.build_steps(
        _args(
            graph_profile="time_fact_core",
            canonical_graph_json=output,
            canonical_report_json=report,
        ),
        layout,
    )
    command = next(step.command for step in steps if step.name == "canonical-graph")

    assert "--graph-profile" in command
    assert command[command.index("--graph-profile") + 1] == "time_fact_core"
    assert command[command.index("--output") + 1] == str(output)
    assert str(command[command.index("--report-json") + 1]) == str(report)


def test_prepare_qa_stage_uses_selected_canonical_graph_path() -> None:
    module = _load_module()
    layout = module.default_navqa_sequence_layout(seq_id="3", cam="0", stride=10)
    output = Path(".scratch/navqa_clean_second_mainline/time_fact_core/seq3/canonical_temporal_memory_graph.json")

    steps = module.build_steps(
        _args(
            stages=["canonical-graph", "qa"],
            graph_profile="time_fact_core",
            canonical_graph_json=output,
        ),
        layout,
    )
    qa_command = next(step.command for step in steps if step.name == "qa")

    assert Path(qa_command[1]).name == "run_navqa_official_qa_on_canonical_graph.py"
    assert qa_command[2] == str(output)
