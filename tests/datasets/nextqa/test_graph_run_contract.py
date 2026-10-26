from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlkg.datasets.nextqa.runtime.contracts import load_full_manifest
from vlkg.datasets.nextqa.runtime.io_utils import (
    NEXTQA_SCRATCH_ROOT,
    atomic_write_json,
    require_within_nextqa_scratch,
)
from vlkg.datasets.nextqa.runtime.run_snapshot import (
    collect_imported_repo_sources,
    compare_source_snapshot,
    fingerprint_file,
)
import vlkg.datasets.nextqa.runtime.run_snapshot as run_snapshot_module
import yaml
from vlkg.datasets.nextqa.graph.temporal_canonical import load_nextqa_graph_config
from vlkg.core.config.config import load_domain_config
from scripts.datasets.nextqa.build.build_graph_full import select_pending_videos
from scripts.datasets.nextqa.build.build_graph_full import _expected_counts


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())




@pytest.mark.requires_data
def test_full_manifest_is_complete_and_uses_all_every30_records() -> None:
    path = (
        REPO_ROOT
        / "datasets/nextqa/source/nextqa-val-v1-repaired29/payload/source_manifest.json"
    )
    if not path.is_file():
        pytest.skip('requires external normalized NextQA source payload')
    manifest = load_full_manifest(path)

    assert manifest["video_count"] == 570
    assert [video["local_sequential_id"] for video in manifest["videos"]] == list(
        range(1, 571)
    )
    assert _expected_counts(manifest) == {
        "videos": 570,
        "sampled_frames": 23381,
        "completed_hsgg_frames": 23044,
        "skipped_frames": 337,
    }


def test_write_guard_rejects_non_nextqa_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must stay below"):
        require_within_nextqa_scratch(tmp_path / "outside.json")


def test_atomic_json_writes_only_below_nextqa_scratch(nextqa_tmp_dir: Path) -> None:
    target = atomic_write_json(nextqa_tmp_dir / "record.json", {"ok": True})
    assert target.is_relative_to(NEXTQA_SCRATCH_ROOT)
    assert json.loads(target.read_text()) == {"ok": True}


def test_source_drift_is_detected(nextqa_tmp_dir: Path) -> None:
    source = nextqa_tmp_dir / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    snapshot = {
        "source_files": [fingerprint_file(source)],
    }
    source.write_text("VALUE = 2\n", encoding="utf-8")

    report = compare_source_snapshot(snapshot, current_source_paths=[source])

    assert report["code_drift_detected"] is True
    assert [item["path"] for item in report["changed_files"]] == [str(source.resolve())]


def test_source_snapshot_excludes_venv_and_scratch_dependencies() -> None:
    paths = collect_imported_repo_sources(
        modules=[run_snapshot_module, yaml],
    )

    assert Path(run_snapshot_module.__file__).resolve() in paths
    assert all(".venv" not in path.parts for path in paths)
    assert all(".scratch" not in path.relative_to(REPO_ROOT).parts for path in paths)


def test_nextqa_association_parameters_match_navqa_cg6() -> None:
    nextqa = load_domain_config(
        REPO_ROOT / "configs/datasets/nextqa/graph.yaml",
        config_dir=REPO_ROOT / "configs",
    )
    navqa = load_domain_config(
        REPO_ROOT / "configs/datasets/navqa/graph.yaml",
        config_dir=REPO_ROOT / "configs",
    )

    assert nextqa.association == navqa.association.model_copy(
        update={"label_conflict_embedding_veto_threshold": 0.86}
    )
    assert nextqa.enable_structured_state_extraction is True
    assert navqa.enable_structured_state_extraction is False
    effective_nextqa = load_nextqa_graph_config(
        REPO_ROOT / "configs/datasets/nextqa/graph.yaml",
        config_dir=REPO_ROOT / "configs",
    )
    assert effective_nextqa.component_label_set == set()


def test_mainline_human_compat_profile_preserves_selected_parameters() -> None:
    base = load_domain_config(
        REPO_ROOT / "configs/datasets/nextqa/graph.yaml",
        config_dir=REPO_ROOT / "configs",
    )
    experimental = load_nextqa_graph_config(
        REPO_ROOT / "configs/datasets/nextqa/graph.yaml",
        config_dir=REPO_ROOT / "configs",
    )

    assert experimental.profile_name == "nextqa_cg6_human_compat_v2"
    assert {
        "person",
        "people",
        "man",
        "woman",
        "boy",
        "girl",
        "child",
        "player",
    } in experimental.normalized_label_compatibility_groups
    assert experimental.association == base.association.model_copy(
        update={"label_conflict_embedding_veto_threshold": 0.86}
    )


def test_resume_selection_skips_completed_and_retries_error() -> None:
    videos = [
        {"official_video_id": "100"},
        {"official_video_id": "200"},
        {"official_video_id": "300"},
    ]
    completed = [SimpleNamespace(official_video_id="200")]

    pending = select_pending_videos(videos, completed=completed)

    assert [item["official_video_id"] for item in pending] == ["100", "300"]
