"""Published controls are verifiable before downloading any dataset payloads."""
import json
from pathlib import Path
import sys

import pytest
import yaml

from scripts.datasets import manage, nextqa_mainline


@pytest.fixture
def controls(tmp_path):
    layers = manage.LAYERS
    registry = {"datasets": {}, "target_layout": {"dataset_template": dict.fromkeys(layers, "unused")}}
    for dataset in manage.DATASET_KEYS:
        ids = {manage.ASSET_ID_FIELDS[layer]: f"{dataset}-{layer}" for layer in layers}
        registry["datasets"][dataset] = {"asset_ids": ids}
        config = tmp_path / f"configs/datasets/{dataset}/graph.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("{}", encoding="utf-8")
        manifest = {"dataset": dataset, "asset_ids": ids,
                    "graph_config": config.relative_to(tmp_path).as_posix(),
                    "target_layout": dict.fromkeys(layers, "unused"), "current_paths": {}}
        for layer in layers:
            asset_id = ids[manage.ASSET_ID_FIELDS[layer]]
            payload = f"datasets/{dataset}/{layer}/{asset_id}/payload/not-downloaded.json"
            manifest["current_paths"]["graph" if layer == "graphs" else layer] = [payload]
            pointer = {"dataset": dataset, "layer": layer, "asset_id": asset_id,
                       "current_paths": [payload], "entrypoints": {"manifest": payload}}
            path = tmp_path / f"datasets/{dataset}/{layer}/{asset_id}/asset_pointer.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(pointer), encoding="utf-8")
        path = tmp_path / f"registry/datasets/{dataset}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (tmp_path / "configs/dataset_asset_registry.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    return tmp_path


def test_metadata_check_does_not_need_payloads_or_history(controls, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("historical verifier must not run")
    monkeypatch.setattr(manage, "verify_metadata_copies", forbidden)
    before = sorted(p.relative_to(controls).as_posix() for p in controls.rglob("*"))
    for dataset in manage.DATASET_KEYS:
        result = manage.inspect_metadata(dataset, controls)
        assert result["metadata_valid"] and not result["issues"]
        assert result["external_assets_checked"] is False
    assert before == sorted(p.relative_to(controls).as_posix() for p in controls.rglob("*"))


@pytest.mark.parametrize("bad_path", ["../outside", "C:/private/data", "/tmp/private", "datasets/../../outside"])
def test_metadata_check_rejects_escaping_paths(controls, bad_path):
    path = controls / "datasets/navqa/source/navqa-source/asset_pointer.json"
    pointer = json.loads(path.read_text())
    pointer["entrypoints"]["manifest"] = bad_path
    path.write_text(json.dumps(pointer))
    assert not manage.inspect_metadata("navqa", controls)["metadata_valid"]


def test_metadata_check_detects_pointer_identity_and_missing_config(controls):
    path = controls / "datasets/navqa/source/navqa-source/asset_pointer.json"
    pointer = json.loads(path.read_text())
    pointer["dataset"] = "nextqa"
    path.write_text(json.dumps(pointer))
    assert any("identity mismatch" in issue for issue in manage.inspect_metadata("navqa", controls)["issues"])
    (controls / "configs/datasets/nextqa/graph.yaml").unlink()
    assert any("missing graph config" in issue for issue in manage.inspect_metadata("nextqa", controls)["issues"])


@pytest.mark.parametrize("stage,group,filename", [
    ("retrieve", "retrieval", "run_graph_retrieval_pilot.py"),
    ("graph-qa", "qa", "run_graph_only_qa_pilot.py"),
    ("visual-evidence", "retrieval", "build_visual_evidence_v2.py"),
    ("visual-base-qa", "qa", "run_visual_enhanced_qa.py"),
    ("compose-C", "qa", "merge_visual_route_policy_v3.py"),
])
def test_nextqa_uses_current_interpreter_and_script_layout(tmp_path, monkeypatch, stage, group, filename):
    script = tmp_path / "scripts/datasets/nextqa" / group / filename
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    monkeypatch.setattr(nextqa_mainline, "asset_file", lambda *args: tmp_path / "external.json")
    command = nextqa_mainline.stage_command(stage, "new-run", root=tmp_path)
    assert command[:2] == [sys.executable, str(script)]
