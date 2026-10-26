"""Checks that guard portable metadata reads against bad paths and corruption."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release/verify_resource_catalog.py"
spec = importlib.util.spec_from_file_location("release_catalog_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.mark.parametrize("value", ["", ".", "../outside", "/outside", "C:/outside", "C:outside", "nested/../outside", "nested\\outside", "https://example.invalid/data"])
def test_rejects_nonportable_references(tmp_path, value):
    with pytest.raises(ValueError, match="unsafe"):
        verifier.safe_reference(tmp_path, value)


def test_relative_reference_uses_selected_root_not_cwd(tmp_path, monkeypatch):
    data_root = tmp_path / "selected"
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    assert verifier.safe_reference(data_root, "datasets/navqa/example.json") == data_root / "datasets/navqa/example.json"


def test_corrupt_catalogue_file_fails_before_resource_loading(tmp_path):
    (tmp_path / "record.json").write_text("{}", encoding="utf-8")
    (tmp_path / "catalog_files.json").write_text(json.dumps({"files": [{"path": "record.json", "bytes": 2, "sha256": "0" * 64}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verifier.verify_catalog(tmp_path)


def test_ledger_cannot_reference_parent_directory(tmp_path):
    (tmp_path / "catalog_files.json").write_text(json.dumps({"files": [{"path": "../outside", "bytes": 0, "sha256": "0" * 64}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        verifier.verify_catalog(tmp_path)
