"""Reject swapped unit bindings and altered archive manifests before data reads."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aligned_catalog_verifier", ROOT / "scripts/release/verify_resource_catalog.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def catalog(tmp_path):
    for name in ("aligned_packages.json", "aligned_units.json", "aligned_packages.SHA256SUMS.txt"):
        shutil.copyfile(ROOT / "catalog" / name, tmp_path / name)
    units = json.loads((ROOT / "catalog/unit_catalog.json").read_text(encoding="utf-8"))["units"]
    lookup = {(u["dataset"], u["shard"], str(u["unit_id"])): u for u in units}
    return tmp_path, lookup


def test_verified_pair_manifest(catalog):
    root, lookup = catalog
    assert verifier.verify_aligned_catalog(root, lookup)["aligned_package_count"] == 60


def test_unit_cannot_borrow_another_groups_archive(catalog):
    root, lookup = catalog
    path = root / "aligned_units.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["units"][0]["visual_archive"] = data["units"][1]["visual_archive"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="another group's"):
        verifier.verify_aligned_catalog(root, lookup)


def test_archive_pair_must_cover_same_units(catalog):
    root, lookup = catalog
    path = root / "aligned_packages.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["packages"][0]["unit_ids"] = ["unknown-unit"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unit coverage"):
        verifier.verify_aligned_catalog(root, lookup)


def test_checksum_text_cannot_disagree_with_catalog(catalog):
    root, lookup = catalog
    (root / "aligned_packages.SHA256SUMS.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum ledger"):
        verifier.verify_aligned_catalog(root, lookup)
