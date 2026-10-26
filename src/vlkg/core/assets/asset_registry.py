"""Symmetric dataset asset registry access.

The registry is metadata only. Payloads remain at their current locations until a
pointer is replaced by a migrated asset manifest. This module deliberately does
not create directories, copy files, move files, or delete files.
"""

from __future__ import annotations

from vlkg.workspace import workspace_root

import json
import hashlib
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = workspace_root()
REGISTRY_PATH = REPO_ROOT / "configs" / "dataset_asset_registry.yaml"
DATASET_MANIFEST_ROOT = REPO_ROOT / "registry" / "datasets"
DATASET_ROOT = REPO_ROOT / "datasets"
DATASET_KEYS = ("navqa", "nextqa", "openeqa")
LAYERS = ("source", "graphs", "indexes", "ir", "retrieval", "runs", "reports", "cache", "scratch")
ASSET_ID_FIELDS = {"source": "source_id", "graphs": "graph_id", "indexes": "index_id",
                   "ir": "ir_id", "retrieval": "retrieval_id", "runs": "qa_id",
                   "cache": "cache_id", "reports": "reports_id", "scratch": "scratch_id"}


def _component(value: str) -> None:
    if not value or value in {".", ".."} or any(c in value for c in '/\\:'):
        raise ValueError(f"invalid asset identifier: {value!r}")


def _local_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"asset path escapes repository: {value}")
    return path


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError("dataset asset registry must contain a datasets mapping")
    missing = [key for key in DATASET_KEYS if key not in payload["datasets"]]
    if missing:
        raise ValueError(f"dataset registry missing datasets: {missing}")
    return payload


def load_dataset_manifest(dataset: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    if dataset not in DATASET_KEYS:
        raise ValueError(f"unsupported dataset: {dataset}")
    path = root / "registry" / "datasets" / f"{dataset}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset:
        raise ValueError(f"dataset manifest mismatch: {path}")
    target_layout = payload.get("target_layout") or {}
    if set(target_layout) != set(LAYERS):
        raise ValueError(f"dataset manifest layers are not symmetric: {path}")
    return payload


def load_pointer(dataset: str, layer: str, asset_id: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    if dataset not in DATASET_KEYS:
        raise ValueError(f"unsupported dataset: {dataset}")
    _component(asset_id)
    if layer not in LAYERS:
        raise ValueError(f"unsupported asset layer: {layer}")
    path = root / "datasets" / dataset / layer / asset_id / "asset_pointer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset") != dataset or payload.get("layer") != layer:
        raise ValueError(f"asset pointer identity mismatch: {path}")
    if payload.get("asset_id") != asset_id:
        raise ValueError(f"asset pointer id mismatch: {path}")
    return payload


def current_paths(dataset: str, layer: str, asset_id: str, root: Path = REPO_ROOT) -> tuple[Path, ...]:
    pointer = load_pointer(dataset, layer, asset_id, root)
    return tuple(_local_path(root, value) for value in pointer.get("current_paths", []))


def metadata_paths(dataset: str, layer: str, asset_id: str, root: Path = REPO_ROOT) -> tuple[Path, ...]:
    pointer = load_pointer(dataset, layer, asset_id, root)
    return tuple(_local_path(root, value) for value in pointer.get("metadata_paths", []))


def graph_config_path(dataset: str, root: Path = REPO_ROOT) -> Path:
    """Return the materialized per-dataset configuration, without loading a model."""
    manifest = load_dataset_manifest(dataset, root)
    return _local_path(root, manifest["graph_config"])


def asset_file(dataset: str, layer: str, name: str, root: Path = REPO_ROOT) -> Path:
    """Resolve a named selected asset without hardcoding a dated run directory."""
    if layer not in ASSET_ID_FIELDS:
        raise ValueError(f"layer has no selected asset identifier: {layer}")
    manifest = load_dataset_manifest(dataset, root)
    asset_id = manifest['asset_ids'][ASSET_ID_FIELDS[layer]]
    pointer = load_pointer(dataset, layer, asset_id, root)
    try:
        value = pointer['entrypoints'][name]
    except KeyError as exc:
        raise ValueError(f"no named {name} entry for {dataset}/{layer}") from exc
    path = _local_path(root, value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def inspect_dataset(dataset: str, root: Path = REPO_ROOT) -> dict[str, Any]:
    """Inspect actual paths for all layers; empty layers are not evidence of completion."""
    manifest = load_dataset_manifest(dataset, root)
    issues: list[str] = []
    assets = []
    required_layers = {"source", "graphs", "indexes", "ir", "retrieval", "runs"}
    owned_required_payloads = set()
    for layer in LAYERS:
        pointers = sorted((root / "datasets" / dataset / layer).glob("*/asset_pointer.json"))
        if len(pointers) != 1:
            issues.append(f"expected one selected pointer for {layer}; found {len(pointers)}")
        for path in pointers:
            asset_id = path.parent.name
            pointer = load_pointer(dataset, layer, asset_id, root)
            paths = current_paths(dataset, layer, asset_id, root)
            metadata = metadata_paths(dataset, layer, asset_id, root)
            dependencies = tuple(_local_path(root, value) for value in pointer.get('dependency_paths', []))
            for item in (*paths, *metadata, *dependencies):
                if not item.exists():
                    issues.append(f"missing: {item.relative_to(root.resolve()).as_posix()}")
            for name, value in pointer.get('entrypoints', {}).items():
                if not _local_path(root, value).is_file():
                    issues.append(f"missing entrypoint: {layer}/{name}: {value}")
            # Correlation with the dataset manifest prevents silent pointer drift.
            key = {"graphs": "graph"}.get(layer, layer)
            expected = {_local_path(root, s) for s in manifest["current_paths"].get(key, [])}
            if set(paths) != expected:
                issues.append(f"manifest/pointer mismatch: {layer}/{asset_id}")
            if layer in required_layers and not paths:
                issues.append(f"unresolved required layer: {layer}/{asset_id}")
            owned_root = (root / "datasets" / dataset / layer).resolve()
            if paths and all(item.resolve().is_relative_to(owned_root) for item in paths):
                if pointer["payload_migrated"] and layer in required_layers:
                    owned_required_payloads.add(layer)
            elif paths:
                issues.append(f"current paths escape the owning layer: {layer}/{asset_id}")
            assets.append({"layer": layer, "asset_id": asset_id, "status": pointer["status"],
                           "current_path_count": len(paths), "metadata_count": len(metadata),
                           "dependency_path_count": len(dependencies),
                           "payload_migrated": pointer["payload_migrated"]})
    cfg = graph_config_path(dataset, root)
    if not cfg.is_file():
        issues.append(f"missing graph config: {cfg}")
    return {"dataset": dataset, "graph_config": cfg.relative_to(root.resolve()).as_posix(),
            "assets": assets, "issues": issues, "paths_valid": not issues,
            "payload_migration_complete": not issues and owned_required_payloads == required_layers,
            "migration_scope": "Selected required payloads exist under their owning layers; deep dependency, history isolation and final acceptance are separate gates."}


def verify_metadata_copies(root: Path = REPO_ROOT) -> list[str]:
    """Check copied evidence against its original byte hashes, not reserialized JSON."""
    from vlkg.core.assets.asset_provenance import historical_control_file
    ledger = json.loads(historical_control_file("registry/migrations/metadata_migration_v1.json", root).read_text(encoding="utf-8"))
    issues = []
    for row in ledger["items"]:
        for side in ("source", "target"):
            path = _local_path(root, row[side])
            if not path.is_file():
                from vlkg.core.assets.asset_provenance import relocated_provenance
                path = relocated_provenance(path, root)
            if not path.is_file():
                issues.append(f"missing: {row[side]}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != row[f"{side}_sha256"]:
                issues.append(f"hash mismatch: {row[side]}")
    return issues


def validate_symmetric_layout(root: Path = REPO_ROOT) -> None:
    registry = load_registry(root / "configs" / "dataset_asset_registry.yaml")
    expected = set(registry["target_layout"]["dataset_template"])
    if expected != set(LAYERS):
        raise ValueError(f"registry layers are not symmetric: {sorted(expected)}")
    for dataset in DATASET_KEYS:
        manifest = load_dataset_manifest(dataset, root)
        if set(manifest["target_layout"]) != expected:
            raise ValueError(f"manifest layers differ for {dataset}")
