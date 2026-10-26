"""Common read-only asset operations for all three datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vlkg.core.assets.asset_registry import (  # noqa: E402
    DATASET_KEYS, graph_config_path, inspect_dataset, validate_symmetric_layout,
    verify_metadata_copies,
    asset_file, LAYERS,
    ASSET_ID_FIELDS, load_registry, load_dataset_manifest, load_pointer,
)


def inspect_metadata(dataset: str, root: Path = ROOT) -> dict:
    """Validate published controls without opening or requiring external payloads."""
    root = root.resolve()
    issues = []
    assets = []

    def relative_path(value, context):
        if (not isinstance(value, str) or not value or "\\" in value
                or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
                or ".." in PurePosixPath(value).parts):
            raise ValueError(f"{context}: expected a portable workspace-relative path")
        path = (root / value).resolve()
        if path == root or not path.is_relative_to(root):
            raise ValueError(f"{context}: path escapes workspace")
        return path

    try:
        validate_symmetric_layout(root)
        registry = load_registry(root / "configs/dataset_asset_registry.yaml")
        manifest = load_dataset_manifest(dataset, root)
        config = relative_path(manifest["graph_config"], "graph_config")
        if not config.is_file():
            issues.append("missing graph config: " + manifest["graph_config"])
        for layer in LAYERS:
            selected = manifest["asset_ids"][ASSET_ID_FIELDS[layer]]
            if registry["datasets"][dataset]["asset_ids"][ASSET_ID_FIELDS[layer]] != selected:
                issues.append(f"registry/manifest selected asset mismatch: {layer}")
            pointers = sorted((root / "datasets" / dataset / layer).glob("*/asset_pointer.json"))
            if len(pointers) != 1:
                issues.append(f"expected one selected pointer for {layer}; found {len(pointers)}")
            pointer = load_pointer(dataset, layer, selected, root)
            for path in pointers:
                if path.parent.name != selected:
                    issues.append(f"unexpected selected pointer: {layer}/{path.parent.name}")
            paths = pointer.get("current_paths", [])
            for field in ("current_paths", "metadata_paths", "dependency_paths"):
                values = pointer.get(field, [])
                if not isinstance(values, list):
                    raise ValueError(f"{layer}/{field}: expected a list")
                for value in values:
                    relative_path(value, f"{layer}/{field}")
            for name, value in pointer.get("entrypoints", {}).items():
                relative_path(value, f"{layer}/entrypoints/{name}")
            for field in ("target_path", "registry", "source_manifest"):
                if field in pointer:
                    relative_path(pointer[field], f"{layer}/{field}")
            expected = manifest["current_paths"].get("graph" if layer == "graphs" else layer, [])
            for value in expected:
                relative_path(value, f"{layer}/manifest current_paths")
            if set(paths) != set(expected):
                issues.append(f"manifest/pointer mismatch: {layer}/{selected}")
            owner = root / "datasets" / dataset / layer
            if any(not relative_path(value, layer).is_relative_to(owner) for value in paths):
                issues.append(f"current paths escape the owning layer: {layer}/{selected}")
            assets.append({"layer": layer, "asset_id": selected})
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    return {"dataset": dataset, "assets": assets, "issues": issues,
            "metadata_valid": not issues, "external_assets_checked": False,
            "scope": "Published registry, selected pointer identities, relative path bounds and graph configs only; external payloads and historical migration evidence are not checked."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "validate", "graph-config", "resolve"))
    parser.add_argument("--dataset", choices=(*DATASET_KEYS, "all"), default="all")
    parser.add_argument("--layer", choices=LAYERS)
    parser.add_argument("--name", default="manifest")
    parser.add_argument("--metadata-only", action="store_true",
                        help="For status/validate: check published controls without external payloads or historical migration evidence.")
    args = parser.parse_args()
    datasets = DATASET_KEYS if args.dataset == "all" else (args.dataset,)
    if args.metadata_only:
        if args.command not in ("status", "validate"):
            parser.error("--metadata-only is available only with status or validate")
        result = {"datasets": [inspect_metadata(d, ROOT) for d in datasets],
                  "external_assets_checked": False}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return int(any(d["issues"] for d in result["datasets"]))
    if args.command == 'resolve':
        if not args.layer:
            parser.error('resolve requires --layer')
        print(json.dumps({d: str(asset_file(d, args.layer, args.name, ROOT)) for d in datasets}, indent=2))
        return 0
    if args.command == "graph-config":
        print(json.dumps({d: str(graph_config_path(d, ROOT)) for d in datasets}, indent=2))
        return 0
    validate_symmetric_layout(ROOT)
    result = {"datasets": [inspect_dataset(d, ROOT) for d in datasets]}
    if args.command == "validate":
        result["metadata_copy_issues"] = verify_metadata_copies(ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(any(d["issues"] for d in result["datasets"]) or bool(result.get("metadata_copy_issues")))


if __name__ == "__main__":
    raise SystemExit(main())
