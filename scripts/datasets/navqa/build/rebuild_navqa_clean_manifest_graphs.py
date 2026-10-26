from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild canonical graphs listed in a clean NaVQA sequence manifest "
            "directly from each sequence temporal_graph.json. Experimental "
            "manifests may also request temporal graph rebuilds first."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--rebuild-temporal",
        action="store_true",
        help=(
            "Rebuild temporal_graph.json paths declared by the manifest before "
            "canonical graph conversion."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser


def repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest_path = repo_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    return payload


def graph_profile(payload: dict[str, Any]) -> str:
    profile = str(payload.get("graph_profile") or "").strip()
    if not profile:
        raise ValueError("manifest missing graph_profile")
    return profile


def temporal_graph_config(payload: dict[str, Any]) -> str:
    return str(payload.get("temporal_graph_config") or "default.yaml")


def stable_identity_merge_mode(payload: dict[str, Any]) -> str | None:
    value = payload.get("stable_identity_merge_mode")
    if value is None:
        return None
    normalized = str(value).strip()
    if normalized not in {"semantic", "none"}:
        raise ValueError(
            "stable_identity_merge_mode must be 'semantic' or 'none'"
        )
    return normalized


def optional_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a boolean when provided")


def sequence_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("sequences")
    if not isinstance(items, list):
        raise ValueError("manifest missing sequences list")
    return [item for item in items if isinstance(item, dict)]


def temporal_graph_path_for_item(item: dict[str, Any]) -> Path:
    if item.get("temporal_graph_json"):
        return repo_path(item["temporal_graph_json"])
    sample_root = repo_path(item["sample_root"])
    return sample_root / "temporal_graph.json"


def temporal_graph_input_dir_for_item(item: dict[str, Any]) -> Path:
    if item.get("hsgg_input_dir"):
        return repo_path(item["hsgg_input_dir"])
    sample_root = repo_path(item["sample_root"])
    local_paths = sample_root / "hsgg_local_paths"
    return local_paths if local_paths.exists() else sample_root


def temporal_command_for_item(item: dict[str, Any], *, config: str) -> list[str]:
    from vlkg.core.assets.asset_registry import asset_file
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build" / "convert_json_to_temporal_graph.py"),
        str(temporal_graph_input_dir_for_item(item)),
        "--output",
        str(temporal_graph_path_for_item(item)),
        "--config",
        config,
        '--embedding-cache-db',
        str(asset_file('navqa', 'cache', 'embeddings')),
    ]
    if item.get("frame_time_index_json"):
        command.extend(["--frame-time-index", str(repo_path(item["frame_time_index_json"]))])
    if item.get("video_id"):
        command.extend(["--video-id", str(item["video_id"])])
    return command


def command_for_item(
    item: dict[str, Any],
    *,
    profile: str,
    stable_identity_merge_mode: str | None = None,
    stable_identity_include_fragments: bool = False,
    stable_identity_include_noise: bool = False,
    compact_scene_label_entities: bool | None = None,
    stable_core_compact_scene_label_entities: bool | None = None,
) -> list[str]:
    temporal_graph_json = temporal_graph_path_for_item(item)
    canonical_graph_json = repo_path(item["canonical_graph_json"])
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build" / "convert_temporal_graph_to_canonical_graph.py"),
        str(temporal_graph_json),
        "--output",
        str(canonical_graph_json),
        "--graph-profile",
        profile,
    ]
    if stable_identity_merge_mode is not None:
        command.extend(
            [
                "--stable-identity-merge-mode",
                stable_identity_merge_mode,
            ]
        )
    if stable_identity_include_fragments:
        command.append("--stable-identity-include-fragments")
    if stable_identity_include_noise:
        command.append("--stable-identity-include-noise")
    if compact_scene_label_entities is True:
        command.append("--compact-scene-label-entities")
    elif compact_scene_label_entities is False:
        command.append("--no-compact-scene-label-entities")
    if profile == "time_fact_core":
        command.extend(
            [
                "--report-json",
                str(canonical_graph_json.with_name("time_fact_core_report.json")),
            ]
        )
        if stable_core_compact_scene_label_entities is False:
            command.append("--no-stable-core-scene-label-compaction")
    return command


def main() -> None:
    args = build_parser().parse_args()
    payload = load_manifest(args.manifest)
    profile = graph_profile(payload)
    temporal_config = temporal_graph_config(payload)
    merge_mode = stable_identity_merge_mode(payload)
    include_fragments = optional_bool(payload, "stable_identity_include_fragments") or False
    include_noise = optional_bool(payload, "stable_identity_include_noise") or False
    compact_scene_labels = optional_bool(payload, "compact_scene_label_entities")
    stable_core_compact_scene_labels = optional_bool(
        payload,
        "stable_core_compact_scene_label_entities",
    )
    items = sequence_items(payload)
    if not items:
        raise SystemExit("manifest has no sequence entries")
    for item in items:
        if args.rebuild_temporal:
            temporal_command = temporal_command_for_item(item, config=temporal_config)
            print(" ".join(temporal_command), flush=True)
            if not args.dry_run:
                subprocess.run(temporal_command, cwd=REPO_ROOT, check=True)
        command = command_for_item(
            item,
            profile=profile,
            stable_identity_merge_mode=merge_mode,
            stable_identity_include_fragments=include_fragments,
            stable_identity_include_noise=include_noise,
            compact_scene_label_entities=compact_scene_labels,
            stable_core_compact_scene_label_entities=stable_core_compact_scene_labels,
        )
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
