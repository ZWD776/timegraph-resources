"""Read-only resolution of relocated dataset sources and derived sidecars."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from vlkg.workspace import workspace_root

DATASETS = frozenset({'navqa', 'nextqa', 'openeqa'})


def _owner(path: Path, root: Path) -> str | None:
    parts = tuple(part.lower() for part in path.relative_to(root).parts)
    if len(parts) >= 2 and parts[0] in {'datasets', '.scratch'} and parts[1] in DATASETS:
        return parts[1]
    if len(parts) >= 2 and parts[0] == 'data':
        if parts[1] in {'navqa', 'nextqa', 'next-qa'}:
            return 'nextqa' if parts[1] in {'nextqa', 'next-qa'} else 'navqa'
        if parts[1] == 'openeqa' or parts[1].startswith('openeqa_'):
            return 'openeqa'
        if len(parts) >= 3 and parts[1:3] == ('workdir', 'openeqa'):
            return 'openeqa'
    return None


def _local_config_path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ValueError('source path aliases require workspace-relative paths')
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('source path alias escapes workspace or names its root')
    return path


@lru_cache(maxsize=16)
def _load_cached(config: Path, mtime_ns: int, size: int) -> tuple[dict[Path, Path], dict[Path, Path]]:
    """mtime and size invalidate the cache; resolving each image only stats config."""
    root = config.parent.parent.resolve()
    payload = json.loads(config.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('source path alias config must be an object')
    rows, sidecar_rows = payload.get('aliases', []), payload.get('derived_sidecars', [])
    if not isinstance(rows, list) or not isinstance(sidecar_rows, list):
        raise ValueError('source aliases and derived_sidecars must be lists')
    mapping: dict[Path, Path] = {}
    for row in rows:
        dataset = row.get('dataset') if isinstance(row, dict) else None
        if dataset not in DATASETS:
            raise ValueError('source path alias requires a supported dataset owner')
        source, target = (_local_config_path(row[field], root) for field in ('source', 'target'))
        if any(owner is not None and owner != dataset for owner in (_owner(source, root), _owner(target, root))):
            raise ValueError('source path alias crosses dataset ownership')
        if source in mapping and mapping[source] != target:
            raise ValueError('source path alias has conflicting targets')
        mapping[source] = target
    sidecars: dict[Path, Path] = {}
    for row in sidecar_rows:
        if not isinstance(row, dict):
            raise ValueError('derived_sidecars entry must be an object')
        source, target = (_local_config_path(row[field], root) for field in ('input_dir', 'processed_attributes_dir'))
        source_owner, target_owner = _owner(source, root), _owner(target, root)
        if source_owner is not None and target_owner is not None and source_owner != target_owner:
            raise ValueError('derived sidecar crosses dataset ownership')
        if source in sidecars and sidecars[source] != target:
            raise ValueError('derived sidecar has conflicting targets')
        sidecars[source] = target
    return mapping, sidecars


def _configuration(root: Path) -> tuple[dict[Path, Path], dict[Path, Path]]:
    path = root / 'registry/source_path_aliases.json'
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}, {}
    return _load_cached(path, stat.st_mtime_ns, stat.st_size)


def _candidate(path: str | Path, root: Path) -> tuple[Path, bool]:
    original = Path(path)
    selected = Path(os.path.abspath(original if original.is_absolute() else root / original))
    # Avoid stat/resolve/config reads for explicit external inputs.
    return (selected, True) if selected.is_relative_to(root) else (original, False)


def resolve_data_path(path: str | Path, root: str | Path | None=None) -> Path:
    """Resolve missing workspace paths by longest component prefix, without writes.

    Existing files/directories win. Explicit external paths are returned untouched
    and are never inspected by this resolver or considered for source aliases.
    """
    repo = Path(root).resolve() if root is not None else workspace_root()
    selected, local = _candidate(path, repo)
    if not local:
        return selected
    if selected.exists():
        return selected
    aliases, _ = _configuration(repo)
    seen_paths: set[Path] = set()
    seen_sources: set[Path] = set()
    while not selected.exists():
        if selected in seen_paths:
            raise ValueError('source path alias cycle detected')
        seen_paths.add(selected)
        # Ancestors are ordered longest first: lookup scales with path depth,
        # rather than scanning every per-video alias for each observed object.
        match = next(((source, aliases[source]) for source in (selected, *selected.parents) if source in aliases), None)
        if match is None:
            break
        source, target = match
        if source in seen_sources:
            raise ValueError('source path alias cycle or recursive expansion detected')
        seen_sources.add(source)
        selected = target / selected.relative_to(source)
        if not selected.is_relative_to(repo):
            raise ValueError('resolved source path escapes workspace')
    return selected


def resolve_processed_attributes_dir(input_dir: str | Path, root: str | Path | None=None) -> Path | None:
    """Locate configured derived attribute storage, never create a directory."""
    repo = Path(root).resolve() if root is not None else workspace_root()
    selected, local = _candidate(input_dir, repo)
    if not local:
        return None
    _, sidecars = _configuration(repo)
    target = sidecars.get(resolve_data_path(selected, repo), sidecars.get(selected))
    return resolve_data_path(target, repo) if target is not None else None


def relocate_image_references(value: Any, root: str | Path | None = None) -> Any:
    """Return a runtime metadata view without rewriting historical index/cache keys."""
    repo = Path(root).resolve() if root is not None else workspace_root()

    def image(reference: Any) -> Any:
        if not isinstance(reference, str) or not reference:
            return reference
        selected, local = _candidate(reference, repo)
        return str(resolve_data_path(selected, repo)) if local else reference

    def walk(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                key: image(child) if key in {'image_path', 'canonical_image_path'}
                else [image(x) for x in child] if key == 'image_paths' and isinstance(child, list)
                else walk(child)
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [walk(child) for child in item]
        return item

    return walk(value)


def relocate_image_references(value: Any, root: str | Path | None = None) -> Any:
    """Return a runtime metadata view without rewriting historical index/cache keys."""
    repo = Path(root).resolve() if root is not None else workspace_root()

    def image(reference: Any) -> Any:
        if not isinstance(reference, str) or not reference:
            return reference
        selected, local = _candidate(reference, repo)
        return str(resolve_data_path(selected, repo)) if local else reference

    if isinstance(value, dict):
        return {
            key: image(item) if key in {'image_path', 'canonical_image_path'}
            else [image(x) for x in item] if key == 'image_paths' and isinstance(item, list)
            else relocate_image_references(item, repo)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [relocate_image_references(item, repo) for item in value]
    return value
