"""NExT-QA-owned filesystem and atomic serialization helpers."""
from __future__ import annotations
from vlkg.workspace import workspace_root
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, BinaryIO
REPO_ROOT = workspace_root()

def select_run_root(value: str | None, *, repo_root: Path=REPO_ROOT) -> Path:
    """New runs use the dataset-owned tree; explicit roots must stay in it."""
    if value is None:
        return (repo_root / 'datasets/nextqa/runs/new_runs/payload').resolve()
    selected = (repo_root / value).resolve()
    allowed = (repo_root / 'datasets/nextqa/runs').resolve()
    if not selected.is_relative_to(allowed) or 'payload' not in selected.relative_to(allowed).parts:
        raise ValueError('parallel NExT-QA output must stay inside datasets/nextqa/runs/.../payload')
    return selected
NEXTQA_SCRATCH_ROOT = select_run_root(os.environ.get('VLKG_NEXTQA_RUN_ROOT'))

def require_within_nextqa_scratch(path: str | Path, *, scratch_root: str | Path=NEXTQA_SCRATCH_ROOT) -> Path:
    """Resolve a path and reject writes outside the NExT-QA scratch tree."""
    resolved = Path(path).resolve()
    root = Path(scratch_root).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f'NExT-QA output must stay below {root}: {resolved}') from exc
    return resolved

def atomic_write_text(path: str | Path, value: str, *, scratch_root: str | Path=NEXTQA_SCRATCH_ROOT) -> Path:
    """Atomically replace a UTF-8 text file inside the NExT-QA scratch tree."""
    target = require_within_nextqa_scratch(path, scratch_root=scratch_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp', delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target

def atomic_write_json(path: str | Path, value: Any, *, scratch_root: str | Path=NEXTQA_SCRATCH_ROOT) -> Path:
    """Atomically write deterministic, human-readable JSON."""
    return atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', scratch_root=scratch_root)

def atomic_copy_binary_stream(path: str | Path, source: BinaryIO, *, scratch_root: str | Path=NEXTQA_SCRATCH_ROOT, chunk_size: int=1024 * 1024) -> Path:
    """Atomically copy a binary stream into the isolated NExT-QA tree."""
    target = require_within_nextqa_scratch(path, scratch_root=scratch_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=target.parent, prefix=f'.{target.name}.', suffix='.tmp', delete=False) as handle:
            temporary_path = Path(handle.name)
            shutil.copyfileobj(source, handle, length=chunk_size)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return target

def write_immutable_json(path: str | Path, value: Any, *, resume: bool, scratch_root: str | Path=NEXTQA_SCRATCH_ROOT) -> Path:
    """Create immutable run metadata or verify it during an explicit resume."""
    target = require_within_nextqa_scratch(path, scratch_root=scratch_root)
    if target.exists():
        if not resume:
            raise FileExistsError(f'refusing to overwrite existing run file: {target}')
        current = json.loads(target.read_text(encoding='utf-8'))
        if current != value:
            raise ValueError(f'resume contract mismatch for immutable file: {target}')
        return target
    if resume:
        raise FileNotFoundError(f'resume requested but immutable file is missing: {target}')
    return atomic_write_json(target, value, scratch_root=scratch_root)
