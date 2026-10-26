"""Immutable run metadata and source-drift checks for NExT-QA jobs."""
from __future__ import annotations
from vlkg.workspace import workspace_root
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, write_immutable_json
REPO_ROOT = workspace_root()

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')

def sha256_file(path: str | Path, chunk_size: int=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()

def fingerprint_file(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {'path': str(resolved), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': sha256_file(resolved)}

def collect_imported_repo_sources(*, extra_paths: Iterable[str | Path]=(), modules: Iterable[ModuleType] | None=None) -> list[Path]:
    """Collect imported Python source files owned by this repository."""
    candidates: set[Path] = {Path(path).resolve() for path in extra_paths}
    module_values = list(modules) if modules is not None else list(sys.modules.values())
    for module in module_values:
        module_file = getattr(module, '__file__', None)
        if not module_file:
            continue
        path = Path(module_file)
        if path.suffix in {'.pyc', '.pyo'}:
            source_candidate = Path(str(path)[:-1])
            if source_candidate.exists():
                path = source_candidate
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if relative.parts and relative.parts[0].lower() in {'.venv', '.scratch'}:
            continue
        if resolved.suffix == '.py' and resolved.is_file():
            candidates.add(resolved)
    return sorted(candidates, key=lambda item: str(item).lower())

def git_status_summary() -> dict[str, Any]:
    result = subprocess.run(['git', 'status', '--short', '--untracked-files=all'], cwd=REPO_ROOT, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    text = result.stdout
    lines = [line for line in text.splitlines() if line.strip()]
    return {'returncode': result.returncode, 'entry_count': len(lines), 'status_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'stderr': result.stderr.strip() or None}

def persist_run_config(path: str | Path, *, contract: dict[str, Any], resume: bool, schema_version: str='nextqa_graph_pilot_run_config_v1') -> dict[str, Any]:
    """Create run_config.json once and enforce an exact resume contract."""
    target = Path(path).resolve()
    if resume:
        if not target.is_file():
            raise FileNotFoundError(f'resume run_config is missing: {target}')
        existing = json.loads(target.read_text(encoding='utf-8'))
        if existing.get('contract') != contract:
            raise ValueError('resume run_config contract does not match this invocation')
        return existing
    payload = {'schema_version': schema_version, 'started_at_utc': utc_now(), 'python_executable': str(Path(sys.executable).resolve()), 'git_status_summary': git_status_summary(), 'contract': contract}
    write_immutable_json(target, payload, resume=False)
    return payload

def persist_source_snapshot(path: str | Path, *, source_paths: Iterable[str | Path], resume: bool) -> dict[str, Any]:
    """Create the start snapshot or validate all hashes before a resume."""
    target = Path(path).resolve()
    if resume:
        if not target.is_file():
            raise FileNotFoundError(f'resume source_snapshot is missing: {target}')
        payload = json.loads(target.read_text(encoding='utf-8'))
        drift = compare_source_snapshot(payload, current_source_paths=source_paths)
        if drift['code_drift_detected']:
            raise RuntimeError(f'cannot resume after source drift: {drift}')
        return payload
    fingerprints = [fingerprint_file(path) for path in source_paths]
    payload = {'schema_version': 'nextqa_source_snapshot_v1', 'captured_at_utc': utc_now(), 'source_files': fingerprints}
    write_immutable_json(target, payload, resume=False)
    return payload

def compare_source_snapshot(snapshot: dict[str, Any], *, current_source_paths: Iterable[str | Path] | None=None) -> dict[str, Any]:
    """Compare current source hashes with an immutable start snapshot."""
    expected = {str(Path(item['path']).resolve()): str(item['sha256']) for item in snapshot.get('source_files', [])}
    paths = [Path(path).resolve() for path in current_source_paths] if current_source_paths is not None else [Path(path) for path in expected]
    current_set = {str(path) for path in paths}
    changed: list[dict[str, Any]] = []
    missing: list[str] = []
    for path_text, expected_hash in expected.items():
        path = Path(path_text)
        if not path.is_file():
            missing.append(path_text)
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            changed.append({'path': path_text, 'start_sha256': expected_hash, 'end_sha256': actual_hash})
    added = sorted(current_set - set(expected))
    return {'checked_at_utc': utc_now(), 'code_drift_detected': bool(changed or missing or added), 'changed_files': changed, 'missing_files': sorted(missing), 'newly_imported_repo_sources': added}

def persist_drift_report(path: str | Path, *, snapshot: dict[str, Any], current_source_paths: Iterable[str | Path]) -> dict[str, Any]:
    report = compare_source_snapshot(snapshot, current_source_paths=current_source_paths)
    atomic_write_json(path, report)
    return report
