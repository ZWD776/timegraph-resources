"""Shared command construction; dataset decisions remain in their adapters."""
from pathlib import Path
import sys

from vlkg.workspace import workspace_root


def command(dataset, stage, filename, *arguments):
    root = workspace_root()
    script = root / 'scripts/datasets' / dataset / stage / filename
    if not script.is_file():
        raise FileNotFoundError(script)
    return [sys.executable, str(script), *map(str, arguments)]


def output_root(dataset, run_id):
    if not run_id or any(part in run_id for part in ('/', '\\', ':', '..')):
        raise ValueError('run identifier must be a single safe path component')
    return workspace_root() / 'datasets' / dataset / 'runs/new_runs/payload' / run_id
