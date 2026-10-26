"""Resolve new-run registry defaults without silently changing resume contracts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def recorded_or_selected(explicit: Path | None, *, resume: bool, run_config: Path,
                         field: str, selected: Callable[[], Path], parent: bool = False) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if not resume:
        return selected().resolve()
    contract = json.loads(run_config.read_text(encoding='utf-8'))['contract']
    recorded = contract[field]
    value = recorded['path'] if isinstance(recorded, dict) else recorded
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'resume contract has no path for {field}')
    path = Path(value).resolve()
    return path.parent if parent else path
