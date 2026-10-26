from __future__ import annotations

import json
import sys
from importlib import import_module
from pathlib import Path

import pytest


SCRIPT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()) / "scripts" / "openeqa"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

shared = import_module("scripts.datasets.openeqa._shared")


@pytest.mark.parametrize("streaming", [False, True])
def test_atomic_json_writes_retry_transient_windows_file_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    output = tmp_path / "output.json"
    original_replace = Path.replace
    calls = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("transient Windows file lock")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(shared.time, "sleep", lambda _seconds: None)

    if streaming:
        shared.write_json_array_atomic(output, iter([{"value": 1}]))
    else:
        shared.write_json_atomic(output, {"value": 1})

    assert calls == 3
    expected = [{"value": 1}] if streaming else {"value": 1}
    assert json.loads(output.read_text(encoding="utf-8")) == expected
