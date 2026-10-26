from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run_relative(relative_path: str) -> None:
    root = Path(__file__).resolve().parent
    repo_root = root.parent
    target = root / Path(relative_path)
    wrapper = Path(sys.argv[0]).resolve()

    if wrapper.parent == root and wrapper.name != "_run_wrapper.py":
        try:
            wrapper_display = wrapper.relative_to(repo_root).as_posix()
        except ValueError:
            wrapper_display = wrapper.name
        try:
            target_display = target.relative_to(repo_root).as_posix()
        except ValueError:
            target_display = str(target)
        print(
            f"info: `{wrapper_display}` is a compatibility wrapper; "
            f"prefer `{target_display}`.",
            flush=True,
        )

    target_dir = str(target.parent)
    if target_dir not in sys.path:
        sys.path.insert(0, target_dir)
    runpy.run_path(str(target), run_name="__main__")
