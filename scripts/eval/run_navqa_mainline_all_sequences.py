"""Compatibility CLI forwarding to the canonical dataset entrypoint."""
from pathlib import Path
import runpy

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
if __name__ == "__main__":
    runpy.run_path(str(ROOT / 'scripts/datasets/navqa/qa/run_navqa_mainline_all_sequences.py'), run_name="__main__")
