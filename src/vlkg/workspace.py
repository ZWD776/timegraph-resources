"""Find the workspace independently of a module's namespace depth."""
from pathlib import Path


def workspace_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / 'pyproject.toml').is_file() and (candidate / 'registry').is_dir():
            return candidate
    raise FileNotFoundError('workspace marker pyproject.toml + registry not found')
