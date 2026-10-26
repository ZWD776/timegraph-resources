from __future__ import annotations

from vlkg.workspace import workspace_root
from vlkg.core.assets.source_paths import resolve_data_path

from pathlib import Path


REPO_ROOT = workspace_root()
DATA_ROOT = REPO_ROOT / "data"
EXAMPLES_ROOT = DATA_ROOT / "examples"
CACHE_ROOT = DATA_ROOT / "cache"
WORKDIR_ROOT = DATA_ROOT / "workdir"

BUILTIN_DATASETS: tuple[str, ...] = ()


def ensure_data_layout() -> None:
    EXAMPLES_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    WORKDIR_ROOT.mkdir(parents=True, exist_ok=True)


def default_cache_db_path() -> Path:
    ensure_data_layout()
    return CACHE_ROOT / "embedding_cache.db"


def default_workdir_root() -> Path:
    ensure_data_layout()
    return WORKDIR_ROOT


def builtin_dataset_dir(dataset_name: str) -> Path:
    ensure_data_layout()
    preferred = EXAMPLES_ROOT / dataset_name
    fallback = REPO_ROOT / dataset_name
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    return preferred


def resolve_repo_input_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return resolve_data_path(candidate, REPO_ROOT)

    parts = candidate.parts
    if parts and parts[0] in BUILTIN_DATASETS:
        redirected = EXAMPLES_ROOT.joinpath(*parts)
        if redirected.exists():
            return redirected

    direct = REPO_ROOT / candidate
    if direct.exists():
        return direct

    if parts and parts[0] in BUILTIN_DATASETS:
        redirected = EXAMPLES_ROOT.joinpath(*parts)
        return redirected

    return resolve_data_path(direct, REPO_ROOT)
