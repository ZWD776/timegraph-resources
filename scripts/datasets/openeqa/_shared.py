from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

@dataclass(frozen=True)
class EpisodeSummary:
    episode_history: str
    data_source: str
    episode_id: str
    image_dir: Path
    frame_count: int
    first_frame: str | None
    last_frame: str | None
    question_count: int
    question_ids: list[str]
    category_counts: dict[str, int]

def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

def write_json_atomic(path: Path, payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    write_json(tmp_path, payload)
    _replace_with_retry(tmp_path, path)

def write_json_array_atomic(path: Path, items: Iterable[Any]) -> None:
    """Write a JSON array without materializing the whole serialized payload.

    Large OpenEQA audit records contain visual candidate/evidence details.  A
    single ``json.dumps(list)`` temporarily duplicates that data as one large
    string and can exhaust memory on full ScanNet runs.  Serializing one item
    at a time keeps the on-disk JSON contract unchanged while bounding the
    extra allocation to the current record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write('[')
        first = True
        for item in items:
            if not first:
                handle.write(',')
            handle.write('\n')
            json.dump(item, handle, ensure_ascii=False, indent=2)
            first = False
        if not first:
            handle.write('\n')
        handle.write(']')
    _replace_with_retry(tmp_path, path)

def _replace_with_retry(tmp_path: Path, path: Path) -> None:
    for attempt in range(6):
        try:
            tmp_path.replace(path)
            return
        except PermissionError:
            if attempt >= 5:
                raise
            time.sleep(0.25 * (attempt + 1))

def load_open_eqa_rows(open_eqa_json: Path) -> list[dict[str, Any]]:
    payload = load_json(open_eqa_json)
    if not isinstance(payload, list):
        raise ValueError(f'OpenEQA JSON must be a list: {open_eqa_json}')
    rows: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(item)
    return rows

def filter_rows_by_data_source(rows: list[dict[str, Any]], *, data_source: str='hm3d-v0') -> list[dict[str, Any]]:
    prefix = f'{data_source}/'
    return [row for row in rows if str(row.get('episode_history') or '').startswith(prefix)]

def group_rows_by_episode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode_history = str(row.get('episode_history') or '').strip()
        if not episode_history:
            continue
        groups[episode_history].append(row)
    return dict(groups)

def episode_image_dir(frames_root: Path, episode_history: str) -> Path:
    return repo_path(frames_root) / Path(episode_history)

def list_episode_frames(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    frames = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(frames, key=lambda path: path.name)

def build_episode_summary(*, episode_history: str, image_dir: Path, question_rows: list[dict[str, Any]]) -> EpisodeSummary:
    episode_history = str(episode_history)
    data_source, _, episode_id = episode_history.partition('/')
    frame_files = list_episode_frames(image_dir)
    question_ids = [str(row.get('question_id')) for row in question_rows if row.get('question_id') is not None]
    category_counts = Counter((str(row.get('category') or 'unknown') for row in question_rows))
    return EpisodeSummary(episode_history=episode_history, data_source=data_source, episode_id=episode_id, image_dir=image_dir, frame_count=len(frame_files), first_frame=frame_files[0].name if frame_files else None, last_frame=frame_files[-1].name if frame_files else None, question_count=len(question_rows), question_ids=question_ids, category_counts=dict(sorted(category_counts.items())))

def episode_summary_to_dict(summary: EpisodeSummary) -> dict[str, Any]:
    return {'episode_history': summary.episode_history, 'data_source': summary.data_source, 'episode_id': summary.episode_id, 'image_dir': str(summary.image_dir), 'frame_count': summary.frame_count, 'first_frame': summary.first_frame, 'last_frame': summary.last_frame, 'question_count': summary.question_count, 'question_ids': summary.question_ids, 'category_counts': summary.category_counts}
