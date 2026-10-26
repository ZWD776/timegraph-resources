from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from vlkg.workspace import workspace_root
DEFAULT_NAVQA_LAYOUT_ROOT = Path('datasets/navqa/source/navqa-coda-every10-v1/payload/navqa')
CODA_SEQUENCE_URL_TEMPLATE = 'https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/CODa_v1/sequences/{seq_id}.zip'

@dataclass(frozen=True)
class NavQASequenceLayout:
    seq_id: str
    cam: str
    stride: int | None
    layout_root: Path
    sequence_root: Path
    sample_root: Path
    metadata_dir: Path
    timestamps_path: Path
    poses_path: Path
    qa_unfilled_json: Path
    images_dir: Path
    hsgg_local_paths_dir: Path
    frame_time_index_json: Path
    frame_time_index_csv: Path
    temporal_graph_json: Path
    canonical_graph_json: Path
    projection_cache: Path
    qa_results_json: Path
    qa_results_csv: Path
    qa_results_md: Path

def normalize_seq_id(seq_id: str | int) -> str:
    normalized = str(seq_id).strip()
    if not normalized:
        raise ValueError('seq_id must not be empty')
    return normalized

def normalize_cam(cam: str | int) -> str:
    normalized = str(cam).strip().lower()
    if normalized.startswith('cam'):
        normalized = normalized[3:]
    if not normalized:
        raise ValueError('cam must not be empty')
    return normalized

def normalize_stride(stride: str | int | None) -> int | None:
    if stride is None:
        return None
    normalized = str(stride).strip().lower()
    if normalized in {'', 'none', 'sampled', 'unknown'}:
        return None
    if normalized.startswith('every'):
        normalized = normalized[5:]
    value = int(normalized)
    if value <= 0:
        raise ValueError('stride must be positive')
    return value

def stride_dir_name(stride: int | None) -> str:
    if stride is None:
        return 'sampled'
    return f'every{stride}'

def coda_sequence_zip_url(seq_id: str | int) -> str:
    return CODA_SEQUENCE_URL_TEMPLATE.format(seq_id=normalize_seq_id(seq_id))

def default_navqa_sequence_layout(*, seq_id: str | int, cam: str | int='0', stride: str | int | None=10, layout_root: str | Path=DEFAULT_NAVQA_LAYOUT_ROOT) -> NavQASequenceLayout:
    seq = normalize_seq_id(seq_id)
    cam_id = normalize_cam(cam)
    stride_value = normalize_stride(stride)
    root = Path(layout_root)
    sequence_root = root / f'seq{seq}'
    sample_root = sequence_root / f'cam{cam_id}' / stride_dir_name(stride_value)
    metadata_dir = sequence_root / 'metadata'
    source_entry = _default_source_entry(seq=seq, cam=cam_id, stride=stride_value, layout_root=root)
    def source_path(field: str, fallback: Path) -> Path:
        value = source_entry.get(field)
        if not value:
            return fallback
        path = Path(value)
        return workspace_root() / path if root.is_absolute() and not path.is_absolute() else path
    return NavQASequenceLayout(seq_id=seq, cam=cam_id, stride=stride_value, layout_root=root, sequence_root=sequence_root, sample_root=sample_root, metadata_dir=metadata_dir, timestamps_path=metadata_dir / 'timestamps' / f'{seq}.txt', poses_path=metadata_dir / 'poses' / f'{seq}.txt', qa_unfilled_json=source_path('qa_unfilled_json', sequence_root / 'qa_unfilled.json'), images_dir=source_path('images_dir', sample_root / 'images'), hsgg_local_paths_dir=source_path('hsgg_input_dir', sample_root / 'hsgg_local_paths'), frame_time_index_json=source_path('frame_time_index_json', sample_root / 'frame_time_index.json'), frame_time_index_csv=sample_root / 'frame_time_index.csv', temporal_graph_json=sample_root / 'temporal_graph.json', canonical_graph_json=sample_root / 'canonical_temporal_memory_graph.json', projection_cache=sample_root / 'canonical_projection_cache.json.gz', qa_results_json=sample_root / 'qa_non_position_live_results.json', qa_results_csv=sample_root / 'qa_non_position_live_results.csv', qa_results_md=sample_root / 'qa_non_position_live_results.md')

def _default_source_entry(*, seq: str, cam: str, stride: int | None, layout_root: Path) -> dict:
    repo_root = workspace_root()
    requested_root = layout_root if layout_root.is_absolute() else repo_root / layout_root
    if requested_root.resolve() != (repo_root / DEFAULT_NAVQA_LAYOUT_ROOT).resolve():
        return {}
    manifest = repo_root / DEFAULT_NAVQA_LAYOUT_ROOT.parent / 'source_manifest.json'
    if not manifest.is_file():
        return {}
    payload = json.loads(manifest.read_text(encoding='utf-8-sig'))
    for entry in payload.get('sequences') or []:
        if normalize_seq_id(entry.get('seq_id', '')) == seq and normalize_cam(entry.get('cam', '0')) == cam and normalize_stride(entry.get('stride', 10)) == stride:
            return entry
    return {}
