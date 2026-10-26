"""Organize each dataset independently, preserving all input and history bytes.

Only the selected source/cache trees inside this workspace are operated on.
No backup directory is inspected. Directory moves use same-volume rename.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
REPORT = ROOT / 'registry/organization/data_sources_20261009'
SOURCE = {
    'navqa': ROOT / 'datasets/navqa/source/navqa-coda-every10-v1/payload',
    'nextqa': ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload',
    'openeqa': ROOT / 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload',
}


def owned(path: Path) -> Path:
    value = path.resolve()
    if not value.is_relative_to(ROOT) or value == ROOT:
        raise ValueError(f'operation escapes workspace: {path}')
    return value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with owned(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(path: Path):
    return json.loads(owned(path).read_text(encoding='utf-8-sig'))


def write(path: Path, data):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.organization_tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


class Organization:
    def __init__(self):
        self.journal_path = REPORT / 'move_journal.json'
        self.journal = read(self.journal_path) if self.journal_path.exists() else {'status': 'running', 'entries': [], 'metadata_originals': {}}
        if self.journal['status'] == 'completed':
            raise RuntimeError('already organized; use --verify')
        self.aliases = list(self.journal['entries'])
        self.sidecars = []

    def save(self):
        write(self.journal_path, self.journal)

    def move(self, source: Path, target: Path, dataset: str):
        source, target = owned(source), owned(target)
        a, b = source.relative_to(ROOT).as_posix(), target.relative_to(ROOT).as_posix()
        prior = next((r for r in self.journal['entries'] if r['source'] == a and r['target'] == b), None)
        if prior and not source.exists() and target.exists():
            return
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(target)
        if source.lstat().st_file_attributes & 0x400:
            raise ValueError(f'reparse source is not permitted: {source}')
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        row = {'dataset': dataset, 'source': a, 'target': b, 'status': 'verified'}
        self.journal['entries'].append(row)
        self.aliases.append(row)
        self.save()

    def preserve_metadata(self, path: Path):
        path = owned(path)
        key = path.relative_to(ROOT).as_posix()
        if key in self.journal['metadata_originals']:
            return
        target = REPORT / 'controls_before' / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        self.journal['metadata_originals'][key] = {'snapshot': target.relative_to(ROOT).as_posix(), 'sha256': digest(path)}
        self.save()


def inventory():
    path = REPORT / 'source_inventory_before.json'
    if path.exists():
        return
    rows = []
    for dataset, base in SOURCE.items():
        for directory, dirs, files in os.walk(base, followlinks=False):
            for name in files:
                p = owned(Path(directory) / name)
                s = p.stat()
                if s.st_file_attributes & 0x400:
                    raise ValueError(f'reparse input: {p}')
                rows.append({'dataset': dataset, 'path': p.relative_to(ROOT).as_posix(), 'bytes': s.st_size, 'file_id': s.st_ino, 'mtime_ns': s.st_mtime_ns})
    write(path, {'files': rows, 'operation': 'project sources only; no backup access'})


def organize_next(op: Organization):
    base = SOURCE['nextqa']
    manifest_path, special_path = base / 'source_manifest.json', base / 'nextqa_special_hsgg_29_manifest_v1.json'
    manifest, special = read(manifest_path), read(special_path)
    if len(manifest['videos']) != 570 or len({x['official_video_id'] for x in manifest['videos']}) != 570:
        raise ValueError('expected 570 unique selected videos')
    op.preserve_metadata(manifest_path)
    op.preserve_metadata(special_path)
    counts = Counter()
    for row in manifest['videos']:
        vid = str(row['official_video_id'])
        old = ROOT / row['normalized_input_dir']
        new = base / 'items' / vid
        objects, trace = read(old / 'objects.json'), read(old / 'source_trace.json')
        for name, record in objects.items():
            for value in (record.get('image_path'), next((x.get('materialized_image_path') for x in trace['sampled_frames'] if x['frame_name'] == name), None)):
                if value and Path(value).is_absolute() and Path(value).resolve().is_relative_to(ROOT):
                    source_dir = Path(value).parent.parent
                    if source_dir != old:
                        op.aliases.append({'dataset': 'nextqa', 'source': source_dir.relative_to(ROOT).as_posix(), 'target': new.relative_to(ROOT).as_posix()})
        for name in ('objects.json', 'source_trace.json'):
            op.move(old / name, base / 'provenance/history/item_metadata' / vid / name, 'nextqa')
        op.move(old, new, 'nextqa')
        for name, record in objects.items():
            record['image_path'] = f'frames/{name}.png'
        for frame in trace['sampled_frames']:
            frame['materialized_image_path'] = f"frames/{frame['frame_name']}.png"
        write(new / 'objects.json', objects)
        write(new / 'source_trace.json', trace)
        origin = {field: row.pop(field) for field in ('archive_half', 'hsgg_archive_root', 'frame_archive_root', 'frame_member_template', 'archive_provenance') if field in row}
        row['origin'] = origin
        row.update(normalized_input_dir=new.relative_to(ROOT).as_posix(), frames_dir=(new / 'frames').relative_to(ROOT).as_posix(), image_path_base='input_directory')
        generation = row['hsgg_generation_pass']
        kind = 'repair_oddheight' if 'oddheight' in generation else 'repair_lowfps' if 'lowfps' in generation else 'primary_' + origin.get('archive_half', 'unknown')
        counts[kind] += 1
        row['selected_generation_kind'] = kind
    by_id = {str(r['official_video_id']): r for r in manifest['videos']}
    for row in special['videos']:
        selected = by_id[str(row['official_video_id'])]
        for field in ('normalized_input_dir', 'frames_dir', 'image_path_base', 'selected_generation_kind'):
            row[field] = selected[field]
        for field in ('hsgg_archive', 'frame_archive', 'archive_half', 'hsgg_archive_root', 'frame_archive_root', 'frame_member_template', 'archive_provenance'):
            if field in row:
                row.setdefault('origin', {})[field] = row.pop(field)
    op.move(special_path, base / 'manifests/repaired29_manifest.json', 'nextqa')
    write(base / 'manifests/repaired29_manifest.json', special)
    manifest['source_layout'] = {'version': 'nextqa_items_v2', 'item_key': 'official_video_id', 'single_selected_version_per_video': True, 'generation_counts': dict(counts)}
    write(manifest_path, manifest)
    old_eval = base / 'nextqa_val_standardized_full4996_20260812a'
    op.move(old_eval, base / 'evaluation', 'nextqa')
    for item in list((base / 'evaluation').iterdir()):
        if item.name not in ('questions.runtime.json', 'labels.evaluator_only.json'):
            op.move(item, base / 'provenance/history/evaluation_run_records' / item.name, 'nextqa')
    for name in ('next-qa', 'runtime', 'nextqa_val_manifest_v1.json', 'special_splice'):
        p = base / name
        if p.exists():
            op.move(p, base / 'provenance/history' / name, 'nextqa')
    op.move(base / 'provenance/archive_retirement_20261008', base / 'provenance/history/archive_retirement_20261008', 'nextqa')
    return dict(counts)


def organize_nav(op: Organization):
    base = SOURCE['navqa']
    p = base / 'source_manifest.json'
    manifest = read(p)
    op.preserve_metadata(p)
    report = []
    for row in manifest['sequences']:
        seq = str(row['seq_id'])
        sample = ROOT / row['sample_root']
        old = ROOT / row['hsgg_input_dir']
        new = sample / 'hsgg'
        for name in ('hsgg_local_paths', 'hsgg_upstream'):
            op.move(sample / name, base / f'provenance/history/seq{seq}' / name, 'navqa')
        op.move(old, new, 'navqa')
        cache = ROOT / f'datasets/navqa/cache/navqa-cache-external-v1/payload/processed_attributes/seq{seq}'
        for item in list(new.iterdir()):
            if item.name.startswith('processed_attributes'):
                op.move(item, cache / item.name, 'navqa')
            elif item.name == 'temporal_graph.json':
                op.move(item, base / f'provenance/history/seq{seq}/runtime_derived' / item.name, 'navqa')
        op.sidecars.append({'input_dir': new.relative_to(ROOT).as_posix(), 'processed_attributes_dir': cache.relative_to(ROOT).as_posix()})
        original_counts = {}
        for name in ('objects', 'attributes', 'relations', 'hierarchy_mapping', 'scene_graphs'):
            old_file = new / f'{name}.json'
            if old_file.exists():
                original_counts[name] = len(read(old_file))
                op.move(old_file, base / f'provenance/history/seq{seq}/runtime_aggregate' / old_file.name, 'navqa')
        tables = {k: {} for k in ('objects', 'attributes', 'relations', 'hierarchy_mapping')}
        frame_files = sorted((new / 'frame_scene_graphs').glob('*.json'))
        for frame in frame_files:
            raw = read(frame)
            for output, input_key in (('objects','objects'), ('attributes','attributes'), ('relations','relations'), ('hierarchy_mapping','hierarchy')):
                tables[output][frame.stem] = raw[input_key]
        for name, data in tables.items():
            write(new / f'{name}.json', data)
        row.update(hsgg_input_dir=new.relative_to(ROOT).as_posix(), images_dir=(sample / 'images').relative_to(ROOT).as_posix(), authoritative_record_source='frame_scene_graphs', completed_frame_count=len(frame_files))
        report.append({'seq_id': seq, 'selected_frames': len(frame_files), 'old_aggregate_counts': original_counts, 'new_aggregate_count': len(frame_files)})
        for item in list(sample.iterdir()):
            if item.is_file() and item.suffix == '.log':
                op.move(item, base / f'provenance/history/seq{seq}/logs' / item.name, 'navqa')
    for name in ('legacy_outputs', 'diagnostics'):
        p0 = base / 'navqa/seq0' / name
        if p0.exists():
            op.move(p0, base / 'provenance/history/seq0' / name, 'navqa')
    manifest['source_layout'] = {'version': 'navqa_selected_hsgg_v2', 'authoritative_record_source': 'frame_scene_graphs', 'legacy_aggregates_preserved': True}
    write(p, manifest)
    return report


def organize_open(op: Organization):
    base = SOURCE['openeqa']
    p = base / 'source_manifest.json'
    manifest = read(p)
    op.preserve_metadata(p)
    for shard, row in manifest['shards'].items():
        old_hsgg, old_frames = ROOT / row['hsgg'], ROOT / row['frames']
        new_hsgg, new_frames = base / shard / 'hsgg', base / shard / 'frames'
        op.move(old_hsgg, new_hsgg, 'openeqa')
        op.move(old_frames, new_frames, 'openeqa')
        for item in list(new_hsgg.iterdir()):
            if item.is_file():
                op.move(item, base / shard / 'provenance/generation' / item.name, 'openeqa')
        for episode in sorted(new_hsgg.glob('*/*')):
            if not episode.is_dir() or not (episode / 'frame_scene_graphs').is_dir():
                continue
            for item in list(episode.iterdir()):
                if item.is_file() and item.name != 'run_manifest.json':
                    target = base / shard / 'provenance/legacy_episode_files' / episode.relative_to(new_hsgg) / item.name
                    op.move(item, target, 'openeqa')
        row.update(hsgg=new_hsgg.relative_to(ROOT).as_posix(), frames=new_frames.relative_to(ROOT).as_posix(), authoritative_record_source='frame_scene_graphs', provenance=(base / shard / 'provenance').relative_to(ROOT).as_posix())
    manifest['source_layout'] = {'version': 'openeqa_independent_shards_v2', 'authoritative_record_source': 'frame_scene_graphs', 'timestamp_policy': 'preserve_ordinal_or_explicit_index_no_fps_inference'}
    write(p, manifest)


def empty_dirs():
    for base in SOURCE.values():
        for directory, dirs, files in os.walk(base, topdown=False):
            p = owned(Path(directory))
            if p != base and not any(p.iterdir()):
                p.rmdir()


def controls(op: Organization):
    paths = [ROOT / 'configs/dataset_asset_registry.yaml', ROOT / 'registry/current_registry_lock.json']
    paths += [ROOT / f'registry/datasets/{d}.yaml' for d in SOURCE]
    paths += [base.parent / 'asset_pointer.json' for base in SOURCE.values()]
    for p in paths:
        op.preserve_metadata(p)
    replacements = {
        'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/normalized_inputs': 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/items',
        'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/nextqa_val_standardized_full4996_20260812a': 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/evaluation',
        'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/nextqa_special_hsgg_29_manifest_v1.json': 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/manifests/repaired29_manifest.json',
        'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/special_splice/original_records': 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/provenance/history/special_splice/original_records',
    }
    def replace(value):
        if isinstance(value,dict):
            return {k: replace(v) if k not in ('previous_paths',) else v for k,v in value.items()}
        if isinstance(value,list):
            return [replace(x) for x in value]
        if isinstance(value,str):
            for a,b in sorted(replacements.items(),key=lambda x:len(x[0]),reverse=True):
                if value == a or value.startswith(a + '/'):
                    return b + value[len(a):]
        return value
    for p in paths:
        if p.name == 'current_registry_lock.json':
            continue
        if p.suffix == '.json':
            write(p, replace(read(p)))
        else:
            text = p.read_text(encoding='utf-8')
            for a,b in replacements.items():
                text = text.replace(a,b)
            p.write_text(text,encoding='utf-8')
    lock_path = ROOT / 'registry/current_registry_lock.json'
    lock = read(lock_path)
    for row in lock['files']:
        p = ROOT / row['path']
        row.update(bytes=p.stat().st_size,sha256=digest(p))
    write(lock_path,lock)


def apply():
    inventory()
    op = Organization()
    next_counts = organize_next(op)
    nav_changes = organize_nav(op)
    organize_open(op)
    controls(op)
    aliases = {}
    for row in op.aliases:
        key = row['source']
        if key in aliases and aliases[key]['target'] != row['target']:
            raise ValueError(f'conflicting alias: {key}')
        aliases[key] = {k: row[k] for k in ('dataset','source','target')}
    write(ROOT / 'registry/source_path_aliases.json', {'version':'independent_source_relocations_v1','aliases':list(aliases.values()),'derived_sidecars':op.sidecars})
    empty_dirs()
    op.journal.update(status='completed', nextqa_selection_counts=next_counts, navqa_aggregate_changes=nav_changes, backup_accessed=False)
    op.save()
    print(json.dumps({'status':'completed','moves':len(op.journal['entries']),'nextqa_selection':next_counts,'navqa':nav_changes,'backup_accessed':False}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.apply and args.verify:
        parser.error('choose either --apply or --verify')
    if args.verify:
        from scripts.datasets.verify_organized_sources import main as verify
        raise SystemExit(verify())
    elif args.apply:
        apply()
    else:
        print(json.dumps({'operation':'dry-run','independent_datasets':list(SOURCE),'source_roots':{k:str(v) for k,v in SOURCE.items()},'backup_accessed':False}))


if __name__ == '__main__':
    main()
