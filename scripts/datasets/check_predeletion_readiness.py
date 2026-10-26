"""Check final cleanup boundaries and simulate archive unavailability; never delete."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'archive/legacy_20260928'


def native(path):
    value = str(path.absolute())
    return Path('\\\\?\\' + value) if os.name == 'nt' and not value.startswith('\\\\?\\') else path


def write_report(name, value):
    path = ROOT / 'registry/audits/predeletion' / (name + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value[key] for key in ('status', 'case', 'issues') if key in value}, ensure_ascii=False), flush=True)


def load_plan():
    plan = json.loads((ROOT / 'registry/gates/final_deletion_plan.json').read_text(encoding='utf-8'))
    if plan['deletion_execution_enabled'] is not False:
        raise ValueError('deletion must remain disabled during preparation')
    all_rows = {}
    for owner, item in plan['ledger_hashes'].items():
        blob = (ROOT / item['path']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item['sha256']:
            raise ValueError('ledger changed: ' + owner)
        rows = json.loads(gzip.decompress(blob))
        if len(rows) != item['files']:
            raise ValueError('ledger count differs: ' + owner)
        for row in rows:
            value = row['path']
            if (not value.startswith('archive/legacy_20260928/') or '\\' in value
                    or any(part in ('', '.', '..') for part in value.split('/'))):
                raise ValueError('unsafe candidate/retention path')
            if value.casefold() in all_rows:
                raise ValueError('duplicate or conflicting Windows path')
            if not row.get('sha256') or row.get('delete_now') is not False:
                raise ValueError('missing hash or unexpected execution flag')
            all_rows[value.casefold()] = {**row, 'owner': owner}
    return plan, all_rows


def boundary_check():
    started = time.monotonic()
    plan, rows = load_plan()
    issues, links, unknown, changed, missing, readonly = [], [], [], [], [], []
    seen = set()
    counts = {ds: Counter() for ds in plan['groups']}
    last_notice = started
    def walk_error(error):
        issues.append({'walk_error': str(error)})
    for directory, dirs, files in os.walk(native(ARCHIVE), onerror=walk_error, followlinks=False):
        for name in list(dirs):
            p = Path(directory, name)
            if p.lstat().st_file_attributes & 0x400:
                links.append(p.relative_to(native(ROOT)).as_posix())
                dirs.remove(name)
        for name in files:
            p = Path(directory, name)
            relative = p.relative_to(native(ROOT)).as_posix()
            info = p.lstat()
            if info.st_file_attributes & 0x400:
                links.append(relative)
                continue
            # Catalog ledgers describe everything else; self-inventory is excluded.
            if relative.startswith('archive/legacy_20260928/migration/final_deletion_catalog/'):
                continue
            key = relative.casefold()
            row = rows.get(key)
            if row is None:
                unknown.append({'path': relative, 'bytes': info.st_size})
                continue
            seen.add(key)
            if info.st_size != row['bytes']:
                changed.append({'path': relative, 'expected': row['bytes'], 'actual': info.st_size})
            if row['decision'] == 'delete_candidate':
                if not stat.S_ISREG(info.st_mode):
                    issues.append({'not_regular_candidate': relative})
                if info.st_file_attributes & 1:
                    readonly.append(relative)
                counts[row['owner']]['delete_files'] += 1
                counts[row['owner']]['delete_bytes'] += info.st_size
            else:
                counts[row['owner']]['retain_files'] += 1
                counts[row['owner']]['retain_bytes'] += info.st_size
        if time.monotonic() - last_notice > 20:
            print('boundary inventory:', len(seen), 'registered files', flush=True)
            last_notice = time.monotonic()
    missing = [row['path'] for key, row in rows.items() if key not in seen]
    link_receipt = json.loads((ROOT / 'registry/gates/retired_compatibility_links_receipt.json').read_text(encoding='utf-8'))
    expected_links = {row['target'] for row in link_receipt['links']}
    if set(links) != expected_links:
        issues.append({'reparse_inventory_differs': links})
    for owner, values in counts.items():
        if dict(values) != plan['groups'][owner]:
            issues.append({'plan_count_or_size_differs': owner, 'actual': dict(values), 'expected': plan['groups'][owner]})
    lock = json.loads((ROOT / 'registry/current_registry_lock.json').read_text(encoding='utf-8'))
    for row in lock['files']:
        if hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() != row['sha256']:
            issues.append({'current_control_changed': row['path']})
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
    from vlkg.core.assets.asset_registry import inspect_dataset, verify_metadata_copies
    from vlkg.core.assets.asset_provenance import relocated_provenance, historical_control_file
    datasets = {ds: inspect_dataset(ds) for ds in ('navqa', 'nextqa', 'openeqa')}
    for ds, result in datasets.items():
        if result['issues']:
            issues.append({'dataset': ds, 'issues': result['issues']})
    metadata_issues = verify_metadata_copies(ROOT)
    if metadata_issues:
        issues.append({'metadata_copy_issues': metadata_issues})
    migration = json.loads(historical_control_file('registry/migrations/metadata_migration_v1.json', ROOT).read_text(encoding='utf-8'))
    integrity_reads_candidates = []
    for row in migration['items']:
        for side in ('source', 'target'):
            p = relocated_provenance(ROOT / row[side], ROOT)
            key = p.relative_to(ROOT).as_posix().casefold()
            if key in rows and rows[key]['decision'] == 'delete_candidate':
                integrity_reads_candidates.append(p.relative_to(ROOT).as_posix())
    if integrity_reads_candidates:
        issues.append({'required_metadata_integrity_depends_on_candidates': integrity_reads_candidates})
    if unknown or changed or missing:
        issues.append({'unregistered': len(unknown), 'size_changed': len(changed), 'missing': len(missing)})
    hsgg = ['HSGG', 'HSGG_pri', 'HSGG-main', 'HSGG_upload.tgz']
    if any(not (ROOT / p).exists() for p in hsgg):
        issues.append({'retained_hsgg_missing': True})
    free = shutil.disk_usage(ROOT).free
    report = {'case': 'boundaries', 'status': 'passed' if not issues else 'needs_review', 'issues': issues,
              'groups': {key: dict(value) for key, value in counts.items()}, 'unknown_files': unknown,
              'changed_sizes': changed, 'missing_files': missing, 'readonly_candidates': readonly,
              'reparse_points': links, 'protected_hsgg_present': True, 'current_registry': datasets,
              'free_bytes': free, 'reserve_bytes': 20 * 1024**3, 'actual_deletion_performed': False,
              'payload_hash_policy': 'Reuse verified movement hashes now; check every full current SHA256 immediately before any later removal.',
              'elapsed_seconds': round(time.monotonic() - started, 2)}
    write_report('boundaries', report)
    return int(bool(issues))


class NoArchiveGuard:
    def __init__(self):
        self.denied, self.network = [], []
    def check(self, raw):
        if raw is None or isinstance(raw, int):
            return
        value = os.fsdecode(raw)
        if value.startswith('file:'):
            value = unquote(value[5:].split('?', 1)[0])
        if value.startswith('\\\\?\\'):
            value = value[4:]
        path = Path(value).resolve()
        if path.is_relative_to(ROOT):
            relative = path.relative_to(ROOT).as_posix().casefold()
            if relative.startswith(('archive/', '.scratch/', 'data/cache/', 'data/examples/', 'data/openeqa/', 'data/next-qa/', 'graphiti-main/', 'cognee-main/')):
                self.denied.append(relative)
                raise PermissionError('simulate deletion/unavailability: ' + relative)
    def hook(self, event, args):
        if event in ('open', 'sqlite3.connect'):
            self.check(args[0])
        elif event == 'socket.connect':
            self.network.append(event)
            raise PermissionError('network disabled for readiness audit')


def runtime_check(dataset):
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
    guard = NoArchiveGuard()
    sys.addaudithook(guard.hook)
    import faiss
    real_read = faiss.read_index
    def checked_read(path, *args, **kwargs):
        guard.check(path)
        return real_read(path, *args, **kwargs)
    faiss.read_index = checked_read
    from importlib import import_module
    from vlkg.core.assets.asset_registry import asset_file
    from PIL import Image
    import numpy as np
    error, probes, frozen = None, [], None
    try:
        frozen = import_module(f'vlkg.datasets.{dataset}.audit.replay').replay()
        if dataset == 'nextqa':
            from scripts.datasets.audit_nextqa_execution import resource_probe
            probes = resource_probe()['probes']
        elif dataset == 'navqa':
            from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import VisualFaissGrounder
            index = VisualFaissGrounder(index_dir=asset_file(dataset, 'indexes', 'visual_manifest').parent)
            try:
                for seq in ('0', '16', '22'):
                    row = index.connection.execute('SELECT faiss_id,image_path FROM object_metadata WHERE seq_id=? LIMIT 1', (seq,)).fetchone()
                    vector = np.asarray(index.index.reconstruct(row['faiss_id']), dtype=np.float32)
                    hits = index._search_by_vector(vector=vector, seq_id=seq, top_k=8, search_k=256, same_sequence_only=True, min_score=0)
                    with Image.open(row['image_path']) as image:
                        image.load()
                    if not hits:
                        raise AssertionError('selected NAVQA index returned no hits')
                    probes.append({'sequence': seq, 'hits': len(hits), 'image_read': True})
            finally:
                index.close()
        else:
            from vlkg.datasets.openeqa.runtime.visual_object_index import ObjectVisualSearchIndex
            for shard in ('hm3d', 'scannet'):
                index = ObjectVisualSearchIndex(index_dir=asset_file(dataset, 'indexes', shard).parent)
                try:
                    manifest = json.loads(asset_file(dataset, 'graphs', shard).read_text(encoding='utf-8'))
                    episodes = manifest['episodes']
                    for offset in sorted({0, len(episodes)//2, len(episodes)-1}):
                        episode = episodes[offset]['episode_history']
                        row = index.connection.execute('SELECT faiss_id,image_path FROM object_metadata WHERE episode_history=? LIMIT 1', (episode,)).fetchone()
                        vector = np.asarray(index.index.reconstruct(row['faiss_id']), dtype=np.float32)
                        hits = index.search_vector(episode_history=episode, vector=vector, top_k=8, min_score=0)
                        with Image.open(row['image_path']) as image:
                            image.load()
                        if not hits:
                            raise AssertionError('selected OpenEQA index returned no hits')
                        probes.append({'shard': shard, 'episode': episode, 'hits': len(hits), 'image_read': True})
                finally:
                    index.close()
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    report = {'case': dataset, 'status': 'passed' if error is None and not guard.denied and not guard.network else 'failed',
              'issues': [error] if error else [], 'denied_accesses': guard.denied, 'network_attempts': guard.network,
              'frozen_replay': frozen, 'probes': probes, 'archive_readable': False,
              'calls_model_or_answer_api': False, 'actual_deletion_performed': False,
              'limits': ['Executed branches only; Python open/sqlite and FAISS read_index instrumented, not an OS-wide sandbox.']}
    write_report(dataset, report)
    return int(report['status'] != 'passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('boundaries', 'navqa', 'nextqa', 'openeqa'))
    args = parser.parse_args()
    return boundary_check() if args.case == 'boundaries' else runtime_check(args.case)


if __name__ == '__main__':
    raise SystemExit(main())
