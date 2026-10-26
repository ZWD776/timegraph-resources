"""Describe sealed deletion candidates and retained information; never delete files."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re

from scripts.datasets.migrate_asset_storage import ROOT, safe_path
from vlkg.core.assets.asset_provenance import historical_control_file

RECEIPTS = (
    'historical_dataset_roots_move_receipt.json',
    'nextqa_superseded_views_move_receipt.json',
    'navqa_source_extras_move_receipt.json',
    'openeqa_unselected_run_children_move_receipt.json',
    'navqa_original_run_information_move_receipt.json',
    'navqa_original_index_metadata_move_receipt.json',
)


def classify(relative):
    name = Path(relative).name.lower()
    path = relative.lower()
    if name.endswith(('.pyc', '.pyo', '.pid', '.tmp', '.cache', '.sqlite-wal', '.sqlite-shm')):
        return 'generated_cache_or_fixture_candidate'
    if 'projection_cache' in name:
        return 'generated_cache_or_fixture_candidate'
    if name.endswith(('.faiss', '.sqlite', '.db', '.npy', '.npz', '.pkl', '.pt')):
        return 'binary_asset_candidate'
    if name.endswith(('.tar.gz', '.tgz', '.zip')):
        return 'container_requires_content_review'
    if name.endswith(('.jpg', '.jpeg', '.webp', '.bmp')) or (
        name.endswith('.png') and any(term in path for term in ('/images/', '/frames/',
            'normalized_inputs/', 'frame_', 'scannet_frames', 'frame_scene_graphs/'))
    ):
        return 'image_asset_candidate'
    if name in ('objects.json', 'attributes.json', 'relations.json', 'hierarchy_mapping.json',
                'frame_time_index.json') or '/frame_scene_graphs/' in path:
        return 'old_structured_input_candidate'
    if name.endswith(('.json', '.json.gz')) and (
        'temporal_graph' in name or re.search(r'(^|_)graph($|_)', name.removesuffix('.json'))
    ) and not any(word in name for word in ('manifest', 'summary', 'report', 'audit', 'config',
                                           'prediction', 'answer', 'question', 'eval', 'qa_', 'results')):
        return 'graph_asset_candidate'
    if name.endswith(('.json', '.json.gz', '.jsonl', '.md', '.txt', '.log', '.py', '.yaml', '.yml',
                      '.xml', '.csv', '.sha256', '.png', '.svg', '.pdf')):
        return 'retain_sealed_information'
    if name.endswith(('.mp4', '.avi', '.mov')):
        return 'retain_original_video_reference'
    return 'requires_manual_file_review'


def main():
    output = ROOT / 'archive/legacy_20260928/migration/deletion_catalog'
    output.mkdir(parents=True, exist_ok=True)
    container_review = output / 'container_review.json'
    decisions = {}
    if container_review.exists():
        review = json.loads(container_review.read_text(encoding='utf-8'))
        if review.get('status') == 'container_decisions_verified':
            decisions = {row['path']: row for row in review['containers']}
    by_dataset = {name: [] for name in ('navqa', 'nextqa', 'openeqa')}
    seen = set()
    for receipt_name in RECEIPTS:
        receipt_path = historical_control_file('registry/migrations/' + receipt_name, ROOT)
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if receipt['status'] != 'verified':
            raise ValueError('incomplete receipt: ' + receipt_name)
        for entry in receipt['entries']:
            dataset = entry.get('dataset') or next((name for name in by_dataset
                                                   if '/' + name + '/' in entry['target']), None)
            if dataset not in by_dataset:
                raise ValueError('dataset ownership unresolved')
            if not entry['target'].startswith('archive/legacy_20260928/'):
                raise ValueError('catalog only describes sealed assets')
            blob = historical_control_file(entry['file_ledger'], ROOT).read_bytes()
            if hashlib.sha256(blob).hexdigest() != entry['file_ledger_sha256']:
                raise ValueError('ledger checksum changed')
            for row in json.loads(gzip.decompress(blob)):
                path = (Path(entry['target']) / row['relative']).as_posix() if row['relative'] else entry['target']
                if path in seen:
                    raise ValueError('overlapping candidate inventories')
                seen.add(path)
                classification = classify(row['relative'] or entry['target'])
                container = decisions.get(path)
                if container:
                    classification = {
                        'duplicate_container_candidate': 'verified_duplicate_container_candidate',
                        'generated_test_fixture_candidate': 'generated_cache_or_fixture_candidate',
                        'retain_original_official_annotation_reference': 'retain_original_official_annotation_reference',
                    }[container['decision']]
                by_dataset[dataset].append({'path': path, 'bytes': row['bytes'],
                    'sha256': row['sha256'], 'classification': classification,
                    'move_receipt': receipt_path.relative_to(ROOT).as_posix(),
                    'delete_now': False})
    summary = {'status': 'classified_inventory_pending_final_dependency_acceptance',
               'scope': 'All files in the six completed sealed dataset receipts; remaining active originals and other archives are not silently assumed classified.',
               'actual_deletion_performed': False, 'datasets': {}}
    for dataset, rows in by_dataset.items():
        ledger = gzip.compress(json.dumps(rows, separators=(',', ':')).encode(), mtime=0)
        path = output / (dataset + '.files.json.gz')
        path.write_bytes(ledger)
        counts, sizes = Counter(), Counter()
        for row in rows:
            counts[row['classification']] += 1
            sizes[row['classification']] += row['bytes']
        summary['datasets'][dataset] = {'files': len(rows), 'bytes': sum(sizes.values()),
            'classifications': {name: {'files': counts[name], 'bytes': sizes[name]} for name in counts},
            'ledger': path.relative_to(ROOT).as_posix(), 'ledger_sha256': hashlib.sha256(ledger).hexdigest(),
            'approval_ready': False}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: {'files': data['files'], 'gib': round(data['bytes'] / 1024**3, 3)}
                      for name, data in summary['datasets'].items()}))


if __name__ == '__main__':
    main()
