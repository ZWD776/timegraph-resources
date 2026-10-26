"""Derive path-relocated JSON views, retaining original copies and vector files."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from vlkg.core.assets.asset_paths import validate_mapping  # noqa: E402


class Relocator:
    def __init__(self, mapping: dict[str, str], root: Path):
        validate_mapping(mapping)
        self.root = root.resolve()
        self.prefix = self.root.as_posix()
        self.mapping = sorted(mapping.items(), key=lambda p: len(p[0]), reverse=True)
        self.cache = {}
        self.changed_fields = Counter()
        self.old_paths = Counter()
        self.references = set()

    def path(self, value: str) -> str:
        if value in self.cache:
            return self.cache[value]
        normalized = value.replace('\\', '/')
        absolute = normalized.casefold().startswith(self.prefix.casefold() + '/')
        relative = normalized[len(self.prefix) + 1:] if absolute else normalized
        result = value
        for source, target in self.mapping:
            if relative.casefold() == source.casefold() or relative.casefold().startswith(source.casefold() + '/'):
                result = self.prefix + '/' + target + relative[len(source):]
                self.references.add(result)
                break
        if result == value and (relative.startswith('.scratch/nextqa/') or relative.startswith('data/next-qa/')):
            self.old_paths[relative] += 1
        # Cache filesystem paths only, never prose or scalar labels.
        if absolute or relative.startswith(('.scratch/', 'data/', 'datasets/')):
            self.cache[value] = result
        return result

    def transform(self, value, key=''):
        if isinstance(value, dict):
            for field in value:
                value[field] = self.transform(value[field], field)
        elif isinstance(value, list):
            for index in range(len(value)):
                value[index] = self.transform(value[index], key)
        elif isinstance(value, str):
            # Avoid normalizing arbitrary text or replacing substrings in sentences.
            normalized = value.replace('\\', '/')
            if '.scratch' not in normalized and 'next-qa' not in normalized and 'datasets/' not in normalized:
                return value
            updated = self.path(value)
            if updated != value:
                self.changed_fields[key] += 1
            return updated
        return value


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def derive_json(source: Path, target: Path, relocator: Relocator) -> dict:
    original_bytes = source.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    payload = json.loads(original_bytes)
    del original_bytes
    before = relocator.changed_fields.copy()
    updated = relocator.transform(payload)
    content = (json.dumps(updated, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    # Parse-roundtrip check guards numeric/list/encoding changes in serialization.
    if json.loads(content) != updated:
        raise ValueError(f'JSON serialization changed values: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(target) != hashlib.sha256(content).hexdigest():
            raise FileExistsError(f'different target already exists: {target}')
    else:
        temporary = target.with_name(target.name + '.writing')
        with temporary.open('xb') as stream:
            stream.write(content)
        temporary.rename(target)
    output_hash = hashlib.sha256(content).hexdigest()
    if sha(target) != output_hash:
        raise RuntimeError(f'written JSON hash differs: {target}')
    return {'source': source.relative_to(relocator.root).as_posix(),
            'target': target.relative_to(relocator.root).as_posix(),
            'source_sha256': original_hash, 'target_sha256': output_hash,
            'bytes': len(content), 'changed_fields': dict(relocator.changed_fields - before)}


def write_new(path: Path, payload):
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f'different manifest already exists: {path}')
    path.write_bytes(content)


def run():
    graph_asset = ROOT / 'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1'
    index_asset = ROOT / 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570'
    graph_source = graph_asset / 'payload/nextqa_cg6_human_v2_full570_20260812a'
    graph_target = graph_asset / 'payload/runtime_paths_v2'
    index_source = index_asset / 'payload/visual_index_manifest.json'
    index_target = index_asset / 'payload/runtime_paths_v2'
    plans = [ROOT / 'registry/migrations/nextqa_mainline_copy_plan.json',
             ROOT / 'registry/migrations/nextqa_repaired_source_copy_plan.json']
    mapping = {row['source']: row['target'] for p in plans for row in json.loads(p.read_text(encoding='utf-8'))['entries']}
    graph_legacy = '.scratch/nextqa/graphs/nextqa_cg6_human_v2_full570_20260812a'
    mapping[graph_legacy + '/graphs'] = (graph_target / 'graphs').relative_to(ROOT).as_posix()
    relocator = Relocator(mapping, ROOT)
    graph_manifest = json.loads((graph_asset / 'payload/runtime/nextqa_graph_build_manifest.json').read_text(encoding='utf-8'))
    rows = []
    last = time.monotonic()
    for i, video in enumerate(graph_manifest['videos'], 1):
        for field in ('temporal_graph_path', 'canonical_graph_path', 'canonical_report_path', 'video_audit_path'):
            source = Path(video[field])
            if not source.is_relative_to(graph_source):
                raise ValueError(f'graph source outside verified copied tree: {source}')
            target = graph_target / source.relative_to(graph_source)
            rows.append(derive_json(source, target, relocator))
            video[field] = str(target)
        if i % 25 == 0 or time.monotonic() - last > 15:
            print(f'graphs relocated {i}/570', flush=True)
            last = time.monotonic()
    graph_manifest['output_root'] = graph_manifest['run_root'] = str(graph_target)
    graph_manifest['source_manifest_path'] = str(ROOT / 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/runtime/nextqa_val_manifest_v1.json')
    graph_manifest['config_path'] = str(ROOT / 'configs/datasets/nextqa/graph.yaml')
    graph_manifest['path_relocation'] = {'source_run_id_preserved': True, 'new_graph_build': False}
    index_manifest = json.loads(index_source.read_text(encoding='utf-8'))
    for i, video in enumerate(index_manifest['videos'], 1):
        for kind, shard in video['shards'].items():
            source = Path(shard['metadata_path'])
            if not source.is_relative_to(index_asset / 'payload/videos'):
                raise ValueError(f'index metadata outside copied tree: {source}')
            target = index_target / 'metadata' / video['video_id'] / (kind + '.metadata.json')
            record = derive_json(source, target, relocator)
            rows.append(record)
            shard['metadata_path'] = str(target)
            shard['metadata_bytes'] = record['bytes']
        # Only graph references change. index_path, vectors, IDs and scores stay untouched.
        for field in ('graph_dir', 'temporal_graph_path'):
            if field in video:
                video[field] = relocator.path(video[field])
        if i % 50 == 0:
            print(f'visual metadata relocated {i}/570', flush=True)
    index_manifest['graph_root'] = str(graph_target)
    missing = sorted(p for p in relocator.references if not Path(p).exists())
    report = {'contract': 'nextqa_runtime_path_rebase_v1',
              'status': 'passed' if not missing and not relocator.old_paths else 'unresolved_references',
              'derived_json_files': len(rows), 'changed_fields': dict(relocator.changed_fields),
              'unique_relocated_paths': len(relocator.references),
              'missing_targets': missing, 'unmapped_old_paths': dict(relocator.old_paths),
              'vectors_modified': False, 'originals_modified': False,
              'graph_manifest': (graph_target / 'nextqa_graph_build_manifest.json').relative_to(ROOT).as_posix(),
              'index_manifest': (index_target / 'visual_index_manifest.json').relative_to(ROOT).as_posix(),
              'mapping': mapping}
    out = ROOT / 'registry/migrations/nextqa_runtime_path_rebase.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')
    (ROOT / 'registry/migrations/nextqa_runtime_path_hashes.gz').write_bytes(gzip.compress(
        (json.dumps(rows, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8'), mtime=0))
    if missing or relocator.old_paths:
        raise RuntimeError(f'Unresolved references: {len(missing)} missing, {len(relocator.old_paths)} unmapped; current pointers unchanged.')
    write_new(graph_target / 'nextqa_graph_build_manifest.json', graph_manifest)
    write_new(index_target / 'visual_index_manifest.json', index_manifest)
    print(json.dumps({k:v for k,v in report.items() if k!='mapping'},indent=2),flush=True)


if __name__ == '__main__':
    run()
