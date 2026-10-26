"""Activate verified mainline copies; keep original run snapshots unchanged."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))
from vlkg.core.assets.asset_paths import translate_path  # noqa: E402
from scripts.datasets.verify_nextqa_mainline import mapped  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def rebase(value, mapping, root=ROOT):
    if isinstance(value, str):
        return translate_path(value, mapping, root)
    if isinstance(value, list):
        return [rebase(v, mapping, root) for v in value]
    if isinstance(value, dict):
        return {k: rebase(v, mapping, root) for k, v in value.items()}
    return value


def validate_source_manifest_paths(manifest, root=ROOT):
    """Check selected runtime inputs, leaving archive provenance optional."""
    from vlkg.datasets.nextqa.graph.hsgg_input_adapter import resolve_normalized_input_paths
    from vlkg.datasets.nextqa.runtime.contracts import graph_safe_video_record, validate_manifest_input_storage

    validate_manifest_input_storage(manifest)
    for video in manifest['videos']:
        record = graph_safe_video_record(video)
        if record.get('input_storage') == 'normalized_directory':
            input_dir, frames_dir = resolve_normalized_input_paths(record, repository_root=root)
            for name in ('objects.json', 'attributes.json', 'relations.json', 'hierarchy_mapping.json', 'timeline.json', 'source_trace.json'):
                if not (input_dir / name).is_file():
                    raise RuntimeError(f'source manifest path missing: {input_dir / name}')
            if not frames_dir.is_dir():
                raise RuntimeError(f'source manifest directory missing: {frames_dir}')
        else:
            for field in ('hsgg_archive', 'frame_archive'):
                path = (root / str(record[field])).resolve()
                if not path.is_file():
                    raise RuntimeError(f'source manifest path missing: {path}')


def main():
    current_graph_pointer = ROOT / 'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/asset_pointer.json'
    if read(current_graph_pointer).get('runtime_path_validation'):
        raise RuntimeError('Initial copy activation has been superseded by relocated runtime views; refusing to restore stale pointers.')
    plan_path = ROOT / 'registry/migrations/nextqa_mainline_copy_plan.json'
    plan = read(plan_path)
    receipt_path = ROOT / 'datasets/nextqa/reports/nextqa-reports-v1/payload/copy_receipt.json'
    receipt = read(receipt_path)
    if receipt['plan_sha256'] != hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise RuntimeError('copy receipt belongs to another plan')
    if receipt['status'] != 'byte_copies_verified':
        raise RuntimeError('copy receipt did not pass')
    replay = read(ROOT / 'registry/migrations/nextqa_mainline_replay.json')
    if replay['status'] != 'passed':
        raise RuntimeError('frozen prediction replay did not pass')
    # The full copy pass checked every byte. Activation rechecks presence/size,
    # and hashes every new operational manifest after deriving its path view.
    for row in receipt['rows']:
        if (ROOT / row['target']).stat().st_size != row['bytes']:
            raise RuntimeError(f'copy missing or modified: {row["target"]}')
    extras = read(ROOT / 'registry/migrations/nextqa_repaired_source_copy_receipt.json')
    for row in extras['rows']:
        if hashlib.sha256((ROOT / row['target']).read_bytes()).hexdigest() != row['sha256']:
            raise RuntimeError('repaired source hash changed')
    entries = plan['entries'] + read(ROOT / 'registry/migrations/nextqa_repaired_source_copy_plan.json')['entries']
    mapping = {e['source']: e['target'] for e in entries}
    manifest_path = ROOT / 'registry/datasets/nextqa.yaml'
    manifest = yaml.safe_load(manifest_path.read_text(encoding='utf-8'))
    ids = manifest['asset_ids']
    roots = {layer: ROOT / 'datasets/nextqa' / layer / ids[id_key] for layer, id_key in [
        ('source', 'source_id'), ('graphs', 'graph_id'), ('ir', 'ir_id'),
        ('retrieval', 'retrieval_id'), ('runs', 'qa_id')]}
    definitions = {
        'source': { 'manifest': '.scratch/nextqa/manifests/nextqa_val_manifest_v1.json',
                   'questions': '.scratch/nextqa/standardized/nextqa_val_standardized_full4996_20260812a/questions.runtime.json',
                   'labels': '.scratch/nextqa/standardized/nextqa_val_standardized_full4996_20260812a/labels.evaluator_only.json'},
        'graphs': {'manifest': '.scratch/nextqa/graphs/nextqa_cg6_human_v2_full570_20260812a/nextqa_graph_build_manifest.json'},
        'ir': {'runtime': '.scratch/nextqa/question_ir/nextqa-ir-v3-luna-full4996-20260817b/question_ir.runtime.json',
               'retrieval': '.scratch/nextqa/question_ir/nextqa-ir-v3-luna-full4996-20260817b/question_ir.retrieval.json'},
        'retrieval': {
            'records': '.scratch/nextqa/graph_retrieval/nextqa-graph-retrieval-v2-irv3-full4996-20260817b/retrieval_records.json',
            'routes': '.scratch/nextqa/visual_routes/nextqa_visual_routes_v2_irv3_full4996_20260818a/route_records.json',
            'frozen_graph': '.scratch/nextqa/visual_graph_baseline/nextqa_visual_graph_baseline_v2_irv3_full4996_20260818a/frozen_graph_evidence.json',
            'visual_evidence': '.scratch/nextqa/visual_evidence/nextqa_visual_evidence_v2_irv3_full4996_20260818a/visual_evidence_records.json'},
        'runs': {
            'A': '.scratch/nextqa/qa/nextqa-graph-qa-luna-irv3-full4996-20260817a/predictions.json',
            'B': '.scratch/nextqa/ablation/nextqa_visual_no_routing_qa_v2_full4996_20260901a/predictions.json',
            'C': '.scratch/nextqa/ablation/nextqa_visual_route_policy_v3_mainline_composite_20260902a/predictions.json'},
    }
    derived = []
    entrypoints = {}
    for layer, names in definitions.items():
        entrypoints[layer] = {}
        for name, source in names.items():
            path = mapped(source, entries, ROOT)
            if layer in ('source', 'graphs') and name == 'manifest':
                original = read(ROOT / source)
                updated = rebase(original, mapping)
                path = roots[layer] / 'payload/runtime' / Path(source).name
                write(path, updated)
                derived.append({'source': source, 'target': path.relative_to(ROOT).as_posix(),
                                'source_sha256': hashlib.sha256((ROOT / source).read_bytes()).hexdigest(),
                                'target_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'transformation': 'decoded path strings only; original manifest copies preserved'})
                if layer == 'graphs':
                    for video in updated['videos']:
                        for field in ('canonical_graph_path', 'temporal_graph_path', 'video_audit_path', 'canonical_report_path'):
                            if not Path(video[field]).is_file():
                                raise RuntimeError(f'graph manifest path missing: {video[field]}')
                else:
                    validate_source_manifest_paths(updated)
            if not path.is_file():
                raise FileNotFoundError(path)
            entrypoints[layer][name] = path.relative_to(ROOT).as_posix()
    # Persist a Git-sized, lossless receipt rather than tracking copied payloads.
    compressed = gzip.compress(receipt_path.read_bytes(), mtime=0)
    (ROOT / 'registry/migrations/nextqa_mainline_copy_receipt.gz').write_bytes(compressed)
    for layer, asset_root in roots.items():
        pointer_path = asset_root / 'asset_pointer.json'
        pointer = read(pointer_path)
        if 'previous_paths' not in pointer:
            pointer['previous_paths'] = pointer['current_paths']
        selected = [e['target'] for e in entries if e['layer'] == layer and e['role'] == 'selected']
        dependencies = [e['target'] for e in entries if e['layer'] == layer and e['role'] != 'selected']
        pointer.update(current_paths=selected, dependency_paths=dependencies,
                       entrypoints=entrypoints[layer], status='verified_copy_legacy_references_retained',
                       payload_migrated=True,
                       remaining_work='Embedded graph/image paths and original run contracts still reference retained source trees.')
        pointer['migration'] = {'copy': True, 'move': False, 'delete': False,
                                'next_step': 'rebase embedded graph/image references and integrate output roots'}
        write(pointer_path, pointer)
        manifest['current_paths'][{'graphs': 'graph'}.get(layer, layer)] = selected
    manifest.update(status='selected_mainline_copied_embedded_references_pending', planning_only=False)
    manifest['migration'] = {'copy_payloads': True, 'move_payloads': False, 'delete_payloads': False,
                             'next_gate': 'embedded_reference_rebase_and_output_root_integration'}
    manifest['entrypoints'] = entrypoints
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, width=105), encoding='utf-8', newline='\n')
    registry_path = ROOT / 'configs/dataset_asset_registry.yaml'
    registry = yaml.safe_load(registry_path.read_text(encoding='utf-8'))
    for layer in roots:
        key = {'graphs': 'graph', 'runs': 'qa'}.get(layer, layer)
        registry['datasets']['nextqa']['selected_current_paths'][key] = manifest['current_paths'][{'graphs': 'graph'}.get(layer, layer)]
    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False, width=105), encoding='utf-8', newline='\n')
    write(ROOT / 'registry/migrations/nextqa_mainline_activation.json', {
        'status': 'copied_selected_assets_activated', 'files': receipt['files'], 'bytes': receipt['bytes'],
        'receipt': 'registry/migrations/nextqa_mainline_copy_receipt.gz',
        'receipt_sha256': hashlib.sha256(compressed).hexdigest(), 'derived_manifests': derived,
        'new_inference_run': False, 'sources_deleted': False,
        'remaining_work': ['Embedded graph/image references still refer to original copies.',
                           'Original run configs are provenance, not relocated resume contracts.',
                           'Existing output write guards still use .scratch/nextqa.',
                           'Historical assets have not been sealed.']})
    print('Selected mainline copies activated; original and embedded-reference dependencies retained.')


if __name__ == '__main__':
    main()
