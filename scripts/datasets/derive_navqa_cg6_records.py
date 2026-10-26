"""Create current CG6-only frozen records while preserving their original files."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from scripts.datasets.migrate_asset_storage import ROOT, safe_path

GRAPH_ID = re.compile(r'cg(\d+)', re.I)
REMOVED = Counter()


def retired(value):
    return any(match.group(1) != '6' for match in GRAPH_ID.finditer(value))


def clean(value, location=''):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if retired(key):
                if item not in (None, False, [], {}, '') and not location.startswith('/experiments'):
                    raise ValueError('nonempty retired field needs review: ' + location + '/' + key)
                REMOVED[key] += 1
                continue
            if isinstance(item, str) and retired(item):
                if key != 'visual_faiss_version_policy':
                    raise ValueError('retired text needs review: ' + location + '/' + key)
                REMOVED[key] += 1
                continue
            result[key] = clean(item, location + '/' + key)
        return result
    if isinstance(value, list):
        return [clean(item, location + '/' + str(index)) for index, item in enumerate(value)]
    return value


def derive(source_root, output_root, receipt_path):
    if not output_root.is_relative_to(ROOT / 'datasets/navqa/runs'):
        raise ValueError('derived records must stay in the NAVQA run layer')
    if output_root.exists():
        raise FileExistsError(output_root)
    groups = {
        'CG6-clean_IR-QA-G-clean_full_20260601': 'A',
        'CG6-clean_IR-QA-F-GER-v3-binding_full_20260601': 'B',
        'CG6_question_router_v4_mainline_full_20260715': 'components/base',
        'CG6_question_router_v4_binary_existence_fix_20260719': 'components/binary',
        'navqa_deepseek_bestline_composite_20260902': 'C',
    }
    mappings = []
    for old, new in groups.items():
        for base in (source_root, ROOT / '.scratch/navqa_cg6_local_tracklet_runs'):
            for before, after in ((str(base / old), str(output_root / new)),
                                  ((base / old).as_posix(), (output_root / new).as_posix())):
                mappings.append((before, after))

    def paths(value):
        if isinstance(value, dict):
            return {key: paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [paths(item) for item in value]
        if isinstance(value, str):
            for old, new in mappings:
                if value == old or value.startswith(old + '/') or value.startswith(old + '\\'):
                    return new + value[len(old):]
        return value

    files = []
    for old, new in groups.items():
        directory = source_root / old
        selected = sorted(directory.glob('seq*/qa_non_position_live_with_question_ir.json'))
        selected += [directory / ('navqa_route_composite.json' if new == 'C'
                                  else 'mainline_all_sequences_summary.json')]
        for original in selected:
            raw = original.read_bytes()
            source = json.loads(raw)
            current = paths(clean(source))
            target = output_root / new / original.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            # The exact same projection must recover every retained field.
            if json.loads(target.read_text(encoding='utf-8')) != current:
                raise ValueError('derived object roundtrip changed')
            files.append({'source': original.relative_to(ROOT).as_posix(),
                          'target': target.relative_to(ROOT).as_posix(),
                          'source_sha256': hashlib.sha256(raw).hexdigest(),
                          'target_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'derived_pending_behavior_validation', 'files': files,
               'removed_empty_fields_or_description_counts': dict(REMOVED),
               'all_other_fields_preserved_except_selected_run_path_relocation': True,
               'originals_modified': False, 'experiment_payloads_deleted': False}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': receipt['status'], 'files': len(files),
                      'removed_entries': sum(REMOVED.values())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    derive(safe_path(args.source_root), safe_path(args.output_root), safe_path(args.receipt))


if __name__ == '__main__':
    main()
