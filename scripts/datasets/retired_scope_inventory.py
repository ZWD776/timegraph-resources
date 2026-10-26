"""Inventory the explicitly retired prototypes/backends without following links."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from scripts.datasets.migrate_asset_storage import ROOT, is_reparse, native_path


RETIRED = (
    'graphiti-main', 'cognee-main', 'data/examples', 'robot_unitree.mp4',
    'docker-compose.basketball-neo4j-deepseek.yml',
    'docker-compose.basketball-neo4j.yml', 'docker-compose.robot-graph-view.yml',
    'docker-compose.robot-neo4j.yml',
)
RETAINED = ('HSGG', 'HSGG_pri', 'HSGG-main', 'HSGG_upload.tgz')


def measure(relative):
    path = ROOT / relative
    result = {'path': relative, 'files': 0, 'bytes': 0, 'links': [], 'errors': []}
    if not path.exists():
        result['missing'] = True
        return result
    if path.is_file():
        result.update(files=1, bytes=path.stat().st_size)
        return result
    def error(exc):
        result['errors'].append(str(exc))
    for directory, dirs, files in os.walk(native_path(path), onerror=error):
        for name in list(dirs):
            child = Path(directory, name)
            if is_reparse(child):
                result['links'].append(str(child))
                dirs.remove(name)
        for name in files:
            child = Path(directory, name)
            if is_reparse(child):
                result['links'].append(str(child))
            else:
                result['files'] += 1
                result['bytes'] += child.stat().st_size
    return result


def main():
    report = {'authorization': 'Keep HSGG; fully retire robot/basketball prototypes and Graphiti/Cognee.',
              'actual_deletion_performed': False,
              'retired': [measure(p) for p in RETIRED],
              'retained': [measure(p) for p in RETAINED],
              'compatibility_links': [],
              'disk': {'free_bytes': shutil.disk_usage(ROOT).free,
                       'reserve_bytes': 20 * 1024**3}}
    for relative in ('basketball_frame', 'robot_unitree_every5'):
        path = ROOT / relative
        report['compatibility_links'].append({'path': relative,
            'is_reparse': is_reparse(path), 'resolved_target': str(path.resolve()),
            'target_within_workspace': path.resolve().is_relative_to(ROOT)})
    out = ROOT / 'registry/audits/retired_scope_inventory.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'retired': report['retired'], 'retained': report['retained'],
                      'free_gib': round(report['disk']['free_bytes'] / 1024**3, 2)}))


if __name__ == '__main__':
    main()
