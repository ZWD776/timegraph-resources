"""Locate moved provenance for integrity audits, never for runtime defaults."""
from __future__ import annotations

import json
from pathlib import Path


def historical_control_file(relative: str, root: Path) -> Path:
    """Resolve archived governance evidence for integrity tools, never payload defaults."""
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError('control path escapes workspace')
    if candidate.is_file():
        return candidate
    archived = (root / 'archive/legacy_20260928/governance' / relative).resolve()
    if not archived.is_relative_to(root.resolve()):
        raise ValueError('archived control path escapes workspace')
    return archived


def relocated_provenance(path: Path, root: Path) -> Path:
    root = root.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError('provenance must stay within workspace')
    selected = path.resolve()
    if selected.is_file():
        return selected
    mapping = {}
    receipts = list((root / 'registry/migrations').glob('*_move_receipt.json'))
    receipts += list((root / 'archive/legacy_20260928/governance/registry/migrations').glob('*_move_receipt.json'))
    receipts += list((root / 'registry/gates').glob('*_move_receipt.json'))
    for receipt in receipts:
        data = json.loads(receipt.read_text(encoding='utf-8'))
        for row in data.get('entries', []):
            if row.get('status') in {'verified', 'moved_pending_verification', 'hashed_before_move'}:
                source, target = root / row['source'], root / row['target']
                if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
                    raise ValueError('move receipt escapes workspace')
                mapping[source.resolve()] = target.resolve()
    seen = set()
    while not selected.is_file():
        if selected in seen:
            raise ValueError('provenance move chain contains a cycle')
        seen.add(selected)
        match = next((source for source in sorted(mapping, key=lambda p: len(p.parts), reverse=True)
                      if selected == source or selected.is_relative_to(source)), None)
        if match is None:
            break
        selected = mapping[match] / selected.relative_to(match)
    return selected
