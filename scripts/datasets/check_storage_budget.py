"""Check a conservative additional-storage budget before migration batches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
GIB = 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--additional-gib', type=float, required=True)
    parser.add_argument('--reserve-gib', type=float, default=20)
    parser.add_argument('--batch', required=True)
    args = parser.parse_args()
    if args.additional_gib < 0 or args.reserve_gib < 0:
        raise ValueError('storage budgets must be nonnegative')
    usage = shutil.disk_usage(ROOT)
    projected = usage.free - int(args.additional_gib * GIB)
    fits = projected >= args.reserve_gib * GIB
    report = {
        'contract': 'migration_storage_budget_v1', 'batch': args.batch,
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'volume': ROOT.anchor, 'total_bytes': usage.total, 'used_bytes': usage.used,
        'free_bytes': usage.free, 'additional_budget_bytes': int(args.additional_gib * GIB),
        'minimum_reserve_bytes': int(args.reserve_gib * GIB),
        'projected_free_bytes': projected, 'fits_with_reserve': fits,
        'on_insufficient_space': 'Complete and validate one dataset; remove only verified deletion candidates before continuing.',
        'scope': 'Same-volume moves do not add payload bytes; budget covers derived JSON/SQLite/cache views and verification outputs.',
        'does_not_authorize_unverified_deletion': True,
    }
    if not args.batch or any(c in args.batch for c in '/\\:') or '..' in args.batch:
        raise ValueError('invalid budget batch identifier')
    out = ROOT / 'registry/audits/storage' / (args.batch + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if fits else 2


if __name__ == '__main__':
    raise SystemExit(main())
