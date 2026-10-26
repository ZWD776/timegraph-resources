"""Check traversal coverage against completed hash ledgers without rehashing payloads."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

from scripts.datasets.migrate_asset_storage import ROOT, is_reparse, native_path, safe_path


def verify(receipt_path: Path):
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['status'] != 'verified':
        raise ValueError('receipt must be complete before checking traversal coverage')
    results = []
    for item in receipt['entries']:
        blob = safe_path(item['file_ledger']).read_bytes()
        if hashlib.sha256(blob).hexdigest() != item['file_ledger_sha256']:
            raise ValueError('ledger fingerprint changed: ' + item['file_ledger'])
        expected = {row['relative']: row['bytes'] for row in json.loads(gzip.decompress(blob))}
        path = native_path(safe_path(item['target']))
        if is_reparse(path):
            raise ValueError('reparse target: ' + item['target'])
        actual = {}
        if path.is_file():
            actual[''] = path.stat().st_size
        else:
            if not path.is_dir():
                raise FileNotFoundError(path)

            def fail(error):
                raise error

            for directory, dirs, files in os.walk(path, onerror=fail):
                for name in dirs + files:
                    child = Path(directory, name)
                    if is_reparse(child):
                        raise ValueError('reparse child: ' + str(child))
                for name in files:
                    child = Path(directory, name)
                    actual[child.relative_to(path).as_posix()] = child.stat().st_size
        if actual != expected:
            raise ValueError('inventory names or sizes changed: ' + item['target'])
        results.append({'target': item['target'], 'files': len(actual),
                        'bytes': sum(actual.values()), 'status': 'passed'})
    return {'status': 'passed', 'receipt': receipt_path.relative_to(ROOT).as_posix(),
            'scope': 'Complete traversal, names, sizes and ledger integrity; payload hashes reuse the completed move verification.',
            'rehashes_payloads': False, 'entries': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(safe_path(args.receipt))
    output = safe_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'roots': len(result['entries']),
                      'files': sum(row['files'] for row in result['entries'])}))


if __name__ == '__main__':
    main()
