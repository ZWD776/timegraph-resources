"""Stream explicit path-prefix substitutions without parsing large graph JSON.

Only a reviewed set of path prefixes may change; file bytes outside matches are
copied directly. The source and resulting file receive independent hashes.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def path_variants(source: str, target: str, root: Path):
    for old, new in (
        (source, target),
        (source.replace('/', '\\'), target.replace('/', '\\')),
        (str(root / source), str(root / target)),
        ((root / source).as_posix(), (root / target).as_posix()),
    ):
        yield old.encode('utf-8'), new.encode('utf-8')
        yield json.dumps(old, ensure_ascii=False)[1:-1].encode('utf-8'), json.dumps(new, ensure_ascii=False)[1:-1].encode('utf-8')


class PathPrefixRewriter:
    def __init__(self, mapping, root: Path):
        self.mapping = dict(mapping)
        pairs = {}
        for old, new in self.mapping.items():
            if not old or not new or old == new:
                raise ValueError('path-prefix mapping must change a nonempty prefix')
            for source, target in path_variants(old, new, root):
                if source in pairs and pairs[source] != target:
                    raise ValueError('ambiguous source-prefix mapping')
                pairs[source] = target
        self.pairs = pairs
        self.max_prefix = max(map(len, pairs))
        self.pattern = re.compile(b'(?:' + b'|'.join(re.escape(p) for p in sorted(pairs, key=len, reverse=True)) + br')(?=[/\\"\s]|$)')

    def rewrite_text(self, value: str) -> str:
        return self.pattern.sub(lambda match: self.pairs[match.group()], value.encode('utf-8')).decode('utf-8')

    def rewrite_value(self, value):
        if isinstance(value, str):
            return self.rewrite_text(value)
        if isinstance(value, list):
            return [self.rewrite_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.rewrite_value(item) for key, item in value.items()}
        return value

    def stream(self, source: Path, target: Path, *, chunk_size=4 * 1024**2, verify_existing=False):
        if (target.exists() and not verify_existing) or source.resolve() == target.resolve():
            raise FileExistsError('runtime views must use a new, distinct file')
        target.parent.mkdir(parents=True, exist_ok=True)
        old_hash, new_hash = hashlib.sha256(), hashlib.sha256()
        counts = Counter()
        carry = b''
        class ComparisonWriter:
            def __enter__(self):
                self.stream = target.open('rb')
                return self
            def write(self, value):
                if self.stream.read(len(value)) != value:
                    raise ValueError(f'existing runtime view differs from explicit rewrite: {target}')
            def __exit__(self, exc_type, exc, traceback):
                try:
                    if exc_type is None and self.stream.read(1):
                        raise ValueError('existing runtime view has unexpected trailing bytes')
                finally:
                    self.stream.close()

        output = ComparisonWriter() if target.exists() and verify_existing else target.open('xb')
        with source.open('rb') as reader, output as writer:
            while True:
                block = reader.read(chunk_size)
                old_hash.update(block)
                buffer = carry + block
                cut = max(0, len(buffer) - self.max_prefix - 1) if block else len(buffer)
                cursor = 0
                for match in self.pattern.finditer(buffer):
                    if match.start() >= cut:
                        break
                    prefix = buffer[cursor:match.start()]
                    replacement = self.pairs[match.group()]
                    writer.write(prefix); writer.write(replacement)
                    new_hash.update(prefix); new_hash.update(replacement)
                    counts[match.group().decode('utf-8')] += 1
                    cursor = match.end()
                end = max(cursor, cut)
                remainder = buffer[cursor:end]
                writer.write(remainder); new_hash.update(remainder)
                carry = buffer[end:]
                if not block:
                    break
        return {'source_sha256': old_hash.hexdigest(), 'target_sha256': new_hash.hexdigest(),
                'source_bytes': source.stat().st_size, 'target_bytes': target.stat().st_size,
                'replacements': dict(counts), 'only_reviewed_prefix_substitutions': True}
