"""Check current publication files or reachable Git objects without printing secrets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


PATTERNS = (
    re.compile(rb"https?://(?:pan|yun)\.baidu\.com/[^\s<>\"']+", re.I),
    re.compile(rb"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    re.compile(rb'"access_code"\s*:\s*"[^"\s][^"]*"'),
)


def check(raw: bytes) -> bool:
    return any(p.search(raw) for p in PATTERNS)


def audit(root: Path, history: bool = False) -> dict:
    def git(*args, raw=None):
        return subprocess.run(["git", "-C", str(root), *args], input=raw,
                              capture_output=True, check=True).stdout
    checked = 0
    findings = []
    if history:
        ids = [line.split()[0] for line in git("rev-list", "--objects", "HEAD").splitlines()]
        batch = git("cat-file", "--batch", raw=b"\n".join(ids) + b"\n")
        offset = 0
        for _ in ids:
            end = batch.index(b"\n", offset)
            oid, kind, size = batch[offset:end].split()
            size = int(size)
            data = batch[end + 1:end + 1 + size]
            offset = end + 2 + size
            checked += 1
            if check(data):
                findings.append({"object": oid.decode(), "type": kind.decode()})
    else:
        names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
        for name in sorted(set(names) - {""}):
            path = root / name
            if not path.is_file():
                continue
            checked += 1
            if check(path.read_bytes()):
                findings.append({"path": name})
    return {"status": "failed" if findings else "passed", "scope": "reachable_git_objects" if history else "current_publication_files",
            "objects_checked": checked, "findings": findings, "secret_values_printed": False,
            "provider_cache_or_prior_copies_checked": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--history", action="store_true")
    args = p.parse_args()
    result = audit(args.root.resolve(), args.history)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
