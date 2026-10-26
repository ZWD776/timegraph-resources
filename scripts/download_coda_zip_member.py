from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from download_navqa_images_from_hsgg_output import get_range, load_zip_index
from vlkg.datasets.navqa.runtime.paths import coda_sequence_zip_url


LOCAL_FILE_HEADER_SIGNATURE = bytes([0x50, 0x4B, 0x03, 0x04])


def download_member(*, url: str, member: str, out: Path) -> dict[str, object]:
    infos = load_zip_index(url)
    if member not in infos:
        matches = [name for name in infos if member.lower() in name.lower()]
        raise SystemExit(
            f"ZIP member not found: {member}\n"
            f"Similar matches: {json.dumps(matches[:20], ensure_ascii=False)}"
        )

    info = infos[member]
    header = get_range(url, info.header_offset, info.header_offset + 29)
    (
        signature,
        _version,
        _flags,
        _method,
        _mtime,
        _mdate,
        _crc,
        _compressed_size,
        _uncompressed_size,
        name_len,
        extra_len,
    ) = struct.unpack("<4s5H3L2H", header)
    if signature != LOCAL_FILE_HEADER_SIGNATURE:
        raise RuntimeError(f"Bad local ZIP header for {member!r}: {signature!r}")

    data_start = info.header_offset + 30 + name_len + extra_len
    payload = get_range(url, data_start, data_start + info.compress_size - 1)
    if info.compress_type == ZIP_STORED:
        data = payload
    elif info.compress_type == ZIP_DEFLATED:
        data = zlib.decompress(payload, -15)
    else:
        raise RuntimeError(f"Unsupported compression type {info.compress_type} for {member!r}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {
        "source_url": url,
        "member": member,
        "compressed_size": info.compress_size,
        "uncompressed_size": len(data),
        "output": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small member from a CODa sequence ZIP via HTTP range.")
    parser.add_argument("member", help="ZIP member path, e.g. timestamps/0.txt")
    parser.add_argument("--seq-id", default="0")
    parser.add_argument("--url", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    url = args.url or coda_sequence_zip_url(args.seq_id)
    result = download_member(url=url, member=args.member, out=args.out)
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
