from __future__ import annotations

import argparse
import re
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED

import requests
from remotezip import RemoteZip

URL = "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/CODa_v1/sequences/0.zip"


def frame_id(name: str) -> int:
    match = re.search(r"_(\d+)\.jpe?g$", name.lower())
    return int(match.group(1)) if match else 10**18


def get_range(url: str, start: int, end: int, tries: int = 8) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    last_error: Exception | None = None
    for _ in range(tries):
        try:
            response = requests.get(url, headers=headers, timeout=120)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"range download failed: {start}-{end}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CODa seq0 cam0 images sampled every 5 frames."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", default=None)
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()

    out = Path(
        args.out
        or f".scratch/navqa/seq0/cam0/every5/images_first{args.limit}"
    )
    out.mkdir(parents=True, exist_ok=True)

    with RemoteZip(args.url) as archive:
        infos = {info.filename: info for info in archive.infolist()}

    names = sorted(
        [
            name
            for name in infos
            if name.startswith("2d_rect/cam0/0/") and name.lower().endswith((".jpg", ".jpeg"))
        ],
        key=frame_id,
    )
    selected = names[::5][: args.limit]

    print(f"cam0 images: {len(names)}", flush=True)
    print(f"selected: {len(selected)}", flush=True)
    print(f"workers: {args.workers}", flush=True)
    print(f"out: {out}", flush=True)

    def fetch_one(item: tuple[int, str]) -> tuple[int, str, str]:
        index, name = item
        info = infos[name]
        dst = out / f"{index:06d}_{Path(name).name}"
        if dst.exists() and dst.stat().st_size > 0:
            return index, name, "exists"

        header = get_range(args.url, info.header_offset, info.header_offset + 29)
        sig, _ver, _flag, _method, _mtime, _mdate, _crc, _csize, _usize, nlen, elen = struct.unpack(
            "<4s5H3L2H", header
        )
        if sig != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {name}")

        data_start = info.header_offset + 30 + nlen + elen
        data_end = data_start + info.compress_size - 1
        payload = get_range(args.url, data_start, data_end)

        if info.compress_type == ZIP_STORED:
            data = payload
        elif info.compress_type == ZIP_DEFLATED:
            data = zlib.decompress(payload, -15)
        else:
            raise RuntimeError(f"unsupported compression {info.compress_type} for {name}")

        dst.write_bytes(data)
        return index, name, f"{len(data)} bytes"

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_one, item) for item in enumerate(selected)]
        for future in as_completed(futures):
            _index, name, status = future.result()
            done += 1
            print(f"[{done}/{len(selected)}] {name} -> {status}", flush=True)

    print(f"elapsed_sec: {time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
