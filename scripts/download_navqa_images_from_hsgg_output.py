from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.datasets.navqa.runtime.paths import (
    coda_sequence_zip_url,
    default_navqa_sequence_layout,
)

CODA_IMAGE_PATTERN = re.compile(
    r"(?P<coda_name>2d_rect_cam(?P<cam>\d+)_(?P<seq>\d+)_(?P<frame>\d+)\.jpe?g)$",
    re.IGNORECASE,
)
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_EOCD_LOCATOR_SIGNATURE = b"PK\x06\x07"
CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"


@dataclass(frozen=True)
class ZipMemberInfo:
    filename: str
    compress_type: int
    compress_size: int
    header_offset: int


def zip64_values(extra: bytes) -> list[int]:
    offset = 0
    while offset + 4 <= len(extra):
        header_id, data_size = struct.unpack("<HH", extra[offset : offset + 4])
        offset += 4
        data = extra[offset : offset + data_size]
        offset += data_size
        if header_id != 0x0001:
            continue
        return [
            value[0]
            for value in struct.iter_unpack("<Q", data[: len(data) - (len(data) % 8)])
        ]
    return []


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


def get_content_length(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=120)
    response.raise_for_status()
    length = response.headers.get("Content-Length")
    if length is None:
        raise RuntimeError("server did not return Content-Length")
    return int(length)


def load_zip_index(url: str) -> dict[str, ZipMemberInfo]:
    file_size = get_content_length(url)
    tail_size = min(file_size, 70_000)
    tail = get_range(url, file_size - tail_size, file_size - 1)
    eocd_offset = tail.rfind(EOCD_SIGNATURE)
    if eocd_offset < 0:
        raise RuntimeError("Could not find ZIP end-of-central-directory record")

    eocd_absolute_offset = file_size - tail_size + eocd_offset
    eocd = tail[eocd_offset : eocd_offset + 22]
    (
        _sig,
        _disk_no,
        _cd_disk_no,
        _disk_entries,
        total_entries,
        cd_size,
        cd_offset,
        comment_len,
    ) = struct.unpack("<4s4H2LH", eocd)
    if eocd_offset + 22 + comment_len > len(tail):
        raise RuntimeError("Incomplete ZIP end-of-central-directory record")

    if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or total_entries == 0xFFFF:
        locator_start = eocd_absolute_offset - 20
        locator = get_range(url, locator_start, locator_start + 19)
        (
            locator_sig,
            _zip64_disk_no,
            zip64_eocd_offset,
            _total_disks,
        ) = struct.unpack("<4sLQL", locator)
        if locator_sig != ZIP64_EOCD_LOCATOR_SIGNATURE:
            raise RuntimeError("Could not find ZIP64 EOCD locator")

        zip64_header = get_range(url, zip64_eocd_offset, zip64_eocd_offset + 55)
        fields = struct.unpack("<4sQ2H2L4Q", zip64_header)
        if fields[0] != ZIP64_EOCD_SIGNATURE:
            raise RuntimeError("Bad ZIP64 EOCD signature")
        total_entries = fields[7]
        cd_size = fields[8]
        cd_offset = fields[9]

    central_dir = get_range(url, cd_offset, cd_offset + cd_size - 1)
    infos: dict[str, ZipMemberInfo] = {}
    offset = 0
    for _ in range(total_entries):
        header = central_dir[offset : offset + 46]
        if len(header) < 46:
            raise RuntimeError("Truncated central directory header")
        fields = struct.unpack("<4s6H3L5H2L", header)
        sig = fields[0]
        if sig != CENTRAL_DIRECTORY_SIGNATURE:
            raise RuntimeError("Bad central directory signature")
        compress_type = fields[4]
        compress_size = fields[8]
        name_len = fields[10]
        extra_len = fields[11]
        comment_len = fields[12]
        header_offset = fields[16]
        name_start = offset + 46
        name_end = name_start + name_len
        filename = central_dir[name_start:name_end].decode("utf-8")
        extra = central_dir[name_end : name_end + extra_len]
        if compress_size == 0xFFFFFFFF or header_offset == 0xFFFFFFFF:
            values = zip64_values(extra)
            value_index = 0
            # ZIP64 extra values are present only for fields that overflow, in this order:
            # uncompressed size, compressed size, local header offset, disk start.
            if fields[7] == 0xFFFFFFFF:
                value_index += 1
            if compress_size == 0xFFFFFFFF:
                compress_size = values[value_index]
                value_index += 1
            if header_offset == 0xFFFFFFFF:
                header_offset = values[value_index]
        infos[filename] = ZipMemberInfo(
            filename=filename,
            compress_type=compress_type,
            compress_size=compress_size,
            header_offset=header_offset,
        )
        offset = name_end + extra_len + comment_len
    return infos


def image_name_from_path(path_value: str) -> str:
    return Path(path_value.replace("\\", "/")).name


def parse_coda_image(image_name: str) -> re.Match[str]:
    match = CODA_IMAGE_PATTERN.search(image_name)
    if not match:
        raise ValueError(f"Unsupported CODa image name: {image_name}")
    return match


def coda_zip_member(image_name: str) -> str:
    match = parse_coda_image(image_name)
    cam = match.group("cam")
    seq = match.group("seq")
    return f"2d_rect/cam{cam}/{seq}/{match.group('coda_name')}"


def load_required_images(hsgg_output_dir: Path) -> list[tuple[str, str]]:
    objects_path = hsgg_output_dir / "objects.json"
    objects = json.loads(objects_path.read_text(encoding="utf-8-sig"))
    image_names = {
        image_name_from_path(frame_payload["image_path"])
        for frame_payload in objects.values()
        if isinstance(frame_payload, dict) and frame_payload.get("image_path")
    }
    return [
        (image_name, coda_zip_member(image_name))
        for image_name in sorted(
            image_names,
            key=lambda name: int(parse_coda_image(name).group("frame")),
        )
    ]


def load_manifest_images(hsgg_output_dir: Path) -> list[tuple[str, str]]:
    manifest_path = hsgg_output_dir / "run_manifest.json"
    if not manifest_path.exists():
        return []

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    frames = manifest.get("frames", {})
    if not isinstance(frames, dict):
        return []

    image_names = {
        image_name_from_path(frame_payload["image_path"])
        for frame_payload in frames.values()
        if isinstance(frame_payload, dict) and frame_payload.get("image_path")
    }
    return [
        (image_name, coda_zip_member(image_name))
        for image_name in sorted(
            image_names,
            key=lambda name: int(parse_coda_image(name).group("frame")),
        )
    ]


def merge_required_images(image_groups: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
    by_image_name: dict[str, str] = {}
    for group in image_groups:
        for image_name, member_name in group:
            by_image_name[image_name] = member_name
    return [
        (image_name, by_image_name[image_name])
        for image_name in sorted(
            by_image_name,
            key=lambda name: int(parse_coda_image(name).group("frame")),
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download only the CODa images referenced by a local HSGG output directory."
    )
    parser.add_argument("hsgg_output_dir", type=Path)
    parser.add_argument("--seq-id", default="0")
    parser.add_argument("--cam", default="0")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--layout-root", type=Path, default=Path(".scratch/navqa"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--url", default=None)
    parser.add_argument(
        "--include-manifest",
        action="store_true",
        help="Also download images listed in run_manifest.json, including skipped/error frames.",
    )
    args = parser.parse_args()

    hsgg_output_dir = args.hsgg_output_dir.resolve()
    layout = default_navqa_sequence_layout(
        seq_id=args.seq_id,
        cam=args.cam,
        stride=args.stride,
        layout_root=args.layout_root,
    )
    out = (args.out or layout.images_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    url = args.url or coda_sequence_zip_url(args.seq_id)

    image_groups = [load_required_images(hsgg_output_dir)]
    if args.include_manifest:
        image_groups.append(load_manifest_images(hsgg_output_dir))
    required_images = merge_required_images(image_groups)
    print(f"required_images: {len(required_images)}", flush=True)
    print(f"out: {out}", flush=True)

    infos = load_zip_index(url)

    missing_members = [member_name for _, member_name in required_images if member_name not in infos]
    if missing_members:
        raise SystemExit(f"Missing members in CODa zip, sample={missing_members[:5]}")

    def fetch_one(item: tuple[int, tuple[str, str]]) -> tuple[int, str, str]:
        index, (local_name, member_name) = item
        info = infos[member_name]
        dst = out / local_name
        if dst.exists() and dst.stat().st_size > 0:
            return index, member_name, "exists"

        header = get_range(url, info.header_offset, info.header_offset + 29)
        sig, _ver, _flag, _method, _mtime, _mdate, _crc, _csize, _usize, nlen, elen = struct.unpack(
            "<4s5H3L2H", header
        )
        if sig != b"PK\x03\x04":
            raise RuntimeError(f"bad local header for {member_name}")

        data_start = info.header_offset + 30 + nlen + elen
        data_end = data_start + info.compress_size - 1
        payload = get_range(url, data_start, data_end)

        if info.compress_type == ZIP_STORED:
            data = payload
        elif info.compress_type == ZIP_DEFLATED:
            data = zlib.decompress(payload, -15)
        else:
            raise RuntimeError(f"unsupported compression {info.compress_type} for {member_name}")

        dst.write_bytes(data)
        return index, member_name, f"{len(data)} bytes"

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch_one, item) for item in enumerate(required_images)]
        for future in as_completed(futures):
            _index, member_name, status = future.result()
            done += 1
            print(f"[{done}/{len(required_images)}] {member_name} -> {status}", flush=True)

    print(f"elapsed_sec: {time.time() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
