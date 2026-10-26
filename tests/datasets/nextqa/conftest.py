from __future__ import annotations

import io
import json
import shutil
import tarfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from vlkg.datasets.nextqa.runtime.io_utils import NEXTQA_SCRATCH_ROOT


@pytest.fixture
def nextqa_tmp_dir() -> Iterator[Path]:
    path = NEXTQA_SCRATCH_ROOT / "pytest" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def add_json_member(archive: tarfile.TarFile, name: str, value: object) -> None:
    raw = json.dumps(value).encode("utf-8")
    member = tarfile.TarInfo(name=name)
    member.size = len(raw)
    archive.addfile(member, io.BytesIO(raw))


def add_png_member(archive: tarfile.TarFile, name: str) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color=(96, 128, 160)).save(buffer, format="PNG")
    raw = buffer.getvalue()
    member = tarfile.TarInfo(name=name)
    member.size = len(raw)
    archive.addfile(member, io.BytesIO(raw))


def sample_frame_record(
    *,
    label: str = "person",
    second_label: str = "chair",
    component_inside_host: bool = False,
    stale_path: str = "/cloud/old.png",
) -> dict:
    second_box = [2, 2, 10, 15] if component_inside_host else [25, 5, 45, 35]
    return {
        "image_path": stale_path,
        "objects": {
            "image_path": stale_path,
            "boxes": [[0, 0, 20, 30], second_box],
            "labels": [label, second_label],
            "scores": [0.95, 0.9],
            "stage2_empty": False,
            "stage3_skipped": False,
            "record_list": [],
            "record_dict": [],
        },
        "hierarchy": {
            "layer1_nodes": [label, second_label],
            "layer2_mapping": {},
            "layer3_mapping": {},
        },
        "attributes": {"0": "Person standing near object", "1": second_label},
        "relations": [
            {
                "idx": [0, 1],
                "subject_label": label,
                "object_label": second_label,
                "predicate": "near",
            }
        ],
    }


def make_archive_and_video(
    root: Path,
    *,
    frame_statuses: list[tuple[str, str]],
    local_id: int = 1,
    video_id: str = "100",
    labels: dict[str, str] | None = None,
    second_labels: dict[str, str] | None = None,
    component_inside_host_frames: set[str] | None = None,
    include_conflicting_aggregate: bool = True,
) -> tuple[Path, dict]:
    hsgg_root = f"hsgg_every30_first_half_default_hsgg/{local_id}_{video_id}"
    frame_root = f"frames_every30_first_half/{video_id}"
    archive_path = root / "hsgg.tar.gz"
    completed = sum(status == "completed" for _, status in frame_statuses)
    skipped = sum(status == "skipped_no_scene_graph" for _, status in frame_statuses)
    run_manifest = {
        "total_images": len(frame_statuses),
        "completed_frames": completed,
        "skipped_no_scene_graph_frames": skipped,
        "error_frames": 0,
        "frames": {
            frame_name: {"status": status, "image_path": "/cloud/stale.png"}
            for frame_name, status in frame_statuses
        },
    }
    with tarfile.open(archive_path, "w:gz") as archive:
        add_json_member(archive, f"{hsgg_root}/run_manifest.json", run_manifest)
        if include_conflicting_aggregate:
            add_json_member(
                archive,
                f"{hsgg_root}/objects.json",
                {"malicious_aggregate_frame": {"labels": ["wrong-source"]}},
            )
        for frame_name, status in frame_statuses:
            if status != "completed":
                continue
            add_json_member(
                archive,
                f"{hsgg_root}/frame_scene_graphs/{frame_name}.json",
                sample_frame_record(
                    label=(labels or {}).get(frame_name, "person"),
                    second_label=(second_labels or {}).get(frame_name, "chair"),
                    component_inside_host=frame_name
                    in (component_inside_host_frames or set()),
                ),
            )
    frame_archive_path = root / "frames.tar.gz"
    with tarfile.open(frame_archive_path, "w:gz") as archive:
        for frame_name, _ in frame_statuses:
            add_png_member(archive, f"{frame_root}/{frame_name}.png")
    timestamps = [float(frame_name.rsplit("_", 1)[1][:-1]) for frame_name, _ in frame_statuses]
    video = {
        "episode_key": f"nextqa-val/{video_id}",
        "official_video_id": video_id,
        "vidor_id": f"0001/{video_id}",
        "local_sequential_id": local_id,
        "archive_half": "first",
        "hsgg_archive": str(archive_path.resolve()),
        "hsgg_archive_root": hsgg_root,
        "frame_archive": str(frame_archive_path.resolve()),
        "frame_archive_root": frame_root,
        "frame_member_template": f"{frame_root}/{{frame_filename}}",
        "hsgg_record_source": "frame_scene_graphs_members",
        "hsgg_generation_pass": "test",
        "hsgg_frame_count": completed,
        "sampled_image_count": len(frame_statuses),
        "skipped_frame_count": skipped,
        "start_timestamp_seconds": timestamps[0],
        "end_timestamp_seconds": timestamps[-1],
        "official_frame_count": 100,
        "inferred_source_fps": 29.97,
        "median_sample_interval_seconds": None,
        "timestamp_consistency_status": "passed",
    }
    return archive_path, video
