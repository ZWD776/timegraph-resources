from __future__ import annotations

import argparse
import json
from pathlib import Path


def image_name_from_path(path_value: str) -> str:
    return Path(path_value.replace("\\", "/")).name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify local NaVQA images match image_path entries in HSGG objects.json."
    )
    parser.add_argument("hsgg_output_dir", type=Path)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--include-manifest",
        action="store_true",
        help="Also verify images listed in run_manifest.json, including skipped/error frames.",
    )
    args = parser.parse_args()

    objects_path = args.hsgg_output_dir / "objects.json"
    objects = json.loads(objects_path.read_text(encoding="utf-8-sig"))
    required_names = [
        image_name_from_path(frame_payload["image_path"])
        for frame_payload in objects.values()
        if isinstance(frame_payload, dict) and frame_payload.get("image_path")
    ]
    if args.include_manifest:
        manifest_path = args.hsgg_output_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            frames = manifest.get("frames", {})
            if isinstance(frames, dict):
                required_names.extend(
                    image_name_from_path(frame_payload["image_path"])
                    for frame_payload in frames.values()
                    if isinstance(frame_payload, dict) and frame_payload.get("image_path")
                )
    unique_required = sorted(set(required_names))
    local_names = {
        path.name for path in args.images_dir.glob("*.jpg")
    } | {
        path.name for path in args.images_dir.glob("*.jpeg")
    }

    missing = [name for name in unique_required if name not in local_names]
    extra = [name for name in sorted(local_names) if name not in set(unique_required)]

    print(f"required_unique_images={len(unique_required)}")
    print(f"local_images={len(local_names)}")
    print(f"missing={len(missing)}")
    print(f"extra={len(extra)}")
    if missing:
        print("missing_sample=" + repr(missing[:10]))
    if extra:
        print("extra_sample=" + repr(extra[:10]))

    if missing:
        raise SystemExit("image verification failed")
    print("image_verification=OK")


if __name__ == "__main__":
    main()
