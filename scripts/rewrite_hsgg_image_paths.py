from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REWRITE_JSON_FILES = {
    "objects.json",
    "scene_graphs.json",
    "run_manifest.json",
}


def image_name_from_path(path_value: str) -> str:
    return Path(path_value.replace("\\", "/")).name


def collect_images(images_dir: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images[path.name] = path.resolve()
    return images


def rewrite_image_paths(value: Any, images_by_name: dict[str, Path]) -> tuple[Any, int, list[str]]:
    if isinstance(value, dict):
        changed = 0
        missing: list[str] = []
        rewritten: dict[str, Any] = {}
        for key, item in value.items():
            if key == "image_path" and isinstance(item, str):
                image_name = image_name_from_path(item)
                local_path = images_by_name.get(image_name)
                if local_path is None:
                    missing.append(image_name)
                    rewritten[key] = item
                else:
                    rewritten[key] = str(local_path)
                    changed += int(item != str(local_path))
                continue

            new_item, item_changed, item_missing = rewrite_image_paths(item, images_by_name)
            rewritten[key] = new_item
            changed += item_changed
            missing.extend(item_missing)
        return rewritten, changed, missing

    if isinstance(value, list):
        changed = 0
        missing: list[str] = []
        rewritten_items: list[Any] = []
        for item in value:
            new_item, item_changed, item_missing = rewrite_image_paths(item, images_by_name)
            rewritten_items.append(new_item)
            changed += item_changed
            missing.extend(item_missing)
        return rewritten_items, changed, missing

    return value, 0, []


def json_files_to_rewrite(output_dir: Path) -> list[Path]:
    paths = [output_dir / name for name in REWRITE_JSON_FILES]
    frame_dir = output_dir / "frame_scene_graphs"
    if frame_dir.exists():
        paths.extend(sorted(frame_dir.glob("*.json")))
    return [path for path in paths if path.exists()]


def copy_hsgg_dir(source_dir: Path, output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(
                f"Output directory already exists: {output_dir}. "
                "Pass --force to replace it."
            )
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an HSGG output directory and rewrite image_path values to point "
            "at a local image directory."
        )
    )
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dir = args.source_dir.resolve()
    images_dir = args.images_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.exists():
        raise SystemExit(f"Source directory does not exist: {source_dir}")
    if not images_dir.exists():
        raise SystemExit(f"Images directory does not exist: {images_dir}")

    images_by_name = collect_images(images_dir)
    if not images_by_name:
        raise SystemExit(f"No images found in: {images_dir}")

    copy_hsgg_dir(source_dir, output_dir, force=args.force)

    total_changed = 0
    missing_names: list[str] = []
    rewritten_files = 0
    for path in json_files_to_rewrite(output_dir):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        rewritten, changed, missing = rewrite_image_paths(data, images_by_name)
        if changed:
            path.write_text(
                json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            rewritten_files += 1
        total_changed += changed
        missing_names.extend(missing)

    unique_missing = sorted(set(name for name in missing_names if name))
    print(f"source_dir={source_dir}")
    print(f"output_dir={output_dir}")
    print(f"images_dir={images_dir}")
    print(f"local_images={len(images_by_name)}")
    print(f"rewritten_files={rewritten_files}")
    print(f"rewritten_image_paths={total_changed}")
    print(f"missing_images={len(unique_missing)}")
    if unique_missing:
        print(f"missing_sample={unique_missing[:10]!r}")
        raise SystemExit("image path rewrite failed")
    print("rewrite_status=OK")


if __name__ == "__main__":
    main()
