from __future__ import annotations

import argparse
import json
from pathlib import Path

from hsgg_standard_bundle import write_hsgg_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Adapt an aggregated scene_graphs.json file into the standard "
            "objects/attributes/relations/hierarchy_mapping JSON tables."
        )
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to the aggregated scene_graphs.json file.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory containing frame PNGs. Defaults to <input_json_parent>/frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write standard JSON tables. Defaults to <input_json_parent>/json.",
    )
    parser.add_argument(
        "--write-rewritten-scene-graphs",
        action="store_true",
        help="Also emit a rewritten aggregated scene_graphs file with normalized image_path values.",
    )
    return parser


def load_scene_graphs(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_image_path(image_path: str, frames_dir: Path) -> Path:
    filename = Path(image_path).name
    normalized = (frames_dir / filename).resolve()
    if not normalized.exists():
        raise FileNotFoundError(
            f"Frame file referenced by scene graph does not exist: {normalized}"
        )
    return normalized


def adapt_scene_graphs(
    scene_graphs: list[dict],
    *,
    frames_dir: Path,
) -> tuple[dict, dict, dict, dict, list[dict]]:
    objects_table: dict[str, dict] = {}
    attributes_table: dict[str, dict] = {}
    relations_table: dict[str, list] = {}
    hierarchy_table: dict[str, dict] = {}
    rewritten_scene_graphs: list[dict] = []

    for item in scene_graphs:
        normalized_path = normalize_image_path(item["image_path"], frames_dir)
        frame_name = normalized_path.stem

        objects_payload = dict(item["objects"])
        objects_payload["image_path"] = str(normalized_path)

        attributes_payload = dict(item["attributes"])
        relations_payload = list(item["relations"])
        hierarchy_payload = dict(item["hierarchy"])

        rewritten_item = {
            "image_path": str(normalized_path),
            "objects": objects_payload,
            "attributes": attributes_payload,
            "relations": relations_payload,
            "hierarchy": hierarchy_payload,
        }

        objects_table[frame_name] = objects_payload
        attributes_table[frame_name] = attributes_payload
        relations_table[frame_name] = relations_payload
        hierarchy_table[frame_name] = hierarchy_payload
        rewritten_scene_graphs.append(rewritten_item)

    return (
        objects_table,
        attributes_table,
        relations_table,
        hierarchy_table,
        rewritten_scene_graphs,
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()

    input_json = args.input_json.resolve()
    input_parent = input_json.parent
    frames_dir = (args.frames_dir or (input_parent / "frames")).resolve()
    output_dir = (args.output_dir or (input_parent / "json")).resolve()

    scene_graphs = load_scene_graphs(input_json)
    (
        objects_table,
        attributes_table,
        relations_table,
        hierarchy_table,
        rewritten_scene_graphs,
    ) = adapt_scene_graphs(
        scene_graphs,
        frames_dir=frames_dir,
    )

    write_json(output_dir / "objects.json", objects_table)
    write_json(output_dir / "attributes.json", attributes_table)
    write_json(output_dir / "relations.json", relations_table)
    write_json(output_dir / "hierarchy_mapping.json", hierarchy_table)
    write_hsgg_bundle(
        output_json_dir=output_dir,
        objects_table=objects_table,
        attributes_table=attributes_table,
        relations_table=relations_table,
        hierarchy_table=hierarchy_table,
        write_legacy_rewritten_scene_graphs=args.write_rewritten_scene_graphs,
    )

    print(f"Adapted {len(scene_graphs)} frames into standard JSON tables.")
    print(f"Frames dir: {frames_dir}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
