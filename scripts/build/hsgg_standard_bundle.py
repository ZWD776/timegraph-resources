from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


FRAME_NAME_PATTERN = re.compile(r"^frame_(?P<frame_idx>\d+)(?:_(?P<rest>.+))?$")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sort_frame_names(frame_names: list[str]) -> list[str]:
    def sort_key(frame_name: str) -> tuple[int, str]:
        match = FRAME_NAME_PATTERN.match(frame_name)
        if match:
            return int(match.group("frame_idx")), frame_name
        return 10**12, frame_name

    return sorted(frame_names, key=sort_key)


def normalize_object_payload(objects_payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(objects_payload)
    record_list = normalized.get("record_list")
    if not isinstance(record_list, list) or len(record_list) != 3:
        record_list = [
            list(record_list[0]) if isinstance(record_list, list) and len(record_list) > 0 else [],
            list(record_list[1]) if isinstance(record_list, list) and len(record_list) > 1 else [],
            list(record_list[2]) if isinstance(record_list, list) and len(record_list) > 2 else [],
        ]
    record_dict = normalized.get("record_dict")
    if not isinstance(record_dict, list) or len(record_dict) != 2:
        record_dict = [
            dict(record_dict[0]) if isinstance(record_dict, list) and len(record_dict) > 0 else {},
            dict(record_dict[1]) if isinstance(record_dict, list) and len(record_dict) > 1 else {},
        ]

    normalized["record_list"] = record_list
    normalized["record_dict"] = record_dict
    normalized["stage2_empty"] = bool(normalized.get("stage2_empty", len(record_list[1]) == 0))
    normalized["stage3_skipped"] = bool(normalized.get("stage3_skipped", len(record_list[2]) == 0))
    return normalized


def build_scene_graph_entries(
    *,
    objects_table: dict[str, Any],
    attributes_table: dict[str, Any],
    relations_table: dict[str, Any],
    hierarchy_table: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for frame_name in sort_frame_names(list(objects_table.keys())):
        objects_payload = normalize_object_payload(dict(objects_table[frame_name]))
        entry = {
            "image_path": objects_payload.get("image_path"),
            "objects": objects_payload,
            "hierarchy": dict(hierarchy_table.get(frame_name, {})),
            "attributes": dict(attributes_table.get(frame_name, {})),
            "relations": list(relations_table.get(frame_name, [])),
        }
        entries.append(entry)
    return entries


def write_hsgg_bundle(
    *,
    output_json_dir: Path,
    objects_table: dict[str, Any],
    attributes_table: dict[str, Any],
    relations_table: dict[str, Any],
    hierarchy_table: dict[str, Any],
    write_legacy_rewritten_scene_graphs: bool = False,
) -> None:
    scene_graphs = build_scene_graph_entries(
        objects_table=objects_table,
        attributes_table=attributes_table,
        relations_table=relations_table,
        hierarchy_table=hierarchy_table,
    )

    write_json(output_json_dir / "scene_graphs.json", scene_graphs)

    frame_scene_graphs_dir = output_json_dir / "frame_scene_graphs"
    frame_scene_graphs_dir.mkdir(parents=True, exist_ok=True)
    for frame_name, scene_graph in zip(
        sort_frame_names(list(objects_table.keys())),
        scene_graphs,
        strict=True,
    ):
        write_json(frame_scene_graphs_dir / f"{frame_name}.json", scene_graph)

    if write_legacy_rewritten_scene_graphs:
        write_json(output_json_dir / "scene_graphs.rewritten.json", scene_graphs)
