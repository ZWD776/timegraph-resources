from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.config.config import load_domain_config
from vlkg.core.models.temporal_graph import BoundingBox
from vlkg.core.temporal.pathing import default_cache_db_path, resolve_repo_input_path
from vlkg.core.transformers import JSONToGraphTransformer
from vlkg.core.transformers.json_to_graph import _FrameObjectInput


REQUIRED_FILES = (
    "objects.json",
    "attributes.json",
    "relations.json",
    "hierarchy_mapping.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether a standard JSON directory is ready for temporal graph construction."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the json/ folder or the standard JSON files directly.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("default.yaml"),
        help="Domain profile YAML relative to configs/ or an absolute path.",
    )
    parser.add_argument(
        "--frame-time-index",
        type=Path,
        default=None,
        help="Optional frame-time index JSON used for timestamp parsing coverage checks.",
    )
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=default_cache_db_path(),
        help="Embedding cache database to inspect.",
    )
    parser.add_argument(
        "--skip-cache-check",
        action="store_true",
        help="Skip exact embedding cache hit/miss counting.",
    )
    return parser


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_json_dir(input_dir: Path) -> Path:
    input_dir = input_dir.resolve()
    return input_dir / "json" if (input_dir / "json").exists() else input_dir


def load_frame_time_index(path: Path | None) -> dict:
    if path is None:
        return {}
    resolved = resolve_repo_input_path(path)
    if not resolved.exists():
        raise SystemExit(f"Missing frame-time-index: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8-sig"))


def count_cache_hits(
    *,
    cache_db: Path,
    requests: list[tuple[str, str]],
) -> tuple[int, int]:
    if not cache_db.exists():
        return 0, len(requests)
    connection = sqlite3.connect(cache_db)
    try:
        hits = 0
        for image_path, bbox_key in requests:
            row = connection.execute(
                """
                SELECT 1
                FROM embedding_cache
                WHERE image_path = ? AND bbox_key = ?
                LIMIT 1
                """,
                (image_path, bbox_key),
            ).fetchone()
            if row is not None:
                hits += 1
        return hits, len(requests) - hits
    finally:
        connection.close()


def main() -> None:
    args = build_parser().parse_args()
    input_dir = resolve_repo_input_path(args.input_dir)
    json_dir = resolve_json_dir(input_dir)
    config = load_domain_config(args.config, config_dir=REPO_ROOT / "configs")
    frame_time_index_payload = load_frame_time_index(args.frame_time_index)

    print(f"input_dir={input_dir.resolve()}")
    print(f"json_dir={json_dir}")

    missing = [name for name in REQUIRED_FILES if not (json_dir / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    objects = load_json(json_dir / "objects.json")
    attributes = load_json(json_dir / "attributes.json")
    relations = load_json(json_dir / "relations.json")
    hierarchy = load_json(json_dir / "hierarchy_mapping.json")

    object_keys = set(objects.keys())
    attr_keys = set(attributes.keys())
    relation_keys = set(relations.keys())
    hierarchy_keys = set(hierarchy.keys())

    print(f"objects_frames={len(object_keys)}")
    print(f"attributes_frames={len(attr_keys)}")
    print(f"relations_frames={len(relation_keys)}")
    print(f"hierarchy_frames={len(hierarchy_keys)}")

    if not (object_keys == attr_keys == relation_keys == hierarchy_keys):
        raise SystemExit("Frame keys are inconsistent across the four standard JSON tables.")

    transformer = JSONToGraphTransformer(
        input_dir=input_dir,
        config=config,
        frame_time_index_path=(
            resolve_repo_input_path(args.frame_time_index)
            if args.frame_time_index is not None
            else None
        ),
    )
    metas = transformer._sorted_frame_metas(object_keys)
    print(f"parsed_frames={len(metas)}")
    print(f"first_frame={metas[0].frame_name}")
    print(f"last_frame={metas[-1].frame_name}")
    if frame_time_index_payload:
        indexed_frames = frame_time_index_payload.get("frames", [])
        print(f"frame_time_index_frames={len(indexed_frames)}")
        graph_frame_indices = {
            int(item.get("graph_frame_index", item.get("frame_index")))
            for item in indexed_frames
            if isinstance(item, dict)
            and item.get("graph_frame_index", item.get("frame_index")) is not None
        }
        parsed_frame_indices = {meta.frame_index for meta in metas}
        missing_time_indices = sorted(parsed_frame_indices - graph_frame_indices)
        print(f"frame_time_index_missing_graph_frames={len(missing_time_indices)}")
        if missing_time_indices:
            print(f"frame_time_index_missing_sample={missing_time_indices[:10]}")

    bad_paths: list[str] = []
    image_paths: list[str] = []
    for frame_name in sorted(object_keys):
        image_path = Path(objects[frame_name]["image_path"])
        image_paths.append(str(image_path))
        if not image_path.exists():
            bad_paths.append(str(image_path))
            if len(bad_paths) >= 5:
                break

    if bad_paths:
        raise SystemExit(f"Found non-existent image_path entries, sample={bad_paths}")
    print(f"image_paths={len(image_paths)}")
    print(f"image_paths_distinct={len(set(image_paths))}")

    sample_frame = metas[0].frame_name
    sample_object_count = len(objects[sample_frame]["labels"])
    sample_attribute_count = len(attributes[sample_frame])
    sample_relation_count = len(relations[sample_frame])
    frame_object_counts = [len(objects[meta.frame_name].get("labels", [])) for meta in metas]
    label_counts = Counter(
        transformer._normalize_label(label)
        for meta in metas
        for label in objects[meta.frame_name].get("labels", [])
    )

    embedding_requests: list[tuple[str, str]] = []
    role_counts: Counter[str] = Counter()
    for frame_rank, meta in enumerate(metas):
        frame_payload = objects[meta.frame_name]
        frame_attributes = attributes.get(meta.frame_name, {})
        hierarchy_entry = hierarchy.get(meta.frame_name, {})
        parent_lookup = transformer._build_parent_lookup(hierarchy_entry)
        prev_label_boxes = transformer._neighbor_label_boxes(
            frame_metas=metas,
            objects_table=objects,
            frame_rank=frame_rank - 1,
        )
        next_label_boxes = transformer._neighbor_label_boxes(
            frame_metas=metas,
            objects_table=objects,
            frame_rank=frame_rank + 1,
        )
        frame_box_data: list[_FrameObjectInput] = []
        for object_index, (box_values, label, score) in enumerate(
            zip(
                frame_payload.get("boxes", []),
                frame_payload.get("labels", []),
                frame_payload.get("scores", []),
                strict=True,
            )
        ):
            normalized_label = transformer._normalize_label(label)
            min_score = transformer.label_min_scores.get(normalized_label, 0.0)
            if float(score) < min_score:
                continue
            bbox = BoundingBox(
                x1=float(box_values[0]),
                y1=float(box_values[1]),
                x2=float(box_values[2]),
                y2=float(box_values[3]),
            )
            parent_label = parent_lookup.get(normalized_label)
            role_decision = transformer._classify_frame_object_role(
                label=normalized_label,
                parent_label=parent_label,
                bbox=bbox,
                hierarchy_entry=hierarchy_entry,
                prev_label_boxes=prev_label_boxes,
                next_label_boxes=next_label_boxes,
            )
            frame_box_data.append(
                _FrameObjectInput(
                    object_index=object_index,
                    bbox=bbox,
                    label=normalized_label,
                    score=float(score),
                    state=transformer._normalize_state_value(
                        frame_attributes.get(str(object_index))
                    ),
                    parent_label=parent_label,
                    is_static=role_decision.is_static,
                    is_noise=role_decision.role == "noise",
                    node_role=role_decision.role,
                )
            )

        if transformer.config.association.identity_role_strategy != "tracklet":
            frame_box_data = transformer._fold_component_states(frame_box_data)
        kept_items = [
            item
            for item in frame_box_data
            if not item.folded_into_host and not item.skip_entity_creation
        ]
        for item in kept_items:
            role_counts[item.node_role] += 1
            embedding_requests.append(
                (
                    str(Path(frame_payload["image_path"])),
                    transformer.embedding_cache.bbox_to_key(item.bbox),
                )
            )

    print(f"sample_frame={sample_frame}")
    print(f"sample_object_count={sample_object_count}")
    print(f"sample_attribute_count={sample_attribute_count}")
    print(f"sample_relation_count={sample_relation_count}")
    print(f"objects_total={sum(frame_object_counts)}")
    print(f"objects_min={min(frame_object_counts)}")
    print(f"objects_max={max(frame_object_counts)}")
    print(f"objects_avg={sum(frame_object_counts) / max(len(frame_object_counts), 1):.2f}")
    print(f"top_labels={label_counts.most_common(20)}")
    print(f"role_counts={dict(role_counts)}")
    print(f"embedding_requests_estimate={len(embedding_requests)}")
    if not args.skip_cache_check:
        hits, misses = count_cache_hits(
            cache_db=resolve_repo_input_path(args.cache_db),
            requests=embedding_requests,
        )
        print(f"embedding_cache_db={resolve_repo_input_path(args.cache_db)}")
        print(f"embedding_cache_hits={hits}")
        print(f"embedding_cache_misses={misses}")
        hit_rate = hits / len(embedding_requests) if embedding_requests else 0.0
        print(f"embedding_cache_hit_rate={hit_rate:.4f}")
    print("preflight_status=READY")


if __name__ == "__main__":
    main()
