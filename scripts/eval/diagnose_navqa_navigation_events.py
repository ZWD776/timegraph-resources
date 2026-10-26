from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.models.canonical_graph import CanonicalSceneNode, CanonicalTemporalMemoryGraph


INDOOR_CUES = {
    "atrium",
    "backpack",
    "bench",
    "ceiling",
    "chair",
    "couch",
    "couches",
    "elevator",
    "floor",
    "hallway",
    "indoor",
    "indoors",
    "lobby",
    "room",
    "sofa",
    "stair",
    "stairs",
    "wall",
}
OUTDOOR_CUES = {
    "bicycle",
    "bush",
    "car",
    "curb",
    "fire hydrant",
    "hydrant",
    "lamppost",
    "loading zone",
    "outdoor",
    "outside",
    "pavement",
    "pole",
    "road",
    "sidewalk",
    "sky",
    "street",
    "streetlight",
    "tree",
    "truck",
    "vehicle",
}
DOORWAY_CUES = {
    "building",
    "door",
    "doors",
    "doorway",
    "elevator door",
    "entrance",
    "exit",
    "steps",
    "window",
    "windows",
}


def label_set(scene: CanonicalSceneNode) -> set[str]:
    labels: set[str] = set()
    fields = [
        scene.label,
        scene.scene_kind_hint,
        *scene.representative_static_labels,
        *scene.representative_dynamic_labels,
        *scene.entry_static_labels,
        *scene.exit_static_labels,
    ]
    for value in fields:
        text = str(value or "").strip().lower()
        if text:
            labels.add(text)
    return labels


def cue_score(labels: set[str], cues: set[str]) -> tuple[int, list[str]]:
    matched = sorted(cue for cue in cues if any(cue in label for label in labels))
    return len(matched), matched


def classify_scene(scene: CanonicalSceneNode) -> dict[str, Any]:
    labels = label_set(scene)
    indoor_score, indoor_matches = cue_score(labels, INDOOR_CUES)
    outdoor_score, outdoor_matches = cue_score(labels, OUTDOOR_CUES)
    doorway_score, doorway_matches = cue_score(labels, DOORWAY_CUES)
    if indoor_score >= outdoor_score + 2 and indoor_score >= doorway_score:
        coarse = "indoor"
    elif outdoor_score >= indoor_score + 2 and outdoor_score >= doorway_score:
        coarse = "outdoor"
    elif doorway_score >= 2:
        coarse = "doorway"
    elif indoor_score > outdoor_score:
        coarse = "maybe_indoor"
    elif outdoor_score > indoor_score:
        coarse = "maybe_outdoor"
    else:
        coarse = "unknown"
    return {
        "scene_id": scene.scene_id,
        "scene_index": scene.scene_index,
        "label": scene.label,
        "start_frame": scene.time_range.start_frame,
        "end_frame": scene.time_range.end_frame,
        "start_ts": scene.time_range.start_ts,
        "end_ts": scene.time_range.end_ts,
        "coarse_location": coarse,
        "indoor_score": indoor_score,
        "doorway_score": doorway_score,
        "outdoor_score": outdoor_score,
        "indoor_matches": indoor_matches,
        "doorway_matches": doorway_matches,
        "outdoor_matches": outdoor_matches,
        "representative_static_labels": list(scene.representative_static_labels),
        "representative_dynamic_labels": list(scene.representative_dynamic_labels),
    }


def find_transition_candidates(scene_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for i in range(1, len(scene_rows) - 1):
        prev_scene = scene_rows[i - 1]
        current = scene_rows[i]
        next_scene = scene_rows[i + 1]
        triple = [
            prev_scene["coarse_location"],
            current["coarse_location"],
            next_scene["coarse_location"],
        ]
        prev_outdoor = "outdoor" in triple[0]
        prev_indoor = "indoor" in triple[0]
        next_outdoor = "outdoor" in triple[2]
        next_indoor = "indoor" in triple[2]
        is_doorway = current["coarse_location"] == "doorway" or current["doorway_score"] >= 2
        event_type = None
        if prev_outdoor and is_doorway and next_indoor:
            event_type = "enter_building_candidate"
        elif prev_indoor and is_doorway and next_outdoor:
            event_type = "leave_building_candidate"
        if event_type is None:
            continue
        confidence = (
            prev_scene["outdoor_score" if prev_outdoor else "indoor_score"]
            + current["doorway_score"]
            + next_scene["indoor_score" if next_indoor else "outdoor_score"]
        )
        candidates.append(
            {
                "event_type": event_type,
                "confidence": confidence,
                "estimated_frame_range": [
                    current["start_frame"],
                    current["end_frame"],
                ],
                "support_chain": [
                    _compact_scene(prev_scene),
                    _compact_scene(current),
                    _compact_scene(next_scene),
                ],
            }
        )
    return sorted(candidates, key=lambda item: item["confidence"], reverse=True)


def _compact_scene(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": row["scene_id"],
        "label": row["label"],
        "frames": [row["start_frame"], row["end_frame"]],
        "coarse_location": row["coarse_location"],
        "scores": {
            "indoor": row["indoor_score"],
            "doorway": row["doorway_score"],
            "outdoor": row["outdoor_score"],
        },
        "matches": {
            "indoor": row["indoor_matches"],
            "doorway": row["doorway_matches"],
            "outdoor": row["outdoor_matches"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether canonical graph scenes support NaVQA navigation events."
    )
    parser.add_argument("canonical_graph_json", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".scratch/navqa_navigation_event_diagnostics.json"),
    )
    args = parser.parse_args()

    graph = CanonicalTemporalMemoryGraph.model_validate_json(
        args.canonical_graph_json.read_text(encoding="utf-8")
    )
    scene_rows = [
        classify_scene(scene)
        for scene in sorted(
            graph.scene_nodes,
            key=lambda item: (
                item.time_range.start_frame if item.time_range.start_frame is not None else 10**12,
                item.scene_id,
            ),
        )
    ]
    candidates = find_transition_candidates(scene_rows)
    payload = {
        "source_graph": str(args.canonical_graph_json),
        "scene_count": len(scene_rows),
        "location_counts": {
            label: sum(1 for row in scene_rows if row["coarse_location"] == label)
            for label in sorted({row["coarse_location"] for row in scene_rows})
        },
        "transition_candidates": candidates,
        "scene_sequence": scene_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"scene_count={len(scene_rows)}")
    print(f"location_counts={payload['location_counts']}")
    print(f"transition_candidates={len(candidates)}")
    for item in candidates[:10]:
        print(json.dumps(item, ensure_ascii=False))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
