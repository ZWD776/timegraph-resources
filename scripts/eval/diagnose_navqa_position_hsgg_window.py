from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "closest",
    "did",
    "for",
    "front",
    "in",
    "is",
    "nearest",
    "not",
    "of",
    "place",
    "see",
    "sit",
    "the",
    "there",
    "to",
    "where",
    "with",
    "you",
}

QUESTION_ALIASES = {
    "closest place to sit": [
        "bench",
        "chair",
        "seat",
        "seating",
        "sofa",
        "couch",
    ],
    "yellow police call pole": [
        "yellow police call pole",
        "police call pole",
        "call pole",
        "emergency phone",
        "yellow pole",
        "police",
    ],
    "statue in front of the bill and melinda gates computer science building": [
        "statue",
        "sculpture",
    ],
    "nearest recycling bin": [
        "recycling bin",
        "recycle bin",
        "trash can",
        "trash",
        "bin",
    ],
    "sculpture with the hanging boats": [
        "sculpture",
        "boat",
        "boats",
    ],
    "closest elevators": [
        "elevator",
        "elevators",
        "lift",
    ],
    "nearest bike rack that is not empty": [
        "bike rack",
        "bicycle rack",
        "bicycle",
        "bike",
        "handlebars",
    ],
    "red fire hydrant": [
        "red fire hydrant",
        "fire hydrant",
        "hydrant",
    ],
}


@dataclass(frozen=True)
class FrameRecord:
    sampled_index: int
    graph_frame_index: int
    source_frame: int
    image_name: str
    timestamp_unix: float
    elapsed_sec: float


def normalize_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_frame_index(path: Path) -> list[FrameRecord]:
    rows: list[FrameRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                FrameRecord(
                    sampled_index=int(row["sampled_index"]),
                    graph_frame_index=int(row["graph_frame_index"]),
                    source_frame=int(row["source_frame"]),
                    image_name=row["image_name"],
                    timestamp_unix=float(row["timestamp_unix"]),
                    elapsed_sec=float(row["elapsed_sec_from_sequence_start"]),
                )
            )
    return rows


def frame_key_from_path(path_text: str) -> str:
    return Path(path_text.replace("\\", "/")).name


def iter_graph_text(scene_graph: dict) -> Iterable[str]:
    objects = scene_graph.get("objects") or {}
    for label in objects.get("labels") or []:
        yield str(label)
    for attr in (scene_graph.get("attributes") or {}).values():
        yield str(attr)
    for rel in scene_graph.get("relations") or []:
        if not isinstance(rel, dict):
            yield str(rel)
            continue
        for key in ("subject_label", "object_label", "predicate", "raw_output"):
            yield str(rel.get(key) or "")
        for candidate in rel.get("all_candidates") or []:
            yield str(candidate)


def load_hsgg_by_frame(scene_graphs_path: Path) -> dict[str, dict]:
    with scene_graphs_path.open(encoding="utf-8") as f:
        scene_graphs = json.load(f)
    by_frame: dict[str, dict] = {}
    for scene_graph in scene_graphs:
        image_name = frame_key_from_path(scene_graph.get("image_path", ""))
        if image_name:
            by_frame[image_name] = scene_graph
    return by_frame


def load_position_reference(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    refs: dict[str, float] = {}
    for item in payload.get("position_answers") or []:
        uuid = str(item.get("uuid") or "")
        timestamp = item.get("answer_timestamp_unix")
        if uuid and isinstance(timestamp, (int, float)):
            refs[uuid] = float(timestamp)
    return refs


def load_position_questions(
    path: Path,
    timestamp_by_uuid: dict[str, float],
    seq_id: str,
) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    questions = [
        row
        for row in rows
        if normalize_text(row.get("Type \n(binary, position, time, text)")) == "position"
        and str(row.get("Seq ID") or "") == seq_id
    ]
    for row in questions:
        uuid = str(row.get("UUID") or "")
        if uuid in timestamp_by_uuid:
            row["_answer_timestamp_unix"] = str(timestamp_by_uuid[uuid])
    return questions


def aliases_for_question(question: str) -> list[str]:
    normalized = normalize_text(question)
    aliases: list[str] = []
    for key, values in QUESTION_ALIASES.items():
        if key in normalized:
            aliases.extend(values)
    if not aliases:
        tokens = [
            token
            for token in normalized.split()
            if len(token) >= 3 and token not in STOPWORDS
        ]
        aliases.extend(tokens)
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        alias = normalize_text(alias)
        if alias and alias not in seen:
            deduped.append(alias)
            seen.add(alias)
    return deduped


def find_nearest_frame(frames: list[FrameRecord], timestamp_unix: float) -> FrameRecord:
    return min(frames, key=lambda frame: abs(frame.timestamp_unix - timestamp_unix))


def nearby_frames(
    frames: list[FrameRecord],
    timestamp_unix: float,
    window_sec: float,
) -> list[FrameRecord]:
    return [
        frame
        for frame in frames
        if abs(frame.timestamp_unix - timestamp_unix) <= window_sec
    ]


def hit_aliases(scene_graph: dict | None, aliases: list[str]) -> tuple[list[str], str, list[str]]:
    if scene_graph is None:
        return [], "", []
    text_items = [normalize_text(item) for item in iter_graph_text(scene_graph)]
    joined = " | ".join(item for item in text_items if item)
    hits = [alias for alias in aliases if alias in joined]
    labels = [str(label) for label in (scene_graph.get("objects") or {}).get("labels") or []]
    return hits, joined, labels


def summarize_labels(labels: Iterable[str], limit: int = 12) -> str:
    counter = Counter(normalize_text(label) for label in labels if normalize_text(label))
    return ", ".join(f"{label}:{count}" for label, count in counter.most_common(limit))


def build_report(args: argparse.Namespace) -> tuple[list[dict], str]:
    frames = load_frame_index(Path(args.frame_index))
    hsgg_by_frame = load_hsgg_by_frame(Path(args.scene_graphs))
    timestamp_by_uuid = load_position_reference(Path(args.position_reference))
    questions = load_position_questions(Path(args.qa_csv), timestamp_by_uuid, args.seq_id)
    rows: list[dict] = []
    md_lines = [
        "# NaVQA Position HSGG Window Diagnosis",
        "",
        f"- position questions: {len(questions)}",
        f"- HSGG frames loaded: {len(hsgg_by_frame)}",
        f"- windows: +/-{args.near_window_sec:g}s and +/-{args.wide_window_sec:g}s",
        "",
        "| # | UUID | Question | Official time | Nearest frame | +/-near hit | +/-wide hit | Hit aliases | Nearby labels |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]

    for index, question_row in enumerate(questions, start=1):
        question = question_row["Question"]
        timestamp = float(question_row["_answer_timestamp_unix"])
        aliases = aliases_for_question(question)
        nearest = find_nearest_frame(frames, timestamp)
        near = nearby_frames(frames, timestamp, args.near_window_sec)
        wide = nearby_frames(frames, timestamp, args.wide_window_sec)

        near_hits: set[str] = set()
        wide_hits: set[str] = set()
        near_labels: list[str] = []
        wide_labels: list[str] = []
        hit_frames: list[str] = []

        for frame in wide:
            scene_graph = hsgg_by_frame.get(frame.image_name)
            hits, _, labels = hit_aliases(scene_graph, aliases)
            if abs(frame.timestamp_unix - timestamp) <= args.near_window_sec:
                near_hits.update(hits)
                near_labels.extend(labels)
            wide_hits.update(hits)
            wide_labels.extend(labels)
            if hits:
                hit_frames.append(
                    f"{frame.image_name}@{frame.timestamp_unix:.3f}:"
                    + ",".join(sorted(set(hits)))
                )

        nearest_hits, _, nearest_labels = hit_aliases(
            hsgg_by_frame.get(nearest.image_name),
            aliases,
        )

        row = {
            "index": index,
            "uuid": question_row["UUID"],
            "question": question,
            "official_timestamp_unix": timestamp,
            "official_timestamp_text": question_row["Timestamp \nwith answer"],
            "completed_position_answer": question_row.get("Completed position answer", ""),
            "nearest_image_name": nearest.image_name,
            "nearest_timestamp_unix": nearest.timestamp_unix,
            "nearest_delta_sec": round(nearest.timestamp_unix - timestamp, 3),
            "aliases": "; ".join(aliases),
            "nearest_hit_aliases": "; ".join(nearest_hits),
            "near_window_hit": bool(near_hits),
            "near_window_hit_aliases": "; ".join(sorted(near_hits)),
            "wide_window_hit": bool(wide_hits),
            "wide_window_hit_aliases": "; ".join(sorted(wide_hits)),
            "hit_frames": " || ".join(hit_frames[:20]),
            "nearest_labels": summarize_labels(nearest_labels),
            "near_window_labels": summarize_labels(near_labels),
            "wide_window_labels": summarize_labels(wide_labels),
        }
        rows.append(row)
        md_lines.append(
            "| {index} | {uuid} | {question} | {time} | {nearest} ({delta:+.1f}s) | {near_hit} | {wide_hit} | {hits} | {labels} |".format(
                index=index,
                uuid=row["uuid"],
                question=question.replace("|", "\\|"),
                time=row["official_timestamp_text"],
                nearest=row["nearest_image_name"],
                delta=row["nearest_delta_sec"],
                near_hit="yes" if row["near_window_hit"] else "no",
                wide_hit="yes" if row["wide_window_hit"] else "no",
                hits=(row["wide_window_hit_aliases"] or "-").replace("|", "\\|"),
                labels=(row["near_window_labels"] or "-").replace("|", "\\|"),
            )
        )

    return rows, "\n".join(md_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-csv",
        default=None,
    )
    parser.add_argument(
        "--frame-index",
        default=None,
    )
    parser.add_argument(
        "--scene-graphs",
        default=None,
    )
    parser.add_argument(
        "--position-reference",
        default=None,
    )
    parser.add_argument(
        "--out-csv",
        default=".scratch/navqa_position_hsgg_window_diagnosis.csv",
    )
    parser.add_argument(
        "--out-md",
        default=".scratch/navqa_position_hsgg_window_diagnosis.md",
    )
    parser.add_argument("--near-window-sec", type=float, default=10.0)
    parser.add_argument("--wide-window-sec", type=float, default=30.0)
    parser.add_argument("--seq-id", default="0")
    args = parser.parse_args()
    sequence_root = Path(".scratch/navqa") / f"seq{args.seq_id}"
    sample_root = sequence_root / "cam0" / "every10"
    diagnostics_dir = sequence_root / "diagnostics"
    if args.qa_csv is None:
        args.qa_csv = str(diagnostics_dir / "official_data_completed_position.csv")
    if args.frame_index is None:
        args.frame_index = str(sample_root / "frame_time_index.csv")
    if args.scene_graphs is None:
        args.scene_graphs = str(sample_root / "hsgg_local_paths" / "scene_graphs.json")
    if args.position_reference is None:
        args.position_reference = str(diagnostics_dir / "position_reference.json")

    rows, markdown = build_report(args)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    out_md = Path(args.out_md)
    out_md.write_text(markdown, encoding="utf-8")

    near_hits = sum(1 for row in rows if row["near_window_hit"])
    wide_hits = sum(1 for row in rows if row["wide_window_hit"])
    print(f"position questions: {len(rows)}")
    print(f"+/-{args.near_window_sec:g}s hits: {near_hits}/{len(rows)}")
    print(f"+/-{args.wide_window_sec:g}s hits: {wide_hits}/{len(rows)}")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_md}")


if __name__ == "__main__":
    main()
