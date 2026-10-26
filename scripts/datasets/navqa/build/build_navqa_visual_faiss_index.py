from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlkg.core.temporal.pathing import default_cache_db_path, resolve_repo_input_path  # noqa: E402


@dataclass(frozen=True)
class SequenceSpec:
    seq_id: str
    cam: str
    stride: int | None
    sample_root: Path
    frame_time_index_json: Path | None
    hsgg_input_dir: Path | None = None
    images_dir: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an experimental object-level FAISS index for NaVQA visual grounding "
            "from existing SigLIP embeddings. This does not compute new embeddings and "
            "does not modify either frozen mainline."
        )
    )
    parser.add_argument(
        "--sequence-manifest",
        type=Path,
        default=Path("configs/datasets/navqa/sequences.json"),
        help="Sequence manifest with sample_root entries.",
    )
    parser.add_argument(
        "--cache-db",
        type=Path,
        default=default_cache_db_path(),
        help="SQLite embedding_cache.db produced by the SigLIP cache.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/navqa/runs/new_runs/payload/indexes"),
        help="Output directory for index.faiss, metadata.jsonl, and manifest.json.",
    )
    parser.add_argument(
        "--index-name",
        default="object_siglip_flat_ip",
        help="Basename used for output files.",
    )
    parser.add_argument(
        "--seq-id",
        action="append",
        default=None,
        help="Optional sequence id filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Number of vectors to add to FAISS per batch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing index/metadata files in the output directory.",
    )
    return parser


def repo_relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_sequences(manifest_path: Path, seq_filter: set[str] | None) -> list[SequenceSpec]:
    manifest_path = resolve_repo_input_path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequences: list[SequenceSpec] = []
    for item in payload.get("sequences") or []:
        seq_id = str(item["seq_id"])
        if seq_filter is not None and seq_id not in seq_filter:
            continue
        sample_root = resolve_repo_input_path(Path(item["sample_root"]))
        frame_time_path = item.get("frame_time_index_json")
        sequences.append(
            SequenceSpec(
                seq_id=seq_id,
                cam=str(item.get("cam", "0")),
                stride=item.get("stride"),
                sample_root=sample_root,
                hsgg_input_dir=resolve_repo_input_path(item.get("hsgg_input_dir") or sample_root / "hsgg_local_paths"),
                images_dir=resolve_repo_input_path(item.get("images_dir") or sample_root / "images"),
                frame_time_index_json=(
                    resolve_repo_input_path(Path(frame_time_path))
                    if frame_time_path
                    else None
                ),
            )
        )
    if not sequences:
        raise ValueError("No sequences selected from manifest.")
    return sequences


def load_frame_index(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return {}
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_name = str(row.get("image_name") or "")
        if image_name:
            by_name[image_name] = row
            by_name[Path(image_name).stem] = row
    return by_name


def bbox_key(box: Iterable[Any]) -> str | None:
    values = list(box)
    if len(values) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(values[index]) for index in range(4))
    except (TypeError, ValueError):
        return None
    return f"{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}"


def bbox_values(box: Iterable[Any]) -> list[float] | None:
    values = list(box)
    if len(values) < 4:
        return None
    try:
        return [float(values[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None


def image_name_from_payload(frame_key: str, payload: dict[str, Any]) -> str:
    image_path = str(payload.get("image_path") or "")
    if image_path:
        return Path(image_path.replace("\\", "/")).name
    if Path(frame_key).suffix:
        return Path(frame_key).name
    return f"{frame_key}.jpg"


def actual_image_path(
    *,
    images_dir: Path,
    frame_key: str,
    payload: dict[str, Any],
) -> Path:
    image_name = image_name_from_payload(frame_key, payload)
    candidate = images_dir / image_name
    if candidate.exists():
        return candidate
    stem = Path(image_name).stem or frame_key
    matches = sorted(images_dir.glob(f"{stem}*"))
    if matches:
        return matches[0]
    return candidate


def cached_embedding(
    cursor: sqlite3.Cursor,
    *,
    image_path: str,
    bbox_cache_key: str,
) -> np.ndarray | None:
    row = cursor.execute(
        """
        SELECT embedding, dim
        FROM embedding_cache
        WHERE image_path = ? AND bbox_key = ?
        """,
        (str(Path(image_path)), bbox_cache_key),
    ).fetchone()
    if row is None:
        return None
    embedding = np.frombuffer(row[0], dtype=np.float32)
    if embedding.size != int(row[1]):
        return None
    return embedding.copy()


def metadata_for_object(
    *,
    faiss_id: int,
    spec: SequenceSpec,
    frame_key: str,
    payload: dict[str, Any],
    frame_info: dict[str, Any],
    object_index: int,
    box: list[float],
    cache_image_path: str,
    cache_bbox_key: str,
    image_path: Path,
) -> dict[str, Any]:
    labels = payload.get("labels") or []
    scores = payload.get("scores") or []
    label = labels[object_index] if object_index < len(labels) else ""
    score = scores[object_index] if object_index < len(scores) else None
    return {
        "faiss_id": faiss_id,
        "seq_id": spec.seq_id,
        "cam": spec.cam,
        "stride": spec.stride,
        "hsgg_input_dir": repo_relative_or_absolute(spec.hsgg_input_dir or spec.sample_root / "hsgg_local_paths"),
        "frame_key": frame_key,
        "image_name": image_path.name,
        "image_path": repo_relative_or_absolute(image_path),
        "cache_image_path": cache_image_path,
        "object_index": object_index,
        "label": str(label),
        "score": float(score) if isinstance(score, (int, float)) else None,
        "bbox": box,
        "bbox_key": cache_bbox_key,
        "sampled_index": frame_info.get("sampled_index"),
        "graph_frame_index": frame_info.get("graph_frame_index"),
        "source_frame": frame_info.get("source_frame"),
        "timestamp_unix": frame_info.get("timestamp_unix"),
        "timestamp_iso": frame_info.get("timestamp_iso"),
        "elapsed_sec_from_sequence_start": frame_info.get(
            "elapsed_sec_from_sequence_start"
        ),
    }


def add_batch(index: faiss.Index, vectors: list[np.ndarray]) -> None:
    if not vectors:
        return
    matrix = np.vstack(vectors).astype("float32", copy=False)
    faiss.normalize_L2(matrix)
    index.add(matrix)


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_repo_input_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / f"{args.index_name}.faiss"
    metadata_path = output_dir / f"{args.index_name}.metadata.jsonl"
    manifest_path = output_dir / f"{args.index_name}.manifest.json"

    for path in (index_path, metadata_path, manifest_path):
        if path.exists() and not args.force:
            raise FileExistsError(f"Output already exists: {path}. Pass --force.")

    cache_db = resolve_repo_input_path(args.cache_db)
    if not cache_db.exists():
        raise FileNotFoundError(f"Embedding cache DB not found: {cache_db}")

    sequences = load_sequences(
        args.sequence_manifest,
        seq_filter=set(args.seq_id) if args.seq_id else None,
    )

    connection = sqlite3.connect(cache_db)
    cursor = connection.cursor()
    index: faiss.Index | None = None
    dim: int | None = None
    faiss_id = 0
    vectors: list[np.ndarray] = []
    sequence_stats: list[dict[str, Any]] = []

    try:
        with metadata_path.open("w", encoding="utf-8") as metadata_handle:
            for spec in sequences:
                objects_path = (spec.hsgg_input_dir or spec.sample_root / "hsgg_local_paths") / "objects.json"
                images_dir = spec.images_dir or spec.sample_root / "images"
                if not objects_path.exists():
                    raise FileNotFoundError(f"Missing objects.json: {objects_path}")
                objects_payload = json.loads(objects_path.read_text(encoding="utf-8"))
                frame_index = load_frame_index(spec.frame_time_index_json)
                seq_total = 0
                seq_indexed = 0
                seq_missing = 0

                for frame_key, payload in objects_payload.items():
                    if not isinstance(payload, dict):
                        continue
                    cache_image_path = str(Path(str(payload.get("image_path") or "")))
                    if not cache_image_path:
                        continue
                    image_path = actual_image_path(
                        images_dir=images_dir,
                        frame_key=str(frame_key),
                        payload=payload,
                    )
                    image_name = image_name_from_payload(str(frame_key), payload)
                    frame_info = (
                        frame_index.get(image_name)
                        or frame_index.get(Path(image_name).stem)
                        or frame_index.get(str(frame_key))
                        or {}
                    )
                    boxes = payload.get("boxes") or []
                    for object_index, raw_box in enumerate(boxes):
                        seq_total += 1
                        key = bbox_key(raw_box)
                        box = bbox_values(raw_box)
                        if key is None or box is None:
                            seq_missing += 1
                            continue
                        embedding = cached_embedding(
                            cursor,
                            image_path=cache_image_path,
                            bbox_cache_key=key,
                        )
                        if embedding is None:
                            seq_missing += 1
                            continue
                        if dim is None:
                            dim = int(embedding.shape[-1])
                            index = faiss.IndexFlatIP(dim)
                        elif int(embedding.shape[-1]) != dim:
                            seq_missing += 1
                            continue

                        metadata = metadata_for_object(
                            faiss_id=faiss_id,
                            spec=spec,
                            frame_key=str(frame_key),
                            payload=payload,
                            frame_info=frame_info,
                            object_index=object_index,
                            box=box,
                            cache_image_path=cache_image_path,
                            cache_bbox_key=key,
                            image_path=image_path,
                        )
                        metadata_handle.write(
                            json.dumps(metadata, ensure_ascii=False) + "\n"
                        )
                        vectors.append(embedding)
                        faiss_id += 1
                        seq_indexed += 1
                        if len(vectors) >= args.batch_size:
                            if index is None:
                                raise RuntimeError("FAISS index was not initialized.")
                            add_batch(index, vectors)
                            vectors = []

                sequence_stats.append(
                    {
                        "seq_id": spec.seq_id,
                        "objects_total": seq_total,
                        "objects_indexed": seq_indexed,
                        "objects_missing_embedding": seq_missing,
                        "hit_rate": seq_indexed / seq_total if seq_total else 0.0,
                    }
                )

            if index is None or dim is None:
                raise RuntimeError("No cached embeddings were indexed.")
            add_batch(index, vectors)
            faiss.write_index(index, str(index_path))
    finally:
        connection.close()

    metadata_count = sum(1 for _ in metadata_path.open(encoding="utf-8"))
    summary = {
        "contract": "navqa_visual_faiss_index_manifest_v1",
        "index_kind": "object_siglip_flat_ip",
        "note": (
            "Experimental visual grounding index. Built from existing cached "
            "SigLIP object embeddings only; missing embeddings were not computed."
        ),
        "sequence_manifest": repo_relative_or_absolute(
            resolve_repo_input_path(args.sequence_manifest)
        ),
        "cache_db": repo_relative_or_absolute(cache_db),
        "index_path": repo_relative_or_absolute(index_path),
        "metadata_path": repo_relative_or_absolute(metadata_path),
        "dimension": dim,
        "faiss_metric": "inner_product_on_l2_normalized_vectors",
        "ntotal": int(index.ntotal),
        "metadata_rows": metadata_count,
        "sequence_stats": sequence_stats,
        "files": {
            "index_bytes": index_path.stat().st_size,
            "metadata_bytes": metadata_path.stat().st_size,
            "manifest_bytes": 0,
        },
    }
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["files"]["manifest_bytes"] = manifest_path.stat().st_size
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = build_index(args)
    total = sum(item["objects_total"] for item in summary["sequence_stats"])
    indexed = sum(item["objects_indexed"] for item in summary["sequence_stats"])
    missing = sum(
        item["objects_missing_embedding"] for item in summary["sequence_stats"]
    )
    print(f"index_path={resolve_repo_input_path(summary['index_path'])}")
    print(f"metadata_path={resolve_repo_input_path(summary['metadata_path'])}")
    print(f"manifest_path={resolve_repo_input_path(args.output_dir) / (args.index_name + '.manifest.json')}")
    print(f"dimension={summary['dimension']}")
    print(f"ntotal={summary['ntotal']}")
    print(f"objects_total={total}")
    print(f"objects_indexed={indexed}")
    print(f"objects_missing_embedding={missing}")
    print(f"hit_rate={indexed / total:.4f}" if total else "hit_rate=0.0000")
    for item in summary["sequence_stats"]:
        print(
            "seq{seq_id}: indexed={objects_indexed}/{objects_total} "
            "missing={objects_missing_embedding} hit_rate={hit_rate:.4f}".format(
                **item
            )
        )


if __name__ == "__main__":
    main()
