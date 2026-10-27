"""Read one existing aligned SigLIP vector and check its observation bindings.

Standard library only; no model, FAISS import, download, extraction, or writes.
The narrow FAISS reader accepts only the exported, unquantized IndexFlatIP form.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import struct
import tarfile
from typing import Any, BinaryIO, Iterator


KINDS = ("object", "frame", "tracklet", "temporal_window", "event_boundary", "before_after_pair")


def relative_name(value: str) -> str:
    p = PurePosixPath(value)
    if not value or value == "." or p.is_absolute() or PureWindowsPath(value).drive or ".." in p.parts or "\\" in value or ":" in value or "\0" in value:
        raise ValueError("Package references must be bounded, relative POSIX paths")
    return p.as_posix()


def bounded(root: Path, value: str) -> Path:
    root = root.resolve()
    path = (root / relative_name(value)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Resource reference escapes its explicit root")
    return path


class Store:
    """A read-only extracted tree, optionally with visual members in one tar.gz."""

    def __init__(self, root: Path, visual_archive: Path | None = None):
        self.root = root.resolve()
        self.archive = tarfile.open(visual_archive, "r:gz") if visual_archive else None

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()

    @contextmanager
    def open(self, name: str) -> Iterator[BinaryIO]:
        name = relative_name(name)
        if self.archive is not None:
            member = self.archive.getmember(name)
            if not member.isfile():
                raise ValueError("Expected a regular archive member")
            handle = self.archive.extractfile(member)
            if handle is None:
                raise ValueError("Unreadable archive member")
        else:
            handle = bounded(self.root, name).open("rb")
        with handle:
            yield handle

    def size(self, name: str) -> int:
        name = relative_name(name)
        if self.archive is not None:
            member = self.archive.getmember(name)
            if not member.isfile():
                raise ValueError("Expected a regular archive member")
            return member.size
        return bounded(self.root, name).stat().st_size

    def json(self, name: str) -> Any:
        # Bindings and one-video metadata are small; never load a graph or index.
        if self.size(name) > 64 * 1024**2:
            raise ValueError("JSON metadata exceeds this example reader's 64 MiB bound")
        with self.open(name) as handle:
            return json.load(handle)


def index_header(store: Store, name: str) -> dict[str, int]:
    with store.open(name) as handle:
        header = handle.read(45)
    if len(header) != 45 or header[:4] != b"IxFI":
        raise ValueError("Only the exported IndexFlatIP format is supported")
    d = struct.unpack_from("<i", header, 4)[0]
    count = struct.unpack_from("<q", header, 8)[0]
    metric = struct.unpack_from("<i", header, 33)[0]
    values = struct.unpack_from("<Q", header, 37)[0]
    if d != 768 or count <= 0 or header[32] != 1 or metric != 0 or values != d * count:
        raise ValueError("Invalid 768D float32 IndexFlatIP header")
    if store.size(name) != 45 + 4 * values:
        raise ValueError("FAISS payload size disagrees with its header")
    return {"dimension": d, "count": count}


def read_index_vector(store: Store, name: str, row_id: int) -> tuple[tuple[float, ...], bytes]:
    header = index_header(store, name)
    if not 0 <= row_id < header["count"]:
        raise ValueError("Local FAISS row ID is outside this index")
    with store.open(name) as handle:
        handle.seek(45 + 4 * header["dimension"] * row_id)
        raw = handle.read(4 * header["dimension"])
    if len(raw) != 4 * header["dimension"]:
        raise ValueError("Truncated vector")
    vector = struct.unpack("<768f", raw)
    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Vector contains non-finite values")
    return vector, raw


def select_metadata(store: Store, binding: dict[str, Any], kind: str, requested_id: int | None) -> tuple[dict[str, Any], str]:
    uid = str(binding["unit_id"])
    if binding["dataset"] == "nextqa":
        base = binding["unit_archive_root"] + "/visual_features/" + kind
        rows = store.json(base + ".metadata.json")
        if not isinstance(rows, list):
            raise ValueError("NextQA metadata must be an array")
        if not rows:
            raise ValueError("Selected candidate kind is an explicitly empty shard; choose another kind")
        selected = next((r for r in rows if requested_id is None or r["local_faiss_id"] == requested_id), None)
        index = base + ".faiss"
    else:
        if kind != "object":
            raise ValueError("NaVQA/HM3D/ScanNet packages contain object vectors only")
        selected = None
        with store.open(binding["group_object_metadata"]) as handle:
            for line in handle:
                row = json.loads(line)
                if str(row["unit_id"]) == uid and (requested_id is None or row["local_faiss_id"] == requested_id):
                    selected = row
                    break
        index = binding["group_object_index"]
    if selected is None:
        raise ValueError("No existing vector matches this unit and requested local row ID")
    if str(selected["unit_id"]) != uid or selected["faiss_id"] != selected["local_faiss_id"]:
        raise ValueError("Feature row and unit/local FAISS IDs disagree")
    return selected, index


def _existing(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(description + ": " + str(path))
    return path


def _has_key(path: Path, key: str) -> bool:
    # Only a single-frame sidecar or one video's normalized table is parsed.
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("HSGG table exceeds this example reader's 64 MiB bound")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return isinstance(value, dict) and key in value


def check_sidecar(path: Path, image_name: str) -> bool | None:
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("HSGG sidecar exceeds this example reader's 64 MiB bound")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("HSGG sidecar must be a JSON object")
    original_image = payload.get("image_path")
    if original_image:
        if not isinstance(original_image, str) or PurePosixPath(original_image.replace("\\", "/")).name != image_name:
            raise ValueError("HSGG sidecar refers to a different image basename")
        return True
    return None


def check_observation(binding: dict[str, Any], row: dict[str, Any], root: Path,
                      images_root: Path | None, hsgg_root: Path | None,
                      legacy_data_root: Path | None) -> dict[str, Any]:
    ds = binding["dataset"]
    unit = binding["unit_archive_root"]
    frame_key = row.get("hsgg_record_key") or row.get("frame_key") or row.get("frame_name")
    image_value = row.get("image_path") or row.get("canonical_image_path")
    if not image_value or not frame_key:
        raise ValueError("Selected candidate lacks a single-frame observation; use --kind object")
    if not isinstance(frame_key, str) or relative_name(frame_key) != frame_key or "/" in frame_key:
        raise ValueError("HSGG frame key must be a single safe filename stem")
    image_name = PurePosixPath(relative_name(image_value)).name
    if images_root is not None:
        image = bounded(images_root, image_name)
    elif legacy_data_root is not None:
        source_image = row.get("current_source_image_path")
        if not source_image:
            raise ValueError("Feature row lacks an explicit legacy source image binding")
        image = bounded(legacy_data_root, source_image)
    else:
        image = bounded(root, image_value)
    _existing(image, "Image is missing; specify the actual --images-root for a legacy flat JPEG package")

    if legacy_data_root is not None and hsgg_root is None:
        if ds == "nextqa":
            source = binding["coverage"]["current_hsgg_reference_root"]
            hsgg_dir = bounded(legacy_data_root, source)
        else:
            hsgg = bounded(legacy_data_root, row["current_source_hsgg_path"])
            hsgg_dir = hsgg.parent
    else:
        hsgg_dir = hsgg_root.resolve() if hsgg_root else bounded(root, unit + "/hsgg")
    if ds == "nextqa":
        # Cloud HSGG packages contain sidecars and aggregate compatibility tables;
        # normalized old-project inputs instead have four dictionaries by frame key.
        sidecar = bounded(hsgg_dir, "frame_scene_graphs/" + frame_key + ".json")
        if sidecar.is_file():
            hsgg = sidecar
            sidecar_image_matches = check_sidecar(hsgg, image.name)
            hsgg_form = "completed_frame_sidecar"
            tables = 0
        else:
            names = ("objects.json", "attributes.json", "relations.json", "hierarchy_mapping.json")
            tables = sum(_has_key(_existing(bounded(hsgg_dir, n), "Missing normalized HSGG table"), frame_key) for n in names)
            if tables != 4:
                raise ValueError("Frame key must exist in all four normalized HSGG tables")
            hsgg = hsgg_dir / "objects.json"
            hsgg_form = "normalized_four_tables"
            sidecar_image_matches = None
    else:
        candidate = bounded(hsgg_dir, frame_key + ".json")
        nested = bounded(hsgg_dir, "frame_scene_graphs/" + frame_key + ".json")
        hsgg = _existing(candidate if candidate.is_file() else nested, "Missing completed HSGG sidecar")
        sidecar_image_matches = check_sidecar(hsgg, image.name)
        hsgg_form, tables = "completed_frame_sidecar", 0
    graph_paths = {}
    for name, reference in binding["timegraph"].items():
        graph = bounded(legacy_data_root, binding["coverage"][name + "_path"]) if legacy_data_root else bounded(root, reference)
        graph_paths[name] = _existing(graph, "Graph file is missing")
    return {"image_exists": True, "hsgg_exists": True, "hsgg_record_key": frame_key,
            "hsgg_representation": hsgg_form, "normalized_tables_containing_key": tables,
            "sidecar_image_basename_matches": sidecar_image_matches,
            "canonical_exists": True, "temporal_exists": True,
            "image_name": image.name, "graph_reference_validation": "file_presence_only_no_graph_body_loaded",
            "path_binding_mode": "explicit_legacy_data_root" if legacy_data_root else "unpacked_packages_with_explicit_overrides"}


def read_unit(root: Path, unit_root: str, *, visual_archive: Path | None = None,
              images_root: Path | None = None, hsgg_root: Path | None = None,
              legacy_data_root: Path | None = None, kind: str = "object",
              row_id: int | None = None, include_vector: bool = False) -> dict[str, Any]:
    unit_root = relative_name(unit_root)
    store = Store(root, visual_archive)
    try:
        binding = store.json(unit_root + "/visual_features/unit_binding.json")
        if binding.get("unit_archive_root") != unit_root or binding.get("dataset") not in {"navqa", "nextqa", "hm3d", "scannet"}:
            raise ValueError("Unit binding does not identify the requested exported unit")
        row, index = select_metadata(store, binding, kind, row_id)
        vector, raw = read_index_vector(store, index, int(row["local_faiss_id"]))
        checks = check_observation(binding, row, root, images_root, hsgg_root, legacy_data_root)
        result = {"status": "passed", "dataset": binding["dataset"], "unit_id": str(binding["unit_id"]),
                  "group_id": binding["group_id"], "candidate_kind": kind,
                  "local_faiss_id": row["local_faiss_id"], "original_faiss_id": row["original_faiss_id"],
                  "dimension": 768, "dtype": "float32", "vector_sha256": hashlib.sha256(raw).hexdigest(),
                  "vector_l2_norm": math.sqrt(sum(x*x for x in vector)), "vector_first8": list(vector[:8]),
                  "binding_checks": checks, "source_files_modified": False, "models_or_apis_called": False,
                  "cloud_download_verified": False, "full_graph_identity_or_topology_validated": False}
        if include_vector:
            result["vector"] = list(vector)
        return result
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Explicit common root of unpacked packages")
    parser.add_argument("--unit-root", required=True, help="Package unit path, e.g. nextqa-val/10001787725")
    parser.add_argument("--visual-archive", type=Path, help="Read visual members directly from one local tar.gz instead of extracting it")
    parser.add_argument("--images-root", type=Path, help="Actual directory containing this unit's image files (needed for flat legacy NaVQA JPEG archives)")
    parser.add_argument("--hsgg-root", type=Path, help="Actual HSGG root for this unit; may contain frame_scene_graphs/ or four normalized tables")
    parser.add_argument("--legacy-data-root", type=Path, help="Explicit optional original datasets root for local validation; never discovered automatically")
    parser.add_argument("--kind", choices=KINDS, default="object")
    parser.add_argument("--local-faiss-id", type=int)
    parser.add_argument("--include-vector", action="store_true")
    args = parser.parse_args()
    try:
        result = read_unit(args.root, args.unit_root, visual_archive=args.visual_archive,
                           images_root=args.images_root, hsgg_root=args.hsgg_root,
                           legacy_data_root=args.legacy_data_root, kind=args.kind,
                           row_id=args.local_faiss_id, include_vector=args.include_vector)
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        parser.exit(2, "Resource check failed: " + str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
