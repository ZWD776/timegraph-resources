"""Export portable metadata from verified local packaging receipts, never data bodies."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def export(audit: Path, catalog: Path):
    report, plan = load(audit / "final_verification.json"), load(audit / "plan.json")
    if report["status"] != "passed" or report["package_count"] != 60:
        raise ValueError("Expected the verified 60-package release")
    packages = report["packages"]
    lookup = {(p["group_id"], p["kind"]): p for p in packages}
    if len(lookup) != 60:
        raise ValueError("Duplicate archive identity")
    units = []
    for group in plan["groups"]:
        for item in group["units"]:
            base = item["archive_root"]
            ds = group["dataset"]
            units.append({
                "dataset": "openeqa" if ds in ("hm3d", "scannet") else ds,
                "shard": ds if ds in ("hm3d", "scannet") else "coda" if ds == "navqa" else "val",
                "unit_id": item["unit_id"], "group_id": group["group_id"],
                "archive_root": base, "paired_archives": group["filenames"],
                "timegraph_archive": lookup[group["group_id"], "timegraph"]["backup_relative_path"],
                "visual_archive": lookup[group["group_id"], "visual_features"]["backup_relative_path"],
                "timegraph": {"canonical": base + "/timegraph/canonical_temporal_memory_graph.json",
                              "temporal": base + "/timegraph/temporal_graph.json",
                              "binding": base + "/timegraph/unit_binding.json"},
                "visual_features": {
                    "binding": base + "/visual_features/unit_binding.json",
                    "layout": "per_video_six_kinds" if ds == "nextqa" else "group_object_index",
                    "index": base + "/visual_features/object.faiss" if ds == "nextqa" else f"__group_features__/{group['group_id']}/object.faiss",
                    "metadata": base + "/visual_features/object.metadata.json" if ds == "nextqa" else f"__group_features__/{group['group_id']}/object.metadata.jsonl",
                },
                "source_files": item["source_files"], "coverage": item["coverage"],
                "legacy_paths_need_binding": True,
                "legacy_navqa_images_flat": ds == "navqa",
                "legacy_navqa_hsgg_root": f"hsgg_navqa_seq{item['unit_id']}_every10_full" if ds == "navqa" else None,
            })
    if len(units) != 729 or Counter(u["dataset"] for u in units) != Counter(navqa=7, nextqa=570, openeqa=152):
        raise ValueError("Aligned unit counts differ")
    catalog.mkdir(parents=True, exist_ok=True)
    common = {"client_date": "2026-10-27", "access_method": "contact_maintainer_separately",
              "download_credentials_included": False,
              "cloud_upload": "confirmed_by_owner", "cloud_bytes_reverified": False}
    dump(catalog / "aligned_packages.json", {
        "schema_version": "portable_aligned_packages_v1", **common,
        "group_count": 30, "package_count": 60,
        "group_counts": dict(Counter(g["dataset"] for g in plan["groups"])),
        "graph_core_counts": report["graph_core_counts"],
        "visual_vector_counts": report["visual_vector_counts"],
        "visual_vector_total": report["visual_vector_total"],
        "archive_bytes": report["archive_bytes"],
        "local_integrity": "all_member_sha256_and_gzip_crc_passed",
        "packages": packages})
    dump(catalog / "aligned_units.json", {"schema_version": "portable_aligned_units_v1", **common,
        "unit_count": 729, "package_root_semantics": "timegraph/visual_features entries are archive members; source_files and coverage retain relative references into the legacy datasets/ tree.",
        "units": units})
    (catalog / "aligned_packages.SHA256SUMS.txt").write_text("".join(
        p["sha256"] + "  " + p["backup_relative_path"] + "\n" for p in packages), encoding="utf-8")
    return {"groups": 30, "packages": 60, "units": len(units)}


def refresh_ledger(catalog: Path):
    rows = []
    for p in sorted(catalog.rglob("*")):
        if p.is_file() and p.name != "catalog_files.json":
            raw = p.read_bytes()
            if re.search(rb"(?i)(?<![a-z])[a-z]:[\\/]|/Users/|/home/", raw):
                raise ValueError("Absolute machine path in catalogue: " + p.name)
            rows.append({"path": p.relative_to(catalog).as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    dump(catalog / "catalog_files.json", {"files": rows, "self_excluded": True,
        "absolute_machine_path_scan_passed": True, "each_file_below_100_MiB": all(x["bytes"] < 100 * 1024**2 for x in rows),
        "download_credentials_included": False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-directory", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    args = p.parse_args()
    result = export(args.audit_directory, args.catalog)
    refresh_ledger(args.catalog)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
