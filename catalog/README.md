# Resource catalogue

This catalogue contains metadata only. The datasets remain independent.
All file references are relative to an explicitly selected resource root containing datasets/, not this catalogue folder.
No sampled images, HSGG bodies, graph bodies, embeddings, QA question text or reference answers are included.

NaVQA timestamps are original-frame Unix timestamps; elapsed seconds are supplied separately.
NextQA timestamp_seconds are FPS-derived nominal seconds, not container PTS. The current normalized item directory is authoritative; HSGG SHA values refer to original per-frame source-trace records, not the normalized objects.json bytes. Read all four normalized tables using hsgg_record_key.
OpenEQA ordinal_time is an old graph ordering value, never real seconds. Missing old time mappings are left empty. sampled_index is the order of supplied images and is not an elapsed time.

Official QA SHA256 fingerprints identify exact versions. Fetch annotations from their official sources under their terms. NaVQA has 210 official questions and 139 non-position questions; history windows are not interchangeable with answer event timestamps, and old retrieval was not a uniform strict cutoff.
Historical graphs can cover fewer completed frames than current HSGG; unit_catalog records coverage. Tracklets are algorithm predictions, not verified identity ground truth.

External images, HSGG, graph memory and visual features are supplied through the same controlled share. Obtain access from the maintainer separately; no share URL or access code belongs in this repository. Upload was confirmed by the owner; the remote archive bytes were not downloaded and reverified.

`aligned_packages.json` and `aligned_packages.SHA256SUMS.txt` describe 60 locally verified graph/feature archives in 30 paired groups. `aligned_units.json` maps all 729 selected units to the package layout and legacy resource references. Groups are independent by dataset: NaVQA 7, NextQA 6, HM3D 7, ScanNet 10. File-name ranges are sorted positions, not episode-number prefixes.

The older frame CSVs and `unit_catalog.json` refer to the normalized `datasets/` tree. Package paths in `aligned_units.json` refer to archive members. These are two explicitly different layouts. Use `unit_binding.json` and the documented reader; direct extraction does not recreate the old runtime tree or its SQLite indexes.
The exporter refuses nonempty output directories. Run it with a new --output directory to regenerate safely.

`qa_resources.json` records three independent QA packages prepared locally for owner upload. It contains metadata only; question text and answers are outside Git. NaVQA legacy-derived time labels are unverified, with one negative result explicitly flagged in the external package.
