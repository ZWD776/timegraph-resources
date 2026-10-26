# Resource catalogue

This catalogue contains metadata only. The datasets remain independent.
All file references are relative to an explicitly selected resource root containing datasets/, not this catalogue folder.
No sampled images, HSGG bodies, graph bodies, embeddings, QA question text or reference answers are included.

NaVQA timestamps are original-frame Unix timestamps; elapsed seconds are supplied separately.
NextQA timestamp_seconds are FPS-derived nominal seconds, not container PTS. The current normalized item directory is authoritative; HSGG SHA values refer to original per-frame source-trace records, not the normalized objects.json bytes. Read all four normalized tables using hsgg_record_key.
OpenEQA ordinal_time is an old graph ordering value, never real seconds. Missing old time mappings are left empty. sampled_index is the order of supplied images and is not an elapsed time.

Official QA SHA256 fingerprints identify exact versions. Fetch annotations from their official sources under their terms. NaVQA has 210 official questions and 139 non-position questions; history windows are not interchangeable with answer event timestamps, and old retrieval was not a uniform strict cutoff.
Historical graphs can cover fewer completed frames than current HSGG; unit_catalog records coverage. Tracklets are algorithm predictions, not verified identity ground truth.

external_resources.template.json contains the user-supplied HSGG/image share link and access code for the private repository. Its contents were not downloaded or verified. Graph/visual-feature links and archive checksums remain pending.
The exporter refuses nonempty output directories. Run it with a new --output directory to regenerate safely.
