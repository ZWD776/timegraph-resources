#!/usr/bin/env python3
"""Export a small, portable catalogue without copying data, QA text or answers.

All paths in the output are relative to the supplied old-project data root.
Large graph files are only stat'ed; at most 4096 header bytes are read for frame_count.
An existing nonempty output directory is refused: use a new directory for every export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    'navqa': 'datasets/navqa/source/navqa-coda-every10-v1/payload/source_manifest.json',
    'nextqa': 'datasets/nextqa/source/nextqa-val-v1-repaired29/payload/source_manifest.json',
    'openeqa': 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/source_manifest.json',
}
FRAME_FIELDS = ['dataset', 'shard', 'unit_id', 'image_path', 'image_available', 'hsgg_path',
                'hsgg_record_key', 'hsgg_available', 'hsgg_status', 'sampled_index',
                'graph_frame_index', 'source_frame_index', 'timestamp_unix', 'elapsed_seconds',
                'timestamp_seconds', 'ordinal_time', 'timestamp_semantics', 'timestamp_is_real_time',
                'legacy_graph_time_mapping_available', 'image_sha256', 'hsgg_sha256',
                'sha256_origin', 'legacy_hsgg_archive_member']
QA_FIELDS = ['dataset', 'question_id', 'unit_id', 'shard', 'official_question_id', 'question_type',
             'selected_non_position', 'start_time_unix', 'end_time_unix', 'history_policy',
             'question_text_included', 'answer_included']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export(root, out):
    root, out = root.resolve(), out.resolve()
    if root == out or root in out.parents and any(part == 'benchmark_data' for part in out.parts):
        raise ValueError('Unsafe output directory')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Refusing to overwrite a nonempty output directory; select a new --output')
    out.mkdir(parents=True, exist_ok=True)
    units, versions, frame_counts, qa_counts = [], [], Counter(), Counter()

    def local(relative):
        path = root / relative
        if root not in path.resolve().parents:
            raise ValueError('Source is outside the project root')
        if 'benchmark_data' in path.parts:
            raise ValueError('benchmark_data is excluded')
        return path

    def rel(path):
        return path.resolve().relative_to(root).as_posix()

    def dump(name, data):
        with (out / name).open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write('\n')

    def writer(name, fields):
        stream = (out / name).open('x', encoding='utf-8', newline='')
        result = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise', lineterminator='\n')
        result.writeheader()
        return stream, result

    def version(path, dataset, role, count=None):
        versions.append(dict(dataset=dataset, role=role, path=rel(path), bytes=path.stat().st_size,
                             sha256=sha256(path), record_count=count,
                             source_content_copied=False))

    graph_inventory = local('registry/audits/graph_storage_20261010/mainline_graph_units.csv')
    graphs = {}
    with graph_inventory.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            # The historical inventory contains absolute paths; only resolve the datasets suffix.
            for field in ('canonical_path', 'temporal_path'):
                value = row[field].replace('\\', '/')
                marker = value.find('datasets/')
                if marker < 0:
                    raise ValueError('Graph inventory lacks a portable datasets path')
                row[field] = value[marker:]
            graph_key = row['unit_id'].split('/')[-1] if row['dataset'] == 'openeqa' else row['unit_id']
            graphs[(row['dataset'], row['shard'], graph_key)] = row

    def unit(dataset, shard, ident, image_count, completed_count, source_base, semantics,
             timestamp_is_real_time, **extra):
        graph = graphs[(dataset, shard, ident)]
        temporal, canonical = local(graph['temporal_path']), local(graph['canonical_path'])
        with temporal.open('rb') as stream:
            header = stream.read(4096).decode('utf-8', errors='replace')
        count = re.search(r'"frame_count"\s*:\s*(\d+)', header)
        old_count = int(count.group(1)) if count else None
        count_source = 'temporal_graph_header' if count else None
        if old_count is None and dataset == 'nextqa':
            audit_path = temporal.parent / 'video_graph_audit.json'
            if audit_path.is_file():
                old_count = read_json(audit_path).get('temporal_graph', {}).get('frame_count')
                count_source = 'video_graph_audit.temporal_graph.frame_count'
        record = dict(dataset=dataset, shard=shard, unit_id=ident,
                      sampled_image_count=image_count, current_completed_hsgg_count=completed_count,
                      current_hsgg_reference_root=source_base, timestamp_semantics=semantics,
                      timestamp_is_real_time=timestamp_is_real_time, graph_asset_id=graph['asset_id'],
                      canonical_path=graph['canonical_path'], temporal_path=graph['temporal_path'],
                      canonical_bytes=canonical.stat().st_size, temporal_bytes=temporal.stat().st_size,
                      legacy_graph_frame_count=old_count,
                      legacy_graph_frame_count_source=count_source,
                      legacy_graph_frame_count_semantics=(
                          'includes_sampled_skipped_slots' if dataset == 'nextqa' else 'completed_hsgg_frames'),
                      graph_is_historical_algorithm_output_not_identity_ground_truth=True,
                      graph_data_included=False)
        record.update(extra)
        units.append(record)

    manifests = {key: read_json(local(path)) for key, path in SOURCES.items()}
    for dataset, path in SOURCES.items():
        version(local(path), dataset, 'source_manifest')

    # NaVQA: use the explicitly provided original-frame/Unix mapping, not numeric names as time.
    nav = manifests['navqa']
    windows = {}
    fs, fw = writer('navqa_frames.csv', FRAME_FIELDS)
    with fs:
        for seq in nav['sequences']:
            ident = str(seq['seq_id'])
            times = read_json(local(seq['frame_time_index_json']))
            version(local(seq['frame_time_index_json']), 'navqa', 'frame_time_mapping', len(times['frames']))
            qa_window_path = local(seq['qa_unfilled_json'])
            qa_window = read_json(qa_window_path)
            version(qa_window_path, 'navqa', 'qa_history_windows', len(qa_window['data']))
            for row in qa_window['data']:
                windows[(ident, row['id'])] = row
            image_dir, hsgg_dir = local(seq['images_dir']), local(seq['hsgg_input_dir'])
            image_paths = sorted(image_dir.glob('*.jpg'))
            records = {path.stem: path for path in (hsgg_dir / 'frame_scene_graphs').glob('*.json')}
            time_by_name = {row['image_name']: row for row in times['frames']}
            statuses = read_json(hsgg_dir / 'run_manifest.json').get('frames', {})
            for image in image_paths:
                tm = time_by_name.get(image.name, {})
                hsgg = records.get(image.stem)
                fw.writerow(dict(dataset='navqa', shard='coda', unit_id=ident, image_path=rel(image),
                    image_available=True, hsgg_path=rel(hsgg) if hsgg else '', hsgg_record_key=image.stem,
                    hsgg_available=hsgg is not None,
                    hsgg_status='completed' if hsgg else statuses.get(image.stem, {}).get('status', 'not_recorded'),
                    sampled_index=tm.get('sampled_index'), graph_frame_index=tm.get('graph_frame_index'),
                    source_frame_index=tm.get('source_frame'), timestamp_unix=tm.get('timestamp_unix'),
                    elapsed_seconds=tm.get('elapsed_sec_from_sequence_start'),
                    timestamp_semantics='unix_timestamp_from_original_frame_metadata' if tm else 'mapping_missing',
                    timestamp_is_real_time=bool(tm)))
                frame_counts['navqa'] += 1
            if set(records) - {image.stem for image in image_paths}:
                raise ValueError('NaVQA HSGG has unmatched images')
            unit('navqa', 'coda', ident, len(image_paths), len(records), rel(hsgg_dir),
                 'unix_timestamp_from_original_frame_metadata', True, source_stride=seq['stride'])

    qs, qw = writer('navqa_qa_ids.csv', QA_FIELDS)
    official = local(nav['official_annotations'])
    with official.open(encoding='utf-8-sig', newline='') as stream:
        nav_questions = list(csv.DictReader(stream))
    with qs:
        for row in nav_questions:
            ident, qid = row['Seq ID'], row['UUID']
            kind = row['Type \n(binary, position, time, text)'].strip()
            window = windows.get((ident, qid), {})
            qw.writerow(dict(dataset='navqa', shard='coda', question_id=qid, unit_id=ident,
                question_type=kind, selected_non_position=kind.lower() != 'position',
                start_time_unix=window.get('start_time'), end_time_unix=window.get('end_time'),
                history_policy='qa_unfilled_start_end_unix; legacy_retrieval_not_uniform_strict_cutoff',
                question_text_included=False, answer_included=False))
            qa_counts['navqa'] += 1
    version(official, 'navqa', 'official_qa_210_before_nonposition_filter', len(nav_questions))

    # NextQA: current normalized items and selected repairs are authoritative; old archive refs are provenance only.
    nxt = manifests['nextqa']
    fs, fw = writer('nextqa_frames.csv', FRAME_FIELDS)
    with fs:
        for video in nxt['videos']:
            ident = video['official_video_id']
            item = local(video['normalized_input_dir'])
            timeline = read_json(item / 'timeline.json')
            trace = read_json(item / 'source_trace.json')
            provenance = {row['frame_name']: row for row in trace['sampled_frames']}
            for row in timeline['frames']:
                src = provenance[row['frame_name']]
                image = item / src['materialized_image_path']
                fw.writerow(dict(dataset='nextqa', shard='val', unit_id=ident, image_path=rel(image),
                    image_available=image.is_file(), hsgg_path=rel(item / 'objects.json') if row['has_scene_graph'] else '',
                    hsgg_record_key=row['frame_name'], hsgg_available=row['has_scene_graph'],
                    hsgg_status=row['hsgg_status'], sampled_index=row['frame_rank'],
                    source_frame_index=row['frame_index'], timestamp_seconds=row['timestamp_sec'],
                    timestamp_semantics='fps_derived_nominal_video_seconds_not_container_pts',
                    timestamp_is_real_time=True, image_sha256=src.get('frame_member_sha256'),
                    hsgg_sha256=src.get('hsgg_member_sha256'),
                    sha256_origin='preexisting_source_trace_not_rehashed_in_export',
                    legacy_hsgg_archive_member=src.get('hsgg_archive_member')))
                frame_counts['nextqa'] += 1
            unit('nextqa', 'val', ident, len(timeline['frames']), timeline['completed_frame_count'],
                 rel(item), 'fps_derived_nominal_video_seconds_not_container_pts', True,
                 selected_generation_kind=video['selected_generation_kind'],
                 local_sequential_id=video['local_sequential_id'],
                 source_record_representation='objects/attrs/relations/hierarchy.json indexed by frame name',
                 source_fps_inferred=video.get('inferred_source_fps'))
    next_payload = local(SOURCES['nextqa']).parent
    questions = read_json(next_payload / 'evaluation/questions.runtime.json')
    qs, qw = writer('nextqa_qa_ids.csv', QA_FIELDS)
    with qs:
        for row in questions:
            qw.writerow(dict(dataset='nextqa', shard='val', question_id=row['question_id'],
                unit_id=row['video_id'], official_question_id=row['official_qid'],
                question_type=row['official_type'], history_policy='entire_selected_video_no_per_question_cutoff',
                question_text_included=False, answer_included=False))
            qa_counts['nextqa'] += 1
    for filename, role in [('official/val.csv', 'official_validation_qa'),
                           ('evaluation/questions.runtime.json', 'runtime_questions'),
                           ('evaluation/labels.evaluator_only.json', 'evaluator_only_reference_labels')]:
        version(next_payload / filename, 'nextqa', role, len(questions))

    # OpenEQA: enumerate all supplied sampled pictures; old graph time indices only cover old completed observations.
    opn = manifests['openeqa']
    fs, fw = writer('openeqa_frames.csv', FRAME_FIELDS)
    with fs:
        for shard, spec in opn['shards'].items():
            frames_root, hsgg_root = local(spec['frames']), local(spec['hsgg'])
            for episode_dir in sorted(path for path in frames_root.glob('*/*') if path.is_dir()):
                episode = episode_dir.relative_to(frames_root).as_posix()
                ident = episode.split('/')[-1]
                graph = graphs[('openeqa', shard, ident)]
                old_time_path = local(graph['temporal_path']).parent / 'frame_time_index.json'
                old_time = read_json(old_time_path)
                tm_by_name = {row['image_name']: row for row in old_time['frames']}
                hsgg_dir = hsgg_root / episode
                records = {path.stem: path for path in (hsgg_dir / 'frame_scene_graphs').glob('*.json')}
                run_path = hsgg_dir / 'run_manifest.json'
                statuses = read_json(run_path).get('frames', {}) if run_path.is_file() else {}
                images = sorted(episode_dir.glob('*.png'))
                for index, image in enumerate(images):
                    tm = tm_by_name.get(image.name, {})
                    hsgg = records.get(image.stem)
                    source_frame = re.match(r'^(\d+)-rgb$', image.stem)
                    fw.writerow(dict(dataset='openeqa', shard=shard, unit_id=ident,
                        image_path=rel(image), image_available=True, hsgg_path=rel(hsgg) if hsgg else '',
                        hsgg_record_key=image.stem, hsgg_available=hsgg is not None,
                        hsgg_status='completed' if hsgg else statuses.get(image.stem, {}).get('status', 'not_recorded'),
                        sampled_index=index, graph_frame_index=tm.get('graph_frame_index'),
                        source_frame_index=int(source_frame.group(1)) if source_frame else None,
                        ordinal_time=tm.get('timestamp_sec'), timestamp_semantics='ordinal_frame_order',
                        timestamp_is_real_time=False, legacy_graph_time_mapping_available=bool(tm)))
                    frame_counts['openeqa'] += 1
                if set(records) - {image.stem for image in images}:
                    raise ValueError('OpenEQA HSGG has unmatched images')
                unit('openeqa', shard, ident, len(images), len(records), rel(hsgg_dir),
                     'ordinal_frame_order', False, episode_history=episode,
                     legacy_graph_time_index_path=rel(old_time_path),
                     legacy_time_mapping_count=len(old_time['frames']),
                     missing_legacy_time_mapping_count=len(images) - len(tm_by_name),
                     seconds_are_unknown=True)
    official = local(opn['official_annotations'])
    questions = read_json(official)
    qs, qw = writer('openeqa_qa_ids.csv', QA_FIELDS)
    with qs:
        for row in questions:
            episode = row['episode_history']
            qw.writerow(dict(dataset='openeqa', shard=episode.split('-')[0],
                question_id=row['question_id'], unit_id=episode.split('/')[-1],
                question_type=row['category'], history_policy='entire_provided_episode_history_no_per_question_cutoff',
                question_text_included=False, answer_included=False))
            qa_counts['openeqa'] += 1
    version(official, 'openeqa', 'open_eqa_v0_official_questions_answers', len(questions))

    expected_frames = dict(navqa=9889, nextqa=23381, openeqa=31742)
    expected_qa = dict(navqa=210, nextqa=4996, openeqa=1636)
    if frame_counts != expected_frames or qa_counts != expected_qa or len(units) != 729:
        raise ValueError(f'Unexpected coverage: frames={frame_counts}, qa={qa_counts}, units={len(units)}')
    dump('unit_catalog.json', dict(schema_version='resource_unit_catalog_v1',
         data_root_semantics='All paths relative to the old project dataset root; no source files are included.',
         units=units))
    dump('source_versions.json', dict(schema_version='source_versions_v1', files=versions,
         note='SHA256 fingerprints identify the exact local annotation/mapping version; no question or answer content is copied.'))
    resources = []
    for dataset in SOURCES:
        for kind in ('hsgg', 'sampled_images', 'graph_memory', 'visual_features'):
            resources.append(dict(dataset=dataset, resource_kind=kind, download_url=None,
                                  access_code=None, archive_sha256=None, archive_bytes=None,
                                  status='external_access_provided_separately', large_binary_in_git=False))
    dump('external_resources.template.json', dict(schema_version='external_resources_v1', resources=resources,
         note='Images/HSGG/graphs/features are supplied externally. Obtain access separately; never store share URLs or access codes here.'))
    summary = dict(schema_version='resource_catalog_export_v1',
                   exported_at_utc=datetime.now(timezone.utc).isoformat(),
                   frame_counts=dict(frame_counts), qa_counts=dict(qa_counts),
                   official_qa_total=sum(qa_counts.values()), unit_count=len(units),
                   current_completed_hsgg_counts=dict(Counter({key: sum(u['current_completed_hsgg_count'] for u in units if u['dataset'] == key) for key in SOURCES})),
                   legacy_graph_frame_counts=dict(Counter({key: sum(u['legacy_graph_frame_count'] or 0 for u in units if u['dataset'] == key) for key in SOURCES})),
                   coverage_note='NextQA graph frame_count includes skipped placeholders; NaVQA/OpenEQA legacy graph frame_count counts completed frames.',
                   original_data_modified=False, benchmark_data_accessed=False,
                   question_text_or_answers_included=False, images_hsgg_graphs_features_copied=False)
    dump('export_summary.json', summary)
    readme = '''# Resource catalogue (staging)\n\nThis catalogue contains metadata only. The datasets remain independent.\nAll file references are relative to the old-project data root, not this catalogue folder.\nNo sampled images, HSGG bodies, graph bodies, embeddings, QA question text or reference answers are included.\n\nNaVQA timestamps are original-frame Unix timestamps; elapsed seconds are supplied separately.\nNextQA timestamp_seconds are FPS-derived nominal seconds, not container PTS. The current normalized item directory is authoritative; HSGG SHA values refer to original per-frame source-trace records, not the normalized objects.json bytes. Read all four normalized tables using hsgg_record_key.\nOpenEQA ordinal_time is an old graph ordering value, never real seconds. Missing old time mappings are left empty. sampled_index is the order of supplied images and is not an elapsed time.\n\nOfficial QA SHA256 fingerprints identify exact versions. Fetch annotations from their official sources under their terms. NaVQA has 210 official questions and 139 non-position questions; history windows are not interchangeable with answer event timestamps, and old retrieval was not a uniform strict cutoff.\nHistorical graphs can cover fewer completed frames than current HSGG; unit_catalog records coverage. Tracklets are algorithm predictions, not verified identity ground truth.\n\nExternal access details are deliberately withheld. Record archive names and checksums, never share URLs or access codes. This base exporter does not infer cloud upload status; export_aligned_catalog.py adds verified local package metadata.\nThe exporter refuses nonempty output directories. Run it with a new --output directory to regenerate safely.\n'''
    with (out / 'README.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(readme)
    inventory = []
    for path in sorted(out.iterdir()):
        if not path.is_file():
            continue
        if path.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError('Output exceeds normal GitHub file limit')
        content = path.read_text(encoding='utf-8')
        if re.search(r'(?i)(?<![a-z])[a-z]:[\\/]|/Users/|/home/|/cloud/|/data/', content):
            raise ValueError(f'Absolute machine path found in {path.name}')
        inventory.append(dict(path=path.name, bytes=path.stat().st_size, sha256=sha256(path)))
    dump('catalog_files.json', dict(files=inventory, self_excluded=True,
                                    absolute_machine_path_scan_passed=True,
                                    each_file_below_100_MiB=True))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Created {len(list(out.iterdir()))} metadata files; total bytes: {sum(p.stat().st_size for p in out.iterdir())}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Explicit external project-data root containing datasets/. No parent-workspace fallback is used.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.data_root, args.output)


if __name__ == '__main__':
    main()
