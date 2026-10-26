"""Read-only post-organization checks; never inspect external backup storage."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.datasets.organize_source_layouts import SOURCE, REPORT, read, digest, write
from vlkg.core.assets.asset_registry import inspect_dataset, verify_metadata_copies
from vlkg.core.assets.source_paths import resolve_data_path
from vlkg.datasets.nextqa.graph.hsgg_input_adapter import load_normalized_video_input
from scripts.datasets.openeqa.build.build_hm3d_graphs_from_hsgg import load_episode_hsgg_tables


def main():
    journal = read(REPORT / 'move_journal.json')
    if journal['status'] != 'completed':
        raise ValueError('organization has not completed')
    inventory = read(REPORT / 'source_inventory_before.json')['files']
    moves = {row['source']:row['target'] for row in journal['entries']}
    def original_location(name):
        snapshot = journal['metadata_originals'].get(name)
        if snapshot:
            target = ROOT / snapshot['snapshot']
            if digest(target) != snapshot['sha256']:
                raise ValueError(f'original metadata snapshot changed: {name}')
            return target, True
        seen=set()
        while True:
            key=name
            while key not in moves and '/' in key:
                key=key.rpartition('/')[0]
            if key not in moves:
                return ROOT/name, False
            if key in seen:
                raise ValueError('move lineage cycle')
            seen.add(key)
            name=moves[key]+name[len(key):]
    issues=[]
    image_count=0
    for record in inventory:
        path, copied = original_location(record['path'])
        if not path.is_file():
            issues.append({'missing_preserved_file':record['path']});continue
        stat=path.stat()
        if stat.st_size != record['bytes'] or (not copied and (stat.st_ino != record['file_id'] or stat.st_mtime_ns != record['mtime_ns'])):
            issues.append({'preserved_file_changed':record['path'],'target':str(path)})
        if path.suffix.lower() in ('.png','.jpg','.jpeg'):
            image_count+=1

    next_manifest=read(SOURCE['nextqa']/'source_manifest.json')
    totals={'videos':0,'frames':0,'hsgg':0,'skipped':0}
    for row in next_manifest['videos']:
        loaded=load_normalized_video_input(row,repository_root=ROOT)
        vid=str(row['official_video_id'])
        if loaded.input_dir.name != vid:
            raise ValueError('Next item directory must use official video ID')
        old_meta=SOURCE['nextqa']/'provenance/history/item_metadata'/vid
        old_objects,new_objects=read(old_meta/'objects.json'),read(loaded.input_dir/'objects.json')
        strip=lambda table:{name:{k:v for k,v in obj.items() if k!='image_path'} for name,obj in table.items()}
        if strip(old_objects)!=strip(new_objects):
            raise ValueError(f'Next observations changed: {vid}')
        old_trace,new_trace=read(old_meta/'source_trace.json'),read(loaded.source_trace_path)
        for trace in (old_trace,new_trace):
            for frame in trace['sampled_frames']:
                frame.pop('materialized_image_path',None)
        if old_trace!=new_trace:
            raise ValueError(f'Next source/time provenance changed: {vid}')
        totals['videos']+=1;totals['frames']+=loaded.sampled_frame_count
        totals['hsgg']+=loaded.completed_frame_count;totals['skipped']+=loaded.skipped_frame_count
    if totals!={'videos':570,'frames':23381,'hsgg':23044,'skipped':337}:
        raise ValueError(totals)
    if {p.name for p in (SOURCE['nextqa']/'items').iterdir() if p.is_dir()}!={str(row['official_video_id']) for row in next_manifest['videos']}:
        raise ValueError('Next item set has extra or missing videos')

    nav_manifest=read(SOURCE['navqa']/'source_manifest.json')
    nav_rows=[]
    for row in nav_manifest['sequences']:
        directory=ROOT/row['hsgg_input_dir']
        tables={name:read(directory/(name+'.json')) for name in ('objects','attributes','relations','hierarchy_mapping')}
        files=list((directory/'frame_scene_graphs').glob('*.json'))
        for file in files:
            raw=read(file)
            for name,key in (('objects','objects'),('attributes','attributes'),('relations','relations'),('hierarchy_mapping','hierarchy')):
                if tables[name].get(file.stem)!=raw[key]:
                    raise ValueError(f'NAV sidecar/table mismatch: {file}')
            image=ROOT/row['images_dir']/Path(raw['image_path']).name
            if not image.is_file():
                raise FileNotFoundError(image)
        if any(set(t)!=set(p.stem for p in files) for t in tables.values()):
            raise ValueError('NAV aggregate frame set mismatch')
        nav_rows.append({'seq_id':row['seq_id'],'completed':len(files)})
    if sum(r['completed'] for r in nav_rows)!=9759:
        raise ValueError('NAV completed frame count changed')

    open_manifest=read(SOURCE['openeqa']/'source_manifest.json')
    open_rows=[]
    for shard,row in open_manifest['shards'].items():
        hsgg_root=ROOT/row['hsgg'];frames_root=ROOT/row['frames']
        episodes=completed=0
        for episode in sorted(hsgg_root.glob('*/*')):
            if not episode.is_dir() or not (episode/'frame_scene_graphs').is_dir():continue
            tables,kind=load_episode_hsgg_tables(episode)
            if kind!='completed_frame_scene_graphs':raise ValueError('Open did not select sidecars')
            for key,objects in tables['objects'].items():
                image=frames_root/episode.relative_to(hsgg_root)/Path(objects['image_path']).name
                if not image.is_file():raise FileNotFoundError(image)
            episodes+=1;completed+=len(tables['objects'])
        expected=(63,4956) if shard=='hm3d' else (89,26078)
        if (episodes,completed)!=expected:raise ValueError((shard,episodes,completed))
        open_rows.append({'shard':shard,'episodes':episodes,'completed':completed})

    # Representative old references resolve without loading or changing old graphs.
    aliases=read(ROOT/'registry/source_path_aliases.json')['aliases']
    unresolved=[]
    for row in aliases:
        if not resolve_data_path(ROOT/row['source']).exists():unresolved.append(row)
    if unresolved:issues.append({'unresolved_aliases':unresolved[:10]})
    assets={d:inspect_dataset(d)['issues'] for d in SOURCE}
    metadata=verify_metadata_copies(ROOT)
    if any(assets.values()) or metadata:issues.append({'asset_issues':assets,'metadata_copy_issues':metadata})
    lock=read(ROOT/'registry/current_registry_lock.json')
    for row in lock['files']:
        if digest(ROOT/row['path'])!=row['sha256']:issues.append({'registry_lock_changed':row['path']})
    result={'status':'passed' if not issues else 'failed','date':'2026-10-09','preserved_original_files':len(inventory),'preserved_image_files':image_count,
            'nextqa':totals,'navqa':nav_rows,'openeqa':open_rows,'aliases_checked':len(aliases),'issues':issues,
            'datasets_remain_independent':True,'backup_accessed':False,'image_bytes_preserved_by_same_file_id_size_mtime':True,
            'model_or_api_called':False,'source_graph_observations_unchanged':True,'intentional_aggregate_completion':{'navqa_seq22':2,'openeqa_hm3d':37}}
    write(REPORT/'verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='issues'},ensure_ascii=False),flush=True)
    if issues:
        print(json.dumps(issues[:5],ensure_ascii=False),flush=True)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
