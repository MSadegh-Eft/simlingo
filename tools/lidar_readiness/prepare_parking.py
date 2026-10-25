"""Versioned pilot extension: audited parking groups, no changes to old artifacts."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path, PurePosixPath
import tarfile

from simlingo_training.lidar.storage import (atomic_json, digest_file, pack_archive,
    merge_index, build_manifest, route_group)
from simlingo_training.lidar.validation import _audit_init, _audit_row, finalize_manifest

ROOT=Path('/ehsan/m.sadegh/simlingo_lidar')
SOURCE=Path('/ehsan/m.sadegh/datasets/simlingo')
OUT=ROOT/'stage2/pilot-parking-v2'
BASE=ROOT/'stage2/pilot'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'ready.json').exists():
        ready=json.loads((OUT/'ready.json').read_text())
        for key,name in [('manifest_sha256','manifest.json'),('validation_sha256','validation.json'),('audit_sha256','audit.json')]:
            if digest_file(OUT/name)!=ready[key]: raise ValueError('Published extension changed')
        print('Extension already complete',flush=True); return
    ready=json.loads((BASE/'ready.json').read_text())
    for key,name in [('manifest_sha256','manifest.json'),('validation_sha256','validation.json'),('audit_sha256','audit.json')]:
        if digest_file(BASE/name)!=ready[key]: raise ValueError('Original pilot provenance changed')
    base=json.loads((BASE/'manifest.json').read_text())
    files=json.loads((SOURCE/'download_manifest.json').read_text())['files']
    entries=[e for e in files if 'parking_lane' in e['path']]
    selected=set(); selection={}
    for split,limit in [('training',32),('validation',16)]:
        e=next(e for e in entries if e['path'].startswith(f'drivelm_simlingo_{split}_'))
        groups=defaultdict(set)
        # The small annotation archive inventories source groups before scanning
        # the large sensor archive. Include every repetition of selected groups.
        with tarfile.open(SOURCE/'archives'/e['path'],'r|gz') as archive:
            for member in archive:
                p=PurePosixPath(member.name)
                if member.isfile() and p.parent.name=='vqa':
                    route='data/'+str(p.parent.parent).split('/',1)[1]
                    groups[route_group(route)].add(route)
        ordered=sorted(groups,key=lambda g:hashlib.sha256(g.encode()).hexdigest())[:limit]
        routes={r for g in ordered for r in groups[g]}
        selected.update(routes)
        selection[split]={'available_groups':len(groups),'selected_groups':ordered,'routes':len(routes)}
    atomic_json(OUT/'selection.json',selection)
    print('Parking selection: '+json.dumps(selection),flush=True)
    index={}
    for e in sorted(entries,key=lambda e:(e['path'].startswith('data_'),e['size'])):
        print('Preparing '+e['path'],flush=True)
        merge_index(OUT,index,pack_archive(SOURCE/'archives'/e['path'],e['sha256'],OUT,selected))
    benchmark=Path(__file__).resolve().parents[2]/'leaderboard/data/bench2drive220.xml'
    extension=build_manifest(OUT,index,benchmark)
    # Persist and reuse only a completely audited extension. Original audited
    # samples are reused verbatim, rather than recomputing their labels.
    cache=OUT/'parking-audited.json'
    fingerprint=digest_file(OUT/'manifest.json')
    if cache.exists():
        audited=json.loads(cache.read_text())
        if audited['manifest_sha256']!=fingerprint: raise ValueError('Parking audit input changed')
        extension['samples']=audited['samples']
    else:
        with ProcessPoolExecutor(max_workers=4,initializer=_audit_init,initargs=(OUT,index),
                mp_context=multiprocessing.get_context('spawn')) as pool:
            for j,(row,(geometry,stats)) in enumerate(zip(extension['samples'],pool.map(_audit_row,extension['samples'],chunksize=16))):
                row['geometry'],row['lidar_stats']=geometry,stats
                if (j+1)%1000==0: print(f'Parking sensor audit {j+1}/{len(extension["samples"])}',flush=True)
        atomic_json(cache,{'manifest_sha256':fingerprint,'samples':extension['samples']})
    old_index={k:{**v,'shard':'../pilot/'+v['shard']} for k,v in base['members'].items()}
    merge_index(OUT,index,old_index)
    combined=build_manifest(OUT,index,benchmark)
    audited={r['id']:r for r in base['samples']+extension['samples']}
    if len(audited)!=len(base['samples'])+len(extension['samples']): raise ValueError('Duplicate sample identity')
    for row in combined['samples']:
        reference=audited[row['id']]
        if row!= {k:v for k,v in reference.items() if k not in ('geometry','lidar_stats')}:
            raise ValueError('Composed sample changed: '+row['id'])
        row['geometry'],row['lidar_stats']=reference['geometry'],reference['lidar_stats']
    if {r['id'] for r in combined['samples']}!=set(audited): raise ValueError('Composition unexpectedly removed samples')
    combined['geometry_audit_complete']=True
    combined['inherited_shards']={'../pilot/'+name:metadata for name,metadata in base['shards'].items()}
    combined['parent_ready']=ready
    combined['sources']=base['sources']+[{'path':e['path'],'sha256':e['sha256']} for e in entries if e['path'].startswith('data_')]
    combined['parking_selection']=selection
    atomic_json(OUT/'manifest.json',combined)
    report=finalize_manifest(OUT/'manifest.json')
    if any(report['missing_buckets'].values()): raise ValueError('Missing training buckets remain')
    print('Parking extension complete '+json.dumps(report['samples']),flush=True)


if __name__=='__main__': main()
