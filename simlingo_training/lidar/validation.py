"""Persist observed-geometry eligibility and fixed, route-balanced task requests."""
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import multiprocessing
import os

import torch

import cv2
import laspy
import numpy as np

from simlingo_training.lidar.data import ManifestDataset, TaskSampler, TASKS, SNAPSHOT
from simlingo_training.lidar.perception import prepare_cloud, geometry_targets
from simlingo_training.lidar.storage import ShardReader, atomic_json, digest_file


def route_balanced(rows,pool,count,seed=42):
    groups=defaultdict(list)
    for i in pool: groups[rows[i]['group']].append(i)
    rng=np.random.default_rng(seed)
    groups={k:list(rng.permutation(v)) for k,v in sorted(groups.items())}
    selected=[]
    while len(selected)<min(count,len(pool)):
        for group in rng.permutation(list(groups)):
            if groups[group]: selected.append(int(groups[group].pop()))
            if len(selected)>=min(count,len(pool)): break
    return selected


def validation_protocol(dataset):
    budgets={'driving':4096,'qa':2048,'commentary':512,'dreamer':512,'geometry_qa':512,'rear_driving':512}
    requests={}
    for task,count in budgets.items():
        selected=route_balanced(dataset.rows,dataset.pools[task],count)
        requests[task]=[{'index':i,'task':task,'replay':False,'seed':int(np.random.SeedSequence([42,j,71]).generate_state(1)[0]),
                         'position':j,'uid':dataset.rows[i]['id']} for j,i in enumerate(selected)]
    return {'schema':1,'manifest_sha256':dataset.manifest_hash,'split':'val','seed':42,'requests':requests,
            'monitor':{task:items[:min(64,len(items))] for task,items in requests.items()},
            'retention':'Use the same requests and text seeds with cloud omitted and geometry losses disabled for RGB-only checks',
            'metric_rule':'Average within source route group before averaging groups; use fixed generated answers for driving'}


_AUDIT_READER=None

def _audit_init(root,index):
    global _AUDIT_READER
    torch.set_num_threads(1); cv2.setNumThreads(0)
    _AUDIT_READER=ShardReader(root,index)


def _audit_row(row):
    reader=_AUDIT_READER
    image=cv2.imdecode(np.frombuffer(reader.read(row['rgb']),np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape!=(512,1024,3): raise ValueError(f'Unexpected RGB dimensions: {row["id"]}')
    measurements=[reader.json(p) for p in row['measurements']]
    matrices=np.asarray([m['ego_matrix'] for m in measurements])
    current=measurements[0]
    route=np.asarray(current['route']); nav=np.asarray([current['target_point'],current['target_point_next']])
    if matrices.shape!=(12,4,4) or not np.isfinite(matrices).all() or nav.shape!=(2,2) or not np.isfinite(nav).all() \
            or route.ndim!=2 or route.shape[1]!=2 or not len(route) or not np.isfinite(route).all() \
            or not np.isfinite(current['speed']) or current['command'] not in range(1,7):
        raise ValueError(f'Invalid current/future measurement: {row["id"]}')
    laz=laspy.read(io.BytesIO(reader.read(row['lidar'])),laz_backend=laspy.LazBackend.Lazrs)
    xyz=np.column_stack((laz.x,laz.y,laz.z)); cloud=prepare_cloud(xyz)
    geometry=geometry_targets(cloud,reader.json(row['boxes']))
    stats={'raw_points':len(xyz),'voxel_points':len(cloud.coord),'furthest_point_metres':float(np.linalg.norm(xyz,axis=1).max())}
    return {k:v.tolist() for k,v in geometry.items()},stats


def finalize_manifest(path):
    path=Path(path); document=json.loads(path.read_text())
    document['preparation_complete']=False
    atomic_json(path,document)
    # Recheck cross-shard duplicates, including annotations, before publishing.
    reader=ShardReader(path.parent,document['members'])
    document['shards']=document.get('inherited_shards',{}).copy()
    for index_path in sorted(path.parent.glob('*.index.json')):
        metadata=json.loads(index_path.read_text()); index=metadata['members']
        shard=index_path.name.removesuffix('.index.json')+'.tar'
        document['shards'][shard]={k:metadata[k] for k in ('source_sha256','shard_sha256','route_filter_hash')}
        candidate=ShardReader(path.parent,index)
        for name,entry in index.items():
            if name not in document['members']: raise ValueError(f'Unaccounted prepared member: {name}')
            if entry!=document['members'][name] and candidate.read(name)!=reader.read(name):
                raise ValueError(f'Conflicting prepared annotation/sensor member: {name}')
        candidate.close()
    reader.close()
    if not document.get('geometry_audit_complete'):
        fingerprint=hashlib.sha256((digest_file(path)+'rgb-future-voxel-geometry-v2').encode()).hexdigest()
        cache=path.parent/f'geometry-{fingerprint}.jsonl'
        completed=0
        if cache.exists():
            with cache.open('r+') as handle:
                while True:
                    position=handle.tell(); line=handle.readline()
                    if not line: break
                    if not line.endswith('\n'): handle.seek(position); handle.truncate(); break
                    record=json.loads(line); row=document['samples'][completed]
                    if record['uid']!=row['id']: raise ValueError('Geometry recovery cache order differs')
                    row['geometry'],row['lidar_stats']=record['geometry'],record['stats']; completed+=1
            print(f'Recovered {completed} audited frames',flush=True)
        os.environ['OPENBLAS_NUM_THREADS']='1'
        remaining=document['samples'][completed:]
        with cache.open('a') as handle,ProcessPoolExecutor(max_workers=4,initializer=_audit_init,initargs=(path.parent,document['members']),mp_context=multiprocessing.get_context('spawn')) as executor:
            for i,(row,result) in enumerate(zip(remaining,executor.map(_audit_row,remaining,chunksize=16)),start=completed):
                row['geometry'],row['lidar_stats']=result
                handle.write(json.dumps({'uid':row['id'],'geometry':result[0],'stats':result[1]})+'\n')
                if (i+1)%1000==0:
                    handle.flush(); os.fsync(handle.fileno())
                    print(f'Geometry eligibility audited: {i+1}/{len(document["samples"])}',flush=True)
            handle.flush(); os.fsync(handle.fileno())
    document['geometry_audit_complete']=True
    document['preparation_complete']=True
    atomic_json(path,document)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    tokenizer=getattr(processor,'tokenizer',processor)
    train=ManifestDataset(path,tokenizer,'train'); val=ManifestDataset(path,tokenizer,'val')
    protocol=validation_protocol(val)
    atomic_json(path.parent/'validation.json',protocol)
    sampler=TaskSampler(train.pools,'joint',manifest_hash=train.manifest_hash,bucket_pools=train.bucket_pools,bucket_weights=train.bucket_weights)
    requests=sampler.take(960)
    report={'manifest_sha256':train.manifest_hash,'samples':{'train':len(train),'val':len(val)},
        'integrity':'Every eligible RGB decoded; LAZ voxelized; boxes, current/navigation and complete future measurements validated',
        'groups':{'train':len({r['group'] for r in train.rows}),'val':len({r['group'] for r in val.rows})},
        'pools':{'train':{k:len(v) for k,v in train.pools.items()},'val':{k:len(v) for k,v in val.pools.items()}},
        'pool_exclusions':{'train':{k:len(train)-len(v) for k,v in train.pools.items()},'val':{k:len(val)-len(v) for k,v in val.pools.items()}},
        'bucket_counts':{task:{k:len(v) for k,v in buckets.items()} for task,buckets in train.bucket_pools.items()},
        'missing_buckets':{task:[k for k,v in buckets.items() if not v] for task,buckets in train.bucket_pools.items()},
        'joint_cycle_tasks':dict(Counter(r.task for r in requests)),'joint_cycle_replay':sum(r.replay for r in requests),
        'validation_counts':{k:len(v) for k,v in protocol['requests'].items()},'benchmark_screen':document['benchmark_screen']}
    report['geometry']={split:{'presence_valid_per_sector':dataset.geometry['presence_valid'].sum(axis=0).tolist(),
                              'distance_valid_per_sector':dataset.geometry['distance_valid'].sum(axis=0).tolist()}
                        for split,dataset in [('train',train),('val',val)]}
    report['lidar_stats']={key:{'minimum':float(np.min(values)),'median':float(np.median(values)),'p99':float(np.percentile(values,99)),
                              'maximum':float(np.max(values))}
        for key in ('raw_points','voxel_points','furthest_point_metres')
        for values in [[row['lidar_stats'][key] for row in document['samples'] if 'lidar_stats' in row]] if values}
    assert not ({r['group'] for r in train.rows}&{r['group'] for r in val.rows})
    assert all(val.pools[t] and train.pools[t] for t in TASKS)
    atomic_json(path.parent/'audit.json',report)
    atomic_json(path.parent/'ready.json',{'manifest_sha256':digest_file(path),'validation_sha256':digest_file(path.parent/'validation.json'),
                                       'audit_sha256':digest_file(path.parent/'audit.json'),'passed':True})
    print(json.dumps({k:v for k,v in report.items() if k in ('samples','groups','pools','validation_counts')}),flush=True)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('manifest',type=Path)
    finalize_manifest(parser.parse_args().manifest)
