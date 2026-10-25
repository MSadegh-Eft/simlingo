"""Indexed, uncompressed tar shards. Compressed JPEG/LAZ/JSON payloads stay intact."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from collections import OrderedDict
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = 1


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.part')
    with temporary.open('w') as f:
        f.write(json.dumps(value, sort_keys=True, indent=2)+'\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, path)


class ShardReader:
    def __init__(self, root, index):
        self.root, self.index = Path(root), index
        self._pid, self._files, self._json = os.getpid(), {}, OrderedDict()
        self._json_bytes=0

    def __getstate__(self):
        return {'root':self.root,'index':self.index}

    def __setstate__(self,state):
        self.__init__(state['root'],state['index'])

    def close(self):
        for fd in self._files.values(): os.close(fd)
        self._files.clear()

    def __del__(self):
        if hasattr(self,'_files'): self.close()

    def read(self, member):
        entry = self.index[member]
        # pread has no shared file position. Reopen inherited descriptors per
        # process; avoid one SMB open/close round trip per tiny JSON read.
        if self._pid != os.getpid():
            self.close(); self._pid = os.getpid(); self._json.clear(); self._json_bytes=0
        shard = entry['shard']
        if shard not in self._files: self._files[shard] = os.open(self.root/shard,os.O_RDONLY)
        value = os.pread(self._files[shard],entry['size'],entry['offset'])
        if len(value) != entry['size']:
            raise IOError(f'Truncated shard payload: {member}')
        return value

    def json(self, member):
        if member in self._json:
            self._json.move_to_end(member)
            return self._json[member][0]
        decoded=gzip.decompress(self.read(member))
        value = json.loads(decoded)
        self._json[member] = (value,len(decoded)); self._json_bytes+=len(decoded)
        while len(self._json)>256 or self._json_bytes>8*1024**2:
            _,(_,size)=self._json.popitem(last=False); self._json_bytes-=size
        return value


def check_space(root, extra=0):
    if shutil.disk_usage(root).free - extra < 150*1024**3:
        raise RuntimeError('150 GiB shared free-space reserve reached')
    cap_root=Path(root).parent if Path(root).parent.name=='stage2' else Path(root)
    size = sum(p.stat().st_size for p in cap_root.rglob('*.tar*') if p.is_file())
    if size + extra > 100*1024**3:
        raise RuntimeError('100 GiB pilot-preparation cap reached')


def pack_archive(source, expected_sha, output, keep_routes=None):
    """Validate archive, then publish a tar and index with offsets after PAX headers."""
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = source.name.removesuffix('.tar.gz')
    shard, index_path = output/(stem+'.tar'), output/(stem+'.index.json')
    filter_hash = None if keep_routes is None else hashlib.sha256('\n'.join(sorted(keep_routes)).encode()).hexdigest()
    if index_path.exists():
        cached = json.loads(index_path.read_text())
        if cached.get('route_filter_hash') != filter_hash:
            raise RuntimeError('Shard route selection changed; prepare into a separate output directory')
        if cached['source_sha256'] == expected_sha and digest_file(shard) == cached['shard_sha256']:
            return cached['members']
        raise RuntimeError(f'Existing shard provenance/integrity differs: {shard}')
    if digest_file(source) != expected_sha:
        raise RuntimeError(f'Source checksum failed: {source}')
    members = {}
    temporary = shard.with_suffix('.tar.part')
    check_space(output,64*1024**2)
    unchecked=0
    # Large userspace buffers avoid a network round trip for each tar block.
    with source.open('rb',buffering=8*1024**2) as input_stream, temporary.open('wb',buffering=8*1024**2) as output_stream, \
            tarfile.open(fileobj=input_stream,mode='r|gz') as src, tarfile.open(fileobj=output_stream,mode='w') as dst:
        for member in src:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsafe archive member: {name}')
            if not member.isfile(): continue
            kind = name.parent.name
            if kind == 'rgb_augmented' or kind not in ('rgb','lidar','measurements','boxes','commentary','vqa','dreamer') and name.name != 'results.json.gz':
                continue
            route = str(name.parent if name.name == 'results.json.gz' else name.parent.parent)
            normalized = 'data/' + '/'.join(PurePosixPath(route).parts[1:])
            if keep_routes is not None and normalized not in keep_routes: continue
            unchecked += ((member.size+511)//512)*512+2048
            if unchecked >= 64*1024**2:
                check_space(output,member.size+64*1024**2)
                unchecked=0
            payload = src.extractfile(member).read()
            info = tarfile.TarInfo(str(name)); info.size = len(payload)
            dst.addfile(info, io.BytesIO(payload))
            offset = dst.fileobj.tell() - ((len(payload)+511)//512)*512
            if str(name) in members: raise ValueError(f'Duplicate member {name}')
            members[str(name)] = {'shard':shard.name,'offset':offset,'size':len(payload)}
    os.replace(temporary, shard)
    atomic_json(index_path, {'source_sha256':expected_sha,'shard_sha256':digest_file(shard),'route_filter_hash':filter_hash,'members':members})
    print(f'Packed {source.name}: {len(members)} members', flush=True)
    return members


def route_group(route):
    name = PurePosixPath(route).name
    town = name.split('_')[0]
    source = re.sub(r'^Town\d+_Rep\d+_', '', name).split('_route')[0]
    # Conservative source-route grouping, independent of repetition and timestamp.
    return f'{town}:{source}'


def quality_pass(result):
    if result['scores']['score_composed'] >= 100: return True
    benign = sum(len(result['infractions'][k]) for k in ('min_speed_infractions','outside_route_lanes'))
    return result['scores']['score_route'] > 94 and result['num_infractions'] == benign


def official_split(route):
    """Recognize both released collector layouts, without inferring from town."""
    parts=PurePosixPath(route).parts
    if 'routes_training' in parts or 'training_parking_lane' in parts: return 'train'
    if 'routes_validation' in parts or 'validation_parking_lane' in parts: return 'val'
    raise ValueError(f'Unrecognized official partition: {route}')


def merge_index(root,index,new):
    previous,current=ShardReader(root,index),ShardReader(root,new)
    for member in index.keys() & new.keys():
        if previous.read(member)!=current.read(member): raise ValueError(f'Conflicting archives: {member}')
    index.update(new)


def benchmark_routes(path):
    result=[]
    for route in ET.parse(path).getroot().findall('route'):
        points=np.array([[float(p.get(k)) for k in ('x','y','z')] for p in route.findall('./waypoints/position')])
        dense=[]
        for a,b in zip(points[:-1],points[1:]):
            dense.extend(a+(b-a)*t for t in np.linspace(0,1,max(2,int(np.linalg.norm(b-a))+1)))
        if dense: result.append((route.get('town'),route.get('id'),cKDTree(np.array(dense))))
    return result


def build_manifest(root, index, benchmark=None, maximum_train=100000):
    reader = ShardReader(root,index)
    route_measurements = {}
    for n in index:
        if '/measurements/' in n:
            route_measurements.setdefault(str(PurePosixPath(n).parent.parent),{})[int(PurePosixPath(n).name.split('.')[0])]=n
    routes = sorted(route_measurements)
    samples, groups, excluded = [], {}, {}
    benchmarks=benchmark_routes(benchmark) if benchmark else []
    excluded_groups=set()
    for route in routes:
        town=PurePosixPath(route).name.split('_')[0]
        candidates=[b for b in benchmarks if b[0]==town]
        if not candidates: continue
        measurements=route_measurements[route]
        positions=np.array([np.asarray(reader.json(measurements[f])['ego_matrix'])[:3,3] for f in sorted(measurements)])
        length=np.linalg.norm(np.diff(positions,axis=0),axis=1).sum()
        if length<30: continue
        # Conservative physical containment screen; do not infer identity from
        # numeric route IDs belonging to different collection XMLs.
        matches=[rid for _,rid,tree in candidates if np.mean(tree.query(positions)[0]<=3.)>=.95]
        if matches:
            excluded[route]=matches; excluded_groups.add(route_group(route))
    for route in routes:
        if route+'/results.json.gz' not in index or not quality_pass(reader.json(route+'/results.json.gz')): continue
        split = official_split(route)
        town = PurePosixPath(route).name.split('_')[0]
        if split is None or (split=='train' and town=='Town13') or (split=='val' and town!='Town13'):
            raise ValueError(f'Unrecognized official partition: {route}')
        group = route_group(route)
        if group in excluded_groups: continue
        if group in groups and groups[group] != split: raise ValueError(f'Split leakage: {group}')
        groups[group] = split
        measurements = route_measurements[route]
        for frame in sorted(measurements):
            if frame < 10: continue
            future = [measurements.get(frame+i) for i in range(12)]
            required = [f'{route}/{kind}/{frame:04}.{suffix}' for kind,suffix in [('rgb','jpg'),('lidar','laz'),('boxes','json.gz')]]
            if any(x is None for x in future) or not all(x in index for x in required): continue
            annotations = {}
            for family,folder in [('commentary','commentary'),('drivelm','vqa'),('dreamer','dreamer')]:
                path = family+'/'+route.removeprefix('data/')+f'/{folder}/{frame:04}.json.gz'
                if path in index: annotations[family] = path
            samples.append({'id':f'{route}:{frame:04}', 'route':route,'frame':frame,'group':group,'split':split,
                            'rgb':required[0],'lidar':required[1],'boxes':required[2],
                            'measurements':future,'annotations':annotations})
    # A fixed group order avoids size-biased selection if the final archive
    # takes us beyond the 100k pilot ceiling. No group is split across datasets.
    train_groups=sorted({r['group'] for r in samples if r['split']=='train'},key=lambda x:hashlib.sha256(x.encode()).hexdigest())
    counts={g:0 for g in train_groups}
    for row in samples:
        if row['split']=='train': counts[row['group']]+=1
    retained=set(); total=0
    for group in train_groups:
        if total+counts[group]<=maximum_train: retained.add(group); total+=counts[group]
    samples=[r for r in samples if r['split']=='val' or r['group'] in retained]
    groups={r['group']:r['split'] for r in samples}
    manifest = {'schema':SCHEMA,'preprocess':'ego-xyz-scale0.2-voxel0.01-first-v1','members':index,
                'samples':samples,'group_splits':groups,'preparation_complete':False,'benchmark_screen':{
                    'xml_sha256':digest_file(benchmark) if benchmark else None,
                    'rule':'Exclude the whole source group if any >=30m recorded trajectory is >=95% within 3m (3D) of a benchmark polyline',
                    'matches':excluded,'excluded_groups':sorted(excluded_groups),
                    'limits':'Physical containment screen; not proof of independent scenario design or of no prior released-checkpoint exposure'}}
    atomic_json(Path(root)/'manifest.json', manifest)
    return manifest


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,default=Path('/ehsan/m.sadegh/datasets/simlingo'))
    parser.add_argument('--include-validation',action='store_true')
    parser.add_argument('--minimum-train',type=int,default=0,help='Prepare additional archives until this many eligible training frames exist (pilot: 50000)')
    parser.add_argument('--maximum-train',type=int,default=100000)
    parser.add_argument('--benchmark',type=Path,default=Path(__file__).resolve().parents[2]/'leaderboard/data/bench2drive220.xml')
    parser.add_argument('--resume-sensors',action='store_true',help='Verify completed sensor shards and resume labels without rebuilding intermediate manifests')
    args = parser.parse_args()
    if not 0<=args.minimum_train<=args.maximum_train: parser.error('Require 0 <= minimum-train <= maximum-train')
    entries = json.loads((args.dataset/'download_manifest.json').read_text())['files']
    train = next(x for x in entries if x['path']=='data_simlingo_lb1_split_routes_training_VehicleTurningRoute_chunk_002.tar.gz')
    sensors = [train]
    if args.minimum_train:
        candidates=sorted((x for x in entries if x['path'].startswith('data_') and 'routes_training_' in x['path'] and x!=train),key=lambda x:x['size'])
        # Seed the pilot with both original towns and Town12 random-weather data.
        seed=next(x for x in candidates if 'training_3_scenarios' in x['path'])
        sensors += [seed]+[x for x in candidates if x!=seed]
    if args.include_validation:
        validation=min((x for x in entries if x['path'].startswith('data_') and 'validation_3_scenarios' in x['path']),key=lambda x:x['size'])
        sensors.insert(0,validation)
    index = {}
    used=[]
    if args.resume_sensors:
        paths={p.name.removesuffix('.index.json')+'.tar.gz' for p in args.output.glob('data_*.index.json')}
        if not paths: raise ValueError('No completed sensor shards to resume')
        for e in sorted((x for x in entries if x['path'] in paths),key=lambda x:x['path']):
            merge_index(args.output,index,pack_archive(args.dataset/'archives'/e['path'],e['sha256'],args.output)); used.append(e)
        if len(used)!=len(paths): raise ValueError('Sensor shard missing from pinned download manifest')
        manifest=build_manifest(args.output,index,args.benchmark,args.maximum_train)
        count=sum(x['split']=='train' for x in manifest['samples'])
        if args.include_validation and not any(x['split']=='val' for x in manifest['samples']): raise ValueError('No official validation shard to resume')
        print(f'Resumed verified sensor shards: {count} eligible training frames',flush=True)
    else:
        for e in sensors:
            new=pack_archive(args.dataset/'archives'/e['path'],e['sha256'],args.output)
            merge_index(args.output,index,new); used.append(e)
            manifest=build_manifest(args.output,index,args.benchmark,args.maximum_train)
            count=sum(x['split']=='train' for x in manifest['samples'])
            print(f'Eligible training frames after benchmark screen: {count}',flush=True)
            if count>0 and count>=args.minimum_train and (not args.include_validation or any(x['split']=='val' for x in manifest['samples'])): break
    if count<args.minimum_train: raise RuntimeError('Not enough training data within the available source archives')
    selected = {str(PurePosixPath(n).parent.parent) for n in index if '/measurements/' in n}
    prefixes={e['path'].removeprefix('data_').rsplit('_chunk_',1)[0] for e in used}
    for e in entries:
        family=e['path'].split('_',1)[0]
        prefix=e['path'].removeprefix(family+'_').rsplit('_chunk_',1)[0]
        if family in ('commentary','drivelm','dreamer') and prefix in prefixes:
            merge_index(args.output,index,pack_archive(args.dataset/'archives'/e['path'],e['sha256'],args.output,selected))
    manifest = build_manifest(args.output,index,args.benchmark,args.maximum_train)
    manifest['sources']=[{'path':e['path'],'sha256':e['sha256']} for e in used]
    manifest['preparation_complete']=not bool(args.minimum_train)
    atomic_json(args.output/'manifest.json',manifest)
    if args.minimum_train:
        from simlingo_training.lidar.validation import finalize_manifest
        finalize_manifest(args.output/'manifest.json')
    print(json.dumps({'samples':len(manifest['samples']), 'train':sum(x['split']=='train' for x in manifest['samples']),
                      'val':sum(x['split']=='val' for x in manifest['samples']), 'routes':len(manifest['group_splits'])}), flush=True)


if __name__ == '__main__': main()
