"""Rehash every final prepared shard; verify its pinned source relationships."""
import argparse
import json
from pathlib import Path

from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.train import verify_dataset


def main(manifest,output):
    ready=verify_dataset(manifest)
    document=json.loads(manifest.read_text())
    sources={e['path']:e['sha256'] for e in json.loads(Path('/ehsan/m.sadegh/datasets/simlingo/download_manifest.json').read_text())['files']}
    actual={}; total=0
    for name,metadata in sorted(document['shards'].items()):
        p=manifest.parent/name
        archive=p.name.removesuffix('.tar')+'.tar.gz'
        if metadata['source_sha256']!=sources[archive]: raise ValueError('Pinned source differs: '+name)
        sha=digest_file(p)
        if sha!=metadata['shard_sha256']: raise ValueError('Prepared shard changed: '+name)
        actual[name]={'sha256':sha,'bytes':p.stat().st_size}; total+=p.stat().st_size
        print(f'Verified {len(actual)}/{len(document["shards"])} shards: {name}',flush=True)
    indexed={v['shard'] for v in document['members'].values()}
    if indexed-set(actual): raise ValueError('Unverified shards in member index')
    atomic_json(output,{'passed':True,'ready':ready,'manifest_sha256':ready['manifest_sha256'],'shards':actual,'total_bytes':total})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); main(a.manifest,a.output)
