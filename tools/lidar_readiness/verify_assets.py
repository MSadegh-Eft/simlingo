"""Offline verification of pinned local pretrained assets and installed encoder."""
import hashlib
import argparse
import json
from pathlib import Path
import tarfile

import utonia

from simlingo_training.lidar.data import SNAPSHOT
from simlingo_training.lidar.model import RELEASED
from simlingo_training.lidar.perception import WEIGHT_SHA256
from simlingo_training.lidar.storage import digest_file,atomic_json

ROOT=Path('/ehsan/m.sadegh/simlingo_lidar')


def main(output=None):
    assets=json.loads((ROOT/'reports/utonia-assets.json').read_text())
    archive=ROOT/'cache/utonia-de1f9deef4ee65e508d74347f8d0eb848d878abb.tar.gz'
    if digest_file(archive)!=assets['source_archive_sha256']: raise ValueError('Pinned Utonia source archive differs')
    with tarfile.open(archive) as tar:
        expected={m.name.split('/',1)[1]:hashlib.sha256(tar.extractfile(m).read()).hexdigest()
                  for m in tar if m.isfile() and m.name.endswith('.py')}
    source=Path(assets['source_directory'])
    actual={str(p.relative_to(source)):digest_file(p) for p in source.rglob('*.py')}
    generated={k:v for k,v in actual.items() if k.startswith('build/')}
    if generated!={'build/lib/utonia/__init__.py':expected['utonia/__init__.py']}:
        raise ValueError('Unexpected generated build source')
    if {k:v for k,v in actual.items() if not k.startswith('build/')}!=expected:
        raise ValueError('Extracted Utonia source differs')
    installed={str(p.relative_to(Path(utonia.__file__).parent)):digest_file(p) for p in Path(utonia.__file__).parent.rglob('*.py')}
    if installed!={k.removeprefix('utonia/'):v for k,v in expected.items() if k.startswith('utonia/')}:
        raise ValueError('Installed Utonia source differs')
    hf={}
    for path in sorted(SNAPSHOT.iterdir()):
        if not path.is_file(): continue
        blob=path.resolve().name
        if len(blob)==64: actual_blob=digest_file(path)
        elif len(blob)==40:
            payload=path.read_bytes(); actual_blob=hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()
        else: raise ValueError('Unexpected snapshot asset identity: '+str(path))
        if actual_blob!=blob: raise ValueError('HF snapshot asset differs: '+path.name)
        hf[path.name]=actual_blob
    released=digest_file(RELEASED); weights=digest_file(assets['weight_path'])
    if released!='ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28' or weights!=WEIGHT_SHA256:
        raise ValueError('Pretrained weights differ')
    report={'passed':True,'utonia_source_files':expected,'installed_utonia_files':installed,'generated_build_copy':generated,
        'hf_revision':SNAPSHOT.name,'hf_assets':hf,'released_sha256':released,'utonia_sha256':weights,
        'verification_source_sha256':digest_file(__file__)}
    atomic_json(output or ROOT/'readiness/assets.json',report); print('All local pretrained assets match their pinned identities',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path)
    main(p.parse_args().output)
