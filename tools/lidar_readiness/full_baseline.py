"""Full released RGB baseline, with durable per-request recovery and no updates."""
import argparse
import json
from pathlib import Path

import torch

from simlingo_training.lidar.data import ManifestDataset
from simlingo_training.lidar.engine import checkpoint_metadata
from simlingo_training.lidar.numerics import make_training_model
from simlingo_training.lidar.runtime import initialize_device,check_resources,METADATA_CACHE
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.validation_run import evaluate_complete,recover_journal


def main(manifest,output):
    from simlingo_training.lidar.train import prelaunch_resources
    prelaunch_resources(output)
    output.mkdir(parents=True,exist_ok=True); initialize_device(output)
    ready=json.loads((manifest.parent/'ready.json').read_text())
    for key,name in [('manifest_sha256','manifest.json'),('validation_sha256','validation.json'),('audit_sha256','audit.json')]:
        if digest_file(manifest.parent/name)!=ready[key]: raise ValueError('Final dataset provenance differs')
    protocol=json.loads((manifest.parent/'validation.json').read_text())
    model=make_training_model().cuda().eval()
    binding=checkpoint_metadata(model,manifest,{'purpose':'Complete released RGB validation, zero updates',
        'numerics':model.lidar_numerics_policy,'mode':'rgb','generation_cap':100})
    binding['entrypoint_sha256']=digest_file(__file__)
    dataset=ManifestDataset(manifest,model.tokenizer,'val',metadata_cache=METADATA_CACHE)
    report=evaluate_complete(model,dataset,protocol['requests'],'rgb',output,binding,check=lambda:check_resources(output))
    report['optimizer_updates']=0; report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    atomic_json(output/'result.json',report)
    print(json.dumps({'status':'complete','samples':report['samples'],'metrics':report['metrics'],
        'coverage_passed':report['coverage_gate']['passed']}),flush=True)
    if not report['coverage_gate']['passed']: raise ValueError('Insufficient structured-target coverage; inspect before training')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from simlingo_training.lidar.launch_guard import training_lock
    with training_lock(): main(args.manifest,args.output)
