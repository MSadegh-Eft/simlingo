"""One bounded joint optimizer update, with frozen-weight and optimizer checks."""
import json
import os
from pathlib import Path

import torch

from simlingo_training.lidar.data import ManifestDataset, Request
from simlingo_training.lidar.model import make_model
from simlingo_training.lidar.engine import make_optimizer, step_window
from simlingo_training.lidar.storage import atomic_json
from tools.lidar_stage2.smoke import frozen_hash


def main(manifest=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and torch.cuda.device_count()==1
    expected='3e4277b2-2a3c-6f9b-6a87-273c09bfd49d'
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==expected
    torch.cuda.set_per_process_memory_fraction(.75); torch.set_num_threads(4); torch.manual_seed(42)
    root=Path('/ehsan/m.sadegh/simlingo_lidar/stage2')
    model=make_model().cuda(); model.set_stage('joint'); model.train()
    dataset=ManifestDataset(manifest or root/'data/manifest.json',model.tokenizer,max_samples=16)
    sample=dataset[Request(dataset.pools['driving'][0],'driving',False,42,0)]
    before=frozen_hash(model)
    optimizer,scheduler=make_optimizer(model,3750)
    # Save one parameter per group; whole frozen-module hashes cover all frozen weights.
    representatives={g['name']:[(p,p.detach().clone()) for p in g['params'][:2]] for g in optimizer.param_groups}
    stats=step_window(model,[sample],optimizer,scheduler)
    changed={name:any(not torch.equal(p.detach(),old) for p,old in pairs) for name,pairs in representatives.items()}
    assert all(changed.values()),changed
    assert frozen_hash(model)==before
    for state in optimizer.state.values():
        for key in ('exp_avg','exp_avg_sq'):
            assert state[key].dtype==torch.float32 and torch.isfinite(state[key]).all()
    model.zero_grad(set_to_none=True)
    report={'passed':True,'manifest_sha256':dataset.manifest_hash,'updates':1,'presentations':1,'groups_changed':changed,'frozen_weights_unchanged':True,
        'fp32_adam_state':True,'finite_losses':stats,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'limits':'One driving presentation; not the Stage 3 full-mixture memory profile'}
    atomic_json(root/'reports/joint-step.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--manifest',type=Path)
    main(parser.parse_args().manifest)
