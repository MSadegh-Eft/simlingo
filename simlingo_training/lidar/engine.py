"""Single-GPU window normalization, explicit optimizer state, and atomic resume."""
import os
from pathlib import Path
import random
import importlib.metadata

import numpy as np
import torch

LOSS_WEIGHTS = {'route':1.,'speed':1.,'language':1.,'presence':.2,'distance':.2}


def checkpoint_metadata(model,manifest,experiment_config):
    """Bind a resumable run to its data, code, task recipe and environment."""
    from simlingo_training.lidar.data import ALIGN_COUNTS, JOINT_COUNTS, SNAPSHOT
    from simlingo_training.lidar.perception import PREPROCESS_VERSION, WEIGHT_SHA256
    from simlingo_training.lidar.storage import digest_file
    root=Path(__file__).resolve().parents[2]
    sources=sorted((root/'simlingo_training').rglob('*.py'))
    code={str(p.relative_to(root)):digest_file(p) for p in sources}
    manifest=Path(manifest)
    return {'schema':1,'stage':model.stage,'generation_cache':getattr(model,'use_cache',False),'manifest_sha256':digest_file(manifest),
        'validation_sha256':digest_file(manifest.parent/'validation.json') if (manifest.parent/'validation.json').exists() else None,
        'preprocess':PREPROCESS_VERSION,'utonia_sha256':WEIGHT_SHA256,
        'hf_snapshot_revision':SNAPSHOT.name,
        'released_config_sha256':digest_file(root/'models/simlingo/simlingo/.hydra/config.yaml'),
        'released_sha256':'ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28',
        'code':code,'task_counts':list(ALIGN_COUNTS if model.stage=='align' else JOINT_COUNTS),
        'loss_weights':LOSS_WEIGHTS.copy(),'maximum_lrs':{g['name']:g['lr'] for g in model.optimizer_groups()},
        'optimizer':{'type':'AdamW','weight_decay':.1,'beta2':.999,'eps':1e-8,'clip_norm':.3},
        'scheduler':{'type':'OneCycleLR','warmup_fraction':.05,'div_factor':25,'final_div_factor':10000,'beta1':[.95,.85,.95]},
        'versions':{k:importlib.metadata.version(k) for k in ('torch','transformers','peft','spconv-cu124','flash-attn')},
        'experiment_config':experiment_config}


def make_optimizer(model,total_updates):
    optimizer = torch.optim.AdamW(model.optimizer_groups(),weight_decay=.1,betas=(.9,.999),eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=[g['lr'] for g in optimizer.param_groups],
        total_steps=total_updates,pct_start=.05,anneal_strategy='cos',div_factor=25,final_div_factor=10000,
        cycle_momentum=True,base_momentum=.85,max_momentum=.95)
    return optimizer,scheduler


def normalized_loss(terms,counts):
    if not terms: raise ValueError('A presentation must have at least one active objective')
    return sum(LOSS_WEIGHTS[k]*v/max(1,counts[k]) for k,v in terms.items())


def step_window(model,samples,optimizer,scheduler=None,device='cuda',generation_limit=100):
    if not samples: raise ValueError('Empty accumulation window')
    counts = {k:sum(model.loss_counts(s)[k] for s in samples) for k in LOSS_WEIGHTS}
    optimizer.zero_grad(set_to_none=True)
    numerators = {k:0. for k in LOSS_WEIGHTS}
    for sample in samples:
        sample = sample.to(device)
        with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=torch.device(device).type=='cuda'):
            terms,_ = model.loss_terms(sample,generation_limit)
            loss = normalized_loss(terms,counts)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite objective; no optimizer update performed')
        loss.backward()
        for k,v in terms.items(): numerators[k] += float(v.detach())
    parameters = [p for g in optimizer.param_groups for p in g['params']]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters,.3,error_if_nonfinite=True)
    optimizer.step()
    if scheduler is not None: scheduler.step()
    return {'loss':sum(LOSS_WEIGHTS[k]*numerators[k]/max(1,counts[k]) for k in LOSS_WEIGHTS),
            'components':{k:numerators[k]/max(1,counts[k]) for k in LOSS_WEIGHTS},
            'counts':counts,'gradient_norm':float(grad_norm)}


def save_checkpoint(path,model,optimizer,scheduler,sampler,updates,metadata):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if any(p.grad is not None for p in model.parameters()):
        # The caller must clear gradients at an optimizer boundary, not mid-window.
        raise ValueError('Clear gradients before checkpointing at a completed update')
    state = {'schema':1,'model':model.state_dict(),'stage':model.stage,'optimizer':optimizer.state_dict(),
             'scheduler':scheduler.state_dict() if scheduler else None,'sampler':sampler.state_dict(),
             'updates':updates,'metadata':metadata,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
             'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}}
    temporary = path.with_suffix(path.suffix+'.part')
    with temporary.open('wb') as f:
        torch.save(state,f); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def load_checkpoint(path,model,optimizer,scheduler,sampler,metadata):
    # Only checkpoints produced by this trusted local training process are accepted.
    state = torch.load(str(path),map_location='cpu',weights_only=False)
    if state['schema']!=1 or state['stage']!=model.stage or state['metadata']!=metadata:
        raise ValueError('Checkpoint stage/config/provenance differs')
    if (state['scheduler'] is None)!=(scheduler is None):
        raise ValueError('Checkpoint scheduler presence differs')
    model.load_state_dict(state['model'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if scheduler: scheduler.load_state_dict(state['scheduler'])
    sampler.load_state_dict(state['sampler'])
    random.setstate(state['rng']['python']); np.random.set_state(state['rng']['numpy']); torch.set_rng_state(state['rng']['torch'])
    if state['rng']['cuda'] is not None: torch.cuda.set_rng_state_all(state['rng']['cuda'])
    return state['updates']
