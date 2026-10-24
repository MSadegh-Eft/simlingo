"""Bounded paired-gradient diagnostic; no optimizer updates or retained weights."""
import hashlib
import json
import os
from pathlib import Path
import signal

import torch

from simlingo_training.lidar.data import ManifestDataset,Request
from simlingo_training.lidar.engine import normalized_loss
from simlingo_training.lidar.model import make_model
from simlingo_training.lidar.storage import atomic_json
from simlingo_training.lidar.numerics import configure_model
from tools.lidar_stage3.profile_pipeline import cpu_copy,compare_tensors,EXPECTED_UUID


def main(output):
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('10-minute diagnostic limit')))
    signal.alarm(600)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2' and torch.cuda.device_count()==1
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')==EXPECTED_UUID.removeprefix('GPU-')
    torch.set_num_threads(4); torch.cuda.set_per_process_memory_fraction(.85)
    torch.manual_seed(42); model=make_model().cuda(); model.set_stage('joint'); model.train()
    dataset=ManifestDataset('/ehsan/m.sadegh/simlingo_lidar/stage2/pilot/manifest.json',model.tokenizer,max_samples=16)
    sample=dataset[Request(dataset.pools['dreamer'][0],'dreamer',False,42,0)].to('cuda')
    initial=cpu_copy(model.state_dict().items())
    rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all(); report={}
    generated=[]; old=model.generate_ids
    def observed(*args,**kwargs):
        result=old(*args,**kwargs); generated.append(result.cpu().tolist()); return result
    model.generate_ids=observed
    for deterministic in (False,True):
        if deterministic:
            configure_model(model)
        runs=[]; reference=None
        for repetition in range(2):
            model.load_state_dict(initial); model.zero_grad(set_to_none=True)
            torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng); generated.clear()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                terms,_=model.loss_terms(sample,100); loss=normalized_loss(terms,model.loss_counts(sample))
            components={k:float(v.detach()) for k,v in terms.items()}
            loss.backward(); norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.3,error_if_nonfinite=True)
            torch.cuda.synchronize()
            gradients=[(n,p.grad) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None]
            run={'components':components,'unclipped_gradient_norm':float(norm),'generated':list(generated),
                 'cuda_rng_sha256':hashlib.sha256(torch.cuda.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
                 'cpu_rng_sha256':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
            if reference is None: reference=cpu_copy(gradients)
            else:
                run['gradient_comparison']=compare_tensors(gradients,reference,atol=1e-7,rtol=1e-3)
                worst=[]; squared_difference=0.
                for name,value in gradients:
                    difference=value.detach().cpu()-reference[name]
                    squared_difference+=float(difference.double().square().sum())
                    worst.append((float(difference.abs().max()),name))
                run['gradient_l2_difference']=squared_difference**.5
                run['worst_gradient_tensors']=sorted(worst,reverse=True)[:10]
            runs.append(run); del terms,loss,gradients
        report[str(deterministic)]={'runs':runs,'rng_equal':runs[0]['cuda_rng_sha256']==runs[1]['cuda_rng_sha256'],
                                    'generation_equal':runs[0]['generated']==runs[1]['generated'],
                                    'forward_components_equal':runs[0]['components']==runs[1]['components']}
        atomic_json(output,report); print(json.dumps({str(deterministic):report[str(deterministic)]},indent=2),flush=True)
        del reference
    signal.alarm(0)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
