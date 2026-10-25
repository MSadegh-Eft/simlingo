"""Focused cold/warm restart diagnostic: features and one gradient, no updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from simlingo_training.lidar.engine import normalized_loss,LOSS_WEIGHTS
from simlingo_training.lidar.train import create_session
from simlingo_training.lidar.runtime import sample_fingerprint,tensor_fingerprint
from simlingo_training.lidar.storage import atomic_json


def main(boundary,output,warm):
    state=torch.load(boundary,map_location='cpu',weights_only=False,mmap=True)
    metadata=state['metadata']; cfg=metadata['experiment_config']
    manifest=Path('/ehsan/m.sadegh/simlingo_lidar/stage2/pilot-parking-v2/manifest.json')
    session=create_session(manifest,output.parent,state['stage'],cfg['total_updates'],cfg['run'])
    model,data=session.model,session.dataset
    if warm:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for i in range(2):
                sample=data[session.sampler.request_at(i)].to('cuda')
                model.loss_terms(sample)
    session.resume(boundary)
    samples=[data[session.sampler.request_at(i)] for i in range(session.sampler.position,session.sampler.position+64)]
    counts={k:sum(model.loss_counts(s)[k] for s in samples) for k in LOSS_WEIGHTS}
    snapshots={}; summary={}
    def encoder_hook(module,args,point):
        hierarchy=[]
        while True:
            hierarchy.append(point)
            if 'pooling_parent' not in point: break
            point=point.pooling_parent
        for i,p in enumerate(reversed(hierarchy)):
            snapshots[f'encoder/{i}']=p.feat.detach().cpu().clone()
    def prefix_hook(module,args,out): snapshots['prefix']=out[0].detach().cpu().clone()
    def llm_hook(module,args,out): snapshots['llm']=out.last_hidden_state.detach().cpu().clone()
    hooks=[model.lidar.encoder.register_forward_hook(encoder_hook),model.lidar.register_forward_hook(prefix_hook),model.backbone().register_forward_hook(llm_hook)]
    for name,module in model.lidar.encoder.enc[-1].named_modules():
        def save(module,args,out,name=name):
            value=out if torch.is_tensor(out) else out.get('feat') if isinstance(out,dict) else getattr(out,'features',None)
            if torch.is_tensor(value): snapshots['final/'+name]=value.detach().cpu().clone()
        hooks.append(module.register_forward_hook(save))
    sample=samples[0].to('cuda')
    before=np.random.get_state()
    with torch.autocast('cuda',dtype=torch.bfloat16): terms,_=model.loss_terms(sample)
    after=np.random.get_state()
    loss=normalized_loss(terms,counts); loss.backward(); torch.cuda.synchronize()
    for h in hooks: h.remove()
    snapshots.update({'gradient/'+n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None})
    torch.save(snapshots,output)
    summary={'warm':warm,'sample':sample.uid,'fingerprint':sample_fingerprint(samples[0]),
        'terms':{k:float(v) for k,v in terms.items()},'np_rng_before':repr(before),'np_rng_after':repr(after),
        'tensor_hashes':{k:tensor_fingerprint([(k,v)]) for k,v in snapshots.items()}}
    from spconv.algo import CONV_CPP
    algorithms={}
    for channels in (54,108,216,432,576):
        result,found=CONV_CPP.get_tuned_algo(0,0,0,0,channels,channels,(8,9))
        algorithms[str(channels)]=str(result.algo_desp) if found else 'not tuned'
    summary['sparse_algorithms']=algorithms
    atomic_json(output.with_suffix('.json'),summary)
    print(json.dumps({k:summary[k] for k in ('warm','sample','terms','tensor_hashes')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--boundary',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--warm',action='store_true')
    a=p.parse_args(); main(a.boundary,a.output,a.warm)
