"""Two separate processes prove the real next accumulation-64 update recovers."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from simlingo_training.lidar.runtime import check_resources,sample_fingerprint
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.train import create_session,prelaunch_resources


def compare(values,expected):
    values=dict(values)
    if values.keys()!=expected.keys(): raise ValueError('State tensor keys differ')
    mismatches=[]; count=0
    for key,value in values.items():
        count+=1
        if value.dtype!=expected[key].dtype or not torch.equal(value.detach().cpu(),expected[key]): mismatches.append(str(key))
    return {'bitwise':not mismatches,'tensors':count,'mismatches':mismatches[:30]}


def main(manifest,output,stage,phase,context_audit=None,total_updates=None,initial_weights=None,handoff=None):
    prelaunch_resources(output)
    output.mkdir(parents=True,exist_ok=True)
    print('Constructing production training session for '+phase,flush=True)
    config={'purpose':'Disposable fresh-process readiness check','entrypoint_sha256':digest_file(__file__)}
    if initial_weights: config['initial_weights_sha256']=digest_file(initial_weights)
    session=create_session(manifest,output,stage,total_updates or (750 if stage=='align' else 3750),config,initial_weights)
    if session.optimizer.state: raise ValueError('New stage must start with a fresh optimizer')
    model,dataset=session.model,session.dataset
    boundary,expected=output/'boundary.pt',output/'expected-next.pt'
    started=time.monotonic()
    def check(): check_resources(output)
    if phase=='reference':
        initial_requests=[asdict(session.sampler.request_at(i)) for i in range(64)]
        initial_start=time.monotonic(); initial_stats=session.update(check=check); initial_seconds=time.monotonic()-initial_start
        save_started=time.monotonic(); session.save(boundary); save_seconds=time.monotonic()-save_started
        atomic_json(output/'boundary.json',{'sha256':digest_file(boundary),'bytes':boundary.stat().st_size,'pid':os.getpid(),'save_seconds':save_seconds})
    else:
        source=json.loads((output/'boundary.json').read_text())
        if source['pid']==os.getpid() or digest_file(boundary)!=source['sha256']: raise ValueError('Not a fresh validated process')
        load_started=time.monotonic(); session.resume(boundary); load_seconds=time.monotonic()-load_started
        loaded=torch.load(boundary,map_location='cpu',weights_only=False,mmap=True)
        a=compare(model.state_dict().items(),loaded['model'])
        b=compare(((f'{i}:{k}',v) for i,s in session.optimizer.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v)),
                  {f'{i}:{k}':v for i,s in loaded['optimizer']['state'].items() for k,v in s.items() if torch.is_tensor(v)})
        if not (a['bitwise'] and b['bitwise']): raise ValueError('Loaded boundary changed')
        del loaded
    observations={}
    def observe(requests,samples):
        observations['requests']=[asdict(r) for r in requests]
        observations['samples']=[{'uid':s.uid,'fingerprint':sample_fingerprint(s),'timestamp':s.cloud.timestamp if s.cloud is not None else None} for s in samples]
    generations=[]; original=model.generate_ids
    def generated(sample,*args,**kwargs):
        ids=original(sample,*args,**kwargs)
        generations.append({'uid':sample.uid,'position':sample.position,'ids':ids.detach().cpu().tolist()})
        return ids
    model.generate_ids=generated
    update_started=time.monotonic(); stats=session.update(observe=observe,check=check); update_seconds=time.monotonic()-update_started
    model.generate_ids=original
    observations.update({'generations':generations,'stats':stats,'pid':os.getpid(),'seconds':time.monotonic()-started,'update_seconds':update_seconds})
    if phase=='reference':
        observations['initial_update']={'seconds':initial_seconds,'stats':initial_stats,'requests':initial_requests}
        session.save(expected)
        atomic_json(output/'reference.json',observations)
        print(f'Reference exited at update {session.updates}; fresh process must now resume boundary',flush=True)
        return
    reference=json.loads((output/'reference.json').read_text())
    expected_state=torch.load(expected,map_location='cpu',weights_only=False,mmap=True)
    model_comparison=compare(model.state_dict().items(),expected_state['model'])
    optimizer_comparison=compare(((f'{i}:{k}',v) for i,s in session.optimizer.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v)),
        {f'{i}:{k}':v for i,s in expected_state['optimizer']['state'].items() for k,v in s.items() if torch.is_tensor(v)})
    rng=expected_state['rng']
    checks={'model':model_comparison['bitwise'],'optimizer':optimizer_comparison['bitwise'],
        'scheduler':session.scheduler.state_dict()==expected_state['scheduler'],
        'optimizer_groups':session.optimizer.state_dict()['param_groups']==expected_state['optimizer']['param_groups'],
        'sampler':session.sampler.state_dict()==expected_state['sampler'],
        'requests':observations['requests']==reference['requests'],'samples':observations['samples']==reference['samples'],
        'generated_tokens':generations==reference['generations'],'loss_stats':stats==reference['stats'],
        'python_rng':random.getstate()==rng['python'],
        'numpy_rng':all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),rng['numpy'])),
        'torch_rng':torch.equal(torch.get_rng_state(),rng['torch']),
        'cuda_rng':all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda'])),
        'completed_boundary':session.updates==expected_state['updates']==2 and session.sampler.position==128}
    checks['tracking']=all(session.run_state[k]==expected_state['run_state'][k]
                           for k in ('seen_indices','task_counts','replay_count','bucket_counts'))
    report={'passed':all(checks.values()),'checks':checks,'stage':stage,'model':model_comparison,'optimizer':optimizer_comparison,
        'reference_pid':reference['pid'],'resume_pid':os.getpid(),'accumulation':64,'generation_cap':100,
        'manifest_sha256':dataset.manifest_hash,'metadata':session.metadata,
        'reference_seconds':reference['seconds'],'resume_seconds':observations['seconds'],
        'reference_initial_update':reference['initial_update'],'reference_next_update_seconds':reference['update_seconds'],
        'resume_update_seconds':update_seconds,
        'boundary_bytes':source['bytes'],'save_seconds':source['save_seconds'],'load_seconds':load_seconds,
        'unique_window_task_counts':dict(Counter(r['task'] for r in reference['initial_update']['requests']+reference['requests'])),
        'unique_window_replay_count':sum(r['replay'] for r in reference['initial_update']['requests']+reference['requests']),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'temporary_weights_selected':False,
        'scope':'Separate interpreter/model/optimizer/data construction. Same physical server; simulates restart without rebooting the shared server.'}
    atomic_json(output/'result.json',report)
    print(json.dumps(report),flush=True)
    if not report['passed']: raise AssertionError('Fresh-process recovery failed; preserve artifacts for diagnosis')
    del expected_state
    if context_audit:
        from tools.lidar_readiness.stress_bounds import run
        run(model,dataset,context_audit,output)
        report['stress_evidence_sha256']=digest_file(output/'global-stress.json')
        atomic_json(output/'result.json',report)
    boundary.unlink()
    if handoff:
        if stage!='align' or handoff.exists(): raise ValueError('Invalid disposable alignment handoff')
        expected.rename(handoff)
    else: expected.unlink()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['align','joint'],required=True); p.add_argument('--phase',choices=['reference','resume'],required=True)
    p.add_argument('--context-audit',type=Path)
    p.add_argument('--total-updates',type=int)
    p.add_argument('--initial-weights',type=Path); p.add_argument('--handoff',type=Path)
    args=p.parse_args()
    from simlingo_training.lidar.launch_guard import training_lock
    with training_lock(): main(args.manifest,args.output,args.stage,args.phase,args.context_audit,args.total_updates,args.initial_weights,args.handoff)
