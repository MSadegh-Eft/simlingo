"""Finite GPU-2 pipeline profiling; no main-training entry point or retained weights."""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

import numpy as np
import psutil
import torch

from simlingo_training.lidar.data import ManifestDataset, Request, TaskSampler, TASKS
from simlingo_training.lidar.engine import make_optimizer, step_window, checkpoint_metadata, save_checkpoint, load_checkpoint
from simlingo_training.lidar.evaluation import evaluate_requests
from simlingo_training.lidar.model import make_model
from simlingo_training.lidar.numerics import prepare_environment,make_training_model
from simlingo_training.lidar.storage import atomic_json, digest_file

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/ehsan/m.sadegh/simlingo_lidar')
EXPECTED_UUID='GPU-3e4277b2-2a3c-6f9b-6a87-273c09bfd49d'
GIB=1024**3


def identity(items): return items


def system_state():
    process=psutil.Process()
    children=process.children(recursive=True)
    return {'process_rss_bytes':process.memory_info().rss,
            'children_rss_bytes':sum(p.memory_info().rss for p in children if p.is_running()),
            'host_available_bytes':psutil.virtual_memory().available,
            'gpu':subprocess.check_output(['nvidia-smi','-i','2',
                '--query-gpu=name,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader'],text=True).strip()}


def verify_stage2():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()=='feat/simlingo-lidar'
    stage2=json.loads((REPO/'tools/lidar_stage2/stage2-results.json').read_text())
    assert stage2['status']=='complete' and not stage2['long_training_started']
    ready=json.loads((ROOT/'stage2/pilot/ready.json').read_text())
    for key,name in [('manifest_sha256','manifest.json'),('validation_sha256','validation.json'),('audit_sha256','audit.json')]:
        assert digest_file(ROOT/'stage2/pilot'/name)==ready[key],name
    assert ready==stage2['pilot_ready'] and ready['passed']
    metadata=json.loads((ROOT/'stage2/reports/checkpoint-metadata.json').read_text())
    for name,expected in metadata['code'].items(): assert digest_file(REPO/name)==expected,name
    return ready


def hash_tensors(values):
    digest=hashlib.sha256()
    for name,value in values:
        digest.update(str(name).encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_copy(values):
    return {str(k):v.detach().cpu().clone() for k,v in values}


def compare_tensors(values,reference,atol,rtol):
    """Compare every state tensor, including small/zero Adam moments.

    Tolerances are declared before this diagnostic runs. Exact checkpoint reload
    is checked separately; GPU next-update repeatability can differ by rounding.
    No tolerance is changed automatically on a failed comparison.
    """
    exact=True; close=True; max_error=0.; different=0; count=0
    for name,value in values:
        a=value.detach().cpu(); b=reference[str(name)]
        equal=torch.equal(a,b); exact=exact and equal; count+=1
        if not equal:
            different+=1
            max_error=max(max_error,float((a-b).abs().max()))
            close=close and torch.allclose(a,b,atol=atol,rtol=rtol,equal_nan=False)
    return {'bitwise':exact,'within_tolerance':bool(close),'max_absolute_error':max_error,
            'different_tensors':different,'compared_tensors':count,'atol':atol,'rtol':rtol}


class Timings:
    """Observe the real step_window with temporary method wrappers.

    CUDA events measure component device intervals, which include waiting caused
    by concurrent jobs. Wall times synchronize only at presentation boundaries.
    Nested component times must not be added to the inclusive forward time.
    """
    def __init__(self,model,check,output):
        self.model,self.check,self.output=model,check,output
        self.rows=[]; self.events=[]; self.phase='learning'; self.active=None; self.original=[]

    def patch(self,obj,name,function):
        old=getattr(obj,name); self.original.append((obj,name,old)); setattr(obj,name,function(old))

    def event_wrapper(self,label):
        def wrap(original):
            def call(*args,**kwargs):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); result=original(*args,**kwargs); end.record()
                self.events.append((label+'_'+self.phase,start,end))
                return result
            return call
        return wrap

    def __enter__(self):
        def forward(original):
            def call(sample,*args,**kwargs):
                self.check(); torch.cuda.synchronize(); self.events=[]
                self.active={'task':sample.task,'replay':sample.cloud is None,'uid':sample.uid,
                             'teacher_tokens':len(sample.ids),'question_tokens':len(sample.question_ids),
                             'voxel_points':len(sample.cloud.coord) if sample.cloud is not None else 0}
                started=time.monotonic(); result=original(sample,*args,**kwargs); torch.cuda.synchronize()
                self.active['forward_seconds']=time.monotonic()-started
                return result
            return call
        def generation(original):
            def call(*args,**kwargs):
                prior=self.phase; self.phase='generation'
                try:
                    result=original(*args,**kwargs)
                    if self.active is not None: self.active['generated_tokens']=int(result.numel())
                    return result
                finally: self.phase=prior
            return call
        def backward(original):
            def call(tensor,*args,**kwargs):
                start=time.monotonic(); result=original(tensor,*args,**kwargs); torch.cuda.synchronize()
                if self.active is not None:
                    row=self.active; row['backward_seconds']=time.monotonic()-start
                    components=defaultdict(float)
                    for name,a,b in self.events: components[name]+=a.elapsed_time(b)/1000
                    row['components_device_seconds']=dict(components)
                    row['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                    row['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
                    self.rows.append(row); self.active=None; self.events=[]
                    with self.output.open('a') as handle: handle.write(json.dumps(row)+'\n')
                return result
            return call
        self.patch(self.model,'loss_terms',forward)
        self.patch(self.model,'generate_ids',generation)
        self.patch(self.model,'hidden',self.event_wrapper('llm'))
        self.patch(self.model.vision_model.image_encoder.model,'extract_feature',self.event_wrapper('rgb'))
        self.patch(self.model.lidar,'forward',self.event_wrapper('lidar'))
        self.patch(torch.Tensor,'backward',backward)
        return self

    def __exit__(self,*args):
        for obj,name,original in reversed(self.original): setattr(obj,name,original)


def timed_window(model,samples,optimizer,scheduler,check):
    check(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    before=system_state(); start=time.monotonic(); optimizer_time=[]
    original=optimizer.step
    def observed_step(*args,**kwargs):
        torch.cuda.synchronize(); begin=time.monotonic(); result=original(*args,**kwargs); torch.cuda.synchronize()
        optimizer_time.append(time.monotonic()-begin); return result
    optimizer.step=observed_step
    try: stats=step_window(model,samples,optimizer,scheduler,generation_limit=100)
    finally: optimizer.step=original
    torch.cuda.synchronize()
    stats.update({'seconds':time.monotonic()-start,'optimizer_seconds':sum(optimizer_time),
                  'presentations':len(samples),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'system_before':before,'system_after':system_state()})
    check(); model.zero_grad(set_to_none=True)
    return stats


def loader_profile(dataset,requests,workers,check):
    begin=time.monotonic()
    kwargs={'num_workers':workers,'prefetch_factor':1,'multiprocessing_context':'spawn'} if workers else {'num_workers':0}
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,sampler=requests,collate_fn=identity,**kwargs)
    iterator=iter(loader); waits=[]; peak_rss=0; fingerprints=[]
    try:
        for _ in requests:
            check(); start=time.monotonic(); sample=next(iterator)[0]; waits.append(time.monotonic()-start)
            fingerprints.append(hash_tensors([('pixels',sample.pixels),('ids',sample.ids)]))
            process=psutil.Process()
            peak_rss=max(peak_rss,process.memory_info().rss+sum(p.memory_info().rss for p in process.children() if p.is_running()))
            del sample
    finally:
        del iterator,loader; gc.collect()
    return {'workers':workers,'prefetch_factor':1 if workers else None,'samples':len(requests),'seconds':time.monotonic()-begin,
            'wait_median_seconds':float(np.median(waits)),'wait_p95_seconds':float(np.percentile(waits,95)),
            'combined_rss_peak_bytes':peak_rss,'fingerprints':fingerprints}


def validation_profile(model,dataset,protocol,path,check):
    result={}
    for mode in ('rgb','paired'):
        check(); started=time.monotonic()
        evaluation=evaluate_requests(model,dataset,protocol['requests'],mode,limit_per_task=4,check=check)
        atomic_json(path/f'validation-{mode}.json',evaluation)
        result[mode]={'seconds':time.monotonic()-started,'samples':len(evaluation['records']),
                      'per_task':{task:{'samples':len(rows),'mean_seconds':float(np.mean([r['total_seconds'] for r in rows])),
                                       'min_seconds':min(r['total_seconds'] for r in rows),'max_seconds':max(r['total_seconds'] for r in rows),
                                       'generated_tokens':[r['generated_tokens'] for r in rows]}
                                  for task in protocol['requests']
                                  for rows in [[r for r in evaluation['records'] if r['task']==task]] if rows}}
    return result


def stress_checks(model,dataset,check,path):
    # Highest known voxel count is exact over the audited pilot. Text lengths
    # below are candidate-based, not a claim of a global maximum over all prompts.
    largest=max(range(len(dataset)),key=lambda i:dataset.rows[i]['lidar_stats']['voxel_points'])
    candidates={}
    for task,annotation in [('qa','drivelm'),('dreamer','dreamer'),('commentary','commentary')]:
        pool=dataset.pools[task]
        ranked=sorted(pool,key=lambda i:dataset.reader.index[dataset.rows[i]['annotations'][annotation]]['size'],reverse=True)[:16]
        choices=[dataset[Request(i,task,False,seed,0)] for i in ranked for seed in (42,1337)]
        candidates[task]=max(choices,key=lambda s:len(s.ids)); del choices
    candidates['max_voxel_driving']=dataset[Request(largest,'driving',False,42,0)]
    # Combine the largest audited cloud with long generated-condition contexts
    # solely for resource stress. Labels/sensors remain intact in ordinary steps.
    for task in ('dreamer','commentary'):
        candidates['max_cloud_'+task]=replace(candidates[task],cloud=candidates['max_voxel_driving'].cloud)
    reports=[]; model.zero_grad(set_to_none=True)
    for name,cpu in candidates.items():
        check(); sample=cpu.to('cuda'); torch.cuda.reset_peak_memory_stats(); begin=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            terms,_=model.loss_terms(sample,100)
            loss=sum(v/max(1,model.loss_counts(sample)[k]) for k,v in terms.items())
        if not torch.isfinite(loss): raise FloatingPointError(name)
        loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.3,error_if_nonfinite=True)
        torch.cuda.synchronize()
        reports.append({'name':name,'teacher_tokens':len(sample.ids),'question_tokens':len(sample.question_ids),
                        'voxel_points':len(sample.cloud.coord),'seconds':time.monotonic()-begin,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        model.zero_grad(set_to_none=True); del sample,terms,loss; check()
    # Explicit 100-token worst-case generation: ignore EOS only in this diagnostic.
    # The sampled training path above still stops at EOS and has a 100-token cap.
    cpu=max([candidates['max_cloud_dreamer'],candidates['max_cloud_commentary']],key=lambda s:len(s.question_ids))
    sample=cpu.to('cuda')
    check(); begin=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    # Keep the teacher-forced graph live during generation, as loss_terms does.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        teacher_terms,_=model.loss_terms(replace(sample,task='qa'),100)
        language_loss=teacher_terms['language']/max(1,model.loss_counts(sample)['language'])
    del teacher_terms
    with model.generation_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        current,_=model.context(sample); tokens=[]
        for _ in range(100):
            hidden=model.hidden(current).last_hidden_state[:,-1]
            token=model.adaptors.language.lm_head(hidden).argmax(-1); tokens.append(token)
            current=torch.cat((current,model.adaptors.language.embed_tokens(token)[None]),dim=1)
        generated=torch.stack(tokens,dim=1)
    del current,hidden,token
    with torch.autocast('cuda',dtype=torch.bfloat16):
        predictions,_=model.driving(sample,generated)
        loss=language_loss+sum(v.float().square().mean() for v in predictions.values())
    loss.backward(); torch.cuda.synchronize()
    if not torch.isfinite(loss): raise FloatingPointError('100-token driving stress')
    reports.append({'name':'forced_100_tokens_then_driving_backward','generated_tokens':100,'question_tokens':len(sample.question_ids),
                    'voxel_points':len(sample.cloud.coord),'seconds':time.monotonic()-begin,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    model.zero_grad(set_to_none=True); del predictions,generated,loss,language_loss,sample,candidates
    check(); atomic_json(path/'stress.json',reports); return reports


def main(output):
    output=Path(output)
    if (output/'progress.json').exists(): raise FileExistsError('Use a fresh run directory; old evidence is never overwritten')
    output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def alarm(*_): raise TimeoutError('Finite Stage 3 profile reached its 45-minute wall limit')
    signal.signal(signal.SIGALRM,alarm); signal.alarm(45*60)
    report={'stage':3,'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
            'purpose':'Disposable bounded profile; not main training','long_training_started':False,
            'limits':{'wall_seconds':2700,'align_updates':1,'joint_step_executions':4,'accumulation':64,'generation_cap':100,
                      'peak_reserved_gib':40,'worker_count_max':2,'prefetch_per_worker':1}}
    def publish(key,value):
        report[key]=value; report['elapsed_seconds']=time.monotonic()-started
        atomic_json(output/'progress.json',report); print(key+': '+json.dumps(value),flush=True)
    def check():
        if time.monotonic()-started>2700: alarm()
        if torch.cuda.is_initialized() and torch.cuda.memory_reserved()>40*GIB: raise MemoryError('40 GiB process reserve cap exceeded')
        if psutil.virtual_memory().available<8*GIB: raise MemoryError('Less than 8 GiB host RAM available')
        if shutil.disk_usage(output).free<150*GIB: raise OSError('Shared 150 GiB free-space reserve reached')
    checkpoints=[]
    try:
        publish('stage2_recheck',verify_stage2())
        prepare_environment()
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2' and torch.cuda.device_count()==1
        actual=str(torch.cuda.get_device_properties(0).uuid)
        assert actual.removeprefix('GPU-')==EXPECTED_UUID.removeprefix('GPU-')
        assert subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==EXPECTED_UUID
        torch.cuda.set_per_process_memory_fraction(min(.88,40*GIB/torch.cuda.get_device_properties(0).total_memory))
        torch.set_num_threads(4); random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        publish('hardware',system_state()); check()
        manifest=ROOT/'stage2/pilot/manifest.json'
        begin=time.monotonic(); model=make_training_model().cuda(); policy=model.lidar_numerics_policy; assert not model.use_cache
        publish('model_load_seconds',time.monotonic()-begin)
        publish('numerics',policy)
        begin=time.monotonic(); print('Constructing the FULL pilot training dataset (metadata reads over SMB)...',flush=True)
        train=ManifestDataset(manifest,model.tokenizer,'train',augment=True)
        publish('train_dataset',{'seconds':time.monotonic()-begin,'samples':len(train),'state':system_state()})
        begin=time.monotonic(); val=ManifestDataset(manifest,model.tokenizer,'val')
        protocol=json.loads((manifest.parent/'validation.json').read_text()); assert protocol['manifest_sha256']==train.manifest_hash
        publish('val_dataset',{'seconds':time.monotonic()-begin,'samples':len(val),'state':system_state()})
        publish('provenance',checkpoint_metadata(model,manifest,{'profile':True,'microbatch':1,'accumulation':64,'cache':False,'numerics':policy}))
        # Evaluate before any update: RGB path has exactly the released weights.
        publish('validation',validation_profile(model,val,protocol,output,check))
        del val; gc.collect()
        sampler=TaskSampler(train.pools,'joint',manifest_hash=train.manifest_hash,bucket_pools=train.bucket_pools,bucket_weights=train.bucket_weights)
        requests=[sampler.request_at(i) for i in range(64)]
        profiles=[]
        for workers in (0,2): profiles.append(loader_profile(train,requests,workers,check))
        assert profiles[0]['fingerprints']==profiles[1]['fingerprints']
        for profile in profiles: del profile['fingerprints']
        publish('loader',profiles)
        model.set_stage('align'); model.train()
        align=TaskSampler(train.pools,'align',manifest_hash=train.manifest_hash,bucket_pools=train.bucket_pools,bucket_weights=train.bucket_weights)
        begin=time.monotonic(); samples=[train[r] for r in align.take(64)]; io=time.monotonic()-begin
        optimizer,scheduler=make_optimizer(model,750)
        frozen_before=hash_tensors((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        with Timings(model,check,output/'align-samples.jsonl') as timing:
            stats=timed_window(model,samples,optimizer,scheduler,check)
        assert frozen_before==hash_tensors((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        stats['io_seconds']=io; stats['task_counts']=dict(Counter(s.task for s in samples)); stats['frozen_unchanged']=True
        publish('alignment',stats); del optimizer,scheduler,samples; gc.collect(); torch.cuda.empty_cache()
        model.set_stage('joint'); model.train(); optimizer,scheduler=make_optimizer(model,3750)
        frozen_before=hash_tensors((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        sampler.position=0; windows=[]
        for number in range(2):
            begin=time.monotonic(); samples=[train[r] for r in sampler.take(64)]; io=time.monotonic()-begin
            with Timings(model,check,output/'joint-samples.jsonl'):
                stats=timed_window(model,samples,optimizer,scheduler,check)
            stats['io_seconds']=io; windows.append(stats); publish('joint_windows',windows); del samples
        assert all(state[k].dtype==torch.float32 for state in optimizer.state.values() for k in ('exp_avg','exp_avg_sq'))
        metadata=checkpoint_metadata(model,manifest,{'profile':True,'microbatch':1,'accumulation':64,'cache':False,'numerics':policy})
        checkpoint=output/'disposable-boundary.pt'; checkpoints.append(checkpoint)
        begin=time.monotonic(); save_checkpoint(checkpoint,model,optimizer,scheduler,sampler,2,metadata)
        checkpoint_report={'bytes':checkpoint.stat().st_size,'save_seconds':time.monotonic()-begin,
                           'sha256':digest_file(checkpoint),'fp32_adam':True}
        # Real 64-sample next-update recovery check, including RNG/augmentation.
        begin=time.monotonic(); samples=[train[r] for r in sampler.take(64)]; io=time.monotonic()-begin
        with Timings(model,check,output/'joint-samples.jsonl'):
            stats=timed_window(model,samples,optimizer,scheduler,check)
        stats['io_seconds']=io; windows.append(stats)
        reference_model=cpu_copy(model.state_dict().items())
        reference_optimizer=cpu_copy((f'{i}:{k}',v) for i,s in optimizer.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v))
        reference_scheduler=scheduler.state_dict(); reference_sampler=sampler.state_dict()
        begin=time.monotonic(); updates=load_checkpoint(checkpoint,model,optimizer,scheduler,sampler,metadata); torch.cuda.synchronize()
        checkpoint_report['load_seconds']=time.monotonic()-begin; assert updates==2
        # Verify the entire loaded state against the trusted serialized boundary.
        boundary=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        checkpoint_report['loaded_model_exact']=all(torch.equal(v.detach().cpu(),boundary['model'][k]) for k,v in model.state_dict().items())
        current=optimizer.state_dict()
        checkpoint_report['loaded_optimizer_exact']=all(torch.equal(v.detach().cpu(),boundary['optimizer']['state'][i][k])
            for i,s in current['state'].items() for k,v in s.items() if torch.is_tensor(v))
        del boundary,current
        assert checkpoint_report['loaded_model_exact'] and checkpoint_report['loaded_optimizer_exact']
        regenerated=[train[r] for r in sampler.take(64)]
        assert [hash_tensors([('pixels',s.pixels),('ids',s.ids)]) for s in samples]==[hash_tensors([('pixels',s.pixels),('ids',s.ids)]) for s in regenerated]
        del samples
        with Timings(model,check,output/'resume-samples.jsonl'):
            resumed=timed_window(model,regenerated,optimizer,scheduler,check)
        model_comparison=compare_tensors(model.state_dict().items(),reference_model,atol=1e-6,rtol=1e-5)
        optimizer_comparison=compare_tensors(((f'{i}:{k}',v) for i,s in optimizer.state_dict()['state'].items() for k,v in s.items() if torch.is_tensor(v)),
                                            reference_optimizer,atol=1e-8,rtol=1e-3)
        del reference_model,reference_optimizer; gc.collect()
        checkpoint_report.update({'resumed_next_update_model':model_comparison,'resumed_next_update_optimizer':optimizer_comparison,
                                  'resumed_next_update_model_bitwise':model_comparison['bitwise'],
                                  'resumed_next_update_optimizer_bitwise':optimizer_comparison['bitwise'],
                                  'scheduler_equal':scheduler.state_dict()==reference_scheduler,'sampler_equal':sampler.state_dict()==reference_sampler,
                                  'resumed_window':resumed,'augmented_samples_identical':True})
        publish('checkpoint',checkpoint_report)
        assert model_comparison['within_tolerance'] and optimizer_comparison['within_tolerance'] and checkpoint_report['scheduler_equal'] and checkpoint_report['sampler_equal']
        assert frozen_before==hash_tensors((n,p) for n,p in model.named_parameters() if not p.requires_grad)
        publish('joint_windows',windows); publish('checkpoint',checkpoint_report)
        del regenerated; gc.collect()
        publish('stress',stress_checks(model,train,check,output))
        assert not any(p.grad is not None for p in model.parameters())
        publish('final_system',system_state())
        publish('status','complete')
    except BaseException as error:
        publish('error',{'type':type(error).__name__,'message':str(error)}); publish('status','failed'); raise
    finally:
        signal.alarm(0)
        for checkpoint in checkpoints:
            checkpoint.unlink(missing_ok=True); checkpoint.with_suffix('.pt.part').unlink(missing_ok=True)
        # No learned profiling weights are published or selected.


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
