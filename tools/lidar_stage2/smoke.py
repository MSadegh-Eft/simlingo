"""Bounded real-data correctness/learning checks, never a training launch."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from dataclasses import replace

import numpy as np
import torch

from simlingo_training.lidar.data import ManifestDataset, Request, TaskSampler, TASKS
from simlingo_training.lidar.engine import step_window, make_optimizer, save_checkpoint, load_checkpoint, checkpoint_metadata
from simlingo_training.lidar.model import make_model
from simlingo_training.lidar.storage import atomic_json

OUTPUT = Path('/ehsan/m.sadegh/simlingo_lidar/stage2')


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cache_difference(model,sample,tokens):
    """Compare both paths on identical forced prefixes, before text can diverge."""
    with model.generation_mode():
        context,_=model.context(sample)
        current,past=context,None
        for step in range(tokens.shape[1]):
            embeds=torch.cat((context,model.adaptors.language.embed_tokens(tokens[:,:step])),dim=1)
            full=model.hidden(embeds).last_hidden_state[:,-1]
            cached=model.hidden(current,past,True)
            a=model.adaptors.language.lm_head(full).float()[0]
            b=model.adaptors.language.lm_head(cached.last_hidden_state[:,-1]).float()[0]
            if a.argmax()!=b.argmax():
                top=a.topk(2).values
                return {'first_different_step':step,'max_logit_error':float((a-b).abs().max()),
                        'uncached_top_two_gap':float(top[0]-top[1]),'uncached_argmax':int(a.argmax()),'cached_argmax':int(b.argmax())}
            past=cached.past_key_values
            current=model.adaptors.language.embed_tokens(tokens[:,step:step+1])
    return {'forced_prefix_argmax_difference':False}


def main(manifest=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
    assert torch.cuda.device_count() == 1
    expected = subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') == expected.removeprefix('GPU-')
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    report = {'physical_gpu_uuid':expected,'checks':{},'limits':'Stage 2 bounded checks, not the full Stage 3 throughput profile'}
    started = time.monotonic()

    def record(name,value=True):
        report['checks'][name] = value
        report['elapsed_seconds'] = time.monotonic()-started
        atomic_json(OUTPUT/'reports/smoke.json',report)
        print(name+': '+json.dumps(value),flush=True)

    model = make_model().cuda().eval()
    record('loaded_released_and_utonia_strictly', {'total_parameters':sum(p.numel() for p in model.parameters()),
        'alignment_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)})
    manifest = Path(manifest) if manifest else OUTPUT/'data/manifest.json'
    train = ManifestDataset(manifest,model.tokenizer,'train',augment=False,max_samples=128)
    val = ManifestDataset(manifest,model.tokenizer,'val',augment=False,max_samples=128)
    record('task_pools',{'train':{k:len(v) for k,v in train.pools.items()},'val':{k:len(v) for k,v in val.pools.items()}})
    record('manifest',{'path':str(manifest),'sha256':train.manifest_hash})
    samples = {task:train[Request(train.pools[task][0],task,False,42,i)] for i,task in enumerate(TASKS)}
    paired = samples['driving'].to('cuda')
    replay = replace(paired,cloud=None)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a,_ = model.context(paired); b,_ = model.context(replay)
        assert a.shape[1] == b.shape[1]+128
        torch.testing.assert_close(a[:,128:],b,rtol=0,atol=0)
        hidden = model.hidden(b).last_hidden_state
        reference = model.language_model.model(inputs_embeds=b,attention_mask=torch.ones(b.shape[:2],device='cuda',dtype=torch.long),
            output_hidden_states=True,use_cache=False,return_dict=True)
        error = (hidden-reference.hidden_states[-1]).abs().max().item()
        torch.testing.assert_close(hidden,reference.hidden_states[-1],rtol=0,atol=0)
        torch.testing.assert_close(model.adaptors.language.lm_head(hidden[:,-1]).float(),reference.logits[:,-1].float(),rtol=.01,atol=.01)
        # Supervision values must not alter the sensor/navigation embeddings.
        changed = replace(paired,waypoints=paired.waypoints+100,route=paired.route+100,
            geometry={k:torch.zeros_like(v) for k,v in paired.geometry.items()})
        other,_ = model.context(changed)
        torch.testing.assert_close(a,other,rtol=0,atol=0)
        predictions,_=model(paired)
        assert predictions['route'].shape==(1,20,2) and predictions['speed_wps'].shape==(1,10,2)
        try: model(object())
        except TypeError: pass
        else: raise AssertionError('Legacy inputs were silently accepted by the LiDAR model')
        del a,b,hidden,reference,other
    record('prefix_backbone_and_no_target_leakage',{'prefix_tokens':128,'backbone_max_error':error})
    mismatches=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for name,sample in [('rgb',replay),('paired',paired)]+[(task,samples[task].to('cuda')) for task in ('commentary','dreamer','geometry_qa')]:
            uncached = model.generate_ids(sample,32,False)
            cached = model.generate_ids(sample,32,True)
            repeated=model.generate_ids(sample,32,False)
            assert torch.equal(uncached,repeated),'Uncached generation is not repeatable'
            equal=torch.equal(uncached,cached)
            details={'equal':equal,'uncached_tokens':uncached.tolist(),'cached_tokens':cached.tolist(),
                     'text':model.tokenizer.decode(uncached[0],skip_special_tokens=False)}
            if not equal:
                mismatches.append(name); details['diagnostic']=cache_difference(model,sample,uncached)
            record('cache_parity_'+name,details)
    model.use_cache = False
    record('generation_cache_policy',{'enabled':False,'mismatched_tasks':mismatches,
        'reason':'Keep the reference uncached path; BF16 cache equivalence has not passed the pilot gate'})
    del replay,paired

    # Separate objectives must reach the connector through the frozen LLM.
    model.train()
    for objective,task in [('route','driving'),('language','geometry_qa')]:
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            terms,_ = model.loss_terms(samples[task].to('cuda'),8)
            objective_loss = terms[objective]
        objective_loss.backward()
        gradient = model.lidar.connector.queries.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        record(objective+'_connector_gradient',float(gradient.norm()))
        del terms,objective_loss
    model.zero_grad(set_to_none=True)
    initial_frozen = frozen_hash(model)
    optimizer = torch.optim.AdamW(model.optimizer_groups(),weight_decay=.1)
    history = []
    for update in range(16):
        stats = step_window(model,[samples['driving'],samples['geometry_qa']],optimizer,generation_limit=8)
        history.append(stats['loss'])
        print(f'tiny update {update+1}/16: {stats}',flush=True)
    assert np.mean(history[-4:]) < np.mean(history[:4]),history
    model.zero_grad(set_to_none=True)
    assert frozen_hash(model)==initial_frozen
    record('tiny_learning_and_frozen_weights',{'updates':16,'presentations':32,'losses':history,'frozen_sha256':initial_frozen})
    # Check real model/optimizer recovery at a completed update boundary.
    sampler = TaskSampler(train.pools,manifest_hash=train.manifest_hash,bucket_pools=train.bucket_pools,bucket_weights=train.bucket_weights)
    sampler.take(32)
    metadata = checkpoint_metadata(model,manifest,{'purpose':'stage2-smoke','seed':42,'max_samples':128,
        'microbatch':1,'accumulation':2,'updates':16,'generation_limit':8,'augmentation':False,'use_cache':False})
    atomic_json(OUTPUT/'reports/checkpoint-metadata.json',metadata)
    checkpoint = OUTPUT/'reports/smoke-resume.pt'
    save_checkpoint(checkpoint,model,optimizer,None,sampler,16,metadata)
    expected_query = model.lidar.connector.queries.detach().clone()
    expected_random = torch.rand(4,device='cuda')
    with torch.no_grad(): model.lidar.connector.queries.add_(1)
    assert load_checkpoint(checkpoint,model,optimizer,None,sampler,metadata)==16
    torch.testing.assert_close(model.lidar.connector.queries,expected_query,rtol=0,atol=0)
    torch.testing.assert_close(torch.rand(4,device='cuda'),expected_random,rtol=0,atol=0)
    assert sampler.position==32
    checkpoint.unlink()  # This check's temporary full checkpoint only.
    record('real_checkpoint_optimizer_and_cuda_rng_resume')
    del optimizer,expected_query
    torch.cuda.empty_cache()

    model.set_stage('joint'); model.train()
    groups = model.optimizer_groups()
    optimizer,scheduler = make_optimizer(model,3750)
    record('joint_optimizer_groups',{g['name']:{'parameters':sum(p.numel() for p in g['params']),
        'max_lr':scheduler.max_lrs[i] if hasattr(scheduler,'max_lrs') else g['max_lr']} for i,g in enumerate(optimizer.param_groups)})
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        terms,_ = model.loss_terms(samples['driving'].to('cuda'),8)
        loss = sum(terms.values())
    loss.backward()
    gradients = {}
    for group in groups:
        grads = [p.grad for p in group['params'] if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),group['name']
        norm = sum(float(g.float().square().sum()) for g in grads)**.5
        assert norm>0,group['name']
        gradients[group['name']] = norm
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    record('joint_backward_all_groups',gradients)
    model.zero_grad(set_to_none=True)
    # Both generated-condition tasks must exercise their separate learning pass.
    model.set_stage('align'); model.train()
    for task in ('qa','commentary','dreamer','rear_driving'):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            terms,_ = model.loss_terms(samples[task].to('cuda'),8)
            loss = sum(terms.values())
        assert torch.isfinite(loss)
        loss.backward(); model.zero_grad(set_to_none=True)
        record('task_forward_backward_'+task)
        del terms,loss
    record('peak_memory_bytes',{'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()})
    record('passed')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--manifest',type=Path)
    main(parser.parse_args().manifest)
