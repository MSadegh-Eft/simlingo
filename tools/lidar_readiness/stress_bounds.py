"""Resource-only combination of exact audited context/cloud maxima; no update."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import time

import torch

from simlingo_training.lidar.data import ManifestDataset,Request,SNAPSHOT
from simlingo_training.lidar.runtime import METADATA_CACHE,check_resources
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.utils.internvl2_utils import get_custom_chat_template
from tools.lidar_readiness.audit_contexts import variants


def run(model,train,audit_path,output):
    audit=json.loads(Path(audit_path).read_text())
    if not audit['passed'] or audit['manifest_sha256']!=train.manifest_hash: raise ValueError('Context audit differs')
    from simlingo_training.lidar import data
    if audit['data_source_sha256']!=digest_file(data.__file__): raise ValueError('Context implementation changed after audit')
    val=ManifestDataset(train.path,model.tokenizer,'val',metadata_cache=METADATA_CACHE)
    wanted={r['uid'] for batch in audit['witness_batches'].values() for r in batch}
    candidates=[]
    for dataset in (train,val):
        for i,row in enumerate(dataset.rows):
            if row['id'] not in wanted: continue
            base=dataset[Request(i,'driving',False,42,0)]
            current=dataset.reader.json(row['measurements'][0]); annotations={k:dataset.reader.json(v) for k,v in row['annotations'].items()}
            for task,prompt,answer in variants(current,annotations):
                conversations=[[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image'}]},
                                {'role':'assistant','content':[{'type':'text','text':answer}]}]]
                teacher,question=get_custom_chat_template(conversations,model.tokenizer,'OpenGVLab/InternVL2-1B',512,conversation_path=str(SNAPSHOT/'conversation.py'))
                ids=teacher['phrase_ids'][0]; mask=teacher['loss_masking'][0]&teacher['phrase_valid'][0]
                candidates.append(replace(base,task='commentary',ids=ids,answer_mask=mask,question_ids=question['phrase_ids'][0]))
    teacher=max(candidates,key=lambda s:len(s.ids)); question=max(candidates,key=lambda s:len(s.question_ids))
    if len(teacher.ids)!=audit['maxima']['teacher'] or len(question.question_ids)!=audit['maxima']['question']:
        raise ValueError('Maximum-context witnesses did not reproduce the full audit')
    largest=max(range(len(train)),key=lambda i:train.rows[i]['lidar_stats']['voxel_points'])
    cloud=train[Request(largest,'driving',False,42,0)]
    cpu=replace(teacher,cloud=cloud.cloud,geometry=cloud.geometry,question_ids=question.question_ids)
    del candidates,base,teacher,question,cloud,val; gc.collect(); torch.cuda.empty_cache()
    check_resources(output); sample=cpu.to('cuda'); model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    # Hold the maximum teacher graph while generating exactly 100 tokens without
    # EOS early exit, then recompute differentiable sensors and driving queries.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        terms,_=model.loss_terms(replace(sample,task='qa'),100)
        language_loss=terms['language']/max(1,model.loss_counts(sample)['language'])
    del terms
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
    if not torch.isfinite(loss): raise FloatingPointError('Global maximum resource stress')
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.3,error_if_nonfinite=True)
    torch.cuda.synchronize()
    report={'passed':True,'manifest_sha256':train.manifest_hash,'context_audit_sha256':digest_file(audit_path),
        'stress_source_sha256':digest_file(__file__),'teacher_tokens':len(sample.ids),'question_tokens':len(sample.question_ids),
        'voxel_points':len(sample.cloud.coord),'generated_tokens':100,'optimizer_state_resident':True,
        'seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'scope':'Synthetic upper resource bound: independently maximal teacher/question texts and training cloud, teacher graph retained, uncached generation and driving backward. No optimizer update or learning claim.'}
    model.zero_grad(set_to_none=True); check_resources(output)
    atomic_json(Path(output)/'global-stress.json',report)
    return report
