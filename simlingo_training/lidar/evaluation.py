"""Fixed-request, generated-condition evaluation; no teacher answers in driving.

Metrics are averaged within source route groups and then across groups. QA
accuracy is restricted to explicitly parseable targets, with coverage reported.
ROUGE-L here is word-token LCS F1 (0..100), without stemming or bootstrapping.
"""
from collections import defaultdict
from dataclasses import replace
import hashlib
import re
import time

import numpy as np
import torch

from simlingo_training.lidar.data import Request


def words(text):
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())


def answer_text(text):
    # loss_masking includes the chat assistant role header in the pinned
    # template. It is formatting, not part of the reference answer.
    text=re.sub(r'^\s*assistant[ \t]*\r?\n','',text,flags=re.I)
    return re.sub(r'^\s*A\s*:\s*', '', text, flags=re.I).strip()


def rouge_l(prediction, reference):
    a,b=words(answer_text(prediction)),words(answer_text(reference))
    if not a or not b: return 100. if a==b else 0.
    previous=[0]*(len(b)+1)
    for x in a:
        current=[0]
        for j,y in enumerate(b):
            current.append(previous[j]+1 if x==y else max(previous[j+1],current[-1]))
        previous=current
    return 200.*previous[-1]/(len(a)+len(b))


def structured_answer(text):
    """Conservative complete-answer grammar; prose/unknown outputs abstain."""
    text=answer_text(text).lower().strip(' .!')
    if text in ('yes','no','red','yellow','green','left','right','straight','none'):
        return ('category',text)
    decision=re.match(r'^(yes|no),\s+(?!but\b)',text)
    if decision: return ('category',decision[1])
    if text.startswith('no observed vehicle; this does not establish that the lane is clear'):
        return ('category','no')
    match=re.fullmatch(r'(0-10|10-20|20-40|40-64)\s+met(?:re|er)s',text)
    if match: return ('distance_bin',match[1])
    match=re.fullmatch(r'(-?\d+(?:\.\d+)?)(?:\s*(m/s|metres|meters|m|km/h))?',text)
    if match: return ('number',float(match[1]),match[2])
    return None


def dreamer_decision(text):
    text=answer_text(text).lower()
    if re.match(r'^following the given instruction\.',text): return 'follow'
    if re.match(r'^(ignore|reject|refuse) (?:the )?instruction\b',text): return 'refuse'
    return None


def language_metrics(task,text,target):
    metrics={}
    if task=='qa':
        reference=structured_answer(target)
        metrics['qa_structured_accuracy']=100.*(structured_answer(text)==reference) if reference is not None else None
        metrics['qa_exact_match']=100.*(words(answer_text(text))==words(answer_text(target)))
    if task=='commentary': metrics['commentary_rouge_l']=rouge_l(text,target)
    if task=='dreamer':
        reference=dreamer_decision(target)
        metrics['dreamer_decision_accuracy']=100.*(dreamer_decision(text)==reference) if reference is not None else None
    if task=='geometry_qa':
        reference=structured_answer(target)
        metrics['geometry_qa_accuracy']=100.*(structured_answer(text)==reference) if reference is not None else None
    return metrics


def group_metrics(records):
    grouped=defaultdict(lambda:defaultdict(list))
    counts=defaultdict(int)
    populations=defaultdict(list)
    for row in records:
        for key,value in row['metrics'].items():
            if value is not None:
                if not np.isfinite(value): raise ValueError(f'Non-finite evaluation metric: {key}')
                grouped[key][row['group']].append(float(value)); counts[key]+=1
                populations[key].append(str((row['uid'],row.get('task'),row.get('seed'))))
    return {key:{'value':float(np.mean([np.mean(v) for v in groups.values()])),
                 'samples':counts[key],'groups':len(groups),
                 'population_sha256':hashlib.sha256('\n'.join(sorted(populations[key])).encode()).hexdigest()}
            for key,groups in grouped.items()}


def retention_gate(candidate, baseline):
    """Require every predefined metric; missing measurements never pass a gate."""
    rules={'route_ade':('relative_error',.02),'speed_ade':('relative_error',.02),
           'qa_structured_accuracy':('points',2.),'dreamer_decision_accuracy':('points',2.),
           'commentary_rouge_l':('points',2.)}
    checks={}
    for name,(kind,limit) in rules.items():
        a=candidate.get(name,{}); b=baseline.get(name,{})
        if not a.get('samples') or not b.get('samples'):
            checks[name]={'passed':False,'reason':'Missing measurement/parseable target coverage'}; continue
        # Gate comparisons require the same per-metric eligibility population.
        if (a['samples'],a['groups'],a.get('population_sha256'))!=(b['samples'],b['groups'],b.get('population_sha256')):
            checks[name]={'passed':False,'reason':'Metric coverage differs'}; continue
        current,reference=a['value'],b['value']
        passed=current<=reference*(1+limit) if kind=='relative_error' else current>=reference-limit
        checks[name]={'passed':bool(passed),'candidate':current,'baseline':reference,'limit':limit,'rule':kind}
    return {'passed':all(c['passed'] for c in checks.values()),'checks':checks}


def evaluate_requests(model,dataset,requests,mode='paired',limit_per_task=None,check=None,on_record=None):
    """Return records including generated examples; caller controls artifact path.

    RGB retention excludes geometry QA: it has no camera-grounded target. Other
    requests retain the exact question, seed and targets while omitting the cloud.
    This also evaluates the released model when called before any update.
    """
    if mode not in ('paired','rgb'): raise ValueError(mode)
    records=[]; flags={m:m.training for m in model.modules()}; model.eval()
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for task,items in requests.items():
                if mode=='rgb' and task=='geometry_qa': continue
                for item in items[:limit_per_task] if limit_per_task is not None else items:
                    if check: check()
                    start=time.monotonic()
                    request=Request(**{k:v for k,v in item.items() if k!='uid'})
                    if mode=='rgb': request=replace(request,replay=True)
                    sample=dataset[request]
                    if sample.uid!=item['uid']: raise ValueError('Fixed validation UID/index mismatch')
                    if mode=='rgb': sample=replace(sample,cloud=None)
                    sample=sample.to('cuda'); torch.cuda.synchronize(); loaded=time.monotonic()
                    generated=model.generate_ids(sample,100) if sample.language_active else None
                    text=model.tokenizer.decode(generated[0],skip_special_tokens=True) if generated is not None else ''
                    target=model.tokenizer.decode(sample.ids[sample.answer_mask],skip_special_tokens=True)
                    metrics={}; geometry=None
                    if sample.driving_active:
                        predictions,geometry=model.driving(sample,generated)
                        for key,predicted,truth in [('route_ade',predictions['route'][0],sample.route),
                                                     ('speed_ade',predictions['speed_wps'][0],sample.waypoints)]:
                            metrics[key]=float(torch.linalg.vector_norm(predicted.float()-truth,dim=-1).mean())
                    elif mode=='paired':
                        _,geometry=model.context(sample)
                    metrics.update(language_metrics(task,text,target))
                    if geometry is not None:
                        for key,prediction in [('presence',(geometry['presence'][0]>0)),('distance',geometry['distance'][0].argmax(-1))]:
                            valid=sample.geometry[key+'_valid']
                            for sector in range(4):
                                if valid[sector]: metrics[f'geometry_{key}_{sector}_accuracy']=100.*float(prediction[sector]==sample.geometry[key][sector])
                    torch.cuda.synchronize(); end=time.monotonic()
                    records.append({'uid':sample.uid,'group':dataset.rows[request.index]['group'],'task':task,'mode':mode,
                        'seed':request.seed,'metrics':metrics,'generated':text,'reference':target,
                        'generated_tokens':int(generated.numel()) if generated is not None else 0,
                        'teacher_tokens':len(sample.ids),'question_tokens':len(sample.question_ids),
                        'io_transfer_seconds':loaded-start,'compute_seconds':end-loaded,'total_seconds':end-start})
                    if on_record: on_record(records[-1])
                    del sample,generated,geometry
    finally:
        for module,flag in flags.items(): module.training=flag
    return {'records':records,'metrics':group_metrics(records),
            'by_task':{task:group_metrics([r for r in records if r['task']==task]) for task in requests},
            'limits':'Bounded subsets establish timings and metric execution only; no retention/safety/improvement claim'}
