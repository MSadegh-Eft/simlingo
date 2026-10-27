"""Durable complete-protocol evaluation shared by baseline and training."""
from collections import Counter
import json
import os
from pathlib import Path
import time

from simlingo_training.lidar.evaluation import evaluate_requests,group_metrics,summarize_diagnostics
from simlingo_training.lidar.storage import atomic_json,digest_file


def identity(row): return (row['uid'],row['task'],row['seed'])


def recover_journal(path,binding,expected):
    """Only a final interrupted line may be discarded; provenance must match."""
    rows=[]
    if not path.exists():
        with path.open('w') as f:
            f.write(json.dumps({'binding':binding})+'\n'); f.flush(); os.fsync(f.fileno())
        return rows
    with path.open('r+') as f:
        if json.loads(f.readline())!={'binding':binding}: raise ValueError('Evaluation recovery provenance differs')
        seen=set()
        while True:
            start=f.tell(); line=f.readline()
            if not line: break
            if not line.endswith('\n'): f.seek(start); f.truncate(); break
            row=json.loads(line); key=identity(row)
            if key not in expected or key in seen: raise ValueError('Unexpected/duplicate validation request')
            seen.add(key); rows.append(row)
    return rows


def evaluate_complete(model,dataset,requests,mode,output,binding,check=None,
                      complete_protocol=True,diagnostics=False,progress=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    requests={t:items for t,items in requests.items() if mode!='rgb' or t!='geometry_qa'}
    expected={identity(r) for items in requests.values() for r in items}
    if len(expected)!=sum(map(len,requests.values())): raise ValueError('Duplicate protocol identity')
    rows=recover_journal(output/'records.jsonl',binding,expected)
    if any(r['mode']!=mode for r in rows): raise ValueError('Journal evaluation mode differs')
    seen={identity(r) for r in rows}
    remaining={t:[r for r in items if identity(r) not in seen] for t,items in requests.items()}
    print(f'{mode} validation: {len(rows)} recovered, {len(expected)-len(rows)} remaining; zero optimizer updates',flush=True)
    started=time.monotonic(); initial=len(rows)
    if progress: progress(len(rows),len(expected))
    with (output/'records.jsonl').open('a') as journal:
        def record(row):
            rows.append(row); journal.write(json.dumps(row)+'\n')
            if progress: progress(len(rows),len(expected))
            if len(rows)%32==0:
                journal.flush(); os.fsync(journal.fileno())
                progress={'status':'running','completed':len(rows),'expected':len(expected),
                    'this_process_seconds':time.monotonic()-started,'this_process_samples':len(rows)-initial}
                atomic_json(output/'progress.json',progress); print(json.dumps(progress),flush=True)
        evaluate_requests(model,dataset,remaining,mode,check=check,on_record=record,diagnostics=diagnostics)
        journal.flush(); os.fsync(journal.fileno())
    if {identity(r) for r in rows}!=expected or len(rows)!=len(expected): raise ValueError('Evaluation incomplete')
    metrics=group_metrics(rows); counts=Counter(r['task'] for r in rows)
    coverage={name:{**m,'target_fraction':m['samples']/counts[task]} for name,task in
              [('qa_structured_accuracy','qa'),('dreamer_decision_accuracy','dreamer')]
              for m in [metrics.get(name,{'samples':0,'groups':0})] if counts[task]}
    coverage_ok=len(coverage)==2 and all(m['samples']>=100 and m['groups']>=10 and m['target_fraction']>=.10 for m in coverage.values())
    uid_index={r['id']:i for i,r in enumerate(dataset.rows)}
    def bucket_rows(bucket):
        return [r for r in rows if bucket in dataset.row_buckets[uid_index[r['uid']]]]
    report={'status':'complete','complete_protocol':complete_protocol,'binding':binding,'mode':mode,
        'samples':len(rows),'task_counts':dict(counts),'metrics':metrics,'structured_target_coverage':coverage,
        'coverage_gate':{'passed':coverage_ok,'minimum_samples':100,'minimum_groups':10,'minimum_fraction':.10},
        'by_task':{t:group_metrics([r for r in rows if r['task']==t]) for t in counts},
        'by_scenario_bucket':{b:{'samples':len(selected),'metrics':group_metrics(selected)}
            for b in dataset.bucket_weights for selected in [bucket_rows(b)]},
        'journal_sha256':digest_file(output/'records.jsonl'),'sum_example_seconds':sum(r['total_seconds'] for r in rows),
        'limits':'Full fixed request set only; structured QA accuracy covers explicitly parseable targets, not every open-ended answer. Engineering metrics do not establish driving safety.'}
    if diagnostics:
        if any('diagnostics' not in r for r in rows): raise ValueError('Journal lacks required diagnostics')
        report['diagnostics']=summarize_diagnostics(rows)
    if not complete_protocol: report['limits']='Fixed interim diagnostics; full offline validation is required for retention.'
    atomic_json(output/'result.json',report)
    atomic_json(output/'progress.json',{'status':'complete','completed':len(rows),'expected':len(expected),'coverage_passed':coverage_ok})
    return report
