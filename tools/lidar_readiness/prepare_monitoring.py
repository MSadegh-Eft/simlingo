"""CPU-only monitoring preparation and exact planned-sampler coverage audit."""
import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer

from simlingo_training.lidar.data import ManifestDataset,TaskSampler,SNAPSHOT
from simlingo_training.lidar.monitor_protocol import build_protocol,baseline_records
from simlingo_training.lidar.runtime import METADATA_CACHE
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.train import verify_dataset


def main(manifest,baseline,output):
    ready=verify_dataset(manifest); output.mkdir(parents=True,exist_ok=True)
    full=json.loads((manifest.parent/'validation.json').read_text())
    binding,rows=baseline_records(baseline)
    if binding['manifest_sha256']!=ready['manifest_sha256']: raise ValueError('Baseline dataset differs')
    expected={(r['uid'],r['task'],r['seed']) for t,items in full['requests'].items() if t!='geometry_qa' for r in items}
    if {(r['uid'],r['task'],r['seed']) for r in rows}!=expected or len(rows)!=len(expected): raise ValueError('Incomplete released baseline journal')
    tokenizer=AutoTokenizer.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    train=ManifestDataset(manifest,tokenizer,'train',augment=False,metadata_cache=METADATA_CACHE)
    monitor=build_protocol(full,rows,train)
    monitor.update(validation_sha256=ready['validation_sha256'],reference_journal_sha256=digest_file(baseline))
    target=output/'monitor-protocol.json'
    if target.exists() and json.loads(target.read_text())!=monitor: raise ValueError('Use a new version directory; monitoring protocol is immutable')
    atomic_json(target,monitor)
    audits={}
    for stage,count in [('align',48000),('joint',74304)]:
        sampler=TaskSampler(train.pools,stage,manifest_hash=train.manifest_hash,bucket_pools=train.bucket_pools,bucket_weights=train.bucket_weights)
        seen=set(); tasks=Counter(); buckets=Counter(); replay=0
        for i in range(count):
            r=sampler.request_at(i); seen.add(r.index); tasks[r.task]+=1; replay+=r.replay
            buckets.update(train.row_buckets[r.index])
        reachable={}
        for task,pool in train.pools.items():
            active=[items for bucket,items in train.bucket_pools.get(task,{}).items() if items and train.bucket_weights.get(bucket,0)>0]
            reachable[task]=len(set(i for items in active for i in items)) if active else len(pool)
        audits[stage]={'presentations':count,'planned_unique_frames':len(seen),'tasks':dict(tasks),
                       'replay_presentations':replay,'scenario_presentations':dict(buckets),'reachable_per_task':reachable}
        print(f'{stage}: {count} presentations, {len(seen)} unique training frames',flush=True)
    train_groups={r['group'] for r in train.rows}
    document=json.loads(manifest.read_text()); val_groups={r['group'] for r in document['samples'] if r['split']=='val'}
    if train_groups&val_groups: raise ValueError('Train/validation source group overlap')
    audit={'passed':True,'manifest_sha256':train.manifest_hash,'monitor_protocol_sha256':digest_file(target),
           'eligible_training_frames':len(train),'training_groups':len(train_groups),'validation_groups':len(val_groups),
           'task_pool_sizes':{k:len(v) for k,v in train.pools.items()},'stages':audits,
           'limits':'Seed-42 sampler forecast, with repetition. Reachability and bucket weighting are reported explicitly; not a claim of global dataset representativeness.'}
    atomic_json(output/'sampling-audit.json',audit)
    train.reader.close()
    print(json.dumps({k:monitor[k] for k in ('coverage','paired_requests','rgb_requests')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--baseline-journal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); main(a.manifest,a.baseline_journal,a.output)
