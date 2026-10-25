"""Prove cold/warm metadata and full sensor/target samples are identical."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoProcessor

from simlingo_training.lidar.data import ManifestDataset,TaskSampler,TASKS,SNAPSHOT
from simlingo_training.lidar.runtime import METADATA_CACHE,sample_fingerprint
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.utils.internvl2_utils import get_custom_chat_template
from tools.lidar_readiness.audit_contexts import variants


def main(manifest,output):
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    tokenizer=getattr(processor,'tokenizer',processor)
    result={'manifest_sha256':digest_file(manifest),'splits':{},'passed':True}
    for split in ('train','val'):
        start=time.monotonic(); cold=ManifestDataset(manifest,tokenizer,split,augment=split=='train')
        cold_seconds=time.monotonic()-start
        # Cache construction must independently reproduce the uncached metadata.
        start=time.monotonic(); cached=ManifestDataset(manifest,tokenizer,split,augment=split=='train',metadata_cache=METADATA_CACHE)
        cached_seconds=time.monotonic()-start
        warm_start=time.monotonic(); warm=ManifestDataset(manifest,tokenizer,split,augment=split=='train',metadata_cache=METADATA_CACHE)
        warm_seconds=time.monotonic()-warm_start
        metadata_ok=all(cold.pools==d.pools and cold.row_buckets==d.row_buckets and cold.bucket_pools==d.bucket_pools
            and cold.bucket_weights==d.bucket_weights and all(np.array_equal(cold.geometry[k],d.geometry[k]) for k in cold.geometry)
            for d in (cached,warm))
        sampler=TaskSampler(cold.pools,'joint',manifest_hash=cold.manifest_hash,bucket_pools=cold.bucket_pools,bucket_weights=cold.bucket_weights)
        requests=[]
        for task in TASKS:
            request=next(sampler.request_at(i) for i in range(960) if sampler.request_at(i).task==task and not sampler.request_at(i).replay)
            # Explicitly cover newly prepared parking examples in every family.
            parking=next(i for i in cold.pools[task] if 'parking_lane' in cold.rows[i]['route'])
            requests.append(replace(request,index=parking))
            if task in TASKS[:4]: requests.append(replace(request,index=parking,replay=True))
        samples_ok=all(sample_fingerprint(cold[r])==sample_fingerprint(cached[r])==sample_fingerprint(warm[r]) for r in requests)
        text_ok=True
        for request in requests:
            if request.replay: continue
            sample=cold[request]; row=cold.rows[request.index]
            annotations={k:cold.reader.json(v) for k,v in row['annotations'].items()}
            current=cold.reader.json(row['measurements'][0]); matches=False
            for task,prompt,answer in variants(current,annotations):
                if task!=('driving' if request.task=='rear_driving' else request.task): continue
                conversation=[[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image'}]},
                               {'role':'assistant','content':[{'type':'text','text':answer}]}]]
                teacher,_=get_custom_chat_template(conversation,tokenizer,'OpenGVLab/InternVL2-1B',len(sample.pixels)*256,conversation_path=str(SNAPSHOT/'conversation.py'))
                if torch.equal(teacher['phrase_ids'][0],sample.ids): matches=True; break
            text_ok=text_ok and matches
        result['splits'][split]={'samples':len(cold),'uncached_seconds':cold_seconds,'cache_construction_seconds':cached_seconds,
            'warm_seconds':warm_seconds,'metadata_identical':metadata_ok,'all_modality_samples_identical':samples_ok,
            'parking_sample_checks':len(requests),'context_enumerator_matches_live_samples':text_ok}
        result['passed']=result['passed'] and metadata_ok and samples_ok and text_ok
        print(json.dumps(result['splits'][split]),flush=True)
        cold.reader.close(); cached.reader.close(); warm.reader.close(); del cold,cached,warm
    atomic_json(output,result)
    if not result['passed']: raise AssertionError('Metadata cache changed data')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); main(a.manifest,a.output)
