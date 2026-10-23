"""Real archive/annotation, original-target, and augmentation worker contracts."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from simlingo_training.dataloader.dataset_base import BaseDataset
from simlingo_training.lidar.data import ManifestDataset, TaskSampler, SNAPSHOT, Request, TASKS
from simlingo_training.lidar.storage import atomic_json
from simlingo_training.lidar.validation import validation_protocol

ROOT=Path('/ehsan/m.sadegh/simlingo_lidar/stage2')


def identity(items): return items


def fingerprint(sample):
    h=hashlib.sha256()
    values=[sample.pixels,sample.ids,sample.answer_mask,sample.question_ids,sample.navigation,sample.waypoints,sample.route]
    values+=list(sample.geometry.values())
    if sample.cloud is not None: values += [sample.cloud.coord,sample.cloud.grid,sample.cloud.feat,sample.cloud.metric]
    for value in values: h.update(value.numpy().tobytes())
    h.update((sample.uid+sample.task).encode())
    return h.hexdigest()


def main(manifest=None):
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(str(SNAPSHOT),trust_remote_code=True,local_files_only=True)
    tokenizer=getattr(processor,'tokenizer',processor)
    manifest=Path(manifest) if manifest else ROOT/'data/manifest.json'
    train=ManifestDataset(manifest,tokenizer,'train',augment=True,max_samples=128)
    val=ManifestDataset(manifest,tokenizer,'val',max_samples=128)
    sampler=TaskSampler(train.pools,'joint',manifest_hash=train.manifest_hash,bucket_pools=train.bucket_pools,bucket_weights=train.bucket_weights)
    # Exercise every task with the same saved seed, including augmented RGB replay.
    requests=[Request(train.pools[t][0],t,False,4294967295,i) for i,t in enumerate(TASKS)]
    requests += [Request(train.pools[t][0],t,True,123,i+6) for i,t in enumerate(TASKS[:4])]
    reference=[fingerprint(s) for batch in torch.utils.data.DataLoader(train,batch_size=1,sampler=requests,num_workers=0,collate_fn=identity) for s in batch]
    worker=[fingerprint(s) for batch in torch.utils.data.DataLoader(train,batch_size=1,sampler=requests,num_workers=2,prefetch_factor=1,collate_fn=identity,
                                                                 multiprocessing_context='spawn') for s in batch]
    assert reference==worker
    for i in range(8):
        row=train.rows[i]
        measurements=[train.reader.json(p) for p in row['measurements']]
        sample=train[Request(i,'driving',False,42,i)]
        np.testing.assert_allclose(sample.waypoints.numpy(),BaseDataset.get_waypoints(None,measurements)[1:-1],atol=1e-5)
        np.testing.assert_allclose(sample.route.numpy(),BaseDataset.equal_spacing_route(None,np.asarray(measurements[0]['route'])),atol=1e-5)
        assert sample.cloud.timestamp==row['frame']*.25
    assert not ({r['group'] for r in train.rows}&{r['group'] for r in val.rows})
    protocol=validation_protocol(val)
    repeated=validation_protocol(val)
    assert protocol==repeated
    for task,items in protocol['requests'].items():
        if items:
            item={k:v for k,v in items[0].items() if k!='uid'}
            assert fingerprint(val[Request(**item)])==fingerprint(val[Request(**item)])
    report={'passed':True,'identical_worker_requests':len(requests),'workers':[0,2],
        'manifest_sha256':train.manifest_hash,'manifest':str(manifest),
        'photometric_augmentation':'Seeded recursively; identical tensors across spawn workers',
        'original_target_comparisons':8,'speed_waypoint_shape':[10,2],'route_shape':[20,2],'saved_cadence_seconds':.25,
        'fixed_validation_requests_repeat':True,'no_group_overlap':True,
        'task_pools':{'train':{k:len(v) for k,v in train.pools.items()},'val':{k:len(v) for k,v in val.pools.items()}}}
    atomic_json(ROOT/'reports/data-checks.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--manifest',type=Path)
    main(parser.parse_args().manifest)
