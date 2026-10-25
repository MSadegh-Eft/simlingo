"""Single physical GPU safeguards and completed-window training boundaries."""
import hashlib
import os
from pathlib import Path
import random
import shutil

import numpy as np
import psutil
import torch

from simlingo_training.lidar.data import TaskSampler
from simlingo_training.lidar.engine import step_window, save_checkpoint, load_checkpoint, make_optimizer, checkpoint_metadata

UUID='3e4277b2-2a3c-6f9b-6a87-273c09bfd49d'
GIB=1024**3
METADATA_CACHE=Path('/data/ghazaleh/.cache/simlingo-lidar/metadata')


def initialize_device(output):
    from simlingo_training.lidar.numerics import prepare_environment
    prepare_environment()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2' or torch.cuda.device_count()!=1:
        raise RuntimeError('Use the isolated launcher exposing only physical GPU 2')
    prop=torch.cuda.get_device_properties(0)
    if str(prop.uuid).removeprefix('GPU-')!=UUID: raise RuntimeError('Physical GPU UUID differs')
    torch.cuda.set_per_process_memory_fraction(min(.88,40*GIB/prop.total_memory))
    torch.set_num_threads(4)
    # imgaug initializes its own global generator lazily from NumPy on first
    # use. Initialize it before seeding/restoring the training RNG, even though
    # every augmentation tree is subsequently seeded by its explicit Request.
    from imgaug.random import get_global_rng
    get_global_rng()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    check_resources(output)


def check_resources(output):
    if psutil.virtual_memory().available<8*GIB: raise MemoryError('Host available RAM below 8 GiB')
    if shutil.disk_usage(output).free<150*GIB: raise OSError('Shared free space below 150 GiB')
    if torch.cuda.is_initialized() and torch.cuda.memory_reserved()>40*GIB:
        raise MemoryError('Training process exceeds 40 GiB GPU reserve cap')


def tensor_fingerprint(values):
    h=hashlib.sha256()
    for name,value in values:
        value=value.detach().cpu().contiguous()
        h.update(str(name).encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def sample_fingerprint(sample):
    values=[(k,getattr(sample,k)) for k in ('pixels','ids','answer_mask','question_ids','navigation','waypoints','route')]
    values.extend(('geometry/'+k,v) for k,v in sorted(sample.geometry.items()))
    if sample.cloud is not None:
        values.extend(('cloud/'+k,getattr(sample.cloud,k)) for k in ('coord','grid','feat','metric'))
    return tensor_fingerprint(values)


class TrainingSession:
    """The same boundary/resume code is used by readiness and the future runner.

    Serial loading deliberately has no prefetch cursor. Requests are read without
    advancing the committed sampler. A failed window poisons the session and can
    only be recovered by reloading its last complete checkpoint in a new process.
    """
    def __init__(self,model,dataset,stage,total_updates,config):
        self.model,self.dataset=model,dataset
        model.set_stage(stage); model.train()
        self.optimizer,self.scheduler=make_optimizer(model,total_updates)
        self.sampler=TaskSampler(dataset.pools,stage,manifest_hash=dataset.manifest_hash,
            bucket_pools=dataset.bucket_pools,bucket_weights=dataset.bucket_weights)
        self.updates=0; self.total_updates=total_updates; self.poisoned=False
        self.metadata=checkpoint_metadata(model,dataset.path,{'microbatch':1,'accumulation':64,'cache':False,
            'numerics':model.lidar_numerics_policy,'total_updates':total_updates,'run':config})

    def update(self,observe=None,check=None):
        if self.poisoned: raise RuntimeError('Failed window requires fresh-process recovery')
        if self.updates>=self.total_updates: raise ValueError('Agreed update budget exhausted')
        if check: check()
        start=self.sampler.position
        try:
            requests=[self.sampler.request_at(i) for i in range(start,start+64)]
            samples=[self.dataset[r] for r in requests]
            if observe: observe(requests,samples)
            stats=step_window(self.model,samples,self.optimizer,self.scheduler,generation_limit=100)
            self.model.zero_grad(set_to_none=True)
            if check: check()
            self.sampler.position=start+64; self.updates+=1
            return stats
        except BaseException:
            self.poisoned=True
            raise

    def save(self,path):
        if self.poisoned: raise RuntimeError('Cannot save an incomplete/failed update')
        if self.sampler.position!=self.updates*64: raise ValueError('Committed sampler differs from optimizer count')
        save_checkpoint(path,self.model,self.optimizer,self.scheduler,self.sampler,self.updates,self.metadata)

    def resume(self,path):
        self.updates=load_checkpoint(path,self.model,self.optimizer,self.scheduler,self.sampler,self.metadata)
        if not 0<=self.updates<=self.total_updates or self.sampler.position!=self.updates*64:
            self.poisoned=True; raise ValueError('Checkpoint boundary/count differs')
        self.poisoned=False
