"""Disposable GPU check: diagnostics must preserve the complete next update."""
import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from simlingo_training.lidar.data import ManifestDataset
from simlingo_training.lidar.runtime import METADATA_CACHE,check_resources
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.train import create_session,prelaunch_resources,atomic_copy
from simlingo_training.lidar.launch_guard import training_lock
from simlingo_training.lidar.validation_run import evaluate_complete
from tools.lidar_readiness.fresh_resume import compare


def main(manifest,protocol_path,output,stage):
    prelaunch_resources(output)
    with training_lock():
        output.mkdir(parents=True,exist_ok=False)
        monitor=json.loads(protocol_path.read_text())
        if monitor['manifest_sha256']!=digest_file(manifest): raise ValueError('Monitoring dataset differs')
        config={'purpose':'Disposable diagnostic invariance check','entrypoint_sha256':digest_file(__file__),
                'monitor_protocol_sha256':digest_file(protocol_path)}
        session=create_session(manifest,output,stage,750 if stage=='align' else 1161,config)
        check=lambda:check_resources(output)
        session.update(check=check)
        saved=time.monotonic(); session.save(output/'boundary.pt'); save_seconds=time.monotonic()-saved
        copied=time.monotonic(); atomic_copy(output/'boundary.pt',output/'copy-check.pt'); copy_seconds=time.monotonic()-copied
        if digest_file(output/'copy-check.pt')!=digest_file(output/'boundary.pt'): raise ValueError('Atomic checkpoint copy differs')
        (output/'copy-check.pt').unlink()
        expected_stats=session.update(check=check); session.save(output/'expected.pt')
        session.resume(output/'boundary.pt')
        train=copy.copy(session.dataset); train.augment=False
        val=ManifestDataset(manifest,session.model.tokenizer,'val',metadata_cache=METADATA_CACHE)
        times={}; reports={}
        for name,dataset,requests,mode in [('paired',val,monitor['requests'],'paired'),
            ('rgb',val,monitor['requests'],'rgb'),('train',train,monitor['train_requests'],'paired')]:
            started=time.monotonic()
            reports[name]=evaluate_complete(session.model,dataset,requests,mode,output/name,
                {'metadata':session.metadata,'checkpoint_sha256':digest_file(output/'boundary.pt'),
                 'mode':mode,'diagnostics':True,'split':name},check=check,diagnostics=True,complete_protocol=False)
            times[name]=time.monotonic()-started
        actual_stats=session.update(check=check)
        expected=torch.load(output/'expected.pt',map_location='cpu',weights_only=False,mmap=True)
        adam=lambda state:{f'{i}:{k}':v for i,s in state['state'].items() for k,v in s.items() if torch.is_tensor(v)}
        model=compare(session.model.state_dict().items(),expected['model'])
        optimizer=compare(adam(session.optimizer.state_dict()).items(),adam(expected['optimizer']))
        rng=expected['rng']
        checks={'model':model['bitwise'],'optimizer':optimizer['bitwise'],'stats':actual_stats==expected_stats,
            'scheduler':session.scheduler.state_dict()==expected['scheduler'],
            'sampler':session.sampler.state_dict()==expected['sampler'],
            'tracking':all(session.run_state[k]==expected['run_state'][k] for k in ('seen_indices','task_counts','replay_count','bucket_counts')),
            'python_rng':random.getstate()==rng['python'],
            'numpy_rng':all(np.array_equal(a,b) for a,b in zip(np.random.get_state(),rng['numpy'])),
            'torch_rng':torch.equal(torch.get_rng_state(),rng['torch']),
            'cuda_rng':all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),rng['cuda']))}
        report={'passed':all(checks.values()),'checks':checks,'model':model,'optimizer':optimizer,
                'stage':stage,'metadata':session.metadata,'monitor_protocol_sha256':digest_file(protocol_path),
                'monitor_seconds':times,'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'checkpoint_save_seconds':save_seconds,'candidate_copy_seconds':copy_seconds,
                'disposable_optimizer_windows':3,'main_training_started':False}
        atomic_json(output/'result.json',report)
        if not report['passed']: raise AssertionError('Monitoring changed the next update; retain diagnostics')
        del expected
        (output/'boundary.pt').unlink(); (output/'expected.pt').unlink()
        print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--monitor-protocol',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['align','joint'],required=True)
    a=p.parse_args(); main(a.manifest,a.monitor_protocol,a.output,a.stage)
