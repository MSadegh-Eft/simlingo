import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image
from torch import nn

from simlingo_training.lidar.data import check_context_lengths,TASKS,TaskSampler
from simlingo_training.lidar.storage import official_split
from simlingo_training.lidar.runtime import TrainingSession,tensor_fingerprint
from tools.lidar_readiness.full_baseline import recover_journal
from simlingo_training.lidar.train import validate_config,main as training_main
from simlingo_training.lidar.engine import save_checkpoint,load_checkpoint
from simlingo_training.lidar import sparse_kernels


class ReadinessContracts(unittest.TestCase):
    def test_sparse_kernel_choices_round_trip_and_reject_changed_architecture(self):
        from spconv.algo import CONV,ALL_CONV_ALGO_DESPS,BestConvAlgoByProfile
        model=SimpleNamespace(lidar_numerics_policy={'sparse_kernel_checkpointing':True})
        originals={name:getattr(CONV,name).copy() for name in sparse_kernels.TABLES}
        descriptor=next(d for d in ALL_CONV_ALGO_DESPS if 'f32f32f32f32f32' in str(d))
        try:
            for name in sparse_kernels.TABLES:
                getattr(CONV,name).clear()
                getattr(CONV,name)[(0,0,0,576,576,8,9,-1,False)]=BestConvAlgoByProfile(descriptor,(8,9),1)
            state=sparse_kernels.dump(model)
            for name in sparse_kernels.TABLES: getattr(CONV,name).clear()
            with patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(major=8,minor=9)):
                sparse_kernels.restore(model,state)
                self.assertEqual(sparse_kernels.dump(model),state)
            with patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(major=9,minor=0)):
                with self.assertRaises(ValueError): sparse_kernels.restore(model,state)
        finally:
            for name in sparse_kernels.TABLES:
                getattr(CONV,name).clear(); getattr(CONV,name).update(originals[name])

    def test_lazy_augmentation_initialization_does_not_change_post_seed_numpy_rng(self):
        from simlingo_training.lidar.runtime import initialize_device,UUID
        from simlingo_training.lidar.data import photometric
        import imgaug.random as iarandom
        import os
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2'}),patch.object(iarandom,'GLOBAL_RNG',None), \
                patch('torch.cuda.device_count',return_value=1),patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(uuid=UUID,total_memory=48*1024**3)), \
                patch('torch.cuda.set_per_process_memory_fraction'),patch('torch.cuda.manual_seed_all'),patch('simlingo_training.lidar.runtime.check_resources'):
            initialize_device(root); expected=np.random.get_state()
            image=Image.fromarray(np.zeros((32,32,3),dtype=np.uint8))
            photometric(image,42); actual=np.random.get_state()
            self.assertTrue(all(np.array_equal(a,b) for a,b in zip(expected,actual)))

    def test_entry_point_saves_initial_boundary_before_first_update(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); manifest=root/'manifest.json'; manifest.write_text('{}')
            import hashlib
            sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
            metadata={'manifest_sha256':sha,'validation_sha256':'v','code':{},'external_sources':{},'versions':{}}
            (root/'validation.json').write_text(json.dumps({'requests':{},'monitor':{}}))
            (root/'readiness.json').write_text(json.dumps({'technical_checks_passed':True,'manifest_sha256':sha}))
            (root/'baseline.json').write_text(json.dumps({'complete_protocol':True,'coverage_gate':{'passed':True},'binding':metadata}))
            config={'schema':1,'stage':'align','budget_reviewed':True,'manifest':str(manifest),'output':str(root/'run'),
                'readiness_report':str(root/'readiness.json'),'baseline_result':str(root/'baseline.json'),
                'total_updates':750,'full_validation_updates':[750],'monitor_updates':[],
                'checkpoint_interval_updates':15,'checkpoint_interval_seconds':1800}
            path=root/'config.json'; path.write_text(json.dumps(config))
            def save(path): path.write_text('completed boundary zero')
            def fail(**kwargs):
                self.assertTrue((root/'run/last.pt').exists())
                raise RuntimeError('Injected interruption before first update')
            session=SimpleNamespace(updates=0,total_updates=750,metadata=metadata,model=SimpleNamespace(tokenizer=None),save=save,update=fail)
            with patch('simlingo_training.lidar.train.create_session',return_value=session),patch('simlingo_training.lidar.train.ManifestDataset'),patch('simlingo_training.lidar.train.check_resources'):
                with self.assertRaisesRegex(RuntimeError,'Injected interruption'): training_main(path)
            self.assertEqual((root/'run/last.pt').read_text(),'completed boundary zero')

    def test_initial_checkpoint_restores_before_adam_has_any_state(self):
        model=nn.Linear(1,1); model.stage='align'; optimizer=torch.optim.AdamW(model.parameters())
        sampler=TaskSampler({t:[0] for t in TASKS}); weight=model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'zero.pt'; save_checkpoint(path,model,optimizer,None,sampler,0,{'run':'initial'})
            model.weight.data.add_(1); sampler.take(64)
            self.assertEqual(load_checkpoint(path,model,optimizer,None,sampler,{'run':'initial'}),0)
            self.assertEqual(sampler.position,0); self.assertFalse(optimizer.state)
            self.assertTrue(torch.equal(model.weight,weight))

    def test_training_requires_reviewed_budget_and_final_validation(self):
        config={'schema':1,'stage':'align','total_updates':750,'budget_reviewed':False,
            'full_validation_updates':[750],'monitor_updates':[250,500],
            'checkpoint_interval_updates':15,'checkpoint_interval_seconds':1800}
        with self.assertRaises(ValueError): validate_config(config)
        config['budget_reviewed']=True; validate_config(config)
        config['full_validation_updates']=[]
        with self.assertRaises(ValueError): validate_config(config)
        config['full_validation_updates']=[750,751]
        with self.assertRaises(ValueError): validate_config(config)

    def test_official_parking_layout_and_unknown_layout(self):
        self.assertEqual(official_split('data/simlingo/training_parking_lane/Town12_short/random_weather/Town12_Rep0_36_route0_time'),'train')
        self.assertEqual(official_split('data/simlingo/validation_parking_lane/Town13_short/random_weather/Town13_Rep0_36_route0_time'),'val')
        self.assertEqual(official_split('data/simlingo/collection/routes_training/random/Town01_Rep0_1_route0_time'),'train')
        with self.assertRaises(ValueError): official_split('data/simlingo/unrecognized/Town13_route0_time')

    def test_generation_and_driving_reserve_are_counted(self):
        self.assertEqual(check_context_lengths(1900,1790),2048)
        with self.assertRaises(ValueError): check_context_lengths(1900,1791)
        self.assertEqual(check_context_lengths(1900,1791,paired=False),1921)
        with self.assertRaises(ValueError): check_context_lengths(2049,10,paired=False)

    def test_failed_window_never_advances_or_saves_committed_cursor(self):
        session=TrainingSession.__new__(TrainingSession)
        session.model=SimpleNamespace(); session.dataset={i:i for i in range(64)}
        session.sampler=TaskSampler({t:list(range(64)) for t in TASKS})
        session.total_updates=750; session.updates=0; session.poisoned=False
        session.optimizer=object(); session.scheduler=object()
        session.dataset=SimpleNamespace()
        class Data:
            def __getitem__(self,request): return request
        session.dataset=Data()
        with patch('simlingo_training.lidar.runtime.step_window',side_effect=RuntimeError('Injected mid-window failure')):
            with self.assertRaises(RuntimeError): session.update()
        self.assertEqual(session.sampler.position,0); self.assertEqual(session.updates,0)
        with self.assertRaises(RuntimeError): session.save('must-not-exist.pt')
        with self.assertRaises(RuntimeError): session.update()

    def test_completed_window_commits_once_after_clearing_gradients(self):
        session=TrainingSession.__new__(TrainingSession)
        session.model=nn.Linear(1,1); session.dataset=SimpleNamespace()
        class Data:
            def __getitem__(self,request): return request
        session.dataset=Data(); session.sampler=TaskSampler({t:[0] for t in TASKS})
        session.optimizer=object(); session.scheduler=object(); session.total_updates=750; session.updates=0; session.poisoned=False
        with patch('simlingo_training.lidar.runtime.step_window',return_value={'loss':1.}): session.update()
        self.assertEqual(session.sampler.position,64); self.assertEqual(session.updates,1)

    def test_validation_journal_resumes_only_matching_complete_records(self):
        binding={'code':'original'}; key=('u','qa',42)
        row={'uid':'u','task':'qa','seed':42}
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'journal.jsonl'
            self.assertEqual(recover_journal(path,binding,{key}),[])
            with path.open('a') as f: f.write(json.dumps(row)+'\n'+ '{"uid":')
            self.assertEqual(recover_journal(path,binding,{key}),[row])
            self.assertTrue(path.read_text().endswith('\n'))
            with self.assertRaises(ValueError): recover_journal(path,{'code':'changed'},{key})
            with path.open('a') as f: f.write(json.dumps(row)+'\n')
            with self.assertRaises(ValueError): recover_journal(path,binding,{key})

    def test_fingerprint_includes_shape_dtype_and_every_value(self):
        a=torch.tensor([1.,2.])
        self.assertNotEqual(tensor_fingerprint([('a',a)]),tensor_fingerprint([('a',a.double())]))
        self.assertNotEqual(tensor_fingerprint([('a',a)]),tensor_fingerprint([('a',a[None])]))
        self.assertNotEqual(tensor_fingerprint([('a',a)]),tensor_fingerprint([('a',a+1)]))
        self.assertTrue(tensor_fingerprint([('scalar',torch.tensor(1.))]))


if __name__=='__main__': unittest.main()
