import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from simlingo_training.lidar.data import TaskSampler, TASKS, photometric, equal_spacing_route
from simlingo_training.lidar.validation import route_balanced
from simlingo_training.lidar.engine import normalized_loss, save_checkpoint, load_checkpoint
from simlingo_training.lidar.perception import prepare_cloud, geometry_targets, sector_ids, SpatialConnector
from simlingo_training.lidar.storage import pack_archive, ShardReader, route_group, build_manifest, merge_index
from simlingo_training.models.utils import summarise_losses
from simlingo_training.models.encoder.vlm import VLMEncoderModel
from simlingo_training.models.driving import DrivingModel
from simlingo_training.dataloader.dataset_base import BaseDataset
from PIL import Image


class LidarContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(4)

    def test_inactive_loss_has_finite_zero_gradient(self):
        p = torch.tensor(3.,requires_grad=True)
        result = summarise_losses({'missing':(p.square()[None],torch.zeros(1))})
        result.loss.backward()
        self.assertEqual(result.loss.item(),0.)
        self.assertEqual(p.grad.item(),0.)

    def test_accumulation_matches_packed_token_gradient(self):
        torch.manual_seed(42)
        model = nn.Linear(3,2)
        reference = copy.deepcopy(model)
        inputs = [torch.randn(n,3) for n in (1,5,2)]
        labels = [torch.randint(0,2,(len(x),)) for x in inputs]
        for x,y in zip(inputs,labels):
            normalized_loss({'language':F.cross_entropy(model(x),y,reduction='sum')},{'language':8}).backward()
        F.cross_entropy(reference(torch.cat(inputs)),torch.cat(labels)).backward()
        for a,b in zip(model.parameters(),reference.parameters()):
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-6)

    def test_coordinate_contract_and_sector_boundaries(self):
        xyz = np.array([[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[1,1,0],[1,-1,0]],np.float32)
        np.testing.assert_array_equal(sector_ids(xyz),[0,1,2,3,1,0])
        diagonal=np.array([[0,0,0],[1,1,0],[-1,1,0],[-1,-1,0],[1,-1,0]],np.float32)
        np.testing.assert_array_equal(sector_ids(diagonal),[0,1,2,3,0])
        np.testing.assert_array_equal(sector_ids(diagonal),sector_ids(torch.from_numpy(diagonal)).numpy())
        cloud = prepare_cloud(xyz)
        torch.testing.assert_close(cloud.coord,cloud.metric*.2)
        self.assertTrue(torch.equal(cloud.grid,prepare_cloud(xyz).grid))
        self.assertTrue(torch.equal(cloud.feat[:,3:],torch.zeros(len(cloud.feat),6)))
        for invalid in (np.zeros((0,3)),np.array([[np.nan,0,0]])):
            with self.assertRaises(ValueError): prepare_cloud(invalid)

    def test_visibility_recount_and_annotation_range(self):
        cloud = prepare_cloud(np.array([[-10,0,0],[-10,.1,0],[-10,-.1,0],[0,10,0],[70,0,0],[70,.1,0],[70,-.1,0]],np.float32))
        box = lambda p,n:{'class':'car','position':p,'extent':[1,1,1],'yaw':0.,'num_points':n}
        g = geometry_targets(cloud,[box([-10,0,0],3),box([0,10,0],1),box([70,0,0],3)])
        self.assertTrue(g['presence'][2])
        self.assertFalse(g['presence_valid'][1])
        self.assertFalse(g['presence'][0])
        self.assertTrue(g['distance_valid'][2])

    def test_spatial_visibility_queries_match_full_sweep_reference(self):
        rng=np.random.default_rng(42)
        cloud=prepare_cloud(rng.uniform(-30,30,(18000,3)).astype(np.float32))
        boxes=[{'class':'car','position':rng.uniform(-30,30,3).tolist(),'extent':rng.uniform(.5,5,3).tolist(),
                'yaw':float(rng.uniform(-np.pi,np.pi)),'num_points':3} for _ in range(40)]
        optimized=geometry_targets(cloud,boxes)
        class FullSweep:
            def __init__(self,xyz): self.count=len(xyz)
            def query_ball_point(self,*args): return list(range(self.count))
        with patch('simlingo_training.lidar.perception.cKDTree',FullSweep): reference=geometry_targets(cloud,boxes)
        for key in optimized: torch.testing.assert_close(optimized[key],reference[key],rtol=0,atol=0)

    def test_empty_sectors_and_independent_connector_gradients(self):
        torch.manual_seed(2)
        connector = SpatialConnector([4,8,16])
        hierarchy = [(torch.randn(3,c),torch.tensor([[1.,0.,0.],[2.,0.,0.],[3.,0.,0.]])) for c in (4,8,16)]
        tokens,geo = connector(hierarchy)
        self.assertEqual(tokens.shape,(1,128,896))
        self.assertTrue(torch.isfinite(tokens).all())
        tokens.square().mean().backward()
        self.assertGreater(connector.queries.grad.abs().sum(),0)
        connector.zero_grad(set_to_none=True)
        _,geo = connector(hierarchy)
        (geo['presence'].square().mean()+geo['distance'].square().mean()).backward()
        self.assertGreater(connector.project[0].weight.grad.abs().sum(),0)

    def test_task_quotas_replay_and_sampler_resume(self):
        pools = {task:list(range(3+i)) for i,task in enumerate(TASKS)}
        sampler = TaskSampler(pools,'joint',seed=42,manifest_hash='fixed')
        requests = sampler.take(960)
        counts = {t:sum(r.task==t for r in requests) for t in TASKS}
        self.assertEqual(list(counts.values()),[384,128,128,128,96,96])
        self.assertEqual(sum(r.replay for r in requests),192)
        self.assertFalse(any(r.replay for r in requests if r.task in TASKS[4:]))
        state = sampler.state_dict()
        restored = TaskSampler(pools,'joint',seed=42,manifest_hash='fixed')
        restored.load_state_dict(state)
        self.assertEqual(sampler.take(64),restored.take(64))
        aligned = TaskSampler(pools,'align').take(64)
        self.assertEqual(sum(r.replay for r in aligned),0)
        self.assertEqual([sum(r.task==t for r in aligned) for t in TASKS],[24,8,8,8,8,8])

    def test_bucket_probability_is_independent_of_bucket_size(self):
        sampler=TaskSampler({t:list(range(101)) for t in TASKS},'joint',bucket_pools={'driving':{'small':[0],'large':list(range(1,101))}},bucket_weights={'small':1,'large':1})
        driving=[r for r in sampler.take(9600) if r.task=='driving']
        share=sum(r.index==0 for r in driving)/len(driving)
        self.assertGreater(share,.45); self.assertLess(share,.55)

    def test_route_balanced_unique_validation_selection(self):
        rows=[{'group':'short' if i<2 else 'long'} for i in range(102)]
        selected=route_balanced(rows,list(range(102)),12)
        self.assertEqual(len(selected),len(set(selected)))
        self.assertEqual(sum(i<2 for i in selected),2)
        self.assertEqual(selected,route_balanced(rows,list(range(102)),12))

    def test_photometric_seed_and_original_target_parity(self):
        image=Image.fromarray(np.random.default_rng(1).integers(0,256,(32,64,3),dtype=np.uint8))
        np.testing.assert_array_equal(np.asarray(photometric(image,4294967295)),np.asarray(photometric(image,4294967295)))
        route=np.array([[2.,1.],[7.,2.],[15.,3.]])
        np.testing.assert_allclose(equal_spacing_route(route),BaseDataset.equal_spacing_route(None,route),atol=1e-6)

    def test_indexed_pax_payload_and_route_grouping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.tar.gz'
            name = 'data/simlingo/lb1_split/routes_training/Scenario/'+('Town10_Rep0_'+'x'*180)+'/rgb/0010.jpg'
            value = b'jpeg-payload'*41
            with tarfile.open(source,'w:gz') as archive:
                info = tarfile.TarInfo(name); info.size=len(value); archive.addfile(info,io.BytesIO(value))
            members = pack_archive(source,hashlib.sha256(source.read_bytes()).hexdigest(),root/'packed')
            self.assertEqual(ShardReader(root/'packed',members).read(name),value)
            other=root/'other.tar.gz'
            with tarfile.open(other,'w:gz') as archive:
                info=tarfile.TarInfo(name); info.size=3; archive.addfile(info,io.BytesIO(b'bad'))
            conflicting=pack_archive(other,hashlib.sha256(other.read_bytes()).hexdigest(),root/'packed')
            with self.assertRaises(ValueError): merge_index(root/'packed',members,conflicting)
        a = route_group('data/Town10_Rep0_Scenario4_28_route0_01_12_02_13_38')
        b = route_group('data/Town10_Rep4_Scenario4_28_route1_02_13_01_12_29')
        self.assertEqual(a,b)

    def test_official_splits_future_boundary_and_benchmark_group_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'sensor.tar.gz'
            benchmark=root/'bench.xml'
            benchmark.write_text('<routes><route id="777" town="Town12"><waypoints><position x="0" y="0" z="0"/><position x="100" y="0" z="0"/></waypoints></route></routes>')
            routes=[('train','Town12',1711,0,0),('train','Town12',1711,1,100),('train','Town12',124,0,100),('val','Town13',1711,0,0)]
            with tarfile.open(source,'w:gz') as archive:
                def add(name,value):
                    info=tarfile.TarInfo(name); info.size=len(value); archive.addfile(info,io.BytesIO(value))
                for split,town,identifier,rep,y in routes:
                    route=f'data/simlingo/collection/routes_{"training" if split=="train" else "validation"}/random/{town}_Rep{rep}_{identifier}_route0_time'
                    add(route+'/results.json.gz',gzip.compress(json.dumps({'scores':{'score_composed':100}}).encode()))
                    for frame in range(22):
                        matrix=np.eye(4); matrix[0,3]=frame*4; matrix[1,3]=y
                        add(route+f'/measurements/{frame:04}.json.gz',gzip.compress(json.dumps({'ego_matrix':matrix.tolist()}).encode()))
                    for folder,suffix in [('rgb','jpg'),('lidar','laz'),('boxes','json.gz')]: add(route+f'/{folder}/0010.{suffix}',b'fixture')
            index=pack_archive(source,hashlib.sha256(source.read_bytes()).hexdigest(),root/'packed')
            manifest=build_manifest(root/'packed',index,benchmark)
            self.assertEqual([(r['split'],r['frame']) for r in manifest['samples']],[('train',10),('val',10)])
            self.assertEqual(manifest['benchmark_screen']['excluded_groups'],['Town12:1711'])
            self.assertEqual(len(manifest['benchmark_screen']['matches']),1)

    def test_rgb_freeze_includes_projector_and_module_mode(self):
        class Vision(nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__(); self.model=nn.Module()
                self.model.mlp1=nn.Linear(3,2); self.model.vision_model=nn.Linear(3,3)
        with patch('simlingo_training.models.encoder.vlm.LingoInternVLModel',Vision):
            vision=VLMEncoderModel({'use_global_img':False},None,None,variant='InternVL2',embed_dim=512,freeze=True)
        vision.train()
        self.assertTrue(all(not p.requires_grad for p in vision.parameters()))
        self.assertTrue(all(not m.training for m in vision.modules()))

    def test_legacy_inference_padding_and_commentary_disabled(self):
        model=DrivingModel.__new__(DrivingModel); nn.Module.__init__(model)
        model.wp_encoder=nn.Identity()
        features=torch.zeros(1,6,8)
        owner=self
        class Adaptors:
            language=SimpleNamespace(embed_tokens=nn.Embedding(4,8),lm_head=nn.Linear(8,4))
            def __init__(self):
                self.driving=SimpleNamespace(get_predictions=lambda *args:{'route':torch.zeros(1,20,2),'speed_wps':torch.zeros(1,10,2)})
            def __call__(self,*args,**kwargs):
                return {'language_inputs':torch.zeros(1,4,8),'language_inputs_mask':torch.tensor([[False,True,True,True]])}
            def split_outputs_by_adaptor(self,dictionary,hidden):
                owner.assertIsInstance(hidden,torch.Tensor)
                return {'driving':hidden}
        class Driving:
            def __call__(self,*args): return {'inputs':torch.zeros(1,2,8)}
            def get_predictions(self,*args): return {'route':torch.zeros(1,20,2),'speed_wps':torch.zeros(1,10,2)}
        model.adaptors=Adaptors(); model.adaptors.driving=Driving()
        model.vision_model=SimpleNamespace(image_encoder=SimpleNamespace(replace_placeholder_tokens=lambda **kw:kw['adaptor_dict']))
        masks=[]
        class Language:
            variant='fixture'
            def greedy_sample(self,embeds,**kw): return torch.tensor([[1,2]]),torch.cat((embeds,torch.zeros(1,2,8)),dim=1)
            def forward(self,embeds,attention_mask=None):
                masks.append(attention_mask); return torch.zeros(1,8,8),torch.zeros(1,8,4)
        model.language_model=Language()
        model.tokenizer=SimpleNamespace(eos_token_id=2,batch_decode=lambda *args,**kw:['test'])
        driving_input=SimpleNamespace(camera_images=torch.zeros(1,3,8,8),prompt_inference=SimpleNamespace(placeholder_values={}))
        model.predict_language=True
        model(driving_input)
        torch.testing.assert_close(masks[0],torch.tensor([[False,True,True,True,True,True,True,True]]))
        model.predict_language=False
        model.forward_model=lambda *args:(features,torch.zeros(1,6,4))
        speed,route,language=model(driving_input)
        self.assertEqual(speed.shape,(1,10,2)); self.assertEqual(route.shape,(1,20,2)); self.assertEqual(language,[])

    def test_atomic_optimizer_rng_and_sampler_resume(self):
        model=nn.Linear(2,1); model.stage='align'
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1)
        model(torch.ones(1,2)).sum().backward(); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        sampler=TaskSampler({t:[0] for t in TASKS},manifest_hash='unit')
        sampler.take(3)
        original=copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'last.pt'
            save_checkpoint(path,model,optimizer,scheduler,sampler,1,{'config':'unit'})
            expected=(random.random(),np.random.rand(),torch.rand(3),sampler.take(3))
            model.weight.data.add_(10)
            load_checkpoint(path,model,optimizer,scheduler,sampler,{'config':'unit'})
            actual=(random.random(),np.random.rand(),torch.rand(3),sampler.take(3))
            self.assertEqual(expected[0],actual[0]); self.assertEqual(expected[1],actual[1])
            torch.testing.assert_close(expected[2],actual[2]); self.assertEqual(expected[3],actual[3])
            for k in original: torch.testing.assert_close(model.state_dict()[k],original[k])
            self.assertFalse(path.with_suffix('.pt.part').exists())


if __name__=='__main__': unittest.main()
