import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch
import uuid
import numpy as np
import torch
from torch import nn
from simlingo_training.lidar.control import Controller,stopping_defaults,driving_score,full_candidate_order
from simlingo_training.lidar.data import Sample,TASKS
from simlingo_training.lidar.evaluation import evaluate_requests,evaluation_state,summarize_diagnostics
from simlingo_training.lidar.monitor_protocol import build_protocol
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.runtime import UUID


def metrics(error=1.):
    return {name:{'value':error if name in ('route_ade','speed_ade') else 95.,
                  'samples':100,'groups':20,'population_sha256':name}
            for name in ('route_ade','speed_ade','qa_structured_accuracy','dreamer_decision_accuracy','commentary_rouge_l')}


class ControlTests(unittest.TestCase):
    def test_plateau_requires_patience_minimum_and_descending_lr(self):
        c=Controller(stopping_defaults('align'))
        for u in (0,125,250,375):
            r=c.observe(u,750,1.,1.,1.,{'passed':True},[.1],{'u':u})
            self.assertIsNone(r['stop_reason'])
        self.assertIsNone(c.observe(500,750,1.,1.,1.,{'passed':True},[.3],{})['stop_reason'])
        self.assertEqual(c.observe(625,750,1.,1.,1.,{'passed':True},[.1],{})['stop_reason'],'validation_plateau')

    def test_geometry_progress_resets_patience_and_state_round_trips(self):
        c=Controller(stopping_defaults('align'))
        c.observe(0,750,1.,1.,1.,{'passed':True},[1.],{})
        c.observe(125,750,1.,1.,1.,{'passed':True},[1.],{})
        c=Controller(c.rules,json.loads(json.dumps(c.state)))
        self.assertEqual(c.state['wait'],1)
        row=c.observe(250,750,1.,1.,.98,{'passed':True},[.1],{})
        self.assertEqual(row['improved'],['geometry']); self.assertEqual(row['wait'],0)

    def test_candidate_ranks_retention_first_and_excludes_untrained_zero(self):
        c=Controller(stopping_defaults('align'))
        c.observe(0,750,.01,1.,1.,{'passed':True},[1.],{})
        self.assertIsNone(c.state['candidate'])
        c.observe(125,750,.8,1.,1.,{'passed':False},[1.],{})
        c.observe(250,750,1.,1.,1.,{'passed':True},[1.],{})
        c.observe(375,750,.7,1.,1.,{'passed':False},[1.],{})
        self.assertEqual(c.state['candidate']['update'],250)

    def test_invalid_or_repeated_observation_cannot_advance_state(self):
        c=Controller(stopping_defaults('align')); original=copy.deepcopy(c.state)
        for x in (float('nan'),None,float('inf')):
            with self.assertRaises(ValueError): c.observe(0,750,x,1.,1.,{'passed':True},[1.],{})
            self.assertEqual(c.state,original)
        c.observe(0,750,1.,1.,1.,{'passed':True},[1.],{})
        with self.assertRaises(ValueError): c.observe(0,750,1.,1.,1.,{'passed':True},[1.],{})

    def test_budget_stops_even_while_learning_and_fallback_is_bounded(self):
        c=Controller(stopping_defaults('align'))
        c.observe(0,750,1.,1.,1.,{'passed':True},[1.],{})
        self.assertEqual(c.observe(750,750,.5,.5,.5,{'passed':True},[.01],{})['stop_reason'],'budget_exhausted')
        self.assertEqual(full_candidate_order(500,625),[500,625])
        self.assertEqual(full_candidate_order(625,625),[625])

    def test_selection_requires_same_metric_population(self):
        a=metrics(); b=metrics(); b['route_ade']['population_sha256']='other'
        with self.assertRaises(ValueError): driving_score(a,b)


class DiagnosticTests(unittest.TestCase):
    def test_rng_and_mixed_module_modes_restore_even_on_exception(self):
        model=nn.Sequential(nn.Linear(2,2),nn.Dropout(.5)); model.train(); model[0].eval()
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        expected=(random.random(),np.random.rand(),torch.rand(3))
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with evaluation_state(model):
                self.assertFalse(model.training)
                random.random(); np.random.rand(); torch.rand(20)
                raise RuntimeError('injected')
        self.assertTrue(model.training); self.assertFalse(model[0].training); self.assertTrue(model[1].training)
        self.assertEqual(random.random(),expected[0]); self.assertEqual(np.random.rand(),expected[1])
        self.assertTrue(torch.equal(torch.rand(3),expected[2]))

    def test_generated_driving_and_teacher_loss_are_separate_and_generation_is_reused(self):
        geometry={'presence':torch.zeros(4),'presence_valid':torch.ones(4,dtype=torch.bool),
                  'distance':torch.zeros(4,dtype=torch.long),'distance_valid':torch.ones(4,dtype=torch.bool)}
        sample=Sample(torch.zeros(1),torch.tensor([1,2]),torch.tensor([False,True]),torch.tensor([1]),
            torch.zeros(1),torch.zeros(10,2),torch.zeros(20,2),None,geometry,'commentary','u',0)
        # A cloud makes geometry supervision active. A lightweight fixture needs only .to().
        class Cloud:
            def to(self,device): return self
        sample.cloud=Cloud()
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.w=nn.Parameter(torch.ones(1)); self.calls=0
                self.tokenizer=SimpleNamespace(decode=lambda *a,**k:'A: Yes.')
            def generate_ids(self,s,n):
                self.calls+=1; random.random(); np.random.rand(); torch.rand(1)
                return torch.tensor([[99]])
            def driving(self,s,generated):
                assert generated.tolist()==[[99]]
                return {'route':torch.zeros(1,20,2),'speed_wps':torch.zeros(1,10,2)},self.geo()
            def geo(self): return {'presence':torch.zeros(1,4),'distance':torch.zeros(1,4,4)}
            def teacher_language_terms(self,s): return self.w.sum(),self.geo()
            def loss_counts(self,s): return {'language':1,'route':20,'speed':10,'presence':4,'distance':4}
        class Data:
            rows=[{'group':'g'}]
            def __getitem__(self,r): return sample
        model=Model(); model.train(); before=torch.get_rng_state()
        result=evaluate_requests(model,Data(),{'commentary':[{'index':0,'task':'commentary','replay':False,'seed':42,'position':0,'uid':'u'}]},diagnostics=True,device='cpu')
        self.assertEqual(model.calls,1); self.assertTrue(model.training)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        summary=summarize_diagnostics(result['records'])
        self.assertEqual(summary['losses']['language']['value'],1.)
        self.assertEqual(summary['geometry_by_sector']['0']['distance_count'],1)

    def test_monitor_selection_never_uses_prediction_correctness(self):
        rows=[{'id':f't{i}','group':f'tg{i%32}'} for i in range(32)]
        train=SimpleNamespace(rows=rows,pools={t:list(range(32)) for t in TASKS},manifest_hash='m')
        requests={t:[{'index':i,'task':t,'replay':False,'seed':i,'position':i,'uid':f'v{i}'} for i in range(120)] for t in TASKS}
        protocol={'requests':requests,'monitor':{t:v[:64] for t,v in requests.items()}}
        baseline=[]
        for task,items in requests.items():
            if task=='geometry_qa': continue
            for item in items:
                baseline.append({**item,'group':f'vg{item["index"]%59}',
                    'reference':'Following the given instruction.' if task=='dreamer' else 'A: Yes.',
                    'generated':'bad','metrics':{},'total_seconds':1.})
        a=build_protocol(protocol,baseline,train)
        for row in baseline: row['generated']='perfect'; row['metrics']={'fake':100.}
        b=build_protocol(protocol,baseline,train)
        self.assertEqual(a['requests'],b['requests'])
        self.assertEqual(a['coverage']['qa']['scored_targets'],100)
        self.assertEqual(sum(map(len,a['train_requests'].values())),192)


if __name__=='__main__': unittest.main()
