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
from simlingo_training.lidar.launch_guard import training_lock
from simlingo_training.lidar.monitor_protocol import build_protocol
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.train import run_monitored,monitor_signals,atomic_copy,prelaunch_resources
from simlingo_training.lidar.runtime import UUID
from simlingo_training.lidar.telemetry import Reporter
from simlingo_training.lidar.training_report import recommendations,svg_curves,write_report
from tools.lidar_training.manage import create_tmux_session


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
        with self.assertRaises(ValueError): monitor_signals({'coverage_gate':{'passed':False}},a)


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


class LoopTests(unittest.TestCase):
    def fixture(self,root,fail_update=None):
        class Data:
            augment=True
            def __len__(self): return 1000
        class Session:
            def __init__(self):
                self.updates=0; self.total_updates=750; self.metadata={'fixture':True}; self.dataset=Data()
                self.model=nn.Linear(1,1); self.sampler=SimpleNamespace(position=0)
                self.optimizer=SimpleNamespace(param_groups=[{'name':'connector','max_lr':1.,'lr':1.}])
                self.run_state={'controller':None,'seen_indices':[],'task_counts':{},'replay_count':0,'bucket_counts':{},'training_seconds':0.}
            def update(self,check=None,progress=None):
                if self.updates==fail_update: raise RuntimeError('injected update failure')
                self.updates+=1; self.sampler.position+=64
                self.optimizer.param_groups[0]['lr']=.1 if self.updates>=600 else 1.
                self.run_state['seen_indices']=list(range(min(1000,self.updates*64)))
                return {'loss':1.}
            def save(self,path):
                torch.save({'updates':self.updates,'model':self.model.state_dict(),'metadata':self.metadata,
                    'sparse_kernels':None,'run_state':copy.deepcopy(self.run_state),'lr':self.optimizer.param_groups[0]['lr']},path)
            def resume(self,path):
                state=torch.load(path,weights_only=False); self.updates=state['updates']; self.sampler.position=self.updates*64
                self.run_state=state['run_state']; self.model.load_state_dict(state['model'])
                self.optimizer.param_groups[0]['lr']=state['lr']
        config={'stage':'align','stopping':stopping_defaults('align'),'monitor_protocol_sha256':'m',
                'checkpoint_interval_updates':15,'total_updates':750}
        return Session(),config

    def evaluator(self,fail_full_candidate=False):
        calls=[]
        def evaluate(model,dataset,requests,mode,path,binding,**kwargs):
            path=Path(path); calls.append(str(path))
            error=1.2 if fail_full_candidate and 'full-125/rgb' in str(path) else 1.
            report={'coverage_gate':{'passed':True},'complete_protocol':kwargs.get('complete_protocol',True),
                'metrics':metrics(error),'diagnostics':{'losses':{k:{'value':1.,'count':100} for k in ('language','presence','distance')}}}
            path.mkdir(parents=True,exist_ok=True); atomic_json(path/'result.json',report)
            return report
        return evaluate,calls

    def run_loop(self,session,config,root,evaluator,resume=None):
        with patch('simlingo_training.lidar.train.evaluate_complete',side_effect=evaluator), \
             patch('simlingo_training.lidar.train.check_resources'), \
             patch('simlingo_training.lidar.sparse_kernels.restore'), \
             patch('simlingo_training.lidar.training_report.write_report',return_value={'fixture':True}):
            return run_monitored(session,config,root,object(),{'requests':{}},
                {'requests':{},'train_requests':{},'baseline_metrics':metrics()},
                {'metrics':metrics()},MagicMock(),resume)

    def test_committed_decision_recovers_once_and_full_failure_uses_final_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); session,config=self.fixture(root,fail_update=126); evaluate,calls=self.evaluator(True)
            with self.assertRaisesRegex(RuntimeError,'injected'): self.run_loop(session,config,root,evaluate)
            self.assertEqual(session.updates,126)
            recovered,config=self.fixture(root); recovered.resume(root/'last.pt')
            self.assertEqual(recovered.updates,125)
            count=calls.count(str(root/'monitor-125/paired'))
            self.run_loop(recovered,config,root,evaluate,root/'last.pt')
            self.assertEqual(calls.count(str(root/'monitor-125/paired')),count)
            self.assertEqual(recovered.updates,625)
            final=json.loads((root/'complete.json').read_text())
            self.assertEqual(final['stop_reason'],'validation_plateau')
            self.assertEqual(final['selected']['update'],625)
            self.assertEqual(len(json.loads((root/'selection.json').read_text())['checks']),2)
            decision=json.loads((root/'decisions/00000625.json').read_text())
            self.assertEqual([r['update'] for r in decision['controller']['history']],[0,125,250,375,500,625])

    def test_interrupted_monitor_resumes_before_another_optimizer_update(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); session,config=self.fixture(root); evaluate,calls=self.evaluator()
            def interrupted(*a,**kw):
                if str(a[4]).endswith('monitor-125/rgb'): raise RuntimeError('interrupted monitor')
                return evaluate(*a,**kw)
            with self.assertRaisesRegex(RuntimeError,'interrupted monitor'): self.run_loop(session,config,root,interrupted)
            self.assertEqual(session.updates,125)
            recovered,config=self.fixture(root); recovered.resume(root/'last.pt')
            self.run_loop(recovered,config,root,evaluate,root/'last.pt')
            self.assertTrue((root/'decisions/00000125.json').exists())
            self.assertEqual(len(json.loads((root/'selection.json').read_text())['checks']),1)

    def test_failed_candidate_copy_is_recoverable_without_false_committed_decision(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); session,config=self.fixture(root); evaluate,_=self.evaluator()
            with patch('simlingo_training.lidar.train.atomic_copy',side_effect=OSError('interrupted checkpoint copy')):
                with self.assertRaises(OSError): self.run_loop(session,config,root,evaluate)
            self.assertFalse((root/'decisions/00000125.json').exists())
            recovered,config=self.fixture(root); recovered.resume(root/'last.pt')
            self.run_loop(recovered,config,root,evaluate,root/'last.pt')
            self.assertTrue((root/'complete.json').exists())


class ReportTests(unittest.TestCase):
    def test_report_artifacts_include_matched_loss_curves_examples_and_evidence_for_advice(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); controller=Controller(stopping_defaults('align'))
            for u,train_loss,val_loss in [(0,2.,2.),(125,1.8,2.2),(250,1.6,2.4)]:
                controller.observe(u,750,1.,1.,1.,{'passed':True},[1.],{})
                for mode,loss in [('train',train_loss),('paired',val_loss),('rgb',val_loss)]:
                    path=root/f'monitor-{u}'/mode; path.mkdir(parents=True)
                    atomic_json(path/'result.json',{'metrics':metrics(),'by_scenario_bucket':{},
                        'diagnostics':{'by_task':{'qa':{'language':{'value':loss,'count':100}}}}})
                (root/f'monitor-{u}/paired/records.jsonl').write_text(json.dumps({'uid':'u','group':'g','task':'qa',
                    'generated':'yes','reference':'no','metrics':{}})+'\n')
            tracking={'seen_indices':[0,1],'task_counts':{'qa':192},'replay_count':0,'bucket_counts':{}}
            result=write_report(root,{'stage':'align'},controller.state,tracking,{'selected':None}, {},{'baseline_metrics':metrics()})
            for name in ('report.json','report.md','report.html','learning-curves.csv','learning-curves.svg'):
                self.assertTrue((root/name).stat().st_size)
            self.assertEqual(result['training_language_loss'],[2.,1.8,1.6])
            self.assertEqual(result['matched_validation_language_loss'],[2.,2.2,2.4])
            self.assertTrue(any('overfitting' in r['action'] for r in result['recommendations']))
            self.assertEqual(result['examples'][0]['reference'],'no')


class TerminalTests(unittest.TestCase):
    def test_atomic_copy_preserves_existing_checkpoint_on_interrupted_write(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); source=root/'source.pt'; target=root/'candidate.pt'
            source.write_bytes(b'new checkpoint'); target.write_bytes(b'previous checkpoint')
            def interrupted(src,dst,*args):
                dst.write(b'partial'); raise OSError('interrupted write')
            with patch('shutil.copyfileobj',side_effect=interrupted):
                with self.assertRaises(OSError): atomic_copy(source,target)
            self.assertEqual(target.read_bytes(),b'previous checkpoint')
            atomic_copy(source,target); self.assertEqual(target.read_bytes(),source.read_bytes())

    def test_non_tty_status_tracks_accumulation_and_writes_plain_log(self):
        with tempfile.TemporaryDirectory() as root,patch('sys.stderr',io.StringIO()):
            reporter=Reporter(root,'align',750)
            try:
                with reporter.accumulation() as advance: advance(16); advance(48)
                reporter.status(force=True)
                reporter.event('fixture_complete',updates=0)
            finally: reporter.close()
            status=json.loads((Path(root)/'status.json').read_text())
            self.assertEqual(status['accumulation_completed'],64)
            self.assertIn('fixture_complete',(Path(root)/'training.log').read_text())
            self.assertNotIn('\x1b',(Path(root)/'training.log').read_text())

    def test_busy_gpu_fails_before_model_construction(self):
        with patch('subprocess.check_output',return_value='GPU-'+UUID+', 2300\n'):
            with self.assertRaisesRegex(MemoryError,'40 GiB'): prelaunch_resources('/tmp')

    def test_single_process_lock_is_released_after_exit(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'run.lock'
            with training_lock(path):
                with self.assertRaises(RuntimeError):
                    with training_lock(path): pass
            with training_lock(path): pass

    def test_tmux_attach_detach_duplicate_rejection_and_retained_cpu_log(self):
        socket='simlingo-test-'+uuid.uuid4().hex[:10]; session='cpu-smoke'
        prefix=['tmux','-L',socket]
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); log=root/'cpu.log'
            code='import time\nfrom pathlib import Path\np=Path('+repr(str(log))+')\nfor i in range(24):\n print("CPU monitoring smoke",i,flush=True)\n with p.open("a") as f: f.write(str(i)+"\\n")\n time.sleep(.1)\n'
            command=['python','-u','-c',code]
            try:
                # Exercise a common user customization: the first window is 1.
                subprocess.run(prefix+['-f','/dev/null','new-session','-d','-s','seed','sleep 30'],check=True)
                subprocess.run(prefix+['set-option','-g','base-index','1'],check=True)
                create_tmux_session(command,root,session,socket)
                with self.assertRaises(RuntimeError): create_tmux_session(command,root,session,socket)
                attached=subprocess.Popen(['script','-q','-c',shlex.join(prefix+['attach-session','-t',session]),'/dev/null'],
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env={**os.environ,'TERM':'xterm'})
                time.sleep(.3); before=len(log.read_text().splitlines())
                out,err=attached.communicate(b'\x02d',timeout=10)
                self.assertEqual(attached.returncode,0,(out,err)); self.assertIn(b'CPU monitoring smoke',out)
                time.sleep(.3); self.assertGreater(len(log.read_text().splitlines()),before)
                deadline=time.monotonic()+8
                while time.monotonic()<deadline:
                    dead=subprocess.check_output(prefix+['display-message','-p','-t',session,'#{pane_dead}'],text=True).strip()
                    if dead=='1': break
                    time.sleep(.1)
                self.assertEqual(dead,'1'); self.assertEqual(len(log.read_text().splitlines()),24)
                pane=subprocess.check_output(prefix+['capture-pane','-p','-t',session],text=True)
                self.assertIn('CPU monitoring smoke',pane)
            finally: subprocess.run(prefix+['kill-server'],capture_output=True)


class ReadinessEvidenceTests(unittest.TestCase):
    def test_missing_or_old_gpu_evidence_cannot_publish_ready(self):
        from tools.lidar_readiness.monitored_readiness import publish
        with tempfile.TemporaryDirectory() as root,redirect_stdout(io.StringIO()):
            root=Path(root); (root/'resume-align').mkdir()
            atomic_json(root/'resume-align/result.json',{'passed':True,'checks':{'model':True},'metadata':{'code':{'old':'hash'}}})
            report=publish(root)
            self.assertFalse(report['technical_checks_passed']); self.assertFalse(report['monitored_training_ready'])
            self.assertEqual(report['checks']['resume-align/result.json'],'failed')
            self.assertEqual(report['checks']['baseline/result.json'],'pending')


if __name__=='__main__': unittest.main()
