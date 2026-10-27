"""TTY progress plus durable, readable logs and atomic machine status."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback
import uuid

import torch
from tqdm import tqdm

from simlingo_training.lidar.storage import atomic_json


class Tee:
    def __init__(self, console, log): self.console,self.log=console,log
    def write(self, value):
        self.console.write(value); self.log.write(value); self.log.flush()
        return len(value)
    def flush(self): self.console.flush(); self.log.flush()
    def isatty(self): return self.console.isatty()
    @property
    def encoding(self): return self.console.encoding


class Reporter:
    def __init__(self, output, stage, total):
        self.output=Path(output); self.output.mkdir(parents=True,exist_ok=True)
        self.stage,self.total=stage,total
        self.attempt_id=uuid.uuid4().hex
        self.started=time.monotonic(); self.last_status=0
        self.state={'stage':stage,'phase':'loading','updates':0,'presentations':0,'total_updates':total,'attempt_id':self.attempt_id}
        self.console=sys.stderr
        self.bar=None; self.micro=None
        self.log=(self.output/'training.log').open('a',buffering=1)
        self.events=(self.output/'events.jsonl').open('a',buffering=1)

    @contextmanager
    def capture(self):
        with redirect_stdout(Tee(sys.stdout,self.log)),redirect_stderr(Tee(sys.stderr,self.log)):
            yield self

    def event(self, name, **values):
        row={'time_utc':datetime.now(timezone.utc).isoformat(),'attempt_id':self.attempt_id,'event':name,**values}
        self.events.write(json.dumps(row,allow_nan=False)+'\n'); self.events.flush()
        message=json.dumps(row,allow_nan=False)
        self.log.write(message+'\n'); self.log.flush()
        tqdm.write(message,file=self.console)

    def status(self, force=False, **values):
        self.state.update(values)
        now=time.monotonic()
        if force or now-self.last_status>=10:
            self.state.update(heartbeat_utc=datetime.now(timezone.utc).isoformat(),process_elapsed_seconds=now-self.started)
            if torch.cuda.is_initialized():
                self.state.update(gpu_allocated_GiB=torch.cuda.memory_allocated()/1024**3,
                                  gpu_reserved_GiB=torch.cuda.memory_reserved()/1024**3)
            atomic_json(self.output/'status.json',self.state); self.last_status=now

    def phase(self,name,**values):
        self.status(force=True,phase=name,**values)
        self.event('phase',phase=name,**values)

    def start_training(self,session):
        if self.bar: self.bar.close()
        self.bar=tqdm(total=self.total,initial=session.updates,desc=self.stage+' updates',
                      unit='update',dynamic_ncols=True,mininterval=1,file=self.console,
                      disable=not self.console.isatty())
        self.synchronize(session)

    def synchronize(self,session,stats=None):
        tracking=session.run_state
        elapsed=tracking['training_seconds']; done=session.sampler.position
        rate=done/elapsed if elapsed>0 else None
        lrs={g['name']:g['lr'] for g in session.optimizer.param_groups}
        values={'updates':session.updates,'presentations':done,
                'sampled_epochs':done/len(session.dataset),'unique_frames':len(tracking['seen_indices']),
                'eligible_frames':len(session.dataset),'task_counts':tracking['task_counts'],
                'replay_count':tracking['replay_count'],'bucket_counts':tracking['bucket_counts'],
                'learning_rates':lrs,'measured_training_presentations_per_second':rate,
                'estimated_training_seconds_remaining':(self.total-session.updates)*64/rate if rate else None,
                'eta_scope':'Training only; validation/checkpoint time is additional',
                'next_monitor_update':min(self.total,(session.updates//125+1)*125)}
        if stats is not None: values['stats']=stats
        self.status(force=True,**values)
        if self.bar:
            self.bar.n=session.updates
            self.bar.set_postfix(epoch=f"{values['sampled_epochs']:.3f}",unique=values['unique_frames'],
                loss=f"{stats['loss']:.4f}" if stats else '?',refresh=False)
            self.bar.refresh()

    @contextmanager
    def accumulation(self):
        self.micro=tqdm(total=64,desc='accumulation',unit='sample',position=1,leave=False,
                        dynamic_ncols=True,mininterval=1,file=self.console,disable=not self.console.isatty())
        completed=0
        def advance(n):
            nonlocal completed
            completed+=n
            self.micro.update(n)
            self.status(accumulation_completed=completed,accumulation_total=64)
        try: yield advance
        finally: self.micro.close(); self.micro=None

    @contextmanager
    def evaluation(self,name):
        bar=tqdm(desc=name,unit='request',position=1,leave=False,dynamic_ncols=True,
                 mininterval=1,file=self.console,disable=not self.console.isatty())
        def advance(completed,total):
            bar.total=total; bar.n=completed; bar.refresh()
            self.status(evaluation=name,evaluation_completed=completed,evaluation_total=total)
        try: yield advance
        finally: bar.close()

    def failed(self,exc):
        self.event('failure',error=str(exc),traceback=traceback.format_exc())
        self.status(force=True,phase='failed',error=str(exc),resume_requires_explicit_command=True)

    def close(self):
        if self.bar: self.bar.close()
        self.events.close(); self.log.close()
