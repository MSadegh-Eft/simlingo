"""Single-GPU staged training. Launch only after review of its concrete budget."""
import argparse
import copy
import hashlib
import subprocess
import psutil
import json
import os
from pathlib import Path
import shutil
import time

import torch

from simlingo_training.lidar.data import ManifestDataset
from simlingo_training.lidar.evaluation import retention_gate
from simlingo_training.lidar.numerics import make_training_model
from simlingo_training.lidar.runtime import (initialize_device,check_resources,TrainingSession,METADATA_CACHE,GIB)
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.validation_run import evaluate_complete
from simlingo_training.lidar.control import Controller,driving_score,monitor_due,full_candidate_order
from simlingo_training.lidar.launch_guard import training_lock
from simlingo_training.lidar.telemetry import Reporter


def verify_dataset(manifest):
    manifest=Path(manifest); ready=json.loads((manifest.parent/'ready.json').read_text())
    if not ready['passed']: raise ValueError('Dataset is not ready')
    for key,name in [('manifest_sha256','manifest.json'),('validation_sha256','validation.json'),('audit_sha256','audit.json')]:
        if digest_file(manifest.parent/name)!=ready[key]: raise ValueError('Dataset artifact changed: '+name)
    return ready


def create_session(manifest,output,stage,total_updates,run_config,initial_weights=None):
    """Production construction path, also exercised by fresh-process checks."""
    verify_dataset(manifest); initialize_device(output)
    model=make_training_model().cuda()
    dataset=ManifestDataset(manifest,model.tokenizer,'train',augment=True,metadata_cache=METADATA_CACHE)
    session=TrainingSession(model,dataset,stage,total_updates,run_config)
    if initial_weights:
        if digest_file(initial_weights)!=run_config.get('initial_weights_sha256'):
            raise ValueError('Selected alignment weights differ from the reviewed joint initialization')
        state=torch.load(initial_weights,map_location='cpu',weights_only=False,mmap=True)
        if stage!='joint' or state['stage']!='align': raise ValueError('Joint initialization requires an alignment checkpoint')
        for key in ('manifest_sha256','validation_sha256','preprocess','utonia_sha256','hf_snapshot_revision','released_config_sha256','released_sha256','code','external_sources','versions'):
            if state['metadata'][key]!=session.metadata[key]: raise ValueError('Alignment initialization provenance differs: '+key)
        if state['metadata']['experiment_config']['numerics']!=model.lidar_numerics_policy:
            raise ValueError('Alignment numerical policy differs')
        model.load_state_dict(state['model'],strict=True)
    from simlingo_training.lidar.sparse_kernels import prime
    from simlingo_training.lidar.data import Request
    # The same full-sweep tensor shape/channel contracts occur in every task.
    # Prime before the initial checkpoint so even a restart at update zero keeps
    # the exact kernels; upper-stage backwards are included for joint training.
    prime(model,dataset[Request(0,'driving',False,42,0)].cloud)
    return session


def validate_config(config):
    if config.get('schema')!=2 or config.get('stage') not in ('align','joint'):
        raise ValueError('Use a monitored training configuration (schema 2)')
    if config.get('budget_reviewed') is not True:
        raise ValueError('The actual training budget has not been reviewed; no launch')
    if config.get('supervisor_permission_confirmed') is not True or not config.get('permission_note','').strip():
        raise ValueError('Explicit confirmation of supervisor permission is required; no launch')
    total=config['total_updates']
    if type(total) is not int or total<40: raise ValueError('Invalid OneCycle budget')
    if config.get('monitor_interval_updates')!=125: raise ValueError('Monitoring interval must be 125 updates')
    if config.get('full_validation_policy')!='candidate_then_final_on_retention_failure':
        raise ValueError('Full selected-candidate validation is required')
    if type(config['checkpoint_interval_updates']) is not int or not 1<=config['checkpoint_interval_updates']<=100:
        raise ValueError('Invalid checkpoint cadence')
    if config['checkpoint_interval_seconds']!=1800: raise ValueError('Keep 30-minute recovery cadence')
    rules=config['stopping']
    if (type(rules.get('patience')) is not int or rules['patience']<3
            or type(rules.get('minimum_updates')) is not int or not 1<=rules['minimum_updates']<=total
            or any(not 0<rules.get(k,0)<1 for k in ('score_relative_delta','loss_relative_delta','maximum_lr_fraction'))):
        raise ValueError('Invalid conservative stopping configuration')
    if not config.get('monitor_protocol_sha256'): raise ValueError('Bind the immutable monitoring protocol')


def prelaunch_resources(output):
    from simlingo_training.lidar.runtime import UUID
    row=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True,timeout=15)
    uuid,free=[s.strip() for s in row.split(',')]
    if uuid.removeprefix('GPU-')!=UUID: raise RuntimeError('Physical GPU UUID differs')
    if int(free)<40*1024: raise MemoryError('GPU 2 needs at least 40 GiB free before launch')
    if psutil.virtual_memory().available<16*GIB: raise MemoryError('Launch requires 16 GiB host RAM available')
    parent=Path(output)
    while not parent.exists(): parent=parent.parent
    if shutil.disk_usage(parent).free<200*GIB: raise OSError('Need 150 GiB reserve plus 50 GiB checkpoint capacity')


def atomic_copy(source,target):
    target=Path(target); temporary=target.with_suffix(target.suffix+'.part')
    with Path(source).open('rb') as src,temporary.open('wb') as dst:
        shutil.copyfileobj(src,dst,8*1024**2); dst.flush(); os.fsync(dst.fileno())
    os.replace(temporary,target)


def state_hash(state):
    return hashlib.sha256(json.dumps(state,sort_keys=True,allow_nan=False).encode()).hexdigest()


def monitor_signals(report,reference):
    if not report['coverage_gate']['passed'] or report['complete_protocol']:
        raise ValueError('Incomplete or wrongly identified monitoring protocol')
    losses=report['diagnostics']['losses']
    language=losses.get('language',{})
    geometry=[losses.get(k,{}) for k in ('presence','distance')]
    if not language.get('count') or any(not x.get('count') for x in geometry):
        raise ValueError('Missing language/geometry monitoring supervision')
    return driving_score(report['metrics'],reference),language['value'],sum(x['value'] for x in geometry)/2


def run_monitored(session,config,output,val,protocol,monitor,baseline,reporter,resume=None):
    """Production loop, also tested with finite CPU fixtures and injected crashes."""
    output=Path(output)
    controller=Controller(config['stopping'],session.run_state.get('controller'))
    previous=controller.state['candidate']
    if previous and not monitor_due(session.updates,session.total_updates) and digest_file(output/'candidate.pt')!=previous['binding']['checkpoint_sha256']:
        raise ValueError('Previously selected monitoring candidate is missing or changed')
    last_saved=session.updates; last_save=time.monotonic()
    boundary=Path(resume) if resume else None
    decisions=output/'decisions'; decisions.mkdir(exist_ok=True)
    if any(int(p.stem)>session.updates for p in decisions.glob('*.json')):
        raise ValueError('Resume predates a committed monitor; choose the latest boundary')
    train=copy.copy(session.dataset); train.augment=False
    reference=monitor['baseline_metrics']
    max_lrs={g['name']:g['max_lr'] for g in session.optimizer.param_groups}
    def check(): check_resources(output)
    def save():
        nonlocal boundary,last_saved,last_save
        reporter.phase('saving',updates=session.updates); check_resources(output,extra_disk_bytes=8*GIB)
        session.run_state['controller']=copy.deepcopy(controller.state)
        if (output/'last.pt').exists(): os.replace(output/'last.pt',output/'recovery.pt')
        session.save(output/'last.pt')
        boundary=output/'last.pt'; last_saved=session.updates; last_save=time.monotonic()
        reporter.event('checkpoint',update=last_saved,path=str(boundary))
    def ensure_boundary():
        if boundary is None or not boundary.exists() or last_saved!=session.updates: save()
    def evaluate(dataset,requests,mode,path,binding,diagnostics=False,full=False):
        reporter.phase('validating',evaluation=str(path),updates=session.updates)
        with reporter.evaluation(path.name) as progress:
            return evaluate_complete(session.model,dataset,requests,mode,path,binding,check=check,
                diagnostics=diagnostics,complete_protocol=full,progress=progress)
    def monitor_boundary():
        ensure_boundary(); update=session.updates
        binding={'checkpoint_sha256':digest_file(boundary),'metadata':session.metadata,
                 'monitor_protocol_sha256':config['monitor_protocol_sha256'],'update':update}
        decision_path=decisions/f'{update:08d}.json'
        if decision_path.exists():
            decision=json.loads(decision_path.read_text())
            if decision['binding']!=binding or decision['prior_state_sha256']!=state_hash(controller.state):
                raise ValueError('Monitor decision/checkpoint/controller provenance differs')
            for path,sha in decision['result_hashes'].items():
                if digest_file(output/path)!=sha: raise ValueError('Completed monitor changed')
            controller.state=decision['controller']
            reporter.event('recovered_monitor_decision',update=update)
        else:
            reports={}
            for mode in ('paired','rgb'):
                reports[mode]=evaluate(val,monitor['requests'],mode,output/f'monitor-{update}'/mode,
                    {**binding,'mode':mode,'diagnostics':True},diagnostics=True)
            reports['train']=evaluate(train,monitor['train_requests'],'paired',output/f'monitor-{update}'/'train',
                {**binding,'mode':'paired','split':'train','diagnostics':True},diagnostics=True)
            score,language,geometry=monitor_signals(reports['paired'],reference)
            if not reports['rgb']['coverage_gate']['passed']: raise ValueError('Monitoring RGB target coverage missing')
            gate=retention_gate(reports['rgb']['metrics'],reference)
            before=state_hash(controller.state)
            fractions=[g['lr']/max_lrs[g['name']] for g in session.optimizer.param_groups]
            row=controller.observe(update,session.total_updates,score,language,geometry,gate,fractions,binding)
            if row['new_candidate']:
                check_resources(output,extra_disk_bytes=boundary.stat().st_size)
                atomic_copy(boundary,output/'candidate.pt')
            decision={'binding':binding,'prior_state_sha256':before,'controller':controller.state,
                      'retention_screen':gate,'result_hashes':{
                          str((output/f'monitor-{update}'/mode/'result.json').relative_to(output)):
                          digest_file(output/f'monitor-{update}'/mode/'result.json') for mode in ('paired','rgb','train')}}
            atomic_json(decision_path,decision)
        candidate=controller.state['candidate']
        if candidate and digest_file(output/'candidate.pt')!=candidate['binding']['checkpoint_sha256']:
            raise ValueError('Monitoring candidate checkpoint differs')
        session.run_state['controller']=copy.deepcopy(controller.state)
        last=controller.state['history'][-1]
        reporter.status(force=True,latest_monitor=last,best_candidate_update=candidate['update'] if candidate else None,
                        stopping_wait=controller.state['wait'],stopping_patience=config['stopping']['patience'])
        reporter.event('monitor_decision',**last)
        return controller.state['stop_reason']
    if boundary is None: save()
    reporter.start_training(session)
    reason=monitor_boundary() if monitor_due(session.updates,session.total_updates) else None
    while reason is None and session.updates<session.total_updates:
        reporter.phase('training',updates=session.updates)
        with reporter.accumulation() as advance: stats=session.update(check=check,progress=advance)
        reporter.synchronize(session,stats)
        reporter.event('optimizer_update',update=session.updates,stats=stats,
                       learning_rates={g['name']:g['lr'] for g in session.optimizer.param_groups})
        if (session.updates-last_saved>=config['checkpoint_interval_updates'] or time.monotonic()-last_save>=1800): save()
        if monitor_due(session.updates,session.total_updates): reason=monitor_boundary()
    ensure_boundary()
    candidate=controller.state['candidate']
    selected=None; full_checks=[]
    for update in full_candidate_order(candidate['update'] if candidate else None,session.updates):
        checkpoint=output/'candidate.pt' if update==candidate['update'] else boundary
        sha=digest_file(checkpoint)
        reporter.phase('loading_validation_checkpoint',candidate_update=update)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        if state['updates']!=update or state['metadata']!=session.metadata:
            raise ValueError('Full validation checkpoint provenance differs')
        session.model.load_state_dict(state['model'],strict=True)
        from simlingo_training.lidar.sparse_kernels import restore
        restore(session.model,state['sparse_kernels']); del state
        reports={mode:evaluate(val,protocol['requests'],mode,output/f'full-{update}'/mode,
                   {'checkpoint_sha256':sha,'metadata':session.metadata,'mode':mode},full=True) for mode in ('paired','rgb')}
        gate=retention_gate(reports['rgb']['metrics'],baseline['metrics'])
        gate['passed']=gate['passed'] and reports['rgb']['coverage_gate']['passed']
        score=driving_score(reports['paired']['metrics'],baseline['metrics'])
        full_checks.append({'update':update,'checkpoint_sha256':sha,'retention':gate,'score':score})
        atomic_json(output/'selection.json',{'checks':full_checks,'selected':None})
        if gate['passed']:
            check_resources(output,extra_disk_bytes=checkpoint.stat().st_size)
            atomic_copy(checkpoint,output/'best.pt')
            selected={'update':update,'sha256':digest_file(output/'best.pt'),'path':str(output/'best.pt'),'score':score}
            break
    selection={'checks':full_checks,'selected':selected}
    atomic_json(output/'selection.json',selection)
    from simlingo_training.lidar.training_report import write_report
    report=write_report(output,config,controller.state,session.run_state,selection,baseline,monitor)
    complete={'status':'awaiting_review','stop_reason':reason,'updates':session.updates,
              'selected':selected,'retention_passing_candidate':selected is not None,
              'next':'Review report; explicit permission is required for any further training'}
    atomic_json(output/'complete.json',complete)
    reporter.phase('stopped',**complete)
    return report


def main(config_path,resume=None,initial_weights=None):
    config_path=Path(config_path); config=json.loads(config_path.read_text()); validate_config(config)
    with training_lock():
        manifest=Path(config['manifest']); output=Path(config['output'])
        readiness=json.loads(Path(config['readiness_report']).read_text())
        if readiness.get('technical_checks_passed') is not True or readiness.get('monitored_training_ready') is not True:
            raise ValueError('Monitored training readiness checks are incomplete')
        from tools.lidar_readiness.verify_cpu import source_hashes
        if readiness.get('source_hashes')!=source_hashes(): raise ValueError('Readiness source snapshot is stale')
        for name,sha in readiness['artifact_hashes'].items():
            if digest_file(Path(config['readiness_report']).parent/name)!=sha:
                raise ValueError('Readiness artifact changed: '+name)
        if readiness['manifest_sha256']!=digest_file(manifest): raise ValueError('Readiness dataset differs')
        baseline=json.loads(Path(config['baseline_result']).read_text())
        if not baseline['complete_protocol'] or not baseline['coverage_gate']['passed']: raise ValueError('Full baseline/coverage missing')
        for name,sha in baseline['binding']['code'].items():
            if digest_file(Path(__file__).resolve().parents[2]/name)!=sha: raise ValueError('Refresh baseline against changed source: '+name)
        if resume and initial_weights: raise ValueError('Choose resume or a new stage')
        if config['stage']=='joint' and not (resume or initial_weights): raise ValueError('Joint stage requires selected alignment weights')
        if digest_file(config['monitor_protocol'])!=config['monitor_protocol_sha256']: raise ValueError('Monitoring protocol changed')
        monitor=json.loads(Path(config['monitor_protocol']).read_text())
        if monitor['manifest_sha256']!=digest_file(manifest): raise ValueError('Monitoring dataset differs')
        if (readiness.get('monitor_protocol_sha256')!=config['monitor_protocol_sha256']
                or readiness.get('monitor_reference_verified') is not True
                or readiness.get('baseline_sha256')!=digest_file(config['baseline_result'])):
            raise ValueError('Fresh baseline has not verified the monitoring reference')
        if output.exists() and not resume and any(output.iterdir()): raise ValueError('Existing output requires explicit resume')
        if resume and (output/'complete.json').exists(): raise ValueError('Completed run needs a reviewed continuation configuration')
        prelaunch_resources(output)
        for name,key in [('readiness_report','readiness_sha256'),('baseline_result','baseline_sha256')]: config[key]=digest_file(config[name])
        config['config_file_sha256']=digest_file(config_path)
        output.mkdir(parents=True,exist_ok=True)
        prior_status=json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
        reporter=Reporter(output,config['stage'],config['total_updates'])
        try:
            with reporter.capture():
                atomic_json(output/'resolved-config.json',config)
                reporter.phase('loading',resume=str(resume) if resume else None)
                session=create_session(manifest,output,config['stage'],config['total_updates'],config,initial_weights)
                for key in ('manifest_sha256','validation_sha256','code','external_sources','versions'):
                    if baseline['binding'][key]!=session.metadata[key]: raise ValueError('Released baseline provenance differs: '+key)
                if resume:
                    session.resume(Path(resume))
                    reporter.event('resume_boundary',update=session.updates,
                        prior_reported_update=prior_status.get('updates'),
                        rolled_back_updates=max(0,prior_status.get('updates',session.updates)-session.updates),
                        accounting='Checkpoint counters are authoritative; older uncommitted log rows remain for audit')
                protocol=json.loads((manifest.parent/'validation.json').read_text())
                val=ManifestDataset(manifest,session.model.tokenizer,'val',metadata_cache=METADATA_CACHE)
                return run_monitored(session,config,output,val,protocol,monitor,baseline,reporter,resume)
        except BaseException as exc:
            reporter.failed(exc); raise
        finally: reporter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume',type=Path); p.add_argument('--initial-weights',type=Path)
    args=p.parse_args(); main(args.config,args.resume,args.initial_weights)
