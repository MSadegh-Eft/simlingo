"""Single-GPU staged training. Launch only after review of its concrete budget."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import torch

from simlingo_training.lidar.data import ManifestDataset
from simlingo_training.lidar.evaluation import evaluate_requests,retention_gate
from simlingo_training.lidar.numerics import make_training_model
from simlingo_training.lidar.runtime import (initialize_device,check_resources,TrainingSession,METADATA_CACHE,GIB)
from simlingo_training.lidar.storage import atomic_json,digest_file
from simlingo_training.lidar.validation_run import evaluate_complete


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
    if config.get('schema')!=1 or config['stage'] not in ('align','joint'): raise ValueError('Invalid stage/config schema')
    if config.get('budget_reviewed') is not True: raise ValueError('The actual training budget has not been reviewed; no launch')
    total=config['total_updates']
    if not isinstance(total,int) or total<40: raise ValueError('Invalid OneCycle budget')
    for key in ('full_validation_updates','monitor_updates'):
        values=config[key]
        if values!=sorted(set(values)) or any(not isinstance(i,int) or not 1<=i<=total for i in values): raise ValueError('Invalid validation schedule')
    if total not in config['full_validation_updates']: raise ValueError('Final full validation is required')
    if set(config['full_validation_updates'])&set(config['monitor_updates']): raise ValueError('Redundant validation schedule')
    if not 1<=config['checkpoint_interval_updates']<=100: raise ValueError('Invalid checkpoint cadence')
    if config['checkpoint_interval_seconds']!=1800: raise ValueError('Keep the profiled 30-minute recovery cadence')


def main(config_path,resume=None,initial_weights=None):
    config=json.loads(config_path.read_text()); validate_config(config)
    manifest=Path(config['manifest']); output=Path(config['output']); output.mkdir(parents=True,exist_ok=True)
    readiness=json.loads(Path(config['readiness_report']).read_text())
    if readiness.get('technical_checks_passed') is not True: raise ValueError('Pre-training readiness checks are incomplete')
    if readiness['manifest_sha256']!=digest_file(manifest): raise ValueError('Readiness dataset differs')
    if config['stage']=='joint' and not (resume or initial_weights): raise ValueError('Joint stage requires selected alignment weights')
    if resume and initial_weights: raise ValueError('Choose resume or a new joint stage')
    baseline=json.loads(Path(config['baseline_result']).read_text())
    if not baseline['complete_protocol'] or not baseline['coverage_gate']['passed']: raise ValueError('Full baseline/coverage missing')
    config['readiness_sha256']=digest_file(config['readiness_report'])
    config['baseline_sha256']=digest_file(config['baseline_result'])
    config['config_file_sha256']=digest_file(config_path)
    # Last, recovery, best and an atomic write must all fit beyond the reserve.
    if shutil.disk_usage(output).free-4*8*GIB<150*GIB: raise OSError('Insufficient checkpoint retention space')
    session=create_session(manifest,output,config['stage'],config['total_updates'],config,initial_weights)
    for key in ('manifest_sha256','validation_sha256','code','external_sources','versions'):
        if baseline['binding'][key]!=session.metadata[key]: raise ValueError('Released baseline provenance differs: '+key)
    if resume: session.resume(resume)
    elif (output/'last.pt').exists() or (output/'progress.json').exists():
        raise ValueError('Existing run output requires explicit resume')
    protocol=json.loads((manifest.parent/'validation.json').read_text())
    val=ManifestDataset(manifest,session.model.tokenizer,'val',metadata_cache=METADATA_CACHE)
    last_saved_update=session.updates; last_save=time.monotonic(); boundary_path=Path(resume) if resume else None
    selection_path=output/'selection.json'
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else {'best_score':None,'best_update':None,'checks':{}}
    if any(int(update)>session.updates for update in selection['checks']):
        raise ValueError('Resume checkpoint predates completed selection; choose the latest boundary')
    def check(): check_resources(output)
    def save_boundary():
        nonlocal last_saved_update,last_save,boundary_path
        check()
        if (output/'last.pt').exists(): os.replace(output/'last.pt',output/'recovery.pt')
        session.save(output/'last.pt'); last_saved_update=session.updates; last_save=time.monotonic(); boundary_path=output/'last.pt'
    def validate_boundary():
        update=session.updates
        if update in config['full_validation_updates']:
            if last_saved_update!=update or boundary_path is None or not boundary_path.exists(): save_boundary()
            checkpoint_sha=digest_file(boundary_path)
            evaluations={mode:evaluate_complete(session.model,val,protocol['requests'],mode,output/f'validation-{update}-{mode}',
                {'checkpoint_sha256':checkpoint_sha,'metadata':session.metadata,'mode':mode},check=check) for mode in ('paired','rgb')}
            gate=retention_gate(evaluations['rgb']['metrics'],baseline['metrics'])
            gate['passed']=gate['passed'] and evaluations['rgb']['coverage_gate']['passed']
            paired=evaluations['paired']['metrics']
            for metric in ('route_ade','speed_ade'):
                if paired[metric]['population_sha256']!=baseline['metrics'][metric]['population_sha256']:
                    raise ValueError('Paired driving selection population differs from the released baseline')
            score=sum(paired[k]['value']/max(1e-12,baseline['metrics'][k]['value']) for k in ('route_ade','speed_ade'))/2
            selection['checks'][str(update)]={'retention':gate,'normalized_driving_error':score}
            if gate['passed'] and (selection['best_score'] is None or score<selection['best_score']):
                session.save(output/'best.pt'); selection.update(best_score=score,best_update=update,best_sha256=digest_file(output/'best.pt'))
            atomic_json(selection_path,selection)
        elif update in config['monitor_updates']:
            for mode in ('paired','rgb'):
                result=evaluate_requests(session.model,val,protocol['monitor'],mode,check=check)
                result['complete_protocol']=False
                atomic_json(output/f'monitor-{update}-{mode}.json',result)
    # A crash during validation resumes this exact completed boundary and its
    # manifest/checkpoint-bound journals before another optimizer update.
    if resume: validate_boundary()
    else: save_boundary()  # Recovery must also work before the first scheduled save.
    while session.updates<session.total_updates:
        started=time.monotonic(); stats=session.update(check=check)
        atomic_json(output/'progress.json',{'status':'running','updates':session.updates,'presentations':session.sampler.position,
            'total_updates':session.total_updates,'last_update_seconds':time.monotonic()-started,'stats':stats})
        print(json.dumps({'update':session.updates,**stats}),flush=True)
        if (session.updates-last_saved_update>=config['checkpoint_interval_updates'] or time.monotonic()-last_save>=1800
                or session.updates in config['full_validation_updates']): save_boundary()
        validate_boundary()
    atomic_json(output/'complete.json',{'status':'complete','updates':session.updates,'selected':selection,
        'retention_passing_candidate':selection['best_update'] is not None,'next':'Review report before any next stage'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume',type=Path); p.add_argument('--initial-weights',type=Path)
    args=p.parse_args(); main(args.config,args.resume,args.initial_weights)
