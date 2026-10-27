"""Publish current readiness; missing new GPU evidence must remain pending."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path

from simlingo_training.lidar.evaluation import group_metrics
from simlingo_training.lidar.monitor_protocol import baseline_records
from simlingo_training.lidar.storage import atomic_json,digest_file
from tools.lidar_readiness.verify_cpu import REPO,source_hashes


def publish(root):
    root=Path(root); current=source_hashes()
    core={k:v for k,v in current.items() if k.startswith('simlingo_training/')}
    checks={}; errors={}; artifacts={}; loaded={}
    def artifact(name,validate):
        path=root/name
        if not path.exists(): checks[name]='pending'; return None
        try:
            data=json.loads(path.read_text()); validate(data)
            checks[name]='passed'; artifacts[name]=digest_file(path); loaded[name]=data
            return data
        except (ValueError,KeyError,TypeError,AssertionError,OSError) as exc:
            checks[name]='failed'; errors[name]=str(exc); return None
    def require(value,message):
        if not value: raise ValueError(message)
    monitor=artifact('monitor-protocol.json',lambda d:require(d['paired_requests']==512 and d['rgb_requests']==448
        and all(d['coverage'][t]['scored_targets']>=100 and d['coverage'][t]['groups']>=10 for t in ('qa','dreamer')),
        'Monitoring populations or coverage differ'))
    integrity=artifact('integrity.json',lambda d:require(d['passed'],'Prepared data integrity failed'))
    artifact('assets.json',lambda d:require(d['passed'] and d['verification_source_sha256']==digest_file(REPO/'tools/lidar_readiness/verify_assets.py'),'Pretrained asset evidence differs'))
    artifact('sampling-audit.json',lambda d:require(d['passed'] and d['monitor_protocol_sha256']==digest_file(root/'monitor-protocol.json'), 'Sampler coverage evidence differs'))
    artifact('cpu-tests.json',lambda d:require(d['passed'] and d['tests']>=51 and d['tmux_cpu_smoke_passed']
        and d['source_hashes']==current and d['log_sha256']==digest_file(root/'unit-tests.log'),'CPU evidence is incomplete or stale'))
    baseline=artifact('baseline/result.json',lambda d:require(d['complete_protocol'] and d['coverage_gate']['passed']
        and d['samples']==7680 and d['optimizer_updates']==0 and d['binding']['code']==core
        and d['journal_sha256']==digest_file(root/'baseline/records.jsonl'),'Fresh full RGB baseline is incomplete or stale'))
    for name in ('resume-align/result.json','resume-joint/result.json','monitor-align/result.json','monitor-joint/result.json'):
        artifact(name,lambda d:require(d['passed'] and all(d['checks'].values()) and d['metadata']['code']==core,
            'New GPU recovery/monitoring evidence is incomplete or stale'))
    artifact('resume-joint/global-stress.json',lambda d:require(d['passed']
        and loaded.get('resume-joint/result.json',{}).get('stress_evidence_sha256')==digest_file(root/'resume-joint/global-stress.json'),
        'Combined resource stress is missing its current recovery binding'))
    reference_verified=False
    if baseline and monitor:
        try:
            _,rows=baseline_records(root/'baseline/records.jsonl')
            keys={(r['uid'],r['task'],r['seed']) for t,items in monitor['requests'].items() if t!='geometry_qa' for r in items}
            subset=[r for r in rows if (r['uid'],r['task'],r['seed']) in keys]
            reference_verified=(len(subset)==len(keys) and group_metrics(subset)==monitor['baseline_metrics'])
            require(reference_verified,'Fresh baseline differs from the fixed monitoring reference; prepare a new protocol version')
            checks['monitor_reference']='passed'
        except (ValueError,KeyError) as exc: checks['monitor_reference']='failed'; errors['monitor_reference']=str(exc)
    else: checks['monitor_reference']='pending'
    if integrity and monitor:
        require(monitor['manifest_sha256']==integrity['manifest_sha256'],'Monitoring/integrity datasets differ')
        for name,data in loaded.items():
            binding=data.get('metadata',data.get('binding',{}))
            if 'manifest_sha256' in binding: require(binding['manifest_sha256']==monitor['manifest_sha256'],'Evidence dataset differs: '+name)
        if baseline:
            for name in ('resume-align/result.json','resume-joint/result.json','monitor-align/result.json','monitor-joint/result.json'):
                if name in loaded:
                    for k in ('versions','external_sources','validation_sha256'):
                        require(loaded[name]['metadata'][k]==baseline['binding'][k],'Evidence provenance differs: '+name+'/'+k)
                    script='fresh_resume.py' if name.startswith('resume-') else 'probe_monitoring.py'
                    require(loaded[name]['metadata']['experiment_config']['run']['entrypoint_sha256']==digest_file(REPO/'tools/lidar_readiness'/script),
                            'Evidence entry point differs: '+name)
            for name in ('monitor-align/result.json','monitor-joint/result.json'):
                if name in loaded: require(loaded[name]['monitor_protocol_sha256']==digest_file(root/'monitor-protocol.json'),'GPU monitor protocol differs')
    ready=all(v=='passed' for v in checks.values())
    historical=json.loads((REPO/'tools/lidar_readiness/readiness-results.json').read_text())
    estimates={}
    for stage,total,interval in [('align',750,15),('joint',1161,12)]:
        proof=loaded.get('monitor-'+stage+'/result.json')
        if proof is None:
            estimates[stage]={'status':'pending_new_monitor_measurement','prior_training_only_hours':historical['budgets']['alignment' if stage=='align' else 'joint_one_sampled_epoch']['train_hours'],
                'limits':'Historical training profile; previous end-to-end estimates do not include the new monitoring policy'}
            continue
        monitor_seconds=sum(proof['monitor_seconds'].values())
        nmonitor=1+math.ceil(total/125)
        train=historical['budgets']['alignment' if stage=='align' else 'joint_one_sampled_epoch']['train_hours']
        recovery=loaded.get('resume-'+stage+'/result.json')
        save=recovery['save_seconds'] if recovery else historical['recovery'][stage]['save_seconds']
        copies=proof['candidate_copy_seconds']; saves=1+math.ceil(total/interval)+math.ceil(total/125)
        full=(baseline['sum_example_seconds'] if baseline else historical['baseline_full_seconds'])+historical['paired_full_forecast_seconds']
        estimate=train+(nmonitor*monitor_seconds+saves*save+(nmonitor+1)*copies+full)/3600
        estimates[stage]={'central_hours':estimate,'planning_low_hours':estimate*.8,'planning_high_hours':estimate*1.75,
            'fallback_additional_hours':full/3600,'monitors_including_zero':nmonitor,'measured_monitor_seconds':monitor_seconds,
            'training_rate_source':'Historical complete-mixture profile; verify during the first approved run',
            'limits':'No early-stop saving assumed; one full selected validation included. Paired full timing remains extrapolated.'}
    report={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),'status':'ready_for_permission_review' if ready else 'verification_pending',
            'technical_checks_passed':ready,'monitored_training_ready':ready,'checks':checks,'errors':errors,
            'manifest_sha256':monitor['manifest_sha256'] if monitor else None,
            'monitor_protocol_sha256':digest_file(root/'monitor-protocol.json') if monitor else None,
            'monitor_reference_verified':reference_verified,'baseline_sha256':digest_file(root/'baseline/result.json') if baseline else None,
            'source_hashes':current,'artifact_hashes':artifacts,'estimates':estimates,
            'main_training_started':False,'budget_reviewed':False,'supervisor_permission_confirmed':False,
            'launch':'Disabled until new readiness, budget review and explicit supervisor permission are all satisfied'}
    atomic_json(root/'result.json',report)
    print(json.dumps({k:report[k] for k in ('status','checks','errors','estimates')},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    publish(p.parse_args().output)
