"""Explicit tmux start/resume/status; never queues a job or grants permission."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

from simlingo_training.lidar.launch_guard import training_lock

REPO=Path(__file__).resolve().parents[2]
SESSION='simlingo-lidar'


def tmux_prefix(socket=None): return ['tmux']+(['-L',socket] if socket else [])


def create_tmux_session(command,cwd,session=SESSION,socket=None):
    prefix=tmux_prefix(socket)
    if subprocess.run(prefix+['has-session','-t','='+session],capture_output=True).returncode==0:
        raise RuntimeError('Session already exists; attach to inspect it: '+session)
    # The pane sets its own retention before exec, even if the job fails quickly.
    script='set -e\n'+shlex.join(prefix+['set-option','-w','-t',session,'remain-on-exit','on'])+'\nexec '+shlex.join(command)
    subprocess.run(prefix+['new-session','-d','-s',session,'-c',str(cwd),
                         shlex.join(['bash','--noprofile','--norc','-c',script])],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='action',required=True)
    for name in ('start','resume','status'):
        q=sub.add_parser(name); q.add_argument('--config',type=Path,required=True)
        if name=='resume': q.add_argument('--checkpoint',type=Path,required=True)
        if name=='start': q.add_argument('--initial-weights',type=Path)
    sub.add_parser('attach')
    a=p.parse_args()
    if a.action=='attach':
        raise SystemExit(subprocess.call(['tmux','attach','-t',SESSION]))
    config=json.loads(a.config.read_text()); output=Path(config['output'])
    if a.action=='status':
        path=output/'status.json'
        print(path.read_text() if path.exists() else json.dumps({'status':'not_started','output':str(output),
            'budget_reviewed':config.get('budget_reviewed',False),
            'supervisor_permission_confirmed':config.get('supervisor_permission_confirmed',False)},indent=2))
        return
    from simlingo_training.lidar.train import validate_config,prelaunch_resources
    validate_config(config)
    ready=json.loads(Path(config['readiness_report']).read_text())
    if ready.get('monitored_training_ready') is not True: raise ValueError('Complete monitored readiness checks first')
    prelaunch_resources(output)
    with training_lock(): pass  # Fail promptly if an existing process owns the GPU lock.
    command=['bash',str(REPO/'tools/lidar_stage1/run.sh'),'-u','-m','simlingo_training.lidar.train',
             '--config',str(a.config.resolve())]
    if a.action=='resume': command+=['--resume',str(a.checkpoint.resolve())]
    elif a.initial_weights: command+=['--initial-weights',str(a.initial_weights.resolve())]
    create_tmux_session(command,REPO)
    print('Training process launched. Attach: tmux attach -t '+SESSION)


if __name__=='__main__':
    try: main()
    except (ValueError,RuntimeError,MemoryError,OSError) as exc: raise SystemExit('Launch blocked: '+str(exc))
