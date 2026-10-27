"""Record CPU-only contract tests against an exact source snapshot."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import subprocess
import sys

from simlingo_training.lidar.storage import atomic_json,digest_file

REPO=Path(__file__).resolve().parents[2]


def source_hashes():
    paths=set((REPO/'simlingo_training').rglob('*.py'))
    paths.update((REPO/'tests').glob('test_lidar*.py'))
    paths.update((REPO/'tools/lidar_readiness').glob('*.py'))
    paths.update((REPO/'tools/lidar_training').glob('*.*'))
    return {str(p.relative_to(REPO)):digest_file(p) for p in sorted(paths) if p.suffix in ('.py','.sh')}


def main(output):
    output.mkdir(parents=True,exist_ok=True); before=source_hashes()
    log=output/'unit-tests.log'
    with log.open('w') as handle:
        result=subprocess.run([sys.executable,'-m','unittest','discover','-v','-s','tests','-p','test_lidar*.py'],
                              cwd=REPO,stdout=handle,stderr=subprocess.STDOUT)
    content=log.read_text(); count=re.search(r'Ran (\d+) tests',content)
    report={'passed':result.returncode==0 and before==source_hashes(),'exit_code':result.returncode,
            'tests':int(count[1]) if count else 0,'source_hashes':before,'log_sha256':digest_file(log),
            'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'tmux_cpu_smoke_passed':'test_tmux_attach_detach_duplicate_rejection_and_retained_cpu_log' in content and result.returncode==0,
            'main_training_started':False}
    atomic_json(output/'cpu-tests.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'}),flush=True)
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
