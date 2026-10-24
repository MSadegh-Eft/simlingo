"""Read-only physical-GPU-2/host sampling until a finite profile process exits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import psutil


def main(pid,output):
    process=psutil.Process(pid); identity=process.create_time(); end=time.monotonic()+2700
    with Path(output).open('x',buffering=1) as handle:
        while time.monotonic()<end:
            try:
                if not process.is_running() or process.create_time()!=identity or process.status()==psutil.STATUS_ZOMBIE: break
                children=process.children(recursive=True)
                row={'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
                     'process_rss_bytes':process.memory_info().rss,
                     'children_rss_bytes':sum(p.memory_info().rss for p in children if p.is_running()),
                     'host_available_bytes':psutil.virtual_memory().available,
                     'gpu':subprocess.check_output(['nvidia-smi','-i','2',
                         '--query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu,power.draw',
                         '--format=csv,noheader'],text=True).strip()}
                handle.write(json.dumps(row)+'\n')
            except (psutil.NoSuchProcess,ProcessLookupError): break
            time.sleep(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--pid',type=int,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); main(args.pid,args.output)
