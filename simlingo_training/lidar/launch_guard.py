"""Process lock shared by direct CLI and tmux launches; no GPU work."""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path

LOCK=Path('/data/ghazaleh/.cache/simlingo-lidar/gpu2-training.lock')


@contextmanager
def training_lock(path=LOCK):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError('Another SimLingo training process holds GPU 2 lock') from exc
        handle.seek(0); handle.truncate(); handle.write(str(os.getpid())+'\n'); handle.flush()
        try: yield
        finally: fcntl.flock(handle,fcntl.LOCK_UN)
