"""Serialize model work across the three native worker processes on Linux."""
from contextlib import contextmanager
import os, time
from core import file_lock as fcntl
from core.runtime_env import SHARED_ROOT, worker_headers

@contextmanager
def gpu_session(owner):
    import requests
    run=SHARED_ROOT/'run';run.mkdir(parents=True,exist_ok=True)
    with (run/'gpu.lock').open('a+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:
            # Idle workers may keep cached weights. Release them before handing over GPU.
            for other in ('image',):
                if other==owner:continue
                url=os.environ.get(f'CREATOR_{other.upper()}_WORKER_URL')
                if url:
                    try:requests.post(url+'/unload',headers=worker_headers(),timeout=10)
                    except requests.RequestException:pass
            yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
