"""Atomic JSON persistence and local advisory file locking for manual source builds.
No installer, release metadata, update engine or uninstall procedures.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json, os, tempfile
from core import file_lock

def atomic_json(path: Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.chmod(temp,0o600)
        os.replace(temp,path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

@contextmanager
def locked(path: Path, *, blocking=False):
    path=Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink(): raise ValueError('Refusing symlink lock file')
    fd=os.open(path,os.O_CREAT|os.O_RDWR|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'r+b') as f:
        if os.fstat(fd.fileno()).st_size==0: f.write(b'\0'); f.flush()
        file_lock.flock(f,file_lock.LOCK_EX | (0 if blocking else file_lock.LOCK_NB))
        try: yield
        finally:file_lock.flock(f,file_lock.LOCK_UN)
