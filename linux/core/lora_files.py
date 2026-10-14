"""Idempotent publication of existing LoRA artifacts (no tensor loading/training).
Never overwrite a different file, never delete an existing weight file.
"""
from __future__ import annotations
import contextlib, hashlib, os, re, tempfile, threading
from pathlib import Path
from core.file_lock import flock, LOCK_EX, LOCK_UN
from core.safe_paths import is_link

_LOCK = threading.RLock()

@contextlib.contextmanager
def library_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=root/'.lora-library.lock'
    if is_link(lock):raise RuntimeError('Unsafe LoRA library lock')
    with _LOCK, lock.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        flock(f,LOCK_EX)
        try:yield
        finally:flock(f,LOCK_UN)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def publish(src, root, stem):
    """Caller holds library_lock. Only compare this artifact's filename family."""
    src=Path(src);root=Path(root)
    if is_link(src) or not src.is_file():raise RuntimeError('Invalid LoRA source file')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',stem):raise RuntimeError('Invalid LoRA output name')
    expected=digest(src);size=src.stat().st_size
    canonical=root/(stem+'.safetensors')
    pattern=re.compile(re.escape(stem)+r'(?:_\d+|_sha256_[0-9a-f]{64})?\.safetensors$')
    candidates=[canonical]+sorted((p for p in root.iterdir() if p!=canonical and pattern.fullmatch(p.name)),key=lambda p:p.name)
    for p in candidates:
        if not is_link(p) and p.is_file() and p.stat().st_size==size and digest(p)==expected:
            return p,True,expected
    # A collision never overwrites another job's result. Content name makes retries stable.
    dst=canonical if not canonical.exists() and not is_link(canonical) else root/(stem+'_sha256_'+expected+'.safetensors')
    if dst.exists() or is_link(dst):raise RuntimeError('LoRA output collision; no files were overwritten')
    fd,tmp=tempfile.mkstemp(prefix='.publish-',suffix='.part',dir=root)
    try:
        h=hashlib.sha256()
        with os.fdopen(fd,'wb') as out,src.open('rb') as inp:
            for chunk in iter(lambda:inp.read(1024*1024),b''):
                out.write(chunk);h.update(chunk)
            out.flush();os.fsync(out.fileno())
        if h.hexdigest()!=expected:raise RuntimeError('LoRA source changed during publication; retry after training finishes')
        # Exclusive creation prevents races even with a process that ignores our lock.
        # Copy the verified temporary bytes only after claiming the destination.
        outfd=os.open(dst,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        try:
            with os.fdopen(outfd,'wb') as out,open(tmp,'rb') as inp:
                for chunk in iter(lambda:inp.read(1024*1024),b''):out.write(chunk)
                out.flush();os.fsync(out.fileno())
        except BaseException:
            dst.unlink(missing_ok=True);raise
        return dst,False,expected
    finally:
        Path(tmp).unlink(missing_ok=True)
