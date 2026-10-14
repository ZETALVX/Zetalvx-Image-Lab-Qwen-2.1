"""Pure helpers for the persistent download manager."""
from __future__ import annotations
import threading, time
ACTIVE={'queued','downloading','validating','installing','cancelling','running','needs_confirmation'}


class DownloadCoordinator:
    """Small process-local gate shared by every model downloader.

    Default is one transfer at a time. The GUI may raise the limit to 2 or 3;
    changing the limit never interrupts an active transfer.
    """
    def __init__(self):
        self._cond=threading.Condition();self._limit=1;self._active={}
    def set_limit(self,value):
        value=int(value)
        if value not in (1,2,3):raise ValueError('Parallel downloads must be 1, 2 or 3')
        with self._cond:self._limit=value;self._cond.notify_all()
        return value
    def limit(self):
        with self._cond:return self._limit
    def acquire(self,key,cancel_event=None):
        with self._cond:
            while len(self._active)>=self._limit:
                if cancel_event is not None and cancel_event.is_set():return False
                self._cond.wait(.35)
            if cancel_event is not None and cancel_event.is_set():return False
            self._active[str(key)]=time.time();return True
    def release(self,key):
        with self._cond:self._active.pop(str(key),None);self._cond.notify_all()
    def snapshot(self):
        with self._cond:return {'limit':self._limit,'active':len(self._active),'active_jobs':sorted(self._active)}

COORDINATOR=DownloadCoordinator()

def normalize_job(source,j):
    created=float(j.get('created_at') or 0);started=float(j.get('started_at') or created or 0)
    done=int(j.get('downloaded_bytes') or 0);total=int(j.get('total_bytes') or 0)
    elapsed=max(0.0,time.time()-started) if started and j.get('status') in ACTIVE|{'complete'} else 0.0
    speed=(done/elapsed) if elapsed>1 and done>0 else 0.0
    eta=((total-done)/speed) if speed>0 and total>done else 0.0
    if source=='vision':
        kind=str(j.get('kind') or 'vision');role=str(j.get('role') or '')
        label='Vision · '+('GGUF' if kind=='gguf' else 'Transformers')
        name=str(j.get('current_file') or j.get('path') or j.get('source') or 'Vision model')
    else:
        kind=str(j.get('kind') or j.get('kind_hint') or 'model')
        label='Models · '+({'checkpoint':'Checkpoint','lora':'LoRA','vae':'VAE'}.get(kind,kind.title() if kind else 'Model'))
        name=str(j.get('filename') or j.get('current_file') or 'Model download');role=''
    return {'id':str(j.get('id') or ''),'source':source,'label':label,'name':name,'kind':kind,'role':role,
        'status':str(j.get('status') or ''),'created_at':created,'updated_at':float(j.get('updated_at') or created),
        'started_at':started,'downloaded_bytes':done,'total_bytes':total,'speed_bytes_sec':speed,'eta_seconds':eta,
        'file_count':int(j.get('file_count') or 1),'completed_files':int(j.get('completed_files') or 0),
        'current_file':str(j.get('current_file') or ''),'path':str(j.get('path') or ''),'error':str(j.get('error') or ''),
        'warning':str(j.get('warning') or ''),'safe_source':str(j.get('source') or ''),'sha256':str(j.get('sha256') or ''),
        'history':list(j.get('history') or [])[-80:],'queue_position':int(j.get('queue_position') or 0)}
