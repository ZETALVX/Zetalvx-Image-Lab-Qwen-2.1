"""Recoverable decoder-input archive. Safetensors + JSON, never pickle.

Stores the EXACT VAE input at the pipeline's final decode boundary: unpacking and
normalisation remain those of the installed upstream pipeline. No external
latent upload/deserialisation endpoint is provided. Cache root is app-private.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib,json,os,re,shutil,time,uuid
from pathlib import Path
from core import file_lock
from core.source_util import atomic_json
from core.qwen_assets import read_object
from core.qwen_components import contract_hash,latent_contract
# Latents are durable experiment data, not a small eviction cache. 1.0.51
# removes the old 64-entry / 1 GiB application quota. The only remaining
# limits are per-record structural safety and actual free disk space.
MAX_TENSOR=128*1024**2
DISK_HEADROOM=64*1024**2


def identifier(value):
    if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{32}',value):raise ValueError('Invalid latent/project identifier.')
    return value


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fp:
        for chunk in iter(lambda:fp.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()

class LatentStore:
    def __init__(self,root):self.root=Path(root).absolute()
    def _safe(self,path):
        p=Path(path)
        if p!=self.root and self.root not in p.parents:raise ValueError('Latent path escapes cache root.')
        for x in (p,*p.parents):
            if x.is_symlink():raise ValueError('Linked latent/cache paths are refused.')
        return p
    @contextmanager
    def locked(self):
        self._safe(self.root);self.root.mkdir(parents=True,exist_ok=True)
        f=self._safe(self.root/'cache.lock').open('a+')
        try:file_lock.flock(f.fileno(),file_lock.LOCK_EX);yield
        finally:file_lock.flock(f.fileno(),file_lock.LOCK_UN);f.close()
    def _folder(self,rid):return self._safe(self.root/identifier(rid))
    def exists(self,rid):return (self._folder(rid)/'metadata.json').is_file()
    def _list(self,project=None):
        if project is not None:identifier(project)
        if not self.root.exists():return []
        out=[]
        for p in sorted(self.root.iterdir()):
            if not re.fullmatch('[a-f0-9]{32}',p.name) or not p.is_dir() or p.is_symlink():continue
            try:
                d=self.metadata(p.name)
                if project is None or d['project_id']==project:out.append(d)
            except (ValueError,OSError,KeyError):continue
        return sorted(out,key=lambda d:d['created_at'],reverse=True)
    def list(self,project=None):
        with self.locked():return self._list(project)
    def quota(self):
        values=self._list();disk_path=self.root if self.root.exists() else self.root.parent
        disk_path.mkdir(parents=True,exist_ok=True);disk=shutil.disk_usage(disk_path)
        return {'count':len(values),'bytes':sum(v['bytes'] for v in values),
            'max_count':None,'max_bytes':None,'unlimited':True,
            'disk_free_bytes':disk.free,'disk_total_bytes':disk.total,'disk_headroom_bytes':DISK_HEADROOM}
    def preflight(self):
        with self.locked():
            # No application quota: preserve as many benchmark/Decode Lab
            # latents as the user's disk can hold. Refuse only when the next
            # record could not be written safely.
            if shutil.disk_usage(self.root).free<MAX_TENSOR+DISK_HEADROOM:
                raise ValueError('Insufficient disk space to preserve the latent before decoding.')
    def metadata(self,rid,project=None):
        folder=self._folder(rid);p=self._safe(folder/'metadata.json');d=read_object(p)
        if d.get('schema')!=1 or d.get('representation')!='qwen21.vae-input.v1' or d.get('id')!=rid:raise ValueError('Unsupported or damaged latent record.')
        identifier(d.get('project_id'))
        if project is not None and d['project_id']!=project:raise ValueError('Latent does not belong to this project.')
        tensor=self._safe(folder/'latent.safetensors')
        if not tensor.is_file() or not 0<tensor.stat().st_size<=MAX_TENSOR:raise ValueError('Latent tensor is missing or oversized.')
        if not re.fullmatch('[a-f0-9]{64}',str(d.get('sha256',''))):raise ValueError('Latent checksum is missing.')
        if contract_hash(latent_contract(d.get('contract',{})))!=d.get('contract_hash'):raise ValueError('Latent contract integrity mismatch.')
        return d
    def save(self,tensor,metadata):
        from safetensors.torch import save_file
        import torch
        rid=identifier(metadata['id']);identifier(metadata['project_id'])
        if tensor.ndim!=5 or tensor.shape[0]!=1 or tensor.shape[1]!=64 or tensor.shape[2]!=1:raise ValueError('Unexpected Qwen decoder latent dimensions.')
        if tensor.numel()*tensor.element_size()>MAX_TENSOR-8192:raise ValueError('Latent exceeds the cache tensor limit.')
        tensor=tensor.detach().to('cpu').contiguous()
        if not bool(torch.isfinite(tensor).all()):raise ValueError('Non-finite latent: refused before VAE decode.')
        with self.locked():
            needed=tensor.numel()*tensor.element_size()+8192
            if shutil.disk_usage(self.root).free<needed+DISK_HEADROOM:
                raise ValueError('Insufficient disk space to preserve this latent.')
            dest=self._folder(rid)
            if dest.exists():raise ValueError('Latent already exists; re-decode it rather than overwrite it.')
            stage=self._safe(self.root/('.writing-'+uuid.uuid4().hex));stage.mkdir(mode=0o700)
            try:
                f=stage/'latent.safetensors';save_file({'latents':tensor},str(f),metadata={'representation':'qwen21.vae-input.v1'})
                os.chmod(f,0o600)
                contract=latent_contract(metadata['contract'])
                record={**metadata,'schema':1,'representation':'qwen21.vae-input.v1','created_at':time.time(),
                    'contract':contract,'contract_hash':contract_hash(contract),'shape':list(tensor.shape),
                    'dtype':str(tensor.dtype),'bytes':f.stat().st_size,'sha256':sha_file(f),'decode_results':[]}
                # Reject unexpected objects, NaNs, excessive JSON before atomic commit.
                if len(json.dumps(record,allow_nan=False).encode())>1024**2:raise ValueError('Latent metadata too large.')
                atomic_json(stage/'metadata.json',record);os.rename(stage,dest)
                return record
            finally:
                if stage.is_dir() and not stage.is_symlink():shutil.rmtree(stage)
    def load(self,rid,project=None):
        from safetensors.torch import load_file
        with self.locked():
            d=self.metadata(rid,project);path=self._folder(rid)/'latent.safetensors'
            if path.stat().st_size!=d['bytes'] or sha_file(path)!=d['sha256']:raise ValueError('Latent checksum mismatch; no decode performed.')
            tensors=load_file(str(path),device='cpu')
            if set(tensors)!={'latents'} or list(tensors['latents'].shape)!=d['shape']:raise ValueError('Latent shape/key mismatch.')
            return tensors['latents'],d
    def result(self,rid,info):
        with self.locked():
            d=self.metadata(rid);d['decode_results']=(d.get('decode_results',[])+[{**info,'created_at':time.time()}])[-30:]
            atomic_json(self._folder(rid)/'metadata.json',d)
    def delete(self,rid,project,*,before_delete=None):
        with self.locked():
            if before_delete:before_delete()
            self.metadata(rid,project);folder=self._folder(rid)
            # Only these two private data files: never recursive user paths.
            for name in ('latent.safetensors','metadata.json'):self._safe(folder/name).unlink()
            folder.rmdir()
        return {'deleted':rid,'images_deleted':False}
