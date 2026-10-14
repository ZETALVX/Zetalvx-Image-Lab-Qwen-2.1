"""Persistent Qwen component profiles; data only, separate from prompt presets."""
from __future__ import annotations
import json,re,time,uuid
from contextlib import contextmanager
from pathlib import Path
from core.source_util import atomic_json
from core.qwen_assets import read_object
from core import file_lock

class ModelProfiles:
    def __init__(self,root):self.root=Path(root).absolute()
    def _safe_root(self):
        if any(x.is_symlink() for x in (self.root,*self.root.parents)):raise ValueError('Linked model profile directory.')
    @contextmanager
    def _lock(self):
        self._safe_root();self.root.mkdir(parents=True,exist_ok=True)
        if self.root.is_symlink() or (self.root/'profiles.lock').is_symlink():raise ValueError('Linked profile paths.')
        with (self.root/'profiles.lock').open('a+') as f:
            file_lock.flock(f.fileno(),file_lock.LOCK_EX)
            try:yield
            finally:file_lock.flock(f.fileno(),file_lock.LOCK_UN)
    def _path(self,rid):
        if not isinstance(rid,str) or not re.fullmatch('[a-f0-9]{32}',rid):raise ValueError('Invalid model profile identifier.')
        self._safe_root();p=self.root/(rid+'.json')
        if p.is_symlink() or self.root.is_symlink():raise ValueError('Linked model profile paths are not allowed.')
        return p
    def list(self):
        self._safe_root()
        if not self.root.exists():return []
        out=[]
        for p in sorted(self.root.glob('*.json'))[:200]:
            if p.is_symlink():continue
            try:
                d=read_object(p)
                if d.get('edition')=='qwen21' and d.get('schema')==1 and self._path(d['id'])==p:out.append(d)
            except (ValueError,KeyError):continue
        return sorted(out,key=lambda d:(d['name'].casefold(),d['id']))
    def get(self,rid):
        d=read_object(self._path(rid))
        if d.get('id')!=rid or d.get('schema')!=1 or d.get('edition')!='qwen21':raise ValueError('Invalid model profile.')
        return d
    def save(self,data,*,rid=None):
        from core.model_registry import normalize_config
        if not isinstance(data,dict) or data.get('edition') not in (None,'qwen21'):raise ValueError('Not a Qwen 2.1 model profile.')
        name=data.get('name','')
        if not isinstance(name,str) or not name.strip() or len(name)>120:raise ValueError('Enter a model profile name (1–120 characters).')
        if not isinstance(data.get('config'),dict):raise ValueError('A model profile must contain its component configuration.')
        config=normalize_config({},data['config'])
        from core.qwen_request import validate_params
        loras=validate_params({'loras':data.get('loras',[])})['loras']
        if loras and str(config.get('model_path','')).lower().endswith('.gguf'):raise ValueError('LoRA with GGUF is not supported. Save an unquantized-transformer profile or remove the LoRA stack.')
        # Export just non-executable LoRA fields, not arbitrary nested data.
        loras=[{'path':l['path'],'strength':l.get('strength',1.0),'name':str(l.get('name') or Path(l['path']).name)[:200]} for l in loras]
        self._safe_root();self.root.mkdir(parents=True,exist_ok=True)
        if self.root.is_symlink():raise ValueError('Linked model profile directory.')
        with self._lock():
            existing=self.get(rid) if rid else None
            if not rid and len(self.list())>=200:raise ValueError('Model profile limit reached (200).')
            rid=rid or uuid.uuid4().hex
            result={'schema':1,'edition':'qwen21','id':rid,'name':name.strip(),
                'description':str(data.get('description') or '')[:1000],'config':config,'loras':loras,
                'created_at':existing['created_at'] if existing else time.time(),'updated_at':time.time()}
            atomic_json(self._path(rid),result)
        return result
    def delete(self,rid):
        with self._lock():
            self.get(rid);self._path(rid).unlink()
        return {'deleted':rid,'model_files_deleted':False}
