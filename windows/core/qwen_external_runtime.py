"""Global configuration for user-supplied external ComfyUI runtimes.

The runtime is never installed or modified by Image Lab. Paths are server-local,
explicitly trusted by the user, and shared by Prompt Enhancer, native ConvRot,
GGUF execution and Benchmark Lab preflight.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from core.source_util import atomic_json

DEFAULTS={
    'schema':1,
    'comfy_root':'',
    'comfy_python':'',
    'runtime_confirmed':False,
    'verified_at':None,
    'verified_key':'',
    'capabilities':{},
}
# `verified` is a derived/public-only field returned by the API. 1.0.44 accepts
# it when reading an older accidentally persisted public object, then drops it.
_DERIVED_SAVED_FIELDS={'verified'}
_LOCK=threading.RLock()


def _string(value,label,limit=4096,allow_empty=True):
    if not isinstance(value,str):
        raise ValueError(f'{label} must be text.')
    if '\0' in value:
        raise ValueError(f'{label} contains an invalid NUL character.')
    if len(value)>limit:
        raise ValueError(f'{label} is too long.')
    value=value.strip()
    if not value and not allow_empty:
        raise ValueError(f'Missing {label}.')
    return value


def _path(value,label):
    value=_string(value,label)
    if not value:return ''
    p=Path(value).expanduser()
    if not p.is_absolute():
        raise ValueError(f'{label} must be an absolute server path. Received: {value}')
    # Normalize '.', '..' and '~' without requiring the target to exist yet.
    return os.path.abspath(os.path.normpath(str(p)))


def runtime_key(cfg):
    raw=(str(cfg.get('comfy_root') or '')+'\0'+str(cfg.get('comfy_python') or '')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def normalize_runtime_config(raw,previous=None,*,saved=False):
    if not isinstance(raw,dict):
        raise ValueError('ComfyUI runtime configuration must be a JSON object.')
    allowed=set(DEFAULTS) if saved else {'schema','comfy_root','comfy_python','runtime_confirmed'}
    raw=dict(raw)
    if saved:
        for key in _DERIVED_SAVED_FIELDS:
            raw.pop(key,None)
    unknown=sorted(set(raw)-allowed)
    if unknown:
        raise ValueError('Unsupported ComfyUI runtime configuration field(s): '+', '.join(unknown)+'.')
    out=copy.deepcopy(DEFAULTS)
    if previous:
        out.update({k:v for k,v in previous.items() if k in DEFAULTS})
    for k in ('comfy_root','comfy_python','runtime_confirmed'):
        if k in raw:out[k]=raw[k]
    out['schema']=1
    out['comfy_root']=_path(out['comfy_root'],'ComfyUI folder')
    out['comfy_python']=_path(out['comfy_python'],'ComfyUI Python')
    if type(out['runtime_confirmed']) is not bool:
        raise ValueError('ComfyUI trust confirmation must be true or false.')
    if saved:
        va=raw.get('verified_at',out.get('verified_at'))
        if va is not None and type(va) not in (int,float):
            raise ValueError('Saved ComfyUI verification timestamp is invalid.')
        out['verified_at']=float(va) if va is not None else None
        vk=raw.get('verified_key',out.get('verified_key',''))
        if vk and (not isinstance(vk,str) or len(vk)!=64 or any(c not in '0123456789abcdef' for c in vk)):
            raise ValueError('Saved ComfyUI verification key is invalid.')
        out['verified_key']=vk or ''
        caps=raw.get('capabilities',out.get('capabilities',{}))
        if not isinstance(caps,dict):
            raise ValueError('Saved ComfyUI capabilities must be an object.')
        if len(json.dumps(caps,ensure_ascii=False))>64000:
            raise ValueError('Saved ComfyUI capabilities are too large.')
        out['capabilities']=copy.deepcopy(caps)
    changed=bool(previous and any(out[k]!=previous.get(k,'') for k in ('comfy_root','comfy_python')))
    if changed or not out['runtime_confirmed']:
        out['verified_at']=None;out['verified_key']='';out['capabilities']={}
    return out


def selection_paths(cfg):
    """Validate only the selected local paths; never launches external code."""
    root=Path(_string(cfg.get('comfy_root',''),'ComfyUI folder',allow_empty=False)).expanduser()
    exe=Path(_string(cfg.get('comfy_python',''),'ComfyUI Python',allow_empty=False)).expanduser()
    if not root.is_absolute():
        raise ValueError(f'ComfyUI folder must be an absolute server path. Received: {root}')
    if not exe.is_absolute():
        raise ValueError(f'ComfyUI Python must be an absolute server path. Received: {exe}')
    root=Path(os.path.abspath(os.path.normpath(str(root))));exe=Path(os.path.abspath(os.path.normpath(str(exe))))
    if not root.exists():
        raise ValueError(f'ComfyUI folder does not exist: {root}')
    if not root.is_dir():
        raise ValueError(f'ComfyUI folder is not a directory: {root}')
    if not (root/'comfy/sd.py').is_file():
        raise ValueError(f'ComfyUI folder is not a compatible installation (missing comfy/sd.py): {root}')
    if not exe.exists():
        raise ValueError(f'ComfyUI Python does not exist: {exe}')
    if not exe.is_file():
        raise ValueError(f'ComfyUI Python is not a file: {exe}')
    if os.name!='nt' and not os.access(exe,os.X_OK):
        raise ValueError(f'ComfyUI Python is not executable: {exe}')
    return root,exe


def runtime_paths(cfg):
    root,exe=selection_paths(cfg)
    if cfg.get('runtime_confirmed') is not True:
        raise ValueError('Confirm trust in this existing local ComfyUI runtime before running it.')
    return root,exe


def native_runtime_paths(cfg):
    """Validate the stricter contract required to launch a ComfyUI server."""
    root,exe=runtime_paths(cfg)
    if not (root/'main.py').is_file():
        raise ValueError(f'The selected ComfyUI folder has no main.py server entrypoint: {root}')
    return root,exe


def discover_runtimes(home=None):
    """Check a small, predictable set of local locations. Never scan arbitrary disks."""
    home=Path(home or Path.home())
    roots=[home/'ai/ComfyUI',home/'ComfyUI']
    found=[];seen=set()
    for root in roots:
        if root in seen or not (root/'comfy/sd.py').is_file():continue
        seen.add(root);choices=[]
        # Names are discovery hints only. Manually selected environments may use
        # any directory name (venv311, venv_new, conda env symlink, etc.).
        for env in ('venv311','venv_new','venv','.venv'):
            for suffix in ('bin/python','Scripts/python.exe'):
                exe=root/env/suffix
                if exe.is_file():choices.append(os.path.abspath(os.path.normpath(str(exe))))
        found.append({'root':os.path.abspath(os.path.normpath(str(root))),'interpreters':choices})
    return found


class ComfyRuntimeStore:
    def __init__(self,shared):
        self.shared=Path(shared);self.path=self.shared/'config/qwen-comfyui-runtime.json';self.legacy=self.shared/'config/qwen-prompt-enhancer.json'
    def _legacy(self):
        if not self.legacy.is_file() or self.legacy.is_symlink():return copy.deepcopy(DEFAULTS)
        try:
            d=json.loads(self.legacy.read_text(encoding='utf-8'))
            return normalize_runtime_config({
                'comfy_root':d.get('comfy_root',''),'comfy_python':d.get('comfy_python',''),
                'runtime_confirmed':bool(d.get('runtime_confirmed',False))})
        except (OSError,ValueError,TypeError,json.JSONDecodeError):return copy.deepcopy(DEFAULTS)
    def config(self):
        if not self.path.exists():return self._legacy()
        if self.path.is_symlink():
            raise ValueError(f'ComfyUI runtime configuration file must not be a symlink: {self.path}')
        if self.path.stat().st_size>100000:
            raise ValueError(f'ComfyUI runtime configuration file is unexpectedly large: {self.path}')
        try:
            raw=json.loads(self.path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ValueError(f'ComfyUI runtime configuration file contains invalid JSON: {self.path}') from exc
        return normalize_runtime_config(raw,saved=True)
    def save(self,raw):
        with _LOCK:
            previous=self.config();cfg=normalize_runtime_config(raw,previous)
            # Saving validates the actual server-local folder and interpreter, but
            # does not execute anything. Verification/probing remains a separate step.
            selection_paths(cfg)
            self.path.parent.mkdir(parents=True,exist_ok=True);atomic_json(self.path,cfg)
            return cfg
    def invalidate(self,raw):
        with _LOCK:
            previous=self.config();cfg=normalize_runtime_config(raw,previous)
            selection_paths(cfg)
            cfg['verified_at']=None;cfg['verified_key']='';cfg['capabilities']={}
            self.path.parent.mkdir(parents=True,exist_ok=True);atomic_json(self.path,cfg)
            return cfg
    def mark_verified(self,raw,capabilities):
        if not isinstance(capabilities,dict):raise ValueError('Invalid ComfyUI probe result.')
        with _LOCK:
            previous=self.config()
            # `raw` may be the complete internal config returned by invalidate().
            # Verification metadata is backend-owned and must never be re-validated
            # as if it were editable user input. Keep the public input schema strict.
            editable={k:raw[k] for k in ('schema','comfy_root','comfy_python','runtime_confirmed') if k in raw}
            cfg=normalize_runtime_config(editable,previous)
            runtime_paths(cfg)
            safe={
                'comfyui_ok':bool(capabilities.get('comfyui_ok')),
                'cuda_available':bool(capabilities.get('cuda_available')),
                'qwen_image_nodes':bool(capabilities.get('qwen_image_nodes')),
                'native_qwen_loader':bool(capabilities.get('native_qwen_loader')),
                'gguf_loader':bool(capabilities.get('gguf_loader')),
                'gguf_qwen21':bool(capabilities.get('gguf_qwen21')),
                'gguf_qwen21_detail':str(capabilities.get('gguf_qwen21_detail') or '')[:500],
                'prompt_enhancer_imports':bool(capabilities.get('prompt_enhancer_imports')),
                'gpu':str(capabilities.get('gpu') or '')[:240],
                'python_version':str(capabilities.get('python_version') or '')[:120],
                'torch_version':str(capabilities.get('torch_version') or '')[:120],
            }
            cfg['verified_at']=time.time();cfg['verified_key']=runtime_key(cfg);cfg['capabilities']=safe
            self.path.parent.mkdir(parents=True,exist_ok=True);atomic_json(self.path,cfg)
            return cfg
    def verified(self,cfg=None):
        cfg=cfg or self.config()
        return bool(cfg.get('runtime_confirmed') and cfg.get('verified_at') and cfg.get('verified_key')==runtime_key(cfg) and cfg.get('capabilities',{}).get('comfyui_ok'))
    def public(self):
        cfg=self.config();return {**cfg,'verified':self.verified(cfg)}
