"""User preset composition for Qwen 2.1. No execution, downloads or global edits.

Existing model/generation stores are reused; full presets snapshot both. Portable
exports contain logical descriptors and explicit local bindings, never server paths.
"""
from __future__ import annotations
import copy,hashlib,json,re,time,uuid
from pathlib import Path
from core.source_util import atomic_json
from core.qwen_profiles import ModelProfiles
from core.qwen_assets import read_object
from core.model_registry import normalize_config
from core.qwen_request import validate_params

PATH_KINDS={'model_dir':'repository','model_path':'transformer','vae_path':'vae',
 'vae_config_path':'vae_config','text_encoder_dir':'text_encoder','text_encoder_path':'text_encoder_file',
 'processor_dir':'processor','scheduler_dir':'scheduler'}
OVERRIDE_FIELDS=set(PATH_KINDS)-{'vae_config_path'}
LOAD_OPTIONS={'device','dtype','memory_mode','gpu_memory','cpu_memory','text_encoder_dtype',
 'quantized_loading','decode_device','vae_dtype','vae_decode_mode','tile_height','tile_width',
 'tile_stride_height','tile_stride_width','retain_latents','scheduler_shift'}
GEN_FIELDS={'width','height','steps','cfg','seed','batch','negative_prompt','cfg_enabled','transparent_output',
 'vae_decode_mode','qwen21_vae_decode_mode','decode_device','vae_dtype','tile_height','tile_width',
 'tile_stride_height','tile_stride_width','retain_latents'}


def clean_generation(data):
    if not isinstance(data,dict):raise ValueError('Generation settings must be an object.')
    raw=data.get('params') or {}
    if not isinstance(raw,dict):raise ValueError('Invalid generation parameters.')
    p=validate_params({k:v for k,v in raw.items() if k in GEN_FIELDS})
    p={k:v for k,v in p.items() if k in GEN_FIELDS}
    include=data.get('include_prompt',False)
    if type(include) is not bool:raise ValueError('include_prompt must be boolean.')
    prompt=data.get('prompt','')
    if not isinstance(prompt,str) or len(prompt)>30000:raise ValueError('Invalid preset prompt.')
    return {'params':p,'include_prompt':include,'prompt':prompt if include else ''}


def _name(data):
    value=data.get('name')
    if not isinstance(value,str) or not value.strip() or len(value)>120:raise ValueError('Enter a preset name (1–120 characters).')
    return value.strip()


class PresetLibrary:
    def __init__(self,root,model_profiles,callbacks,current):
        self.root=Path(root);self.models=model_profiles;self.cb=callbacks;self.current=current
        self.full=ModelProfiles(self.root/'full') # locking/path safety only; custom full schema below
    def builtins(self):
        cfg=normalize_config({},self.current())
        return [
          {'id':'@configured','kind':'model','name':'Current configuration','description':'Uses the current Models configuration without changing it.','config':cfg,'loras':[],'readonly':True},
          *[{'id':'@'+key,'kind':'generation','name':name,'readonly':True,'description':'Starting settings; quality and memory use depend on the selected components.',
           **clean_generation({'params':{'steps':steps,'seed':-1,'width':1024,'height':1024,'vae_decode_mode':'full_frame','retain_latents':True}})}
           for key,name,steps in [('preview','Preview · 20 steps',20),('balanced','Balanced · 40 steps',40),('quality','Quality · 50 steps',50)]]]
    def list(self):
        out=self.builtins()
        out += [{**x,'kind':'model','readonly':False} for x in self.models.list()]
        for item in self.cb['generation_list']():
            if item.get('edition') not in (None,'qwen21'):continue
            try:out.append({**item,**clean_generation(item),'kind':'generation','readonly':False})
            except ValueError:continue
        self.full._safe_root()
        if self.full.root.exists():
            for p in sorted(self.full.root.glob('*.json'))[:200]:
                if p.is_symlink():continue
                try:
                    x=read_object(p)
                    if x.get('kind')=='full' and x.get('edition')=='qwen21' and self.full._path(x['id'])==p:out.append(x)
                except (ValueError,KeyError):continue
        return out
    def get(self,kind,rid):
        if kind not in ('model','generation','full'):raise ValueError('Invalid preset kind.')
        item=next((x for x in self.list() if x['kind']==kind and x['id']==rid),None)
        if not item:raise ValueError('Preset not found.')
        return copy.deepcopy(item)
    def save(self,data,rid=None):
        if not isinstance(data,dict):raise ValueError('Invalid preset.')
        kind=data.get('kind');name=_name(data);desc=str(data.get('description') or '')[:1000]
        if rid:
            old=self.get(kind,rid)
            if old.get('readonly'):raise ValueError('Built-in presets are read-only. Save a copy.')
        if kind=='model':return {**self.models.save({**data,'name':name,'description':desc},rid=rid),'kind':'model','readonly':False}
        if kind=='generation':
            d={**clean_generation(data),'name':name,'description':desc,'edition':'qwen21'}
            rid=rid or uuid.uuid4().hex
            d=self.cb['generation_normalize'](d,preset_id=rid)
            p=self.cb['generation_path'](rid)
            if p.is_symlink():raise ValueError('Linked preset files are not allowed.')
            p.parent.mkdir(parents=True,exist_ok=True);atomic_json(p,d)
            return {**d,'kind':kind,'readonly':False}
        if kind!='full':raise ValueError('Invalid preset kind.')
        cfg=normalize_config({},data.get('config') or {})
        loras=validate_params({'loras':data.get('loras',[])})['loras']
        if loras and str(cfg.get('model_path','')).endswith('.gguf'):raise ValueError('GGUF + LoRA is not supported.')
        with self.full._lock():
            if not rid and len(list(self.full.root.glob('*.json')))>=200:raise ValueError('Full preset limit reached (200).')
            rid=rid or uuid.uuid4().hex
            result={'schema':1,'edition':'qwen21','kind':'full','id':rid,'readonly':False,'name':name,'description':desc,
                'config':cfg,'loras':[{'path':x['path'],'strength':x.get('strength',1)} for x in loras],
                'generation':clean_generation(data.get('generation') or {}),'updated_at':time.time()}
            atomic_json(self.full._path(rid),result)
        return result
    def delete(self,kind,rid):
        item=self.get(kind,rid)
        if item.get('readonly'):raise ValueError('Built-in presets are read-only.')
        if kind=='model':return self.models.delete(rid)
        if kind=='generation':p=self.cb['generation_path'](rid)
        else:p=self.full._path(rid)
        if p.is_symlink():raise ValueError('Linked preset file.')
        p.unlink();return {'deleted':rid,'model_files_deleted':False}


def catalog_entry(assets,cfg,rid,kind):
    entry=next((x for x in assets.inventory(cfg) if x['id']==rid and x['kind']==kind and x.get('present') and x.get('selectable',True)),None)
    if not entry:raise ValueError('Component missing or wrong type. Register it in Models → Add first.')
    return entry


def resolve_request(setup,library,assets,current):
    if not isinstance(setup,dict) or set(setup)-{'model_preset_id','overrides','load_options','use_profile_loras'}:raise ValueError('Invalid per-job model setup.')
    rid=setup.get('model_preset_id') or '@configured'
    if not isinstance(rid,str):raise ValueError('Invalid model preset identifier.')
    profile=library.get('full',rid[6:]) if rid.startswith('@full:') else library.get('model',rid)
    cfg=normalize_config({},profile['config']);loras=copy.deepcopy(profile.get('loras') or [])
    overrides=setup.get('overrides') or {};options=setup.get('load_options') or {}
    if not isinstance(overrides,dict) or set(overrides)-OVERRIDE_FIELDS:raise ValueError('Invalid component override.')
    if not isinstance(options,dict) or set(options)-LOAD_OPTIONS:raise ValueError('Invalid loading options.')
    for field,choice in overrides.items():
        if not isinstance(choice,str):raise ValueError('Component choice must be a catalog identifier.')
        if choice:
            entry=catalog_entry(assets,current,choice,PATH_KINDS[field]);cfg[field]=entry['path']
        elif field=='model_dir':raise ValueError('A base Qwen repository is required.')
        else:cfg[field]=''
        if field=='vae_path':
            cfg['vae_config_path']=''
            if cfg[field] and Path(cfg[field]).is_file():
                cp=Path(cfg[field]).parent/'config.json'
                # A Diffusers single-file override may use the original VAE config
                # only when chosen explicitly by this action; strict loader checks remain.
                original=Path(cfg.get('model_dir',''))/'vae/config.json'
                if cp.is_file():cfg['vae_config_path']=str(cp)
                elif original.is_file():cfg['vae_config_path']=str(original)
                else:raise ValueError('Select a VAE profile with a matching config.json first.')
    if overrides.get('text_encoder_path') and 'text_encoder_dir' not in overrides:
        cfg['text_encoder_dir']=''
    if overrides.get('text_encoder_dir') and 'text_encoder_path' not in overrides:
        cfg['text_encoder_path']=''
    cfg=normalize_config(cfg,options)
    if type(setup.get('use_profile_loras',False)) is not bool:raise ValueError('use_profile_loras must be boolean.')
    return cfg,loras,{'preset_id':rid,'preset_name':profile['name'],'overrides':copy.deepcopy(overrides),'load_options':copy.deepcopy(options)}


def export_portable(item):
    data={'schema':'zetalvx-qwen21-preset-1','kind':item['kind'],'name':item['name'],'description':item.get('description','')}
    refs={}
    if item['kind'] in ('model','full'):
        cfg=copy.deepcopy(item['config']);extra_cfg=cfg.pop('vae_config_path','')
        for key,kind in PATH_KINDS.items():
            path=cfg.pop(key,'')
            if path:refs[key]={'kind':kind,'name':Path(path).name}
        data['config']=cfg
        data['loras']=[]
        for i,x in enumerate(item.get('loras',[])):
            key='lora_'+str(i);refs[key]={'kind':'lora','name':Path(x['path']).name};data['loras'].append({'binding':key,'strength':x.get('strength',1)})
        if extra_cfg and 'vae_path' in refs:data['vae_config_required']=True
    if item['kind']=='generation':data['generation']=clean_generation(item)
    elif item['kind']=='full':data['generation']=clean_generation(item['generation'])
    data['components']=refs
    return data


def import_portable(payload,assets,current,bindings=None):
    if not isinstance(payload,dict) or payload.get('schema')!='zetalvx-qwen21-preset-1':raise ValueError('Not a portable Qwen 2.1 preset.')
    if len(json.dumps(payload))>100000:raise ValueError('Preset too large.')
    kind=payload.get('kind')
    if kind not in ('model','generation','full'):raise ValueError('Invalid preset kind.')
    data={'kind':kind,'name':_name(payload),'description':str(payload.get('description') or '')[:1000]}
    refs=payload.get('components') or {};bindings=bindings or {};resolved={};missing=[]
    if not isinstance(refs,dict) or len(refs)>12 or not isinstance(bindings,dict):raise ValueError('Invalid component bindings.')
    inventory=assets.inventory(current)
    for key,descriptor in refs.items():
        expected=PATH_KINDS.get(key) if not re.fullmatch('lora_[0-3]',key) else 'lora'
        if expected in (None,'vae_config') or not isinstance(descriptor,dict) or descriptor.get('kind')!=expected:raise ValueError('Invalid component descriptor.')
        name=descriptor.get('name')
        if not isinstance(name,str) or len(name)>255 or any(c in name for c in ('/','\\','\0')):raise ValueError('Portable references must be filenames, not paths.')
        choice=bindings.get(key)
        candidates=[x for x in inventory if x['kind']==expected and x.get('present') and (x['id']==choice if choice else Path(x['path']).name==name)]
        if len(candidates)==1:resolved[key]=candidates[0]['path']
        else:missing.append({'key':key,**descriptor,'choices':[{'id':x['id'],'name':x['name']} for x in inventory if x['kind']==expected and x.get('present')]})
    if missing:return None,missing
    if kind in ('model','full'):
        raw=payload.get('config') or {}
        if not isinstance(raw,dict) or set(raw)&set(PATH_KINDS):raise ValueError('Portable imports cannot supply machine paths.')
        cfg=normalize_config({},raw);cfg.update({k:v for k,v in resolved.items() if k in PATH_KINDS})
        if not cfg.get('model_dir'):raise ValueError('Bind the base repository before importing a model preset.')
        if cfg.get('vae_path') and Path(cfg['vae_path']).is_file():
            cp=Path(cfg['vae_path']).parent/'config.json';cp=cp if cp.is_file() else Path(cfg['model_dir'])/'vae/config.json'
            if not cp.is_file():raise ValueError('Missing compatible VAE config.json.')
            cfg['vae_config_path']=str(cp)
        data['config']=cfg;data['loras']=[]
        for x in payload.get('loras',[]):
            if not isinstance(x,dict) or x.get('binding') not in resolved:raise ValueError('Invalid LoRA binding.')
            data['loras'].append({'path':resolved[x['binding']],'strength':x.get('strength',1)})
    if kind=='generation':data.update(clean_generation(payload.get('generation') or {}))
    if kind=='full':data['generation']=clean_generation(payload.get('generation') or {})
    return data,[]
