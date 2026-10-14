"""Qwen Image 2.1 registry. Preserves existing user configuration during updates."""
import json,re,threading
from pathlib import Path
from core.paths import APP_ROOT,MODELS_CONFIG,MODELS_ROOT
from core.runtime_env import atomic_json
from core.qwen_assets import repository_status,QwenAssetError,inspect_component
_LOCK=threading.RLock()

def ensure_config():
    with _LOCK:
        if MODELS_CONFIG.exists():return
        data=json.loads((APP_ROOT/'config/models.defaults.json').read_text(encoding='utf-8'))
        cfg=data['models'][0]['config'];cfg['model_dir']=str(MODELS_ROOT/'Qwen-Image-2.1')
        atomic_json(MODELS_CONFIG,data)

def load_registry():
    ensure_config();data=json.loads(MODELS_CONFIG.read_text(encoding='utf-8'))
    if not isinstance(data,dict) or not isinstance(data.get('models'),list):raise ValueError('Invalid Qwen 2.1 model registry.')
    data['models']=[m for m in data['models'] if m.get('provider')=='qwen_image21']
    if not data['models']:raise ValueError('Qwen Image 2.1 is missing from this edition registry.')
    return data

def save_registry(data):
    if any(m.get('provider')!='qwen_image21' for m in data.get('models',[])):raise ValueError('This edition supports Qwen Image 2.1 only.')
    with _LOCK:atomic_json(MODELS_CONFIG,data)

def model_by_id(mid):return next((m for m in load_registry()['models'] if m.get('id')==mid),None)

def normalize_config(existing,incoming):
    import math
    from core.qwen_components import EXTRA_KEYS,normalize_extra,decode_options
    if not isinstance(incoming,dict):raise ValueError('Expected a configuration object.')
    cfg=dict(existing)
    for key,value in incoming.items():
        if key in EXTRA_KEYS:cfg[key]=normalize_extra(key,value)
        elif key in ('model_dir','model_path'):
            if not isinstance(value,str) or '\0' in value or len(value)>4096:raise ValueError('Invalid model path.')
            value=value.strip()
            if value and not Path(value).expanduser().is_absolute():raise ValueError('Model paths must be absolute server paths.')
            cfg[key]=str(Path(value).expanduser()) if value else ''
        elif key in ('device','dtype','memory_mode'):
            choices={'device':('cuda','cpu'),'dtype':('float16','bfloat16','float32'),
                'memory_mode':('balanced','cpu_offload','model_cpu_offload','full_gpu')}
            if value not in choices[key]:raise ValueError('Invalid '+key)
            cfg[key]=value
        elif key in ('gpu_memory','cpu_memory'):
            if not isinstance(value,str) or not re.fullmatch(r'[1-9][0-9]{0,3}(?:GiB|MiB)',value):raise ValueError('Memory limit must use GiB or MiB.')
            cfg[key]=value
        elif key in ('default_steps','default_cfg'):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Invalid '+key)
            lo,hi=(1,120) if key=='default_steps' else (1,20)
            if not lo<=value<=hi or key=='default_steps' and int(value)!=value:raise ValueError('Invalid '+key)
            cfg[key]=int(value) if key=='default_steps' else float(value)
        elif key=='cpu_offload':
            if type(value) is not bool:raise ValueError('cpu_offload must be boolean.')
            # Derived from memory mode; old exported configs may contain it.
        elif key in ('auto_priority','runtime','input_media','runtime_python','supports_strength'):
            # Historical read-only registry metadata are not executable profile settings.
            continue
        else:raise ValueError('Unsupported Qwen setting: '+key)
    # Older forms could store a safetensors file in the directory field.
    encoder_dir=str(cfg.get('text_encoder_dir') or '')
    if encoder_dir.lower().endswith('.safetensors'):
        if cfg.get('text_encoder_path') and cfg['text_encoder_path']!=encoder_dir:
            raise ValueError('Conflicting encoder selections: choose one weight file.')
        cfg['text_encoder_path']=encoder_dir;cfg['text_encoder_dir']=''
    # Selecting a new file alone means weights + base-repository configuration,
    # not a stale folder inherited from another profile. Explicit config is kept.
    if incoming.get('text_encoder_path') and 'text_encoder_dir' not in incoming:
        cfg['text_encoder_dir']=''
    # Recover profiles/configs written by older builds that paired a valid
    # single-file encoder with the generic browser parent directory. A file
    # override gets its architecture from the base repository unless an
    # explicitly complete encoder folder (config.json present) was selected.
    if cfg.get('text_encoder_path') and cfg.get('text_encoder_dir'):
        _te_dir=Path(str(cfg['text_encoder_dir'])).expanduser()
        if _te_dir.is_dir() and not (_te_dir/'config.json').is_file():
            cfg['text_encoder_dir']=''
    cfg['cpu_offload']=cfg.get('memory_mode') in ('balanced','cpu_offload','model_cpu_offload')
    decode_options(cfg)
    return cfg


def update_model(mid,patch):
    if not isinstance(patch,dict):raise ValueError('Expected a model settings object.')
    with _LOCK:
        data=load_registry()
        for m in data['models']:
            if m.get('id')!=mid:continue
            if 'enabled' in patch:
                if type(patch['enabled']) is not bool:raise ValueError('enabled must be boolean.')
                m['enabled']=patch['enabled']
            m['config']=normalize_config(m.get('config') or {},patch.get('config') or {})
            save_registry(data);return m
    return None


def validate_model(model):
    if not model or model.get('provider')!='qwen_image21':return {'ready':False,'fields':[],'missing':['Qwen Image 2.1 model']}
    cfg=model.get('config') or {};result=repository_status(cfg.get('model_dir'),cfg.get('model_path'))
    # Original repository files are not required for components explicitly
    # replaced with another locally validated component. The pipeline root,
    # transformer config and all non-overridden components are still checked.
    from core.qwen_components import component_status
    replaced={kind for kind,key in [('vae','vae_path'),('text_encoder','text_encoder_dir'),('processor','processor_dir'),('scheduler','scheduler_dir')] if cfg.get(key)}
    if replaced:
        fields=[f for f in result['fields'] if f['field'].split('/')[0] not in replaced]
        extra=component_status(cfg)
        fields.extend(f for f in extra['fields'] if f['field'].split(' / ')[0] in replaced)
        result.update(fields=fields,ready=all(f['exists'] for f in fields if f.get('required',True)),
            missing=[f['field'] for f in fields if f.get('required',True) and not f['exists']])
    if cfg.get('text_encoder_path'):
        fields=[f for f in result['fields'] if not f['field'].startswith('text_encoder/') or f['field']=='text_encoder/config.json']
        try:
            inspect_component(cfg['text_encoder_path'],'text_encoder_file');ok=True;detail='Single-file encoder; strict runtime keys/shapes check required.'
        except ValueError as exc:ok=False;detail=str(exc)
        fields.append({'field':'text_encoder_override','value':cfg['text_encoder_path'],'exists':ok,'required':True,'detail':detail})
        if not cfg.get('text_encoder_dir'):
            from core.qwen_components import inspect_folder,component_location
            config_dir=component_location(cfg,'text_encoder')
            try:
                inspect_folder(config_dir,'text_encoder',require_weights=False);config_ok=True;config_detail='Encoder architecture comes from the base repository; no config.json is required beside the weight file.'
            except (ValueError,OSError) as exc:config_ok=False;config_detail=str(exc)
            fields=[f for f in fields if f['field']!='text_encoder/config.json']
            fields.append({'field':'text_encoder/config.json','value':str(config_dir/'config.json'),'exists':config_ok,'required':True,'detail':config_detail})
        result.update(fields=fields,ready=all(f['exists'] for f in fields if f.get('required',True)),missing=[f['field'] for f in fields if f.get('required',True) and not f['exists']])
    return {**result,'id':model['id'],'name':model.get('name','Qwen Image 2.1'),'status':'ready' if result['ready'] else 'missing_files'}
