"""Qwen 2.1 component contracts; JSON/header inspection only (no torch import).

Overrides are local data, not Python plugins. A decoder is a part of the VAE,
not an interchangeable generic Stable-Diffusion decoder. Runtime state-dict and
round-trip checks are performed by services.qwen_components_runtime.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
from core.qwen_assets import read_object,QwenAssetError

COMPONENT_PATHS={'vae_path','vae_config_path','text_encoder_dir','processor_dir','scheduler_dir','text_encoder_path'}
DECODE_DEFAULTS={'decode_device':'configured','vae_dtype':'auto','vae_decode_mode':'auto',
    'tile_height':1024,'tile_width':1024,'tile_stride_height':768,'tile_stride_width':768,
    'retain_latents':True}
EXTRA_DEFAULTS={**DECODE_DEFAULTS,'vae_path':'','vae_config_path':'','text_encoder_dir':'',
    'processor_dir':'','scheduler_dir':'','text_encoder_dtype':'auto','scheduler_shift':None,
    'text_encoder_path':'','quantized_loading':'reject'}
EXTRA_KEYS=set(EXTRA_DEFAULTS)
LATENT_KEYS=('z_dim','in_channels','out_channels','scale_factor_spatial','scale_factor_temporal','latents_mean','latents_std')


def normalize_extra(key,value):
    if key in COMPONENT_PATHS:
        if not isinstance(value,str) or '\0' in value or len(value)>4096:raise ValueError('Invalid component path: '+key)
        value=value.strip()
        if value and not Path(value).expanduser().is_absolute():raise ValueError('Use an absolute server path: '+key)
        return str(Path(value).expanduser()) if value else ''
    choices={'quantized_loading':('reject','native_comfyui','reference_dequantize'), 'decode_device':('configured','cuda','cpu'),
        'vae_dtype':('auto','bfloat16','float16','float32'),
        'text_encoder_dtype':('auto','bfloat16','float16','float32'),
        'vae_decode_mode':('auto','full_frame','tiled','tiled_custom')}
    if key in choices:
        if value not in choices[key]:raise ValueError('Invalid '+key)
        return value
    if key.startswith('tile_'):
        if type(value) is not int or not 64<=value<=4096 or value%16:raise ValueError(key+' must be a multiple of 16 from 64 to 4096.')
        return value
    if key=='retain_latents':
        if type(value) is not bool:raise ValueError('retain_latents must be boolean.')
        return value
    if key=='scheduler_shift':
        if value is None:return None
        if type(value) not in (float,int) or not math.isfinite(value) or not 0<value<=20:raise ValueError('Scheduler shift must be > 0 and <= 20 (or default).')
        return float(value)
    raise ValueError('Unknown Qwen component setting: '+key)


def decode_options(cfg,job=None):
    # Model settings supply defaults; an explicit per-job selection wins.
    values={k:cfg.get(k,v) for k,v in DECODE_DEFAULTS.items()}
    for k,v in (job or {}).items():
        if k in DECODE_DEFAULTS:values[k]=v
    values={k:normalize_extra(k,v) for k,v in values.items()}
    for axis in ('height','width'):
        if values['tile_stride_'+axis]>=values['tile_'+axis]:
            raise ValueError('Custom tiling requires positive overlap: stride must be smaller than tile '+axis+'.')
    return values


def latent_contract(config):
    # Pipeline 2.1 requires these exact latent dimensions and normalisation.
    z=config.get('z_dim');scale=config.get('scale_factor_spatial',16)
    if z!=64 or scale!=16:raise QwenAssetError('Qwen Image 2.1 requires 64 latent channels and 16x spatial compression.')
    for name in ('latents_mean','latents_std'):
        values=config.get(name)
        if not isinstance(values,(tuple,list)) or len(values)!=z or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise QwenAssetError('Invalid '+name+' for Qwen Image 2.1.')
        if name=='latents_std' and any(v<=0 for v in values):raise QwenAssetError('Latent standard deviations must be positive.')
    if config.get('in_channels',4)!=4 or config.get('out_channels',4)!=4:raise QwenAssetError('This edition requires the Qwen 2.1 RGBA VAE (4 channels).')
    return {k:config.get(k,{'in_channels':4,'out_channels':4,'scale_factor_spatial':16,'scale_factor_temporal':8}.get(k)) for k in LATENT_KEYS}


def contract_hash(contract):
    # Numeric JSON encodings are normalised so 0 and 0.0 compare consistently.
    c=dict(contract)
    for key in ('latents_mean','latents_std'):c[key]=[float(x) for x in c[key]]
    return hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def assert_same_latent_space(candidate,reference):
    c=latent_contract(candidate);r=latent_contract(reference)
    if contract_hash(c)!=contract_hash(r):raise QwenAssetError('VAE latent space / normalisation differs from the original. Re-decode is refused; no guessed rescaling.')
    return c


def weight_files(folder):
    folder=Path(folder);indexes=sorted(folder.glob('*.safetensors.index.json'))
    if len(indexes)>1:raise QwenAssetError('Multiple weight indexes found; use one unambiguous component folder.')
    if indexes:
        mapping=read_object(indexes[0]).get('weight_map')
        if not isinstance(mapping,dict) or not mapping:raise QwenAssetError('Empty weight index.')
        names=set(mapping.values())
        if any(not isinstance(n,str) or '/' in n or '\\' in n or '..' in n or not n.endswith('.safetensors') for n in names):raise QwenAssetError('Unsafe weight shard path.')
        paths=[folder/n for n in sorted(names)]
    else:paths=sorted(folder.glob('*.safetensors'))
    if not paths or any(not p.is_file() or p.stat().st_size<16 for p in paths):raise QwenAssetError('Missing safetensors weights/shards: '+str(folder))
    return paths


def component_location(cfg,kind):
    root=Path(str(cfg.get('model_dir') or '')).expanduser()
    mapping={'vae':'vae_path','text_encoder':'text_encoder_dir','processor':'processor_dir','scheduler':'scheduler_dir'}
    path=Path(cfg.get(mapping[kind]) or root/kind).expanduser()
    return path


def inspect_vae(path,config_path='',reference=None):
    path=Path(path).expanduser()
    if not path.is_absolute():raise QwenAssetError('VAE path must be absolute.')
    if path.is_dir():
        cp=path/'config.json';weights=weight_files(path)
        # A from_pretrained folder must be self-contained; no separate config
        # override which the runtime would silently ignore.
        if config_path and Path(config_path).expanduser().resolve()!=cp.resolve():raise QwenAssetError('A VAE folder uses its own config.json. Separate config is only for single safetensors files.')
    else:
        if not path.is_file() or path.suffix.lower()!='.safetensors':raise QwenAssetError('Choose a Diffusers VAE folder or a .safetensors VAE file.')
        if not config_path:raise QwenAssetError('A single-file VAE needs its matching Qwen 2.1 config.json.')
        cp=Path(config_path).expanduser();weights=[path]
    config=read_object(cp)
    if config.get('_class_name')!='AutoencoderKLQwenImage21':raise QwenAssetError('Expected AutoencoderKLQwenImage21, not another Qwen/SDXL VAE.')
    contract=latent_contract(config)
    if reference is not None:assert_same_latent_space(config,reference)
    return {'path':str(path),'config_path':str(cp),'class':'AutoencoderKLQwenImage21','config':config,
        'contract':contract,'contract_hash':contract_hash(contract),'weights':[str(p) for p in weights],
        'note':'Configuration and shards checked. Strict weights + miniature encode/decode are checked at runtime before first use.'}


def inspect_folder(path,kind,reference=None,*,require_weights=True):
    p=Path(path).expanduser()
    if not p.is_absolute() or not p.is_dir():raise QwenAssetError('Component folder not found: '+str(p))
    name='scheduler_config.json' if kind=='scheduler' else 'config.json'
    if kind=='processor':
        # Files needed by AutoProcessor are data only. Runtime uses the fixed
        # Qwen3VLProcessor class, never trust_remote_code / arbitrary auto_map.
        configs=[x for x in (p/'processor_config.json',p/'preprocessor_config.json') if x.is_file()]
        if not configs:raise QwenAssetError('Processor configuration is missing.')
        for f in configs:
            obj=read_object(f)
            if obj.get('processor_class') not in (None,'Qwen3VLProcessor'):raise QwenAssetError('Processor must be Qwen3VLProcessor.')
            if obj.get('auto_map'):raise QwenAssetError('Remote-code processors are not supported.')
        if not (p/'tokenizer_config.json').is_file():raise QwenAssetError('Processor folder must include its tokenizer_config.json.')
        return {'path':str(p),'kind':kind,'config':read_object(configs[0])}
    cfg=read_object(p/name)
    if kind=='text_encoder':
        if cfg.get('model_type')!='qwen3_vl':raise QwenAssetError('Only Qwen3-VL encoders matching the original Qwen 2.1 embedding interface are supported.')
        if cfg.get('auto_map'):raise QwenAssetError('Remote-code encoders are not supported.')
        if require_weights:weight_files(p)
        if reference:
            for section,keys in [('text_config',('hidden_size','vocab_size')),('vision_config',('out_hidden_size',))]:
                for key in keys:
                    expected=reference.get(section,{}).get(key)
                    if expected is not None and cfg.get(section,{}).get(key)!=expected:raise QwenAssetError('Text encoder interface mismatch: '+section+'.'+key)
    elif kind=='scheduler':
        if cfg.get('_class_name')!='FlowMatchEulerDiscreteScheduler':raise QwenAssetError('Only FlowMatchEulerDiscreteScheduler is supported for this Qwen pipeline.')
    else:raise QwenAssetError('Unsupported component folder.')
    return {'path':str(p),'kind':kind,'config':cfg}


def component_status(cfg,decode_only=False):
    fields=[];details={}
    def run(kind,fn):
        path=component_location(cfg,kind)
        try:
            info=fn(path);details[kind]=info
            fields.append({'field':kind+' / active component','value':str(path),'exists':True,'required':True,'detail':info.get('note','Configuration checked; model loading is a separate test.')})
        except (ValueError,OSError) as exc:
            fields.append({'field':kind+' / active component','value':str(path),'exists':False,'required':True,'detail':str(exc)})
    base=Path(str(cfg.get('model_dir') or ''))
    reference=None
    if cfg.get('vae_path'):
        try:reference=read_object(base/'vae/config.json')
        except (ValueError,OSError):pass
    run('vae',lambda path:inspect_vae(path,cfg.get('vae_config_path',''),reference))
    if not decode_only:
        for kind in ('text_encoder','processor','scheduler'):
            if not cfg.get({'text_encoder':'text_encoder_dir','processor':'processor_dir','scheduler':'scheduler_dir'}[kind]):continue
            ref=None
            if kind=='text_encoder':
                try:ref=read_object(base/'text_encoder/config.json')
                except (ValueError,OSError):pass
            run(kind,lambda path,k=kind,r=ref:inspect_folder(path,k,r,require_weights=not (k=='text_encoder' and bool(cfg.get('text_encoder_path')))))
    return {'ready':all(f['exists'] for f in fields),'fields':fields,'missing':[f['field'] for f in fields if not f['exists']],'details':details}
