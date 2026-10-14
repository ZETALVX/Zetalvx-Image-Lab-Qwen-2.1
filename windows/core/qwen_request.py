"""One Qwen form: text-only or 1–10 ordered reference images. Pure validation."""
import math
from pathlib import Path

MAX_IMAGES=10
PARAM_KEYS={'width','height','steps','cfg','seed','batch','negative_prompt','cfg_enabled',
    'transparent_output','vae_decode_mode','qwen21_vae_decode_mode','qwen21_transformer_mode',
    'qwen21_transformer_path','qwen21_text_encoder_path','qwen21_vae_path','source_artifact_id','reference_artifact_ids','loras','ui_qwen21',
    'decode_device','vae_dtype','tile_height','tile_width','tile_stride_height','tile_stride_width','retain_latents','oom_cpu_decode_fallback'}

def validate_params(params):
    if not isinstance(params,dict):raise ValueError('Parameters must be an object.')
    p={k:v for k,v in params.items() if k in PARAM_KEYS}
    for key,default,lo,hi in [('width',1024,256,4096),('height',1024,256,4096),('steps',40,1,120),('batch',1,1,8),('seed',-1,-1,2**53-1)]:
        v=p.get(key,default)
        if type(v) not in (int,float) or not math.isfinite(v) or int(v)!=v or not lo<=v<=hi:raise ValueError('Invalid '+key)
        p[key]=int(v)
    if p['width']%16 or p['height']%16:raise ValueError('Width and height must be multiples of 16.')
    v=p.get('cfg',1.0)
    if type(v) not in (int,float) or not math.isfinite(v) or not 1<=v<=20:raise ValueError('CFG must be between 1 and 20.')
    p['cfg']=float(v)
    for key in ('transparent_output','cfg_enabled','oom_cpu_decode_fallback'):
        if key in p and type(p[key]) is not bool:raise ValueError(key+' must be boolean.')
    if 'cfg_enabled' not in p:p['cfg_enabled']=p['cfg']>1 and bool(p.get('negative_prompt'))
    if not isinstance(p.get('negative_prompt',''),str) or len(p.get('negative_prompt',''))>20000:raise ValueError('Invalid negative prompt.')
    if not p['cfg_enabled']:p['cfg']=1.0;p['negative_prompt']=''
    mode=p.get('qwen21_transformer_mode','configured')
    if mode not in ('configured','original','gguf','safetensors'):raise ValueError('Invalid transformer mode.')
    p['qwen21_transformer_mode']=mode
    raw=p.get('qwen21_transformer_path','')
    if not isinstance(raw,str) or '\0' in raw:raise ValueError('Invalid transformer path.')
    if mode in ('gguf','safetensors'):
        path=Path(raw).expanduser()
        suffix='.gguf' if mode=='gguf' else '.safetensors'
        if not raw or not path.is_absolute() or path.suffix.lower()!=suffix:raise ValueError('Select an absolute '+suffix+' transformer path.')
        p['qwen21_transformer_path']=str(path)
    for key,label in (('qwen21_text_encoder_path','Text Encoder'),('qwen21_vae_path','VAE')):
        raw_comp=p.get(key,'')
        if raw_comp is None:raw_comp=''
        if not isinstance(raw_comp,str) or '\0' in raw_comp:raise ValueError('Invalid '+label+' path.')
        raw_comp=raw_comp.strip()
        if raw_comp:
            cp=Path(raw_comp).expanduser()
            if not cp.is_absolute() or cp.suffix.lower()!='.safetensors':raise ValueError(label+' override must be an absolute .safetensors file.')
            p[key]=str(cp)
        else:p[key]=''
    vae=p.get('qwen21_vae_decode_mode',p.get('vae_decode_mode','auto'))
    if vae not in ('auto','tiled','full_frame','tiled_custom'):raise ValueError('Invalid VAE decode mode.')
    p['qwen21_vae_decode_mode']=vae;p['vae_decode_mode']=vae
    refs=p.get('reference_artifact_ids') or []
    if not isinstance(refs,list) or any(not isinstance(x,str) or not x for x in refs):raise ValueError('Input references must be image identifiers.')
    source=p.get('source_artifact_id')
    if source is not None and not isinstance(source,str):raise ValueError('Invalid primary image.')
    ordered=[]
    for ref in ([source] if source else [])+refs:
        if ref not in ordered:ordered.append(ref)
    if len(ordered)>MAX_IMAGES:raise ValueError('Qwen Image 2.1 accepts at most 10 images per job.')
    # The unified form passes only ordered references. No hidden source precedence.
    p['source_artifact_id']=None;p['reference_artifact_ids']=ordered
    loras=p.get('loras') or []
    if not isinstance(loras,list) or len(loras)>4:raise ValueError('Use at most 4 compatible LoRA adapters.')
    for entry in loras:
        if not isinstance(entry,dict):raise ValueError('Invalid LoRA entry.')
        path=entry.get('path');weight=entry.get('strength',1)
        if not isinstance(path,str) or not Path(path).is_absolute() or not path.lower().endswith('.safetensors'):raise ValueError('LoRA must be an absolute safetensors path.')
        if type(weight) not in (int,float) or not math.isfinite(weight) or not -2<=weight<=2:raise ValueError('Invalid LoRA weight.')
    if loras and mode=='gguf':raise ValueError('LoRA with a GGUF transformer is not enabled in this integration. Use a Diffusers/safetensors transformer.')
    from core.qwen_components import normalize_extra,decode_options
    for key in ('decode_device','vae_dtype','tile_height','tile_width','tile_stride_height','tile_stride_width','retain_latents'):
        if key in p:p[key]=normalize_extra(key,p[key])
    decode_options({},p)
    p['loras']=loras;p['ui_qwen21']=True
    return p

def auto_tool(params):
    n=len(params.get('reference_artifact_ids') or [])
    return 'generate_image' if n==0 else ('reference' if n==1 else 'multi_image')

def effective_prompt(prompt,transparent=False):
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>30000:raise ValueError('Enter a prompt/instruction (maximum 30000 characters).')
    prompt=prompt.strip()
    if transparent:
        return 'Generate an RGBA image with real transparency. '+prompt+' The background must be transparent and the output must retain its alpha channel.'
    return prompt
