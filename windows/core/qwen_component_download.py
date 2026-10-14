"""Explicit, commit-pinned HF component bundles in the existing download queue.
No model is bundled, downloaded at startup, or activated by downloading it.
"""
from __future__ import annotations
import re,time,uuid
from pathlib import Path,PurePosixPath
from urllib.parse import urlsplit,unquote,quote
from core.download_security import DownloadError,clean_user_url
from core.qwen_repository_download import MAX_FILE,MAX_TOTAL
KINDS={'vae','text_encoder','processor','scheduler'}


def allowed_component_path(name,kind):
    if kind not in KINDS or not isinstance(name,str) or len(name)>220 or any(c in name for c in '/\\:') or '..' in name or name.endswith((' ','.')) or any(ord(c)<32 for c in name):return False
    if name in ('LICENSE','LICENSE.md','LICENSE.txt','NOTICE','NOTICE.md','NOTICE.txt','README.md'):return True
    suffix=PurePosixPath(name).suffix.lower()
    # Single component folders intentionally have a flat safetensors/config
    # layout; reject weights.py, pickle, bins and unrelated gallery files.
    allowed={'.json','.safetensors'} if kind in ('vae','text_encoder') else {'.json','.txt','.model','.tiktoken','.jinja'}
    return suffix in allowed and not name.startswith('.')


def inspect_bundle(dl,url,kind):
    if kind not in KINDS:raise DownloadError('Unsupported Qwen component bundle.')
    parsed=urlsplit(clean_user_url(url));parts=parsed.path.strip('/').split('/')
    if parsed.hostname!='huggingface.co' or parsed.query or len(parts) not in (2,4) or len(parts)==4 and parts[2]!='tree':raise DownloadError('Use the Hugging Face repository URL (or /tree/revision).')
    if any(not re.fullmatch('[A-Za-z0-9._-]+',p) for p in parts[:2]):raise DownloadError('Invalid repository name.')
    repo='/'.join(parts[:2]);rev=unquote(parts[3]) if len(parts)==4 else 'main'
    data=dl.transport.json('https://huggingface.co/api/models/'+repo+'/revision/'+quote(rev,safe='')+'?blobs=true')
    if not isinstance(data,dict) or not re.fullmatch('[a-f0-9]{40}',str(data.get('sha',''))):raise DownloadError('Source commit could not be pinned.')
    sha=data['sha'];siblings=data.get('siblings')
    if not isinstance(siblings,list) or len(siblings)>6000:raise DownloadError('Invalid or oversized component file list.')
    names={x.get('rfilename') for x in siblings if isinstance(x,dict)}
    key='scheduler_config.json' if kind=='scheduler' else 'config.json'
    if kind=='processor':key='processor_config.json' if 'processor_config.json' in names or kind+'/processor_config.json' in names else 'preprocessor_config.json'
    prefix=kind+'/' if kind+'/'+key in names else ''
    if prefix+key not in names:raise DownloadError('Matching component configuration is absent: '+key)
    base='https://huggingface.co/'+repo+'/resolve/'+sha+'/'
    cfg=dl.transport.json(base+prefix+key)
    if not isinstance(cfg,dict):raise DownloadError('Invalid component configuration.')
    expected={'vae':'AutoencoderKLQwenImage21','scheduler':'FlowMatchEulerDiscreteScheduler'}
    if kind in expected and cfg.get('_class_name')!=expected[kind]:raise DownloadError('Wrong component class: expected '+expected[kind])
    if kind=='vae':
        from core.qwen_components import latent_contract
        latent_contract(cfg)
    if kind=='text_encoder' and cfg.get('model_type')!='qwen3_vl':raise DownloadError('Only a compatible Qwen3-VL text encoder is supported.')
    if cfg.get('auto_map'):raise DownloadError('Remote-code components are not supported.')
    files=[];seen=set();total=0
    for item in siblings:
        if not isinstance(item,dict):continue
        name=item.get('rfilename')
        if not isinstance(name,str):continue
        if prefix:
            if name.startswith(prefix):relative=name[len(prefix):]
            elif name in ('LICENSE','LICENSE.md','LICENSE.txt','NOTICE','NOTICE.md','NOTICE.txt','README.md'):relative=name
            else:continue
        else:relative=name
        if not allowed_component_path(relative,kind):continue
        if relative.casefold() in seen:raise DownloadError('Duplicate component path.')
        seen.add(relative.casefold());lfs=item.get('lfs') or {}
        size=lfs.get('size') or item.get('size') or 0
        if type(size) is not int or not 0<size<=MAX_FILE:raise DownloadError('Missing or invalid file size: '+relative)
        digest=str(lfs.get('sha256') or lfs.get('oid') or '').removeprefix('sha256:').lower()
        if digest and not re.fullmatch('[a-f0-9]{64}',digest):raise DownloadError('Invalid source digest.')
        if relative.endswith('.safetensors') and not digest:raise DownloadError('Component weights need a source SHA-256.')
        total+=size
        if total>MAX_TOTAL:raise DownloadError('Component bundle too large.')
        files.append({'name':relative,'size':size,'sha256':digest,'url':base+quote(name,safe='/')})
    if not files or len(files)>1000:raise DownloadError('Empty/oversized component bundle.')
    # Pick only standard Diffusers weights when the repo also ships a duplicate
    # ComfyUI export (e.g. Texture-Fix); never load both accidentally.
    if kind=='vae' and any(x['name']=='diffusion_pytorch_model.safetensors' for x in files):
        files=[x for x in files if (not x['name'].endswith('.safetensors') or x['name']=='diffusion_pytorch_model.safetensors') and not x['name'].endswith('.safetensors.index.json')]
        total=sum(x['size'] for x in files)
    card=data.get('cardData') if isinstance(data.get('cardData'),dict) else {}
    licence=str(card.get('license_name') or card.get('license') or 'Source terms must be checked')[:200]
    plan={'id':uuid.uuid4().hex,'expires':time.time()+1800,'kind':'repository','kind_hint':kind,'component_kind':kind,
        'provider':'huggingface','repo':repo,'revision':sha,'source':'https://huggingface.co/'+repo,
        'filename':repo.split('/')[-1]+'-'+kind,'size':total,'sha256':'','license':licence,
        'license_url':'https://huggingface.co/'+repo+'/tree/'+sha,'file_count':len(files),'files':files,'builtin':False}
    with dl.lock:
        dl.plans={k:v for k,v in dl.plans.items() if v['expires']>time.time()}
        if len(dl.plans)>=40:raise DownloadError('Too many open download previews.')
        dl.plans[plan['id']]=plan
    return dl.preview(plan)


def validate_bundle(stage,kind):
    from core.qwen_components import inspect_vae,inspect_folder
    if kind=='vae':inspect_vae(stage)
    else:inspect_folder(stage,kind)
    return {'ready':True,'missing':[]}
