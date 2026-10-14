"""Qwen-only multi-file HF download using the existing pinned-IP transport and queue.
Never executes repository code; only explicit config/tokenizer/safetensors files.
"""
import hashlib,json,os,re,shutil,time,uuid
from pathlib import Path,PurePosixPath
from urllib.parse import urlsplit,unquote,quote
from core.download_security import DownloadError,Cancelled,clean_user_url
from core.runtime_env import atomic_json
from core.qwen_assets import repository_status
MAX_TOTAL=200*1024**3
MAX_FILE=32*1024**3

def allowed_path(name,exclude_transformer=False):
    if not isinstance(name,str) or len(name)>220 or '\\' in name or ':' in name or name.startswith('/') or any(ord(c)<32 for c in name):return False
    parts=name.split('/')
    if any(x in ('','.','..') or x.endswith((' ','.')) for x in parts):return False
    if name in ('model_index.json','LICENSE','LICENSE.md','LICENSE.txt'):return True
    if parts[0] not in ('transformer','text_encoder','processor','tokenizer','scheduler','vae'):return False
    if exclude_transformer and parts[0]=='transformer' and name.endswith(('.safetensors','.safetensors.index.json')):return False
    return PurePosixPath(name).suffix.lower() in ('.json','.safetensors','.txt','.model','.tiktoken')

def inspect_repository(dl,url,exclude_transformer=False):
    url=clean_user_url(url);parsed=urlsplit(url)
    if parsed.hostname!='huggingface.co' or parsed.query:raise DownloadError('Use a Hugging Face Qwen Image 2.1 repository URL without query parameters.')
    bits=parsed.path.strip('/').split('/')
    if len(bits) not in (2,4) or len(bits)==4 and bits[2]!='tree':raise DownloadError('Use https://huggingface.co/owner/repository (or /tree/revision).')
    if any(not re.fullmatch(r'[A-Za-z0-9._-]+',x) for x in bits[:2]):raise DownloadError('Invalid repository identifier.')
    repo='/'.join(bits[:2]);revision=unquote(bits[3]) if len(bits)==4 else 'main'
    metadata=dl.transport.json('https://huggingface.co/api/models/'+repo+'/revision/'+quote(revision,safe='')+'?blobs=true')
    if not isinstance(metadata,dict):raise DownloadError('Invalid repository metadata.')
    sha=metadata.get('sha','')
    if not isinstance(sha,str) or not re.fullmatch('[a-f0-9]{40}',sha):raise DownloadError('The source did not provide a pinned commit; download not started.')
    base='https://huggingface.co/'+repo+'/resolve/'+sha+'/'
    index=dl.transport.json(base+'model_index.json')
    if not isinstance(index,dict) or index.get('_class_name')!='QwenImage21Pipeline':raise DownloadError('This repository is not a QwenImage21Pipeline model.')
    records=metadata.get('siblings') or []
    if not isinstance(records,list) or len(records)>6000:raise DownloadError('Repository file listing is invalid or too large.')
    files=[];seen=set();total=0
    for item in records:
        if not isinstance(item,dict):continue
        name=item.get('rfilename')
        if not allowed_path(name,exclude_transformer):continue
        if name.casefold() in seen:raise DownloadError('Duplicate repository path.')
        seen.add(name.casefold());lfs=item.get('lfs') or {};size=int(lfs.get('size') or item.get('size') or 0)
        if not 0<size<=MAX_FILE:raise DownloadError('Missing or oversized file length: '+name)
        total+=size
        if total>MAX_TOTAL:raise DownloadError('Repository exceeds 200 GiB.')
        digest=str(lfs.get('sha256') or lfs.get('oid') or '').removeprefix('sha256:').lower()
        if digest and not re.fullmatch('[a-f0-9]{64}',digest):raise DownloadError('Invalid file checksum in source metadata.')
        if name.endswith('.safetensors') and not digest:raise DownloadError('Weights must have a source SHA-256: '+name)
        files.append({'name':name,'size':size,'sha256':digest,'url':base+quote(name,safe='/')})
    if not files or len(files)>1000 or 'model_index.json' not in {x['name'] for x in files}:raise DownloadError('Incomplete Qwen repository listing.')
    plan={'id':uuid.uuid4().hex,'expires':time.time()+1800,'kind':'repository','kind_hint':'repository','provider':'huggingface',
        'repo':repo,'revision':sha,'source':'https://huggingface.co/'+repo,'filename':repo.split('/')[-1],
        'size':total,'sha256':'','license':'Terms of this pinned model revision; verify before use.',
        'license_url':'https://huggingface.co/'+repo+'/tree/'+sha,'file_count':len(files),'files':files,
        'exclude_transformer':bool(exclude_transformer),'builtin':False}
    with dl.lock:
        dl.plans={k:v for k,v in dl.plans.items() if v['expires']>time.time()}
        if len(dl.plans)>=40:raise DownloadError('Too many open download previews.')
        dl.plans[plan['id']]=plan
    return dl.preview(plan)

def run_repository(dl,jid,plan):
    stage=dl.incoming/(jid+'.repository');cancel=dl.events[jid]
    try:
        if cancel.is_set():raise Cancelled('Download cancelled.')
        if stage.exists() or stage.is_symlink():raise DownloadError('Repository staging already exists; inspect the previous attempt.')
        if shutil.disk_usage(dl.root).free<plan['size']+512*1024**2:raise DownloadError('Insufficient disk space for the repository.')
        stage.mkdir(mode=0o700);done=0;last=0
        dl._update(jid,status='downloading',file_count=len(plan['files']),completed_files=0,total_bytes=plan['size'])
        for count,item in enumerate(plan['files']):
            if cancel.is_set():raise Cancelled('Download cancelled.')
            from core.qwen_component_download import allowed_component_path
            allowed=allowed_component_path(item['name'],plan['component_kind']) if plan.get('component_kind') else allowed_path(item['name'],plan.get('exclude_transformer',False))
            if not allowed:raise DownloadError('Unexpected repository/component file.')
            target=stage/item['name'];target.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256();received=0
            dl._update(jid,current_file=item['name'],completed_files=count)
            with dl.transport.open(item['url'],cancel=cancel) as response, target.open('xb') as fp:
                for chunk in response.chunks():
                    if cancel.is_set():raise Cancelled('Download cancelled.')
                    received+=len(chunk);done+=len(chunk)
                    if received>item['size'] or received>MAX_FILE or done>plan['size']:raise DownloadError('Repository file exceeds its declared size.')
                    if shutil.disk_usage(dl.root).free<len(chunk)+64*1024**2:raise DownloadError('Insufficient disk space.')
                    fp.write(chunk);digest.update(chunk)
                    if time.monotonic()-last>.6:
                        dl._update(jid,downloaded_bytes=done);last=time.monotonic()
                fp.flush();os.fsync(fp.fileno())
            if received!=item['size']:raise DownloadError('Truncated repository file: '+item['name'])
            if item['sha256'] and digest.hexdigest()!=item['sha256']:raise DownloadError('Checksum mismatch: '+item['name'])
            if target.suffix=='.json':
                if target.stat().st_size>8*1024**2:raise DownloadError('Oversized JSON configuration.')
                try:json.loads(target.read_text())
                except (ValueError,UnicodeError):raise DownloadError('Invalid JSON configuration: '+item['name']) from None
        dl._update(jid,status='validating',downloaded_bytes=done,completed_files=len(plan['files']))
        if plan.get('component_kind'):
            from core.qwen_component_download import validate_bundle
            check=validate_bundle(stage,plan['component_kind'])
        else:check=repository_status(stage)
        missing=set(check.get('missing',[]))
        if not check['ready'] and not (plan.get('exclude_transformer') and missing=={'transformer/weights'}):
            raise DownloadError('Incomplete repository: '+', '.join(sorted(missing)))
        if cancel.is_set():raise Cancelled('Download cancelled.')
        destroot=dl.root/('Qwen-Components/'+plan['component_kind'] if plan.get('component_kind') else 'Qwen-Repositories')
        if not destroot.resolve().is_relative_to(dl.root.resolve()):raise DownloadError('Refusing an external linked repository destination.')
        destroot.mkdir(parents=True,exist_ok=True)
        name=re.sub('[^A-Za-z0-9_.-]','_',plan['filename'])+'-'+plan['revision'][:8]+'-'+jid[:8]
        dest=destroot/name
        if dest.exists() or dest.is_symlink():raise DownloadError('Repository destination already exists.')
        dl._update(jid,status='installing');os.rename(stage,dest)
        atomic_json(dest/'.image-lab-source.json',{'source':plan['source'],'revision':plan['revision'],'terms_confirmed':True,
            'files':[{'name':x['name'],'sha256':x['sha256']} for x in plan['files']],'exclude_transformer':plan.get('exclude_transformer',False)})
        warning='Select this repository in My models. Add a transformer override to use GGUF.' if plan.get('exclude_transformer') else 'Select this repository in My models to use it.'
        if plan.get('component_kind'):warning='Component downloaded and catalogued; select it explicitly in Models → Components. Licence terms still apply.'
        try:dl.on_installed(plan.get('component_kind','repository'),dest)
        except Exception:warning+=' Catalog registration failed; link its server path manually.'
        return dl._update(jid,status='complete',path=str(dest),kind=plan.get('component_kind','repository'),downloaded_bytes=done,total_bytes=done,error='',warning=warning)
    except Exception as exc:
        if stage.is_dir() and not stage.is_symlink():shutil.rmtree(stage)
        message=str(exc) if isinstance(exc,DownloadError) else 'Repository download/validation failed. No model was activated.'
        return dl._update(jid,status='cancelled' if isinstance(exc,Cancelled) else 'failed',error=message)
