"""Reproducible benchmark plans executed by the existing serial application queue.

Plans snapshot configuration, prompts and inputs before confirmation. Deterministic
job IDs and one database transaction make repeated confirmation idempotent.
"""
from __future__ import annotations
import copy,csv,hashlib,io,json,secrets,time,uuid
from pathlib import Path
from core.db import con
from core.source_util import atomic_json
from core.qwen_profiles import ModelProfiles
from core.qwen_assets import read_object
from core.qwen_studio import resolve_request,catalog_entry,PATH_KINDS
from core.qwen_request import validate_params,effective_prompt,auto_tool
from core.qwen_components import component_status,decode_options,assert_same_latent_space
from core.qwen_external_runtime import ComfyRuntimeStore

def _quantized_safetensors(path):
    """Return whether an existing safetensors component uses the reviewed quantized layout."""
    raw=str(path or '')
    if not raw.lower().endswith('.safetensors'):return False
    p=Path(raw).expanduser()
    if not p.is_file():return False
    try:
        from core.qwen_weight_format import inspect_weights
        return bool(inspect_weights(p).get('quantized'))
    except (OSError,ValueError):
        return False


def _diffusers_encoder_issue(cfg):
    """Cheap interface check for a single-file encoder before a large batch is queued.

    The native Qwen 2.1 Diffusers pipeline uses the baseline text-encoder config.
    A 4B/other encoder file can be a perfectly valid Qwen3-VL checkpoint while
    still being incompatible with that exact embedding interface.  Inspect only
    the embedding tensor shape; the worker retains the strict full-key check.
    """
    raw=str(cfg.get('text_encoder_path') or '')
    if not raw.lower().endswith('.safetensors'):return ''
    p=Path(raw).expanduser();cp=Path(str(cfg.get('model_dir') or ''))/'text_encoder/config.json'
    if not p.is_file() or not cp.is_file():return ''
    try:
        ref=read_object(cp);tc=ref.get('text_config') or {}
        vocab=tc.get('vocab_size');hidden=tc.get('hidden_size')
        if not isinstance(vocab,int) or not isinstance(hidden,int):return ''
        # The web runtime intentionally does not depend on the ML safetensors
        # package.  Read only the small safetensors JSON header here so opening
        # old benchmark reports can never fail because an ML runtime dependency
        # is absent from the app/web environment.
        with p.open('rb') as fh:
            raw_len=fh.read(8)
            if len(raw_len)!=8:return ''
            header_len=int.from_bytes(raw_len,'little',signed=False)
            if header_len<=0 or header_len>64*1024*1024:return ''
            header=json.loads(fh.read(header_len).decode('utf-8'))
        key=next((k for k in ('model.language_model.embed_tokens.weight','model.embed_tokens.weight','language_model.embed_tokens.weight') if k in header),None)
        if not key:return ''
        got=tuple(header[key].get('shape') or ());expected=(vocab,hidden)
        if got!=expected:
            return (f'Text Encoder interface mismatch for Diffusers Qwen 2.1: {Path(raw).name} has '
                    f'embed_tokens {got}, baseline expects {expected}. This is a component incompatibility, not a generation failure.')
    except (OSError,ValueError,TypeError,UnicodeDecodeError,json.JSONDecodeError):
        return ''
    return ''


def _resolve_benchmark_backend(cfg):
    """Resolve one benchmark combination without silently changing its components.

    Native ComfyUI is mandatory for GGUF/ConvRot.  A fully explicit trio can
    also stay on Native ComfyUI for apples-to-apples BF16/native comparisons.
    Baseline repository components remain Diffusers; they are never substituted
    with a different catalog file just to satisfy the native bridge.
    """
    model=str(cfg.get('model_path') or '');encoder=str(cfg.get('text_encoder_path') or '');vae=str(cfg.get('vae_path') or '')
    policy=str(cfg.get('quantized_loading') or 'reject')
    gguf=model.lower().endswith('.gguf')
    qmodel=_quantized_safetensors(model);qencoder=_quantized_safetensors(encoder)
    native_required=gguf or qmodel or qencoder
    explicit_model=bool(model and (model.lower().endswith('.gguf') or model.lower().endswith('.safetensors')))
    explicit_encoder=encoder.lower().endswith('.safetensors')
    explicit_vae=vae.lower().endswith('.safetensors')
    explicit_trio=bool(explicit_model and explicit_encoder and explicit_vae)
    if native_required or (policy=='native_comfyui' and explicit_trio):
        cfg['quantized_loading']='native_comfyui'
        missing=[]
        if not explicit_model:missing.append('Transformer')
        if not explicit_encoder:missing.append('Text Encoder')
        if not explicit_vae:missing.append('VAE')
        if missing:
            return 'incompatible',('Native ComfyUI quantized execution needs explicit single-file companions: '+', '.join(missing)+'. Baseline repository components are not silently replaced.')
        return 'native_comfyui',''
    # A ComfyUI-layout single-file VAE can still be tested cleanly with baseline
    # Transformer/Text Encoder without substituting those companions.  Sampling
    # remains in the baseline Diffusers pipeline until the exact upstream
    # vae.decode input boundary; only that captured latent is decoded by the
    # selected VAE in the verified external ComfyUI runtime.
    if vae.lower().endswith('.safetensors') and policy=='native_comfyui' and not explicit_trio and not model and not encoder:
        cfg['quantized_loading']='reject'
        cfg['_benchmark_vae_backend']='comfyui_hybrid'
        return 'hybrid_comfy_vae',''
    # "native_comfyui" is a preference, not a reason to force an otherwise
    # baseline/non-quantized job through the custom single-file path.
    if policy=='native_comfyui':cfg['quantized_loading']='reject'
    issue=_diffusers_encoder_issue(cfg)
    if issue:return 'incompatible',issue
    return 'diffusers',''


def _benchmark_lora_variants(data, assets, current):
    """Freeze selected catalog LoRAs and their strengths into reproducible variants.

    With no selected adapter the legacy profile-LoRA behavior is unchanged.
    Otherwise the test matrix starts with a *true* no-LoRA baseline, then
    distinct individual adapters (or explicit bounded subsets).
    """
    import itertools
    import math
    ids = data.get('lora_ids') or []
    strengths = data.get('lora_strengths') or {}
    mode = data.get('lora_combinations', 'single')
    if not isinstance(ids, list) or len(ids) > 8 or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Select at most 8 distinct registered LoRA adapters.')
    if not isinstance(strengths, dict) or any(not isinstance(k, str) or k not in ids for k in strengths):
        raise ValueError('LoRA strengths must refer to the selected adapters only.')
    if mode not in ('single', 'all'):
        raise ValueError('Invalid LoRA combination mode.')
    if mode == 'all' and len(ids) > 4:
        raise ValueError('All LoRA subsets require at most 4 selected adapters (runtime limit).')
    if not ids:
        if strengths:raise ValueError('LoRA strengths supplied without selected adapters.')
        return None
    items = []
    for rid in ids:
        entry = catalog_entry(assets, current, rid, 'lora')
        path = Path(entry['path'])
        if not path.is_file() or path.suffix.lower() != '.safetensors':
            raise ValueError('LoRA catalog entry must resolve to a readable safetensors file.')
        raw = strengths.get(rid, 1.0)
        if type(raw) not in (int, float) or not math.isfinite(raw) or not -2 <= raw <= 2:
            raise ValueError('LoRA strength must be finite and between -2 and 2.')
        items.append((str(entry['name']), {'path': str(path), 'strength': float(raw)}))
    variants = [('No LoRA', [])]
    sizes = range(1, len(items) + 1) if mode == 'all' else (1,)
    for n in sizes:
        for chosen in itertools.combinations(items, n):
            label = ' + '.join(f'{name} @{value["strength"]:g}' for name, value in chosen)
            variants.append((label, [copy.deepcopy(value) for _, value in chosen]))
    return variants


class BenchmarkStore:
    def __init__(self,root,library,assets,current,cb,latents):
        self.files=ModelProfiles(root);self.library=library;self.assets=assets;self.current=current;self.cb=cb;self.latents=latents
    def get(self,rid):
        data=read_object(self.files._path(rid))
        if data.get('id')!=rid or data.get('schema')!=1:raise ValueError('Invalid benchmark report.')
        return data
    def list(self,pid):
        self.project(pid);out=[];self.files._safe_root()
        if self.files.root.exists():
            for p in self.files.root.glob('*.json'):
                if p.is_symlink():continue
                try:
                    d=read_object(p)
                    if d.get('project_id')==pid:out.append({k:d.get(k) for k in ('id','name','created_at','mode','count','seed')})
                except ValueError:continue
        return sorted(out,key=lambda d:d['created_at'],reverse=True)[:100]
    def _project_names(self):
        return {r['id']:(r.get('name') or r['id']) for r in self.cb['rows'](
            'SELECT id,name FROM projects WHERE COALESCE(trashed,0)=0 ORDER BY sort_order,created_at,id')}
    def _job_map(self,ids):
        out={};values=list(dict.fromkeys(x for x in ids if isinstance(x,str)))
        for start in range(0,len(values),400):
            part=values[start:start+400]
            if not part:continue
            marks=','.join('?' for _ in part)
            for row in self.cb['rows']('SELECT id,project_id,status,phase,progress,output_artifact_id,updated_at,completed_at FROM jobs WHERE id IN ('+marks+')',tuple(part)):
                out[row['id']]=row
        return out
    def _visibility_file(self,rid):
        # Only benchmark ids from the existing validated store become filenames.
        self.get(rid)
        return self.files.root.parent/'qwen21-visibility'/(rid+'.json')
    def visibility(self,rid):
        data=self.get(rid);self.project(data['project_id'])
        path=self._visibility_file(rid)
        if not path.is_file():return []
        if path.is_symlink():raise ValueError('Invalid visibility metadata path.')
        try:
            value=read_object(path)
            if value.get('benchmark_id')!=rid:return []
            allowed={x['job_id'] for x in data.get('entries') or []}
            return sorted(x for x in value.get('hidden',[]) if isinstance(x,str) and x in allowed)
        except (ValueError,TypeError,KeyError):return []
    def set_visibility(self,rid,job_id,hidden):
        if type(hidden) is not bool:raise ValueError('hidden must be a boolean.')
        with self.files._lock():
            d=self.get(rid);self.project(d['project_id'])
            valid={x.get('job_id') for x in d.get('entries') or []}
            if job_id not in valid:raise ValueError('Result does not belong to this benchmark.')
            path=self._visibility_file(rid)
            if path.is_symlink():raise ValueError('Invalid visibility metadata path.')
            values=set(self.visibility(rid))
            if hidden:values.add(job_id)
            else:values.discard(job_id)
            atomic_json(path,{'schema':1,'benchmark_id':rid,'hidden':sorted(values)})
            return {'benchmark_id':rid,'job_id':job_id,'hidden':hidden,'count':len(values)}
    def archive(self,pid=None):
        if pid:self.project(pid)
        projects=self._project_names();out=[];self.files._safe_root()
        if not self.files.root.exists():return []
        for path in self.files.root.glob('*.json'):
            if path.is_symlink():continue
            try:
                d=read_object(path)
                if d.get('schema')!=1 or d.get('id')!=path.stem:continue
                project_id=d.get('project_id')
                if project_id not in projects or (pid and project_id!=pid):continue
                entries=d.get('entries') or [];jobs=self._job_map([x.get('job_id') for x in entries])
                counts={};images=0;active=0
                prompts=set();resolutions=set()
                for entry in entries:
                    row=jobs.get(entry.get('job_id')) or {};status=row.get('status') or 'planned'
                    counts[status]=counts.get(status,0)+1
                    if status in ('queued','running','cancelling'):active+=1
                    if row.get('output_artifact_id'):images+=1
                    prompt=str(entry.get('prompt') or '').strip()
                    if prompt:prompts.add(prompt)
                    params=entry.get('params') or {};w=params.get('width');h=params.get('height')
                    if isinstance(w,int) and isinstance(h,int):resolutions.add(f'{w}×{h}')
                out.append({'id':d['id'],'name':d.get('name') or 'Qwen benchmark','project_id':project_id,
                    'project_name':projects[project_id],'created_at':d.get('created_at',0),'mode':d.get('mode'),
                    'count':len(entries),'seed':d.get('seed'),'status_counts':counts,'image_count':images,
                    'active_count':active,'prompt_count':len(prompts),'resolutions':sorted(resolutions),
                    'complete':bool(entries) and not active and all((jobs.get(x.get('job_id')) or {}).get('status') in ('completed','failed','interrupted','cancelled') for x in entries)})
            except (ValueError,OSError,KeyError,TypeError):continue
        return sorted(out,key=lambda d:(d.get('created_at') or 0,d['id']),reverse=True)
    def rename(self,rid,name):
        if not isinstance(name,str) or not name.strip():raise ValueError('Enter a benchmark name.')
        name=name.strip()
        if len(name)>120:raise ValueError('Benchmark name is limited to 120 characters.')
        with self.files._lock():
            data=self.get(rid);self.project(data['project_id']);data['name']=name;data['updated_at']=time.time()
            atomic_json(self.files._path(rid),data)
        return {'id':rid,'name':name}
    def comparison(self,ids):
        if (not isinstance(ids,list) or not ids or len(ids)>100 or any(not isinstance(x,str) for x in ids)
                or len(set(ids))!=len(ids)):
            raise ValueError('Choose 1–100 different benchmarks to compare.')
        projects=self._project_names();benchmarks=[];items=[]
        for rid in ids:
            data=self.report(rid);project_id=data.get('project_id')
            if project_id not in projects:continue
            hidden_jobs=set(self.visibility(rid))
            benchmarks.append({'id':rid,'name':data.get('name'),'project_id':project_id,
                'project_name':projects[project_id],'created_at':data.get('created_at'),'mode':data.get('mode')})
            for entry in data.get('entries') or []:
                params=entry.get('params') or {};prompt=str(entry.get('prompt') or '').strip();compact=' '.join(prompt.split())
                prompt_key=hashlib.sha256(compact.encode('utf-8')).hexdigest()[:16] if compact else 'no-prompt'
                prompt_title=str(entry.get('prompt_name') or '').strip()
                if not prompt_title:
                    prompt_title=(compact[:94]+'…') if len(compact)>95 else (compact or data.get('name') or 'Benchmark example')
                components=entry.get('components') or {}
                settings={k:params.get(k) for k in ('width','height','steps','cfg','cfg_enabled','negative_prompt','seed','transparent_output')}
                # Duplicate means same exact reproducible generation settings, not identical filename/image bytes.
                cfg=params.get('_model_config_snapshot') or {}
                signature={'prompt':prompt,'parameters':{k:params.get(k) for k in (
                    'width','height','steps','cfg','cfg_enabled','negative_prompt','seed',
                    'transparent_output','reference_artifact_ids','source_artifact_id','loras',
                    'vae_decode_mode','decode_device','vae_dtype')},
                    'components':{k:cfg.get(k) for k in ('model_dir','model_path','text_encoder_path',
                    'text_encoder_dir','vae_path','vae_config_path','scheduler_dir','quantized_loading','dtype','memory_mode')}}
                fingerprint=hashlib.sha256(json.dumps(signature,sort_keys=True,default=str).encode()).hexdigest()[:24]
                def component(key,baseline):
                    value=str(components.get(key) or '')
                    return baseline if not value or value=='Original' else value
                transformer=component('model_path','Baseline transformer')
                encoder=component('text_encoder_path','Baseline encoder')
                vae=component('vae_path','Baseline VAE')
                items.append({'benchmark_id':rid,'benchmark_name':data.get('name'),'project_id':project_id,
                    'project_name':projects[project_id],'created_at':data.get('created_at'),'job_id':entry.get('job_id'),
                    'label':entry.get('label'),'prompt_key':prompt_key,'prompt_title':prompt_title,'prompt':prompt,
                    'status':entry.get('status'),'raw_status':entry.get('raw_status'),'url':entry.get('url'),
                    'artifact_id':entry.get('artifact_id'),'duration_seconds':entry.get('duration_seconds'),
                    'components':components,'transformer':transformer,'encoder':encoder,'vae':vae,
                    'configuration':{'baseline_repository':Path(str(cfg.get('model_dir') or '')).name or '—',
                        'dtype':cfg.get('dtype') or '', 'memory_mode':cfg.get('memory_mode') or ''},
                    'settings':settings,'lora_summary':', '.join(Path(x['path']).name+f' @{x.get("strength",1):g}' for x in (params.get('loras') or [])) or 'No LoRA','metrics':entry.get('metrics') or {},'result':entry.get('result') or {},
                    'fingerprint':fingerprint,'hidden':entry.get('job_id') in hidden_jobs,
                    'compatibility_reason':entry.get('compatibility_reason') or '','error_present':bool(entry.get('error'))})
        return {'benchmarks':benchmarks,'items':items,'count':len(items)}
    def project(self,pid):
        if not isinstance(pid,str) or not self.cb['one']('SELECT id FROM projects WHERE id=? AND COALESCE(trashed,0)=0',(pid,)):raise ValueError('Choose an active project.')
    def image(self,pid,aid):
        row=self.cb['one']("SELECT id,path FROM artifacts WHERE id=? AND project_id=? AND type='image'",(aid,pid))
        if not row or not Path(row['path']).is_file():raise ValueError('Choose an existing image in this project.')
        p=Path(row['path']);h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        return {'id':aid,'sha256':h.hexdigest(),'bytes':p.stat().st_size}
    def preflight(self,entries):
        if not isinstance(entries,list):raise ValueError('Invalid benchmark plan.')
        runtime=ComfyRuntimeStore(self.files.root.parent.parent).public()
        caps=runtime.get('capabilities') or {};verified=bool(runtime.get('verified'))
        native_jobs=[];gguf_jobs=[];convrot_jobs=[];hybrid_vae_jobs=[];blocked=[]
        from core.qwen_weight_format import inspect_weights
        for entry in entries:
            cfg=((entry.get('params') or {}).get('_model_config_snapshot') or {})
            hybrid_vae=cfg.get('_benchmark_vae_backend')=='comfyui_hybrid'
            if cfg.get('quantized_loading')!='native_comfyui' and not hybrid_vae:continue
            model=str(cfg.get('model_path') or '');encoder=str(cfg.get('text_encoder_path') or '');vae=str(cfg.get('vae_path') or '')
            if hybrid_vae:
                hybrid_vae_jobs.append(entry)
                reasons=[]
                if not verified:reasons.append('global ComfyUI runtime is not verified')
                if verified and not caps.get('cuda_available'):reasons.append('CUDA is unavailable in the verified ComfyUI runtime')
                if reasons:blocked.append({'job_id':entry.get('job_id'),'label':entry.get('label'),'reason':'; '.join(reasons)})
                continue
            native=False;convrot=False;gguf=model.lower().endswith('.gguf')
            if gguf:native=True
            single=all(x.lower().endswith('.safetensors') for x in (model,encoder,vae) if x) and all((model,encoder,vae))
            if single:native=True
            for raw in (model,encoder):
                if not raw.lower().endswith('.safetensors'):continue
                try:
                    if Path(raw).is_file() and inspect_weights(Path(raw)).get('quantized'):native=True;convrot=True
                except (OSError,ValueError):pass
            if not native:continue
            native_jobs.append(entry);gguf_jobs.extend([entry] if gguf else []);convrot_jobs.extend([entry] if convrot else [])
            reasons=[]
            if not verified:reasons.append('global ComfyUI runtime is not verified')
            if verified and not caps.get('cuda_available'):reasons.append('CUDA is unavailable in the verified ComfyUI runtime')
            if verified and not caps.get('qwen_image_nodes'):reasons.append('Qwen Image native nodes are unavailable')
            if verified and not caps.get('native_qwen_loader'):reasons.append('native Qwen loader is unavailable')
            if gguf and verified and not caps.get('gguf_loader'):reasons.append('ComfyUI-GGUF / UnetLoaderGGUF is unavailable')
            if gguf and verified and caps.get('gguf_loader') and not caps.get('gguf_qwen21'):reasons.append('ComfyUI-GGUF does not advertise qwen_image21 support')
            if reasons:blocked.append({'job_id':entry.get('job_id'),'label':entry.get('label'),'reason':'; '.join(reasons)})
        total=len(entries);ready=total-len(blocked)
        checks=[{'key':'components','label':'Selected components','required':True,'ok':True,'detail':'Model/component files passed plan validation.'}]
        if native_jobs:
            checks.extend([
                {'key':'comfy_runtime','label':'Global ComfyUI runtime','required':True,'ok':verified,'detail':'Verified in Settings → External runtimes.' if verified else 'Configure, trust and verify ComfyUI in Settings → External runtimes.'},
                {'key':'cuda','label':'CUDA in ComfyUI','required':True,'ok':bool(verified and caps.get('cuda_available')),'detail':caps.get('gpu') or 'CUDA must be available for native benchmark jobs.'},
                {'key':'qwen_nodes','label':'Qwen Image native nodes','required':True,'ok':bool(verified and caps.get('qwen_image_nodes') and caps.get('native_qwen_loader')),'detail':'CLIPLoader qwen_image, TextEncodeQwenImage21 and UNETLoader.'},
            ])
        if convrot_jobs:checks.append({'key':'convrot','label':'Native ConvRot','required':True,'ok':bool(verified and caps.get('native_qwen_loader') and caps.get('qwen_image_nodes')),'detail':str(len(convrot_jobs))+' planned job(s) use quantized safetensors.'})
        if gguf_jobs:checks.append({'key':'gguf','label':'ComfyUI-GGUF','required':True,'ok':bool(verified and caps.get('gguf_loader') and caps.get('gguf_qwen21')),'detail':str(len(gguf_jobs))+' planned job(s) require UnetLoaderGGUF + qwen_image21. '+str(caps.get('gguf_qwen21_detail') or '')})
        if hybrid_vae_jobs:checks.append({'key':'hybrid_vae','label':'ComfyUI VAE decode','required':True,'ok':bool(verified and caps.get('cuda_available')),'detail':str(len(hybrid_vae_jobs))+' planned job(s) keep baseline Diffusers sampling and decode the captured latent with the selected single-file VAE in the verified ComfyUI runtime.'})
        return {'ready':not blocked,'jobs_total':total,'jobs_ready':ready,'native_jobs':len(native_jobs),'hybrid_vae_jobs':len(hybrid_vae_jobs),'blocked_jobs':blocked[:100],'checks':checks,'runtime_verified':verified}
    def preflight_report(self,rid):
        data=self.get(rid);self.project(data['project_id']);return self.preflight(data.get('entries') or [])
    def plan(self,data):
        if not isinstance(data,dict):raise ValueError('Expected a benchmark request.')
        pid=data.get('project_id');self.project(pid);mode=data.get('mode','compare')
        if mode not in ('compare','transformer','encoder','matrix','auto','decode','roundtrip'):raise ValueError('Invalid benchmark mode.')
        count=data.get('repeats',1)
        if type(count) is not int or not 1<=count<=3:raise ValueError('Use 1–3 repetitions.')
        params=validate_params(data.get('params') or {});params['batch']=1
        params['qwen21_transformer_mode']='configured';params['qwen21_transformer_path']=''
        seed=params['seed'] if params['seed']>=0 else secrets.randbelow(2**31);params['seed']=seed
        params['retain_latents']=bool(params.get('retain_latents',True))
        snapshots=[self.image(pid,x) for x in params['reference_artifact_ids']]
        if type(data.get('use_profile_loras',False)) is not bool:raise ValueError('Invalid LoRA selection.')
        selected_lora_variants=_benchmark_lora_variants(data,self.assets,self.current())
        if mode in ('decode','roundtrip') and selected_lora_variants is not None:
            raise ValueError('LoRA compares denoising/sampling. Use a generation benchmark rather than a decode-only test.')
        profile_ids=data.get('profile_ids') or ['@configured']
        if not isinstance(profile_ids,list) or not 1<=len(profile_ids)<=12 or len(set(profile_ids))!=len(profile_ids):raise ValueError('Choose 1–12 different model presets.')
        setups=[];current=self.current();shared_options=data.get('load_options') or {}
        for rid in profile_ids:
            cfg,loras,label=resolve_request({'model_preset_id':rid,'load_options':shared_options},self.library,self.assets,current)
            setups.append((label['preset_name'],cfg,loras))

        def apply_component(cfg,choice,field,kind):
            if choice not in ('@baseline','@original') and not isinstance(choice,str):raise ValueError('Invalid benchmark component choice.')
            if choice=='@baseline':return 'Baseline '+kind
            if choice=='@original':
                cfg[field]=''
                if field=='text_encoder_path':cfg['text_encoder_dir']=''
                if field=='vae_path':cfg['vae_config_path']=''
                return 'Original '+kind
            entry=catalog_entry(self.assets,current,choice,kind);cfg[field]=entry['path']
            if field=='text_encoder_path':cfg['text_encoder_dir']=''
            if field=='vae_path':
                cfg['vae_config_path']='';vp=Path(entry['path'])
                if vp.is_file():
                    local=vp.parent/'config.json';original=Path(str(cfg.get('model_dir') or ''))/'vae/config.json'
                    if local.is_file():cfg['vae_config_path']=str(local)
                    elif original.is_file():cfg['vae_config_path']=str(original)
                    else:raise ValueError('Single-file VAE needs a matching config.json or the baseline Qwen 2.1 VAE config.')
            return entry['name']

        if mode=='compare':
            from core.qwen_components import component_location
            reference=setups[0][1]
            for flag in ('lock_vae','lock_scheduler'):
                if type(data.get(flag,True)) is not bool:raise ValueError('Invalid component lock.')
            for _,cfg,_ in setups:
                if data.get('lock_vae',True):
                    cfg['vae_path']=str(component_location(reference,'vae'));cfg['vae_config_path']=reference.get('vae_config_path','')
                if data.get('lock_scheduler',True):
                    cfg['scheduler_dir']=str(component_location(reference,'scheduler'));cfg['scheduler_shift']=reference.get('scheduler_shift')

        if mode in ('transformer','encoder','matrix','auto'):
            if len(setups)!=1:raise ValueError('Component batch tests use exactly one baseline model preset.')
            base=setups[0];fixed_vae=data.get('fixed_vae','@baseline')
            if fixed_vae not in ('@baseline','@original') and not isinstance(fixed_vae,str):raise ValueError('Invalid fixed VAE.')
            if mode=='transformer':
                ts=data.get('transformers') or []
                if not isinstance(ts,list) or not 1<=len(ts)<=12 or len(set(ts))!=len(ts):raise ValueError('Choose 1–12 different transformer variants.')
                fixed_encoder=data.get('fixed_encoder','@baseline');setups=[]
                for t in ts:
                    c=copy.deepcopy(base[1]);apply_component(c,fixed_encoder,'text_encoder_path','text_encoder_file');apply_component(c,fixed_vae,'vae_path','vae')
                    label=apply_component(c,t,'model_path','transformer');setups.append((label,c,copy.deepcopy(base[2])))
            elif mode=='encoder':
                es=data.get('encoders') or []
                if not isinstance(es,list) or not 1<=len(es)<=12 or len(set(es))!=len(es):raise ValueError('Choose 1–12 different encoder variants.')
                fixed_transformer=data.get('fixed_transformer','@baseline');setups=[]
                for e in es:
                    c=copy.deepcopy(base[1]);apply_component(c,fixed_transformer,'model_path','transformer');apply_component(c,fixed_vae,'vae_path','vae')
                    label=apply_component(c,e,'text_encoder_path','text_encoder_file');setups.append((label,c,copy.deepcopy(base[2])))
            elif mode=='matrix':
                ts=data.get('transformers') or [];es=data.get('encoders') or []
                if not isinstance(ts,list) or not isinstance(es,list) or not ts or not es or len(ts)*len(es)>24:raise ValueError('Choose transformer and encoder variants (maximum 24 combinations).')
                if len(set(ts))!=len(ts) or len(set(es))!=len(es):raise ValueError('Duplicate matrix components.')
                setups=[]
                for t in ts:
                    for e in es:
                        c=copy.deepcopy(base[1]);apply_component(c,fixed_vae,'vae_path','vae')
                        tn=apply_component(c,t,'model_path','transformer');en=apply_component(c,e,'text_encoder_path','text_encoder_file')
                        setups.append((tn+' × '+en,c,copy.deepcopy(base[2])))
            else:
                ts=data.get('transformers') or [];es=data.get('encoders') or [];vs=data.get('vaes') or []
                for values,label in ((ts,'transformer'),(es,'encoder'),(vs,'VAE')):
                    if not isinstance(values,list) or not values or len(values)>12 or len(set(values))!=len(values):raise ValueError('Choose 1–12 different '+label+' variants.')
                strategy=data.get('auto_strategy','full')
                if strategy not in ('full','screening'):raise ValueError('Invalid automatic benchmark strategy.')
                setups=[]
                def configured(t,e,v,prefix=''):
                    c=copy.deepcopy(base[1]);tn=apply_component(c,t,'model_path','transformer');en=apply_component(c,e,'text_encoder_path','text_encoder_file');vn=apply_component(c,v,'vae_path','vae')
                    return ((prefix+' · ' if prefix else '')+tn+' × '+en+' × '+vn,c,copy.deepcopy(base[2]))
                # Native ConvRot uses ComfyUI's single-file loaders. A repository
                # baseline is sharded and cannot be passed to those loaders, so a
                # one-variable screening needs explicit BF16 single-file companions.
                # Pick them only from components the user already selected; never
                # scan the disk or silently download/guess another model.
                native_mode=shared_options.get('quantized_loading')=='native_comfyui'
                def explicit_choice(values,kind,need_float=False):
                    if not native_mode:return None
                    from core.qwen_weight_format import inspect_weights
                    for choice in values:
                        if choice in ('@baseline','@original'):continue
                        try:entry=catalog_entry(self.assets,current,choice,kind)
                        except ValueError:continue
                        path=Path(entry['path'])
                        if not (path.is_file() and path.suffix.lower()=='.safetensors'):continue
                        if kind in ('transformer','text_encoder_file') and need_float:
                            try:
                                if inspect_weights(path).get('quantized'):continue
                            except (OSError,ValueError):continue
                        return choice
                    return None
                def is_native_quantized_choice(choice,kind):
                    if not native_mode or choice in ('@baseline','@original'):return False
                    from core.qwen_weight_format import inspect_weights
                    try:
                        entry=catalog_entry(self.assets,current,choice,kind);path=Path(entry['path'])
                        if kind=='transformer' and path.suffix.lower()=='.gguf':return True
                        return path.suffix.lower()=='.safetensors' and bool(inspect_weights(path).get('quantized'))
                    except (ValueError,OSError):return False
                native_t=explicit_choice(ts,'transformer',True);native_e=explicit_choice(es,'text_encoder_file',True);native_v=explicit_choice(vs,'vae',False)
                if strategy=='full':
                    if len(ts)*len(es)*len(vs)>128:raise ValueError('Automatic full matrix is limited to 128 component combinations before prompts/repetitions.')
                    for t in ts:
                        for e in es:
                            for v in vs:setups.append(configured(t,e,v))
                else:
                    setups.append(configured('@baseline','@baseline','@baseline','Baseline'))
                    for t in ts:
                        if t=='@baseline':continue
                        if is_native_quantized_choice(t,'transformer'):
                            if not native_e or not native_v:raise ValueError('Native ConvRot/GGUF Transformer screening needs a selected non-quantized single-file Text Encoder and a selected single-file VAE as explicit ComfyUI companions.')
                            setups.append(configured(t,native_e,native_v,'Transformer native'))
                        else:setups.append(configured(t,'@baseline','@baseline','Transformer'))
                    for e in es:
                        if e=='@baseline':continue
                        if is_native_quantized_choice(e,'text_encoder_file'):
                            if not native_t or not native_v:raise ValueError('Native ConvRot Encoder screening needs a selected non-quantized single-file Transformer and a selected single-file VAE as explicit ComfyUI companions.')
                            setups.append(configured(native_t,e,native_v,'Encoder native'))
                        else:setups.append(configured('@baseline',e,'@baseline','Encoder'))
                    setups.extend(configured('@baseline','@baseline',v,'VAE') for v in vs if v!='@baseline')

        skipped_native=[]
        if mode in ('compare','transformer','encoder','matrix','auto'):
            compatible=[]
            for item in setups:
                backend,issue=_resolve_benchmark_backend(item[1])
                if not issue:
                    compatible.append(item);continue
                skipped_native.append({'label':item[0],'reason':issue,'backend':'incompatible'})
            setups=compatible
            if not setups:
                reason=(skipped_native[0]['reason'] if skipped_native else 'No compatible component combination remains.')
                raise ValueError('No compatible benchmark combinations remain. '+reason)

        if mode in ('decode','roundtrip') and data.get('vaes') is not None:
            if len(setups)!=1:raise ValueError('VAE batch tests use exactly one baseline model preset.')
            vs=data.get('vaes') or []
            if not isinstance(vs,list) or not 1<=len(vs)<=12 or len(set(vs))!=len(vs):raise ValueError('Choose 1–12 different VAE variants.')
            base=setups[0];setups=[]
            for v in vs:
                c=copy.deepcopy(base[1]);label=apply_component(c,v,'vae_path','vae');setups.append((label,c,copy.deepcopy(base[2])))
        prompt=data.get('prompt','');effective='';latent=None;source=None;prompt_variants=[]
        if mode=='auto':
            raw=data.get('prompts') or []
            if not isinstance(raw,list) or not 1<=len(raw)<=8:raise ValueError('Choose 1–8 benchmark prompts.')
            seen=set()
            for i,item in enumerate(raw,1):
                if not isinstance(item,dict):raise ValueError('Invalid benchmark prompt.')
                text=str(item.get('prompt') or '').strip();name=str(item.get('name') or f'Prompt {i}').strip()[:80]
                if not text or len(text)>8000:raise ValueError('Every benchmark prompt must contain 1–8000 characters.')
                key=(name,text)
                if key in seen:raise ValueError('Duplicate benchmark prompt.')
                seen.add(key);prompt_variants.append({'name':name,'prompt':text,'effective':effective_prompt(text,params.get('transparent_output',False))})
        elif mode in ('compare','transformer','encoder','matrix'):
            effective=effective_prompt(prompt,params.get('transparent_output',False));prompt_variants=[{'name':'Prompt','prompt':prompt,'effective':effective}]
        elif mode=='decode':latent=self.latents.metadata(data.get('latent_id'),pid)
        else:source=self.image(pid,data.get('input_artifact_id'))
        cycles=data.get('cycles',[1,5,10]) if mode=='roundtrip' else [1]
        if not isinstance(cycles,list) or not cycles or any(type(n) is not int or n not in (1,5,10) for n in cycles) or len(set(cycles))!=len(cycles):raise ValueError('Select round-trip cycles 1, 5 or 10.')
        # LoRA is an independent, explicit benchmark component.  The native
        # ComfyUI bridge (including GGUF/ConvRot) currently rejects adapter
        # injection, therefore skip only those combinations, keeping their
        # baseline jobs, rather than queueing knowingly broken LoRA jobs.
        expanded=[]
        smart=(mode=='auto' and data.get('auto_strategy','full')=='screening')
        # Smart screening adds selected LoRAs on the unchanged baseline instead
        # of multiplying *every* one-component-at-a-time experiment by LoRAs.
        smart_anchor=next((i for i,(_,cfg,_) in enumerate(setups)
                           if _resolve_benchmark_backend(cfg)[0]=='diffusers'),None) if smart else None
        if smart and smart_anchor is None and selected_lora_variants is not None:
            skipped_native.extend({'label':name+' · '+label,'backend':'incompatible',
                'reason':'Smart screening has no Diffusers baseline supporting LoRA; native ComfyUI LoRA injection is not enabled.'}
                for name,_,_ in setups for label,adapters in selected_lora_variants if adapters)
        for index,(name,cfg,model_loras) in enumerate(setups):
            if selected_lora_variants is None:
                inherited=copy.deepcopy(model_loras if data.get('use_profile_loras',False) else [])
                if inherited and _resolve_benchmark_backend(cfg)[0]!='diffusers':
                    skipped_native.append({'label':name,'backend':'incompatible',
                        'reason':'Profile LoRA is not supported by the native ComfyUI/GGUF benchmark bridge.'})
                    continue
                expanded.append((name,cfg,inherited))
                continue
            backend,_=_resolve_benchmark_backend(cfg)
            variants=selected_lora_variants if not smart or index==smart_anchor else selected_lora_variants[:1]
            for lora_label,adapters in variants:
                label=name+' · '+lora_label
                if adapters and backend!='diffusers':
                    skipped_native.append({'label':label,'backend':'incompatible',
                        'reason':'LoRA injection is not implemented for the native ComfyUI/GGUF/ConvRot benchmark backend. The no-LoRA baseline remains testable.'})
                    continue
                expanded.append((label,copy.deepcopy(cfg),copy.deepcopy(adapters)))
        setups=expanded
        if not setups:
            raise ValueError('No compatible benchmark jobs remain after applying LoRA compatibility checks.')
        job_total=len(setups)*len(cycles)*count*(len(prompt_variants) if prompt_variants else 1)
        limit=512 if mode=='auto' else 32
        if job_total>limit:raise ValueError(f'Maximum {limit} jobs for this benchmark. Reduce variants, LoRAs, prompts or repetitions.')
        entries=[]
        for name,cfg,loras in setups:
            if loras and str(cfg.get('model_path','')).lower().endswith('.gguf'):
                # Defensive check: unsupported GGUF adapters may never be queued.
                raise ValueError('GGUF + LoRA is not enabled in this Image Lab bridge.')
            if mode in ('compare','matrix','auto'):
                if str(cfg.get('model_path') or '').lower().endswith('.gguf') and cfg.get('quantized_loading')!='native_comfyui':
                    raise ValueError(name+': Qwen Image 2.1 GGUF requires Native ComfyUI GGUF execution; Diffusers GGUF loading is not used for these ComfyUI-GGUF checkpoints.')
                v=self.cb['validate_model']({'id':'qwen_image_21','provider':'qwen_image21','config':cfg})
                if not v.get('ready'):raise ValueError(name+': '+', '.join(v.get('missing',[])))
                from core.qwen_weight_format import require_supported_weights
                for field in ('model_path','text_encoder_path'):
                    path=cfg.get(field)
                    if path and str(path).endswith('.safetensors'):require_supported_weights(path,cfg.get('quantized_loading','reject'))
            else:
                check=component_status(cfg,decode_only=True)
                if not check.get('ready'):raise ValueError(name+': '+ '; '.join(f['detail'] for f in check['fields'] if not f['exists']))
                if latent:assert_same_latent_space(check['details']['vae']['config'],latent['contract'])
            variants=prompt_variants or [{'name':'','prompt':prompt,'effective':effective}]
            for variant in variants:
                for cycle in cycles:
                    for repeat in range(1,count+1):
                        p=copy.deepcopy(params);p['loras']=copy.deepcopy(loras);p['user_prompt']=variant['prompt'];p['_model_config_snapshot']=copy.deepcopy(cfg)
                        if mode in ('compare','transformer','encoder','matrix','auto'):tool=auto_tool(p);job_prompt=variant['effective']
                        else:
                            p={**decode_options(cfg,params),'width':params['width'],'height':params['height'],'_model_config_snapshot':copy.deepcopy(cfg)}
                            if latent:tool='qwen_decode';p['latent_id']=latent['id'];p['width']=latent.get('width',params['width']);p['height']=latent.get('height',params['height']);job_prompt='Benchmark: same latent, different VAE'
                            else:tool='qwen_roundtrip';p.update(input_artifact_id=source['id'],cycles=cycle);job_prompt='Benchmark: VAE round-trip'
                        label=name+((' · '+variant['name']) if mode=='auto' else '')+(f' · {cycle}×' if mode=='roundtrip' else '')+(f' · #{repeat}' if count>1 else '')
                        entries.append({'job_id':uuid.uuid4().hex,'label':label,'prompt_name':variant['name'] if mode=='auto' else None,'repeat':repeat,'cycles':cycle if mode=='roundtrip' else None,'tool':tool,'prompt':job_prompt,'params':p})
        rid=uuid.uuid4().hex
        result={'schema':1,'id':rid,'project_id':pid,'name':str(data.get('name') or 'Qwen benchmark')[:120],
            'created_at':time.time(),'mode':mode,'suite':str(data.get('suite') or mode)[:80],'seed':seed,'count':len(entries),'inputs':snapshots+([source] if source else []),
            'entries':entries,'skipped_incompatible_combinations':skipped_native,'limitations':['No automatic quality/identity score.','Worker CUDA allocator peaks are not total board memory.',
            'Timing includes loading. Native ComfyUI jobs report isolated process GPU memory; reference reconstruction is not native INT8.','VAE round-trips use deterministic posterior mode and RGBA8 between cycles.']}
        result['preflight']=self.preflight(entries)
        with self.files._lock():
            atomic_json(self.files._path(rid),result)
        return result
    def launch(self,rid):
        with self.files._lock():
            data=self.get(rid);self.project(data['project_id'])
            preflight=self.preflight(data.get('entries') or [])
            if not preflight['ready']:
                first=(preflight.get('blocked_jobs') or [{}])[0].get('reason') or 'benchmark preflight failed'
                raise ValueError(f"Preflight: {preflight['jobs_ready']}/{preflight['jobs_total']} jobs ready. {first}. Verify the external runtime or create a new compatible plan before queueing.")
            # Inputs remain exactly those reviewed. Do not silently use edited/replaced media.
            for item in data['inputs']:
                now=self.image(data['project_id'],item['id'])
                if now!=item:raise ValueError('A benchmark input changed. Create a new plan.')
            ids=[x['job_id'] for x in data['entries']]
            with con() as c:
                c.execute('BEGIN IMMEDIATE')
                n=c.execute('SELECT COALESCE(MAX(queue_order),0) FROM jobs').fetchone()[0]
                for i,item in enumerate(data['entries']):
                    old=c.execute('SELECT project_id FROM jobs WHERE id=?',(item['job_id'],)).fetchone()
                    if old:
                        if old['project_id']!=data['project_id']:raise ValueError('Benchmark job ownership mismatch.')
                        continue
                    params={**item['params'],'_benchmark_id':rid,'_benchmark_label':item['label']}
                    c.execute('''INSERT INTO jobs(id,project_id,tool,model_id,status,phase,progress,prompt,params_json,queue_order,cancel_requested)
                        VALUES(?,?,?,'qwen_image_21','queued','queued',0,?,?,?,0)''',
                        (item['job_id'],data['project_id'],item['tool'],item['prompt'],json.dumps(params,ensure_ascii=False),n+i+1))
            for jid in ids:
                job=self.cb['one']('SELECT status FROM jobs WHERE id=?',(jid,))
                if job and job['status']=='queued':self.cb['start_job'](jid)
        return {'id':rid,'jobs':ids,'count':len(ids)}
    def retry(self,rid,*,include_cancelled=False):
        """Resume an existing benchmark without repeating successful work."""
        with self.files._lock():
            data=self.get(rid);self.project(data['project_id'])
            for item in data.get('inputs') or []:
                if not item.get('id'):continue
                now=self.image(data['project_id'],item['id'])
                if now!=item:raise ValueError('A benchmark input changed. Create a new plan instead of retrying this batch.')
            candidates=[];skipped=[];kept_completed=0;kept_active=0;kept_cancelled=0;forced_incompatible=[]
            for entry in data.get('entries') or []:
                row=self.cb['one']('SELECT id,project_id,status FROM jobs WHERE id=?',(entry['job_id'],))
                if row and row.get('project_id')!=data['project_id']:raise ValueError('Benchmark job ownership mismatch.')
                status=(row or {}).get('status') or 'missing'
                if status=='completed':kept_completed+=1;continue
                if status in ('queued','running','cancelling'):kept_active+=1;continue
                if status=='cancelled' and not include_cancelled:kept_cancelled+=1;continue
                if status not in ('missing','failed','interrupted','cancelled'):
                    skipped.append({'job_id':entry['job_id'],'label':entry.get('label'),'status':status,'reason':'Job status is not retryable.'});continue
                cfg=copy.deepcopy((entry.get('params') or {}).get('_model_config_snapshot') or {})
                backend,issue=_resolve_benchmark_backend(cfg)
                if issue:
                    forced_incompatible.append({'job_id':entry['job_id'],'label':entry.get('label'),'status':status,'reason':issue,'backend':backend})
                candidates.append((entry,row,status,cfg))
            if candidates:
                pf_entries=[]
                for entry,_,_,cfg in candidates:
                    checked=copy.deepcopy(entry);checked.setdefault('params',{})['_model_config_snapshot']=copy.deepcopy(cfg);pf_entries.append(checked)
                pf=self.preflight(pf_entries)
                if not pf['ready']:
                    first=(pf.get('blocked_jobs') or [{}])[0].get('reason') or 'benchmark preflight failed'
                    raise ValueError(f"Retry preflight: {pf['jobs_ready']}/{pf['jobs_total']} jobs ready. {first}.")
                with con() as c:
                    c.execute('BEGIN IMMEDIATE');order=c.execute('SELECT COALESCE(MAX(queue_order),0) FROM jobs').fetchone()[0]
                    for index,(entry,row,status,cfg) in enumerate(candidates,1):
                        params={**entry['params'],'_model_config_snapshot':copy.deepcopy(cfg),'_benchmark_id':rid,'_benchmark_label':entry['label']}
                        if row is None:
                            c.execute("""INSERT INTO jobs(id,project_id,tool,model_id,status,phase,progress,prompt,params_json,queue_order,cancel_requested)
                                VALUES(?,?,?,'qwen_image_21','queued','queued',0,?,?,?,0)""",
                                (entry['job_id'],data['project_id'],entry['tool'],entry['prompt'],json.dumps(params,ensure_ascii=False),order+index))
                        else:
                            c.execute("""UPDATE jobs SET status='queued',phase='queued',progress=0,step_current=0,step_total=0,
                                prompt=?,params_json=?,output_artifact_id=NULL,error=NULL,started_at=NULL,completed_at=NULL,duration_seconds=0,
                                queue_order=?,cancel_requested=0,updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                                (entry['prompt'],json.dumps(params,ensure_ascii=False),order+index,entry['job_id']))
                shared=self.files.root.parent.parent
                for entry,_,_,_ in candidates:
                    jid=entry['job_id']
                    for path in (shared/'cache/job_cancel'/(jid+'.cancel'),shared/'cache/job_progress'/(jid+'.json'),shared/'cache/qwen-metrics'/(jid+'.json')):
                        try:path.unlink(missing_ok=True)
                        except OSError:pass
                    self.cb['start_job'](jid)
        return {'id':rid,'retried':[e['job_id'] for e,_,_,_ in candidates],'count':len(candidates),
            'kept_completed':kept_completed,'kept_active':kept_active,'kept_cancelled':kept_cancelled,
            'skipped_incompatible':skipped,'forced_incompatible':forced_incompatible}
    def report(self,rid):
        data=self.get(rid);out={k:v for k,v in data.items() if k!='entries'};items=[]
        for entry in data['entries']:
            job=self.cb['one']('SELECT * FROM jobs WHERE id=?',(entry['job_id'],)) or {}
            artifact=self.cb['one']('SELECT id,path,metadata_json FROM artifacts WHERE id=?',(job.get('output_artifact_id'),)) if job.get('output_artifact_id') else None
            artifact_ready=bool(artifact and Path(str(artifact.get('path') or '')).is_file())
            result={}
            if artifact:
                try:result=json.loads(artifact['metadata_json']).get('runtime_result') or {}
                except (ValueError,TypeError):pass
            metrics=result.get('metrics') or {}
            mp=self.files.root.parent.parent/'cache/qwen-metrics'/(entry['job_id']+'.json')
            # Main runtime stores SHARED/cache/qwen-metrics; derive via supplied root.
            if not metrics and mp.is_file():
                try:metrics=read_object(mp)
                except ValueError:pass
            cfg=((entry.get('params') or {}).get('_model_config_snapshot') or {})
            raw_status=job.get('status','planned');display_status=raw_status;compatibility_reason=''
            if raw_status in ('failed','interrupted','planned'):
                _,compatibility_reason=_resolve_benchmark_backend(copy.deepcopy(cfg))
                if compatibility_reason:display_status='incompatible'
            items.append({**entry,'status':display_status,'raw_status':raw_status,'compatibility_reason':compatibility_reason,'error':job.get('error') or '',
                'duration_seconds':job.get('duration_seconds'),'progress':job.get('progress',0),'result':result,'metrics':metrics,
                'artifact_id':artifact.get('id') if artifact_ready else None,'url':'/api/artifacts/'+artifact['id']+'/file' if artifact_ready else None,
                'components':{k:Path(cfg[k]).name if cfg.get(k) else 'Original' for k in PATH_KINDS if k!='vae_config_path'},
                'precision_policy':cfg.get('quantized_loading','reject')})
        out['entries']=items;return out
    def cancel(self,rid):
        data=self.get(rid)
        for entry in data['entries']:
            j=self.cb['one']('SELECT status FROM jobs WHERE id=?',(entry['job_id'],))
            if j and j['status'] in ('queued','running'):self.cb['request_cancel'](entry['job_id'])
        return {'cancel_requested':rid}
    def delete(self,rid,*,delete_contents=False):
        with self.files._lock():
            d=self.get(rid);self.project(d['project_id']);job_ids=[x.get('job_id') for x in d.get('entries') or [] if isinstance(x,dict) and x.get('job_id')]
            jobs=self._job_map(job_ids)
            if any((jobs.get(jid) or {}).get('project_id') not in (None,d['project_id']) for jid in job_ids):
                raise ValueError('Benchmark job ownership mismatch.')
            if any((jobs.get(jid) or {}).get('status') in ('queued','running','cancelling') for jid in job_ids):
                raise ValueError('Cancel/wait for this benchmark first.')
            if not delete_contents:
                visibility_path=self._visibility_file(rid)
                self.files._path(rid).unlink()
                try:visibility_path.unlink(missing_ok=True)
                except OSError:pass
                return {'deleted':rid,'artifacts_deleted':False,'jobs_deleted':False,'latents_deleted':0}
            artifact_ids=[(jobs.get(jid) or {}).get('output_artifact_id') for jid in job_ids]
            artifact_ids=[x for x in artifact_ids if x]
            artifacts=[]
            for start in range(0,len(artifact_ids),400):
                part=artifact_ids[start:start+400];marks=','.join('?' for _ in part)
                artifacts.extend(self.cb['rows']('SELECT id,project_id,path FROM artifacts WHERE id IN ('+marks+')',tuple(part)))
            shared=self.files.root.parent.parent;artifact_root=(shared/'artifacts'/d['project_id']).resolve()
            paths=[]
            for artifact in artifacts:
                path=Path(str(artifact.get('path') or ''))
                try:resolved=path.resolve(strict=False)
                except OSError:continue
                if artifact.get('project_id')!=d['project_id'] or artifact_root not in resolved.parents or path.is_symlink():
                    raise ValueError('Benchmark artifact path is outside the project artifact store; nothing was deleted.')
                paths.append(path)
            # Delete concrete output files first; inputs referenced by the plan are never included.
            for path in paths:
                if path.is_file():path.unlink()
            with con() as c:
                c.execute('BEGIN IMMEDIATE')
                for start in range(0,len(artifact_ids),400):
                    part=artifact_ids[start:start+400];marks=','.join('?' for _ in part)
                    try:c.execute("UPDATE project_state SET active_artifact_id='' WHERE active_artifact_id IN ("+marks+")",tuple(part))
                    except Exception:pass
                    c.execute('UPDATE jobs SET output_artifact_id=NULL WHERE output_artifact_id IN ('+marks+')',tuple(part))
                    c.execute('DELETE FROM artifacts WHERE id IN ('+marks+')',tuple(part))
                for start in range(0,len(job_ids),400):
                    part=job_ids[start:start+400];marks=','.join('?' for _ in part)
                    c.execute('DELETE FROM jobs WHERE id IN ('+marks+')',tuple(part))
            latent_count=0
            for jid in job_ids:
                try:
                    if self.latents.exists(jid):self.latents.delete(jid,d['project_id']);latent_count+=1
                except (ValueError,OSError):pass
                for path in (shared/'cache/job_cancel'/(jid+'.cancel'),shared/'cache/job_progress'/(jid+'.json'),
                    shared/'cache/qwen-metrics'/(jid+'.json'),shared/'logs/qwen-native-comfy'/(jid+'.log')):
                    try:path.unlink(missing_ok=True)
                    except OSError:pass
            visibility_path=self._visibility_file(rid)
            self.files._path(rid).unlink()
            try:visibility_path.unlink(missing_ok=True)
            except OSError:pass
        return {'deleted':rid,'artifacts_deleted':len(artifacts),'jobs_deleted':len(job_ids),'latents_deleted':latent_count}
    def public_report(self,rid):
        d=self.report(rid)
        # Exports omit paths, raw exception traces and credentials; prompts are opted in by creation.
        keep=('job_id','label','repeat','cycles','tool','status','duration_seconds','metrics','components','precision_policy','artifact_id')
        for i,e in enumerate(d['entries']):
            item={k:e.get(k) for k in keep};result=e['result']
            item['timing']={k:result.get(k) for k in ('load_seconds','conditioning_and_sampling_seconds','offload_seconds','decode_seconds')}
            item['execution']={'backend':result.get('execution_backend') or result.get('pipeline') or 'Image Lab worker',
                'mode':result.get('execution_mode'),'native_int8_benchmark':bool(result.get('native_int8_benchmark')),
                'sampler':result.get('sampler_name'),'scheduler':result.get('scheduler_name'),
                'memory_scope':(result.get('metrics') or {}).get('memory_scope')}
            item['roundtrip']=result.get('roundtrip');item['settings']={k:v for k,v in e['params'].items() if k in ('seed','width','height','steps','cfg','negative_prompt','transparent_output','vae_decode_mode')}
            item['loras']=[{'name':Path(x['path']).name,'strength':x.get('strength',1)} for x in e['params'].get('loras',[])];item['prompt']=e['prompt'];item['error_present']=bool(e.get('error'));d['entries'][i]=item
        return d
    def csv(self,rid):
        d=self.public_report(rid);f=io.StringIO();w=csv.writer(f)
        w.writerow(['label','status','seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_gpu_process_bytes','peak_rss_bytes','precision_policy','execution_backend','native_int8_benchmark','seed','steps','width','height','loras'])
        for e in d['entries']:
            safe=lambda x:"'"+x if isinstance(x,str) and (x[:1] in ('\t','\r','\n') or x.lstrip()[:1] in ('=','+','-','@')) else x
            w.writerow([safe(e['label']),e['status'],e['duration_seconds'],*[e['metrics'].get(k) for k in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_gpu_process_bytes','peak_rss_bytes')],e['precision_policy'],safe(e['execution'].get('backend')),e['execution'].get('native_int8_benchmark'),*[e['settings'].get(k) for k in ('seed','steps','width','height')],safe('; '.join(x['name']+f' @{x["strength"]:g}' for x in e.get('loras',[])))])
        return f.getvalue()
