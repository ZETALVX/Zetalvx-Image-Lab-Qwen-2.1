"""Data-only validation/catalog for Qwen Image 2.1. No torch/model code imports."""
from __future__ import annotations
import hashlib,json,os,re,threading
from pathlib import Path
from core.runtime_env import atomic_json

class QwenAssetError(ValueError):pass

KINDS={'repository','transformer','lora','vae','text_encoder','processor','scheduler','text_encoder_file','prompt_enhancer_t2i','prompt_enhancer_i2i'}
DESTINATIONS={'transformer':'Qwen-2.1/transformers','lora':'Qwen-2.1/loras','vae':'Qwen-2.1/vae','text_encoder_file':'Qwen-2.1/text_encoders','prompt_enhancer_t2i':'Qwen-2.1/prompt_enhancer/t2i','prompt_enhancer_i2i':'Qwen-2.1/prompt_enhancer/i2i'}
_LOCK=threading.RLock()

def browse_directory(path):
    """Browser-only path normalization; never used to save an asset/config path."""
    p=Path(path).expanduser().resolve()
    if p.is_file() or (not p.exists() and p.suffix.lower() in {'.safetensors','.gguf','.json','.bin','.pt','.pth','.ckpt'} and p.parent.is_dir()):
        return p.parent
    return p


def read_object(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>8*1024**2:raise QwenAssetError('Missing or oversized JSON: '+str(path))
    try:value=json.loads(path.read_text(encoding='utf-8'))
    except (ValueError,UnicodeError) as exc:raise QwenAssetError('Invalid JSON: '+str(path)) from exc
    if not isinstance(value,dict):raise QwenAssetError('Expected a JSON object: '+str(path))
    return value

def repository_status(root,override=''):
    """Check pipeline class and every shard named by an index, without loading weights."""
    raw=str(root or '').strip();base=Path(raw).expanduser() if raw else None
    fields=[]
    def record(field,path,ok,required=True,detail=''):
        fields.append({'field':field,'value':str(path or ''),'exists':bool(ok),'required':required,'detail':detail})
    if not base or not base.is_dir():
        record('model_dir',raw,False);return {'ready':False,'fields':fields,'missing':['model_dir']}
    def check_json(relative,field=None,expected=None):
        path=base/relative
        try:
            value=read_object(path)
            ok=not expected or value.get('_class_name')==expected
            detail='' if ok else 'Expected '+str(expected)+', found '+str(value.get('_class_name'))
        except QwenAssetError as exc:ok=False;detail=str(exc);value={}
        record(field or relative,path,ok,detail=detail);return value
    check_json('model_index.json',expected='QwenImage21Pipeline')
    check_json('scheduler/scheduler_config.json')
    processor=base/'processor'
    record('processor',processor,processor.is_dir() and any(processor.glob('*.json')))
    for component in ('text_encoder','vae','transformer'):
        check_json(component+'/config.json')
        if component=='transformer' and override:
            try:
                inspected=inspect_component(override,'transformer');ok=True;detail=inspected['note']
            except QwenAssetError as exc:ok=False;detail=str(exc)
            record('transformer_override',override,ok,detail=detail)
            continue
        folder=base/component;indexes=sorted(folder.glob('*.safetensors.index.json'))
        if indexes:
            for index in indexes:
                try:
                    obj=read_object(index);weightmap=obj.get('weight_map')
                    if not isinstance(weightmap,dict) or not weightmap:raise QwenAssetError('Empty shard index')
                    shards=sorted(set(weightmap.values()))
                    if any(not isinstance(x,str) or '/' in x or '\\' in x or '..' in x or not x.endswith('.safetensors') for x in shards):raise QwenAssetError('Unsafe shard index')
                    for shard in shards:
                        sp=folder/shard;record(component+'/'+shard,sp,sp.is_file() and sp.stat().st_size>16)
                except (QwenAssetError,TypeError,ValueError) as exc:record(component+'/shards',index,False,detail=str(exc))
        else:
            weights=list(folder.glob('*.safetensors'))
            record(component+'/weights',folder,bool(weights) and all(x.stat().st_size>16 for x in weights))
    return {'ready':all(f['exists'] for f in fields if f['required']),'fields':fields,
            'missing':[f['field'] for f in fields if f['required'] and not f['exists']],
            'note':'VAE, text encoder, processor and transformer configuration come from this Qwen Image 2.1 repository.'}

def inspect_component(path,kind):
    if kind in ('prompt_enhancer_t2i','prompt_enhancer_i2i'):
        from core.qwen_prompt_enhancer import inspect_pe_file
        try:return inspect_pe_file(path,kind)
        except ValueError as exc:raise QwenAssetError(str(exc)) from exc
    if kind not in DESTINATIONS:raise QwenAssetError('Choose a Qwen transformer, LoRA or VAE file.')
    p=Path(str(path)).expanduser()
    if not p.is_absolute() or not p.is_file():raise QwenAssetError('Model file not found. Use an absolute server path.')
    if p.suffix.lower()=='.gguf':
        if kind!='transformer':raise QwenAssetError('GGUF is supported only as a Qwen 2.1 transformer override.')
        import struct
        with p.open('rb') as fp:head=fp.read(24)
        if len(head)!=24 or head[:4]!=b'GGUF':raise QwenAssetError('Invalid GGUF header.')
        version,tensors,metadata=struct.unpack('<IQQ',head[4:])
        if version not in (2,3) or not 1<=tensors<=1000000 or metadata>1000000 or p.stat().st_size<=24:
            raise QwenAssetError('Invalid or empty GGUF file.')
        # No claim of architecture compatibility based solely on a filename/header.
        return {'kind':'transformer','family':'unknown','format':'gguf','tensor_count':tensors,
                'note':'GGUF header checked; confirm that its source specifies Qwen Image 2.1 (not another Qwen model).'}
    if p.suffix.lower()!='.safetensors':raise QwenAssetError('Only .safetensors and transformer .gguf files are accepted; no pickle/bin checkpoints.')
    if kind=='text_encoder_file':
        from core.qwen_weight_format import inspect_weights
        info=inspect_weights(p)
        if any('linear_attn.' in k for k in info['keys']):raise QwenAssetError('Qwen3.5 Prompt Enhancer weights must be added as PE-T2I or PE-I2I, not as a diffusion Text encoder.')
        if not any('layers.' in k or 'language_model' in k for k in info['keys']):raise QwenAssetError('This file does not contain Qwen text-encoder layers.')
        return {'kind':kind,'family':'unknown','format':'safetensors','note':'Encoder file registered. Qwen3-VL configuration and all tensor keys/shapes will be checked when loading.'}
    from core.model_downloads import classify_safetensors
    try:info=classify_safetensors(p)
    except Exception as exc:raise QwenAssetError(str(exc)) from exc
    if info['family']=='other':raise QwenAssetError('This file declares a different model family; Qwen Image 2.1 is required.')
    if kind!='text_encoder_file' and info['kind'] not in ('unknown',kind):raise QwenAssetError('The detected component does not match the selected type.')
    return {**info,'format':'safetensors','note':('Qwen Image 2.1 metadata detected.' if info['family']=='qwen21' else 'File structure checked; Qwen Image 2.1 compatibility must be confirmed from its source.')}

# Scan only. Never download, execute, copy or automatically import a model.
BULK_FILE_KINDS={'transformer','lora','vae','text_encoder_file','prompt_enhancer_t2i','prompt_enhancer_i2i'}
BULK_DIR_KINDS={'repository','vae','text_encoder','processor','scheduler'}
SCAN_MAX_FILES=6000
SCAN_MAX_CANDIDATES=500
SCAN_MAX_DEPTH=6
SCAN_SKIP={'__pycache__','.git','.venv','venv','node_modules','.cache','cache','.trash','trash'}

def _guess_file_kind(path):
    names=[x.lower() for x in path.parts[-5:-1]]
    fname=path.name.lower()
    if path.suffix.lower()=='.gguf':return 'transformer'
    if 'pe_i2i' in fname:return 'prompt_enhancer_i2i'
    if 'pe_t2i' in fname:return 'prompt_enhancer_t2i'
    if any('lora' in n for n in names) or 'lora' in fname:return 'lora'
    if any('text_encoder' in n or 'text-encoder' in n for n in names) or 'qwen3vl' in fname:return 'text_encoder_file'
    if any(n=='vae' or 'vae_' in n for n in names) or 'vae' in fname:return 'vae'
    if any('transformer' in n or 'diffusion_models' in n or 'checkpoint' in n or 'unet' in n for n in names):return 'transformer'
    # Ambiguous safetensors should never be silently classified as a checkpoint.
    return ''

def scan_model_folder(root):
    """Bounded shallow read-only scan of server files. No symlink traversal."""
    raw=str(root or '').strip()
    if not raw:raise QwenAssetError('Select an existing server folder.')
    path=Path(raw).expanduser()
    if not path.is_absolute() or not path.is_dir() or path.is_symlink():
        raise QwenAssetError('Select an absolute, non-linked server folder.')
    root=path.resolve()
    candidates=[];seen=set();visited=0;truncated=False
    def collect(path,kind):
        if len(candidates)>=SCAN_MAX_CANDIDATES:return False
        value=str(path)
        if value in seen:return True
        seen.add(value)
        candidates.append({'path':value,'name':path.name,'kind':kind,'selected':bool(kind),
                           'folder':path.is_dir(),'relative':str(path.relative_to(root)) if path!=root else '.'})
        return True
    for base,dirs,files in os.walk(root,followlinks=False):
        current=Path(base);depth=len(current.relative_to(root).parts)
        dirs[:]=sorted(d for d in dirs if d.lower() not in SCAN_SKIP and not (current/d).is_symlink()) if depth<SCAN_MAX_DEPTH else []
        files=sorted(files)
        visited+=len(files)
        if visited>SCAN_MAX_FILES:
            truncated=True;break
        names=set(files)
        if 'model_index.json' in names:collect(current,'repository')
        elif 'config.json' in names and current!=root:
            folder_kind=({ 'vae':'vae', 'text_encoder':'text_encoder','processor':'processor','scheduler':'scheduler'}).get(current.name.lower())
            if folder_kind:collect(current,folder_kind)
        for name in files:
            fp=current/name
            if fp.is_symlink() or fp.suffix.lower() not in ('.safetensors','.gguf'):continue
            if fp.stat().st_size<=24:continue
            if not collect(fp,_guess_file_kind(fp)):
                truncated=True;break
        if truncated or len(candidates)>=SCAN_MAX_CANDIDATES:
            truncated=True;break
    return {'root':str(root),'candidates':candidates,'total':len(candidates),'scanned_files':visited,
            'truncated':truncated,'note':'Preview only. Select supported Qwen 2.1 components before linking; no files are copied or loaded.'}


class QwenAssets:
    def __init__(self,root,state):
        self.root=Path(root).resolve();self.state=Path(state)
    def _load(self):
        if not self.state.exists():return {'schema':1,'entries':[]}
        value=read_object(self.state)
        if value.get('schema')!=1 or not isinstance(value.get('entries'),list):raise QwenAssetError('Invalid Qwen model catalog.')
        return value
    def add(self,path,kind,entry_type='file',accept_unknown=False):
        raw=str(path or '').strip();p=Path(raw).expanduser()
        if not raw or not p.is_absolute():raise QwenAssetError('Use an absolute path on the server.')
        p=p.resolve()
        # The UI exposes one Text encoder choice because users may have either a
        # complete Diffusers/Qwen3-VL folder or a standalone safetensors weight
        # file (BF16/ConvRot). Normalize the latter to the dedicated file kind.
        if kind=='text_encoder' and p.is_file():
            kind='text_encoder_file'
        if kind=='repository':
            check=repository_status(p)
            # A partial repository is useful with a separate GGUF transformer; keep it
            # linkable, but never call it ready until validate_model includes override.
            index=read_object(p/'model_index.json')
            if index.get('_class_name')!='QwenImage21Pipeline':raise QwenAssetError('Not a QwenImage21Pipeline repository.')
            info={'kind':'repository','family':'qwen21','note':'Repository linked; component readiness is shown in the active model.'}
        elif p.is_dir() and kind in ('vae','text_encoder','processor','scheduler'):
            if kind=='text_encoder' and not (p/'config.json').is_file():
                # A generic ComfyUI/text_encoders directory is a browser location,
                # not an encoder component. Never save it as one. If it contains
                # exactly one weight file we can safely use that explicit file;
                # with multiple candidates the user must choose the intended one.
                candidates=sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower()=='.safetensors')[:501]
                if len(candidates)==1:
                    p=candidates[0];kind='text_encoder_file';info=inspect_component(p,kind)
                    if info['family']!='qwen21' and not accept_unknown:
                        raise QwenAssetError('Confirm the source identifies this encoder as compatible with Qwen Image 2.1.')
                else:
                    count='500+' if len(candidates)>500 else str(len(candidates))
                    raise QwenAssetError('This is a parent or incomplete encoder folder, not a Qwen3-VL component. Select a .safetensors file inside it. Candidate weight files found: '+count+'.')
            else:
                from core.qwen_components import inspect_vae,inspect_folder
                inspected=inspect_vae(p) if kind=='vae' else inspect_folder(p,kind)
                info={'kind':kind,'family':'qwen21','note':'Component configuration checked; runtime compatibility test is separate.'}
        else:
            info=inspect_component(p,kind)
            if info['family']!='qwen21' and not accept_unknown:raise QwenAssetError('Confirm the source identifies this component as Qwen Image 2.1.')
        ident=hashlib.sha256((kind+'\0'+str(p)).encode()).hexdigest()[:24]
        entry={'id':ident,'path':str(p),'kind':kind,'name':p.name,'inspection':info,'compatibility_confirmed':bool(accept_unknown)}
        with _LOCK:
            state=self._load();state['entries']=[e for e in state['entries'] if e.get('id')!=ident]+[entry]
            atomic_json(self.state,state)
        return entry
    def add_lora_folder(self, path, accept_unknown=False):
        """Import individual LoRA files from a local directory; never catalog the folder as a LoRA.

        Reuse the bounded, no-symlink scan and the same per-file validation as
        server-file linking, browser uploads and completed downloads.
        """
        raw = str(path or '').strip()
        folder = Path(raw).expanduser()
        if not raw or not folder.is_absolute() or not folder.is_dir() or folder.is_symlink():
            raise QwenAssetError('Choose an existing absolute server folder for LoRA files.')
        scan = scan_model_folder(raw)
        # Do not represent a truncated scan as a successful full-folder import.
        if scan['truncated']:
            raise QwenAssetError('Folder scan limit reached. Select a smaller folder or use Scan server folder for a preview.')
        files = [r for r in scan['candidates'] if not r['folder']
                 and Path(r['path']).suffix.lower() == '.safetensors']
        if not files:
            raise QwenAssetError('No .safetensors files found in the selected LoRA folder.')
        if len(files) > 250:
            raise QwenAssetError('A folder import accepts at most 250 LoRA files. Split the selection.')
        linked, errors = [], []
        for item in files:
            try:
                # Each file, even when its name does not say LoRA, is checked
                # using the normal signature/metadata validator.
                linked.append(self.add(item['path'], 'lora', accept_unknown=accept_unknown))
            except (QwenAssetError, ValueError, OSError) as exc:
                errors.append({'path':item['path'], 'kind':'lora', 'error':str(exc)[:500]})
        return {'linked':len(linked), 'failed':len(errors), 'entries':linked,
                'errors':errors, 'root':scan['root'], 'scanned':len(files)}

    def remove(self,rid):
        with _LOCK:
            state=self._load();entry=next((e for e in state['entries'] if e.get('id')==rid),None)
            if not entry:raise QwenAssetError('Catalog entry not found.')
            state['entries']=[e for e in state['entries'] if e.get('id')!=rid];atomic_json(self.state,state)
        return {'removed':rid,'files_deleted':False,'path':entry['path']}
    def inspect(self,path,kind):
        info=inspect_component(path,kind)
        return {**info,'path':str(path),'needs_confirmation':info['family']!='qwen21'}
    def inventory(self,cfg=None):
        cfg=cfg or {};entries=list(self._load()['entries']);known={e.get('path') for e in entries}
        for kind,relative in DESTINATIONS.items():
            folder=self.root/relative
            if not folder.is_dir():continue
            for path in sorted(folder.iterdir())[:500]:
                if not path.is_file() or path.suffix.lower() not in ('.safetensors','.gguf') or str(path) in known:continue
                entries.append({'id':hashlib.sha256((kind+'\0'+str(path)).encode()).hexdigest()[:24],
                    'path':str(path),'kind':kind,'name':path.name,'managed':True,'registered':False})
        for field,kind in [('model_dir','repository'),('model_path','transformer'),('vae_path','vae'),('text_encoder_dir','text_encoder'),('processor_dir','processor'),('scheduler_dir','scheduler'),('text_encoder_path','text_encoder_file')]:
            raw=str(cfg.get(field) or '').strip()
            if raw and raw not in {e.get('path') for e in entries}:
                entries.append({'id':'configured-'+field,'path':raw,'kind':kind,'name':Path(raw).name,'registered':False})
        out=[]
        for e in entries:
            path=Path(e['path']);e=dict(e);e.setdefault('registered',True);e['present']=path.exists()
            e['active']=str(path) in [str(cfg.get(k)) for k in ('model_dir','model_path','vae_path','text_encoder_dir','processor_dir','scheduler_dir','text_encoder_path')]
            # Read old catalog records without rewriting IDs or touching model files.
            if e.get('kind')=='text_encoder' and path.suffix.lower()=='.safetensors':
                e['kind']='text_encoder_file'
            if e.get('kind')=='text_encoder_file' and ('pe_i2i' in path.name.lower() or 'pe_t2i' in path.name.lower()):
                e['kind']='prompt_enhancer_i2i' if 'pe_i2i' in path.name.lower() else 'prompt_enhancer_t2i'
            e['selectable']=bool(e['present'])
            e['issue']=''
            if e.get('kind')=='text_encoder' and path.is_dir() and not (path/'config.json').is_file():
                # Old releases could accidentally register the browser parent folder.
                # Keep the record visible so it can be unlinked, but never offer it
                # as a model/profile choice and never look for config.json there.
                e['selectable']=False
                e['issue']='Parent/incomplete text-encoder folder. Select a .safetensors file inside it or link a complete encoder folder with config.json.'
            elif e.get('kind')=='text_encoder_file' and not (path.is_file() and path.suffix.lower()=='.safetensors'):
                e['selectable']=False
                e['issue']='Encoder weight file is missing or invalid.'
            if e.get('kind') in ('prompt_enhancer_t2i','prompt_enhancer_i2i'):
                try:inspect_component(path,e['kind'])
                except (ValueError,OSError) as exc:e['selectable']=False;e['issue']=str(exc)
            e['managed']=path.resolve().is_relative_to(self.root) if path.exists() else False
            out.append(e)
        return out
