"""Single-queue model imports, persistent status, bounded safetensors validation.
No inference or Python deserialization of downloaded content is performed.
"""
from __future__ import annotations
import hashlib, json, math, os, queue, re, shutil, struct, threading, time, uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode, unquote, quote
from email.message import Message
from core.download_security import DownloadError, Cancelled, safe_name, clean_user_url, safe_source
from core.model_policy import external_reason, POLICY_VERSION
from core.runtime_env import atomic_json
from core.download_manager import COORDINATOR

MAX_BYTES=32*1024**3
ACTIVE={'queued','downloading','validating','installing','cancelling'}
DESTINATIONS={'transformer':'Qwen-2.1/transformers','lora':'Qwen-2.1/loras','vae':'Qwen-2.1/vae','text_encoder_file':'Qwen-2.1/text_encoders'}
DTYPE_BYTES={'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'I32':4,'U32':4,'I64':8,'U64':8,
             'F16':2,'BF16':2,'F32':4,'F64':8,'F8_E4M3':1,'F8_E5M2':1,'F8_E4M3FN':1}

def classify_safetensors(path):
    """Validate header/offsets without torch. Architecture detection is a heuristic.
    Unknown family requires an explicit user confirmation; known mismatches fail.
    """
    size=Path(path).stat().st_size
    with Path(path).open('rb') as f:
        raw=f.read(8)
        if len(raw)!=8:raise DownloadError('File incompleto: manca l’header safetensors.')
        n=struct.unpack('<Q',raw)[0]
        if not 2<=n<=8*1024*1024 or n+8>=size:raise DownloadError('Il file non è un safetensors valido (o è una pagina HTML).')
        try:
            def unique_pairs(items):
                d={}
                for k,v in items:
                    if k in d:raise ValueError('duplicate key')
                    d[k]=v
                return d
            header=json.loads(f.read(n),object_pairs_hook=unique_pairs)
        except (ValueError,UnicodeError):raise DownloadError('Header safetensors non valido.') from None
    if not isinstance(header,dict):raise DownloadError('Header safetensors non valido.')
    tensors={k:v for k,v in header.items() if k!='__metadata__'}
    if not 1<=len(tensors)<=100000:raise DownloadError('Nessun tensore valido nel file.')
    intervals=[]
    for k,v in tensors.items():
        if not isinstance(v,dict):raise DownloadError('Descrittore tensore non valido.')
        shape=v.get('shape');offset=v.get('data_offsets');dtype=v.get('dtype')
        if not isinstance(shape,list) or len(shape)>16 or any(type(x)!=int or x<0 or x>2**40 for x in shape):
            raise DownloadError('Forma tensore non valida.')
        if not isinstance(offset,list) or len(offset)!=2 or any(type(x)!=int for x in offset):raise DownloadError('Offset tensore non valido.')
        a,b=offset
        if a<0 or b<a or b>size-8-n:raise DownloadError('File incompleto: dati tensore fuori limite.')
        if dtype not in DTYPE_BYTES or math.prod(shape)*DTYPE_BYTES[dtype]!=b-a:
            raise DownloadError('Dimensione o tipo del tensore non supportato dal validatore.')
        if b>a:intervals.append((a,b))
    pos=0
    for a,b in sorted(intervals):
        if a!=pos:raise DownloadError('Dati safetensors sovrapposti o incompleti.')
        pos=b
    if pos!=size-8-n:raise DownloadError('Lunghezza safetensors non coerente.')
    md=header.get('__metadata__') or {}
    if not isinstance(md,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in md.items()):
        raise DownloadError('Metadati safetensors non validi.')
    keys=list(tensors);joined=' '.join(keys[:20000]).lower()
    architecture=' '.join(str(v) for v in md.values()).lower()
    q21=any(x in architecture for x in ('qwen-image-2.1','qwen_image_2.1','qwenimage21','qwen_image21','qwen-image-21'))
    other=any(x in architecture or x in joined for x in ('sdxl','stable-diffusion','flux','conditioner.embedders','lora_te2_','controlnet_cond_embedding'))
    family='qwen21' if q21 and not other else ('other' if other else 'unknown')
    if any(x in joined for x in ('lora_down','lora_up','lora_a.','lora_b.','lora.down.','lora.up.')):kind='lora'
    elif any(k.startswith(('encoder.','decoder.','vae.encoder.','vae.decoder.')) for k in keys):kind='vae'
    elif any(x in joined for x in ('transformer_blocks.','single_transformer_blocks.','single_blocks.','blocks.')):kind='transformer'
    else:kind='unknown'
    return {'kind':kind,'family':family,'tensor_count':len(tensors),
            'metadata':{str(k):str(v)[:240] for k,v in list(md.items())[:30]}}

def classify_model_file(path):
    with Path(path).open('rb') as fp:header=fp.read(24)
    if header[:4]==b'GGUF':
        if len(header)!=24:raise DownloadError('Incomplete GGUF header.')
        version,tensors,metadata=struct.unpack('<IQQ',header[4:])
        if version not in (2,3) or not 1<=tensors<=1000000 or metadata>1000000 or Path(path).stat().st_size<=24:raise DownloadError('Invalid GGUF header.')
        return {'kind':'transformer','family':'unknown','format':'gguf','tensor_count':tensors,'validation':'header-only; confirm Qwen Image 2.1 source'}
    return classify_safetensors(path)


def normalize_link(url):
    url=clean_user_url(str(url).strip());p=urlsplit(url);host=(p.hostname or '').lower()
    reason=external_reason(url)
    if reason:raise DownloadError(reason)
    if host=='huggingface.co':
        bits=p.path.strip('/').split('/')
        if len(bits)<5 or bits[2] not in ('blob','resolve'):
            raise DownloadError('For a component, use its Hugging Face file link. For the whole model choose Repository.')
        repo='/'.join(bits[:2]);rev=unquote(bits[3]);filename=unquote('/'.join(bits[4:]))
        name=safe_name(filename)
        return {'provider':'huggingface','repo':repo,'revision':rev,'repo_file':filename,'filename':name,
                'url':f'https://huggingface.co/{repo}/resolve/{quote(rev,safe="")}/{quote(filename,safe="/")}',
                'source':f'https://huggingface.co/{repo}/blob/{quote(rev,safe="")}/{quote(filename,safe="/")}',
                'kind_hint':'auto','license':'Non dichiarata / da verificare'}
    if host in ('civitai.com','www.civitai.com'):
        match=re.fullmatch(r'/api/download/models/(\d+)/?',p.path)
        version=match.group(1) if match else (parse_qs(p.query).get('modelVersionId') or [''])[0]
        if not re.fullmatch(r'\d+',version):
            raise DownloadError('Civitai: copia il link Download della versione, oppure un URL con modelVersionId. La pagina del modello da sola non identifica il file.')
        return {'provider':'civitai','version':version,'url':url,'source':safe_source(url),'kind_hint':'auto','license':'Termini dell’autore (vedi pagina)'}
    return {'provider':'direct','url':url,'source':safe_source(url),'kind_hint':'auto','license':'Non verificata: consulta la fonte'}

class ModelDownloads:
    def __init__(self,root,state_root,transport,on_installed=None,start_worker=True):
        self.root=Path(root);self.state_root=Path(state_root);self.transport=transport
        self.on_installed=on_installed or (lambda *_:None)
        self.lock=threading.RLock();self.jobs={};self.events={};self.plans={};self.pending=queue.Queue()
        self.incoming=self.root/'.incoming'
        if self.incoming.is_symlink():raise DownloadError('La cartella temporanea download non può essere un link simbolico.')
        self.incoming.mkdir(parents=True,exist_ok=True,mode=0o700);self.incoming.chmod(0o700)
        self.state_root.mkdir(parents=True,exist_ok=True)
        for p in self.state_root.glob('*.json'):
            try:
                j=json.loads(p.read_text());jid=j['id']
                if not re.fullmatch('[a-f0-9]{32}',jid):continue
                if j.get('status') in ACTIVE:
                    j['status']='interrupted';j['error']='App riavviata: ripeti l’importazione dal link.'
                    (self.incoming/(jid+'.part')).unlink(missing_ok=True);atomic_json(p,j)
                if not isinstance(j.get('history'),list):j['history']=[]
                self.jobs[jid]=j
            except (OSError,ValueError,KeyError):continue
        if start_worker:
            for i in range(3):threading.Thread(target=self._loop,daemon=True,name=f'model-download-queue-{i+1}').start()
    def inspect(self,url,kind='auto',filename='',builtin=False):
        if kind in ('vae_repository','text_encoder_repository','processor_repository','scheduler_repository'):
            from core.qwen_component_download import inspect_bundle
            return inspect_bundle(self,url,kind.removesuffix('_repository'))
        if kind=='repository':
            from core.qwen_repository_download import inspect_repository
            return inspect_repository(self,url,exclude_transformer=(filename=='without-transformer'))
        if kind not in ('auto',*DESTINATIONS):raise DownloadError('Choose Transformer, LoRA, VAE or Repository.')
        plan=normalize_link(url);plan['builtin']=bool(builtin);plan['kind']=kind;plan['sha256']='';plan['size']=0
        if plan['provider']=='huggingface':
            d=self.transport.json(f'https://huggingface.co/api/models/{plan["repo"]}/revision/{quote(plan["revision"],safe="")}?blobs=true')
            if not isinstance(d,dict):raise DownloadError('Metadati repository non validi.')
            sha=str(d.get('sha') or '')
            if re.fullmatch('[a-f0-9]{40}',sha):
                plan['revision']=sha;plan['url']=f'https://huggingface.co/{plan["repo"]}/resolve/{sha}/{quote(plan["repo_file"],safe="/")}'
            cd=d.get('cardData') or {};reported=cd.get('license') if isinstance(cd,dict) else None
            if reported:plan['license']=str(reported)[:240]+' · metadata della fonte, non verificata'
            record=next((x for x in d.get('siblings',[]) if x.get('rfilename')==plan['repo_file']),{})
            lfs=record.get('lfs') or {};plan['size']=int(lfs.get('size') or record.get('size') or 0)
            candidate=str(lfs.get('sha256') or lfs.get('oid') or '').removeprefix('sha256:').lower()
            if re.fullmatch('[a-f0-9]{64}',candidate):plan['sha256']=candidate
            plan['license_url']=f'https://huggingface.co/{plan["repo"]}'
        elif plan['provider']=='civitai':
            d=self.transport.json(f'https://civitai.com/api/v1/model-versions/{plan["version"]}')
            if not isinstance(d,dict):raise DownloadError('Metadati Civitai non validi.')
            files=[f for f in d.get('files',[]) if str(f.get('name','')).lower().endswith(('.safetensors','.gguf'))]
            # Honour direct-link format/fp/size selectors; never silently choose another variant.
            params=parse_qs(urlsplit(url).query)
            for key in ('type','format','fp','size'):
                if key in params:
                    files=[f for f in files if str((f.get('metadata') or {}).get(key,f.get(key,''))).lower()==params[key][0].lower()]
            if len(files)>1:
                primary=[f for f in files if f.get('primary')]
                if len(primary)==1:files=primary
            if len(files)!=1:raise DownloadError('This version contains multiple files or no supported weights. Paste the exact file download link.')
            item=files[0];plan['filename']=safe_name(item['name'])
            download=item.get('downloadUrl')
            if not download:raise DownloadError('Nessun download disponibile per questo file.')
            plan['url']=clean_user_url(download)
            if (urlsplit(plan['url']).hostname or '').lower() not in ('civitai.com','www.civitai.com'):
                raise DownloadError('URL di download Civitai inatteso.')
            plan['size']=int(float(item.get('sizeKB') or 0)*1024)
            plan['sha256']=str((item.get('hashes') or {}).get('SHA256') or '').lower()
            base=str(d.get('baseModel') or '');plan['base_model']=base
            air=str(d.get('air') or '').lower()
            if base and not any(x in base.lower().replace(' ','').replace('-','') for x in ('qwenimage2.1','qwen2.1','qwenimage21')):
                raise DownloadError('The source declares a different base model. Qwen Image 2.1 is required.')
            typ=str((d.get('model') or {}).get('type') or '').lower()
            if typ in ('checkpoint','lora','vae'):plan['kind_hint']='transformer' if typ=='checkpoint' else typ
            model_id=d.get('modelId');plan['source']=f'https://civitai.com/models/{model_id}?modelVersionId={plan["version"]}' if str(model_id).isdigit() else safe_source(url)
            plan['license_url']=plan['source']
            plan['license']='Termini dell’autore: verifica la pagina della versione (non certificati dall’app)'
        else:
            with self.transport.open(plan['url']) as r:
                ct=str(r.headers.get('Content-Type') or '').lower()
                if any(x in ct for x in ('text/html','application/json','text/plain')):raise DownloadError('Il link restituisce una pagina, non un file modello.')
                m=Message();m['content-disposition']=r.headers.get('Content-Disposition','')
                plan['filename']=safe_name(filename or m.get_filename() or Path(urlsplit(plan['url']).path).name)
                plan['size']=int(r.headers.get('Content-Length') or 0)
        if filename:plan['filename']=safe_name(filename)
        plan['filename']=safe_name(plan.get('filename',''))
        if not 0<=plan['size']<=MAX_BYTES:raise DownloadError('File oltre il limite di 32 GiB.')
        if plan['sha256'] and not re.fullmatch('[a-f0-9]{64}',plan['sha256']):raise DownloadError('Hash SHA-256 della fonte non valido.')
        if builtin:raise DownloadError('Choose the Qwen repository instead of a legacy built-in checkpoint.')
        plan['id']=uuid.uuid4().hex;plan['expires']=time.time()+1800
        with self.lock:
            self.plans={k:v for k,v in self.plans.items() if v['expires']>time.time()}
            if len(self.plans)>40:raise DownloadError('Troppe anteprime aperte; attendi o riavvia.')
            self.plans[plan['id']]=plan
        return self.preview(plan)
    def preview(self,p):
        return {k:p.get(k,'') for k in ('id','filename','size','source','provider','license','license_url','kind','kind_hint','base_model','sha256','revision','builtin','file_count','exclude_transformer','component_kind')}
    def list(self):
        with self.lock:return sorted((dict(v) for v in self.jobs.values()),key=lambda j:j.get('created_at',0),reverse=True)[:100]
    def get(self,jid):
        with self.lock:
            if jid not in self.jobs:raise DownloadError('Download non trovato.')
            return dict(self.jobs[jid])
    def _update(self,jid,**change):
        with self.lock:
            job=self.jobs[jid];now=time.time();old_status=job.get('status')
            new_status=change.get('status',old_status)
            if new_status=='downloading' and not job.get('started_at'):change.setdefault('started_at',now)
            history=job.get('history') if isinstance(job.get('history'),list) else []
            if new_status!=old_status:
                message=str(change.get('error') or change.get('warning') or '')[:500]
                history.append({'at':now,'status':new_status,'message':message})
                change['history']=history[-80:]
            job.update(change,updated_at=now);atomic_json(self.state_root/(jid+'.json'),job)
            return dict(job)
    def start(self,preview_id,accepted=False):
        if not accepted:raise DownloadError('Conferma di aver verificato i termini della fonte per il tuo utilizzo.')
        with self.lock:
            plan=self.plans.get(preview_id)
            if not plan or plan['expires']<time.time():raise DownloadError('Anteprima scaduta: analizza di nuovo il link.')
            if sum(j.get('status') in ACTIVE for j in self.jobs.values())>=30:raise DownloadError('La coda contiene già 30 download.')
            jid=uuid.uuid4().hex;self.events[jid]=threading.Event()
            now=time.time()
            self.jobs[jid]=dict(self.preview(plan),id=jid,status='queued',created_at=now,updated_at=now,downloaded_bytes=0,
                               total_bytes=plan.get('size',0),error='',policy_version=POLICY_VERSION,terms_confirmed=True,
                               history=[{'at':now,'status':'queued','message':''}])
            self._update(jid)
            self.pending.put((jid,dict(plan)))
            return dict(self.jobs[jid])
    def cancel(self,jid):
        with self.lock:
            if jid not in self.jobs:raise DownloadError('Download non trovato.')
            j=self.jobs[jid]
            if j['status'] in ACTIVE:self.events.setdefault(jid,threading.Event()).set();return self._update(jid,status='cancelling')
            if j['status']=='needs_confirmation':
                (self.incoming/(jid+'.part')).unlink(missing_ok=True);return self._update(jid,status='cancelled')
            return dict(j)
    def _loop(self):
        while True:
            jid,plan=self.pending.get();key='models:'+jid;event=self.events.setdefault(jid,threading.Event());acquired=False
            try:
                acquired=COORDINATOR.acquire(key,event)
                if not acquired:
                    self._update(jid,status='cancelled',error='Download annullato prima dell’avvio.')
                    continue
                self._run(jid,plan)
            except Exception:
                # Never leak URLs/credentials in logs, and don't kill the entire queue.
                try:self._update(jid,status='failed',error='Errore locale nel salvataggio del download; controlla spazio e permessi.')
                except Exception:pass
            finally:
                if acquired:COORDINATOR.release(key)
                self.pending.task_done()
    def _run(self,jid,plan):
        if plan.get('kind')=='repository':
            from core.qwen_repository_download import run_repository
            return run_repository(self,jid,plan)
        part=self.incoming/(jid+'.part');cancel=self.events[jid]
        try:
            if cancel.is_set():raise Cancelled('Download annullato.')
            if shutil.disk_usage(self.root).free<max(int(plan.get('size',0)),128*1024**2)+128*1024**2:
                raise DownloadError('Spazio disco insufficiente.')
            self._update(jid,status='downloading',current_file=str(plan.get('filename') or ''))
            digest=hashlib.sha256();done=0;last=0
            with self.transport.open(plan['url'],cancel=cancel) as r:
                ct=str(r.headers.get('Content-Type') or '').lower()
                if any(x in ct for x in ('text/html','application/json','text/plain')):raise DownloadError('Il server ha restituito una pagina invece del modello.')
                total=int(r.headers.get('Content-Length') or 0)
                if total>MAX_BYTES:raise DownloadError('File oltre il limite di 32 GiB.')
                with part.open('xb') as f:
                    for chunk in r.chunks():
                        if cancel.is_set():raise Cancelled('Download annullato.')
                        done+=len(chunk)
                        if done>MAX_BYTES:raise DownloadError('Download oltre il limite di 32 GiB.')
                        if shutil.disk_usage(self.root).free<len(chunk)+64*1024**2:raise DownloadError('Spazio disco insufficiente.')
                        f.write(chunk);digest.update(chunk)
                        if time.monotonic()-last>0.6:
                            self._update(jid,downloaded_bytes=done,total_bytes=total or plan.get('size',0));last=time.monotonic()
                    f.flush();os.fsync(f.fileno())
                if total and total!=done:raise DownloadError('Download troncato: dimensione differente da quella dichiarata.')
            if cancel.is_set():raise Cancelled('Download annullato.')
            sha=digest.hexdigest();expected=plan.get('sha256','')
            if expected and sha!=expected:raise DownloadError('SHA-256 non corrispondente alla fonte: file scartato.')
            self._update(jid,status='validating',sha256=sha,downloaded_bytes=done,total_bytes=done)
            if plan.get('kind')=='text_encoder_file':
                from core.qwen_weight_format import inspect_weights
                inspected=inspect_weights(part)
                if not any('layers.' in k or 'language_model' in k for k in inspected['keys']):raise DownloadError('This file does not contain text encoder layers.')
                detected={'kind':'text_encoder_file','family':'unknown'}
            else:detected=classify_model_file(part)
            if detected['family']=='other' or detected['kind']=='unsupported':raise DownloadError('This file declares an incompatible model family or component.')
            chosen=plan['kind'] if plan['kind']!='auto' else detected['kind']
            if chosen=='unknown':chosen=plan.get('kind_hint','auto')
            if chosen in DESTINATIONS and detected['kind'] not in ('unknown',chosen):raise DownloadError('Il tipo rilevato non corrisponde alla destinazione scelta.')
            confirmed=detected['family']=='qwen21' and chosen in DESTINATIONS and detected['kind']==chosen
            self._update(jid,detected=detected,kind=chosen)
            if not confirmed and not plan.get('builtin'):
                self._update(jid,status='needs_confirmation',error='File structure checked; confirm that its source identifies this component as Qwen Image 2.1 before installing it.')
                return
            self.finalize(jid,chosen,accept_unknown=bool(plan.get('builtin')))
        except Cancelled as e:
            part.unlink(missing_ok=True);self._update(jid,status='cancelled',error=str(e))
        except Exception as e:
            part.unlink(missing_ok=True)
            msg=str(e) if isinstance(e,DownloadError) else 'Errore download/verifica locale. Nessun modello è stato installato; riprova.'
            self._update(jid,status='failed',error=msg)
    def finalize(self,jid,kind,accept_unknown=False):
        if kind not in DESTINATIONS:raise DownloadError('Choose a Qwen transformer, LoRA or VAE.')
        with self.lock:
            j=self.jobs.get(jid)
            if not j or j['status'] not in ('needs_confirmation','validating'):raise DownloadError('Questo download non è pronto per l’installazione.')
            if self.events.get(jid,threading.Event()).is_set():raise Cancelled('Download annullato.')
            detected=j.get('detected') or {}
            if detected.get('kind') not in (kind,'unknown'):raise DownloadError('Tipo file non compatibile con la cartella.')
            if detected.get('family')!='qwen21' and not accept_unknown:raise DownloadError('Conferma prima la compatibilità indicata dalla fonte.')
            part=self.incoming/(jid+'.part')
            if not part.is_file() or part.is_symlink():raise DownloadError('File temporaneo non disponibile; ripeti il download.')
            dest_dir=self.root/DESTINATIONS[kind];dest_dir.mkdir(parents=True,exist_ok=True)
            if not dest_dir.resolve().is_relative_to(self.root.resolve()):raise DownloadError('La destinazione download non può essere un link esterno. Usa la configurazione manuale.')
            filename=safe_name(j['filename']);dest=dest_dir/filename
            if dest.exists() or dest.is_symlink():dest=dest_dir/(dest.stem+'-'+jid[:8]+dest.suffix)
            self._update(jid,status='installing')
            # Hard-link publish is atomic and cannot overwrite any existing file.
            os.link(part,dest);part.unlink()
            meta={'source':j['source'],'license_reported':j['license'],'sha256':j['sha256'],
                  'downloaded_at':time.time(),'terms_confirmed':True,'policy_version':POLICY_VERSION,
                  'detected':detected,'not_a_license_certification':True}
            warning=''
            try:atomic_json(dest.with_suffix('.source.json'),meta)
            except OSError:warning='File installato, ma impossibile salvare i metadati di provenienza.'
            try:self.on_installed(kind,dest)
            except Exception:warning+=' File installato; usa Aggiorna elenco o selezionalo manualmente.'
            return self._update(jid,status='complete',kind=kind,path=str(dest),error='',warning=warning)
