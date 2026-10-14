"""Authenticated Qwen-only models hub. Account and download transport preserved."""
import os,threading,shutil
from pathlib import Path
from flask import request,jsonify
from core.runtime_env import DATA_ROOT,MODELS_ROOT,SHARED_ROOT
from core.csrf import token,valid
from core.download_security import PublicHTTPS,TokenStore,DownloadError
from core.model_downloads import ModelDownloads
from core.model_uploads import BrowserModelUploads
from core.qwen_assets import QwenAssets,QwenAssetError,scan_model_folder,BULK_FILE_KINDS,BULK_DIR_KINDS
from core.model_policy import catalog,POLICY_VERSION
from core.image_runtime import health as worker_health
from core.task_adapters import support_matrix

def register_model_hub(app,cb):
    tokens=TokenStore(DATA_ROOT/'secrets');transport=PublicHTTPS(tokens)
    assets=QwenAssets(MODELS_ROOT,SHARED_ROOT/'config/qwen-files.json')
    dl=ModelDownloads(MODELS_ROOT,SHARED_ROOT/'downloads/model-hub',transport,
        lambda kind,path:assets.add(str(path),kind,accept_unknown=True),
        start_worker=os.environ.get('SDXL_STUDIO_NO_BACKGROUND')!='1')
    uploads=BrowserModelUploads(MODELS_ROOT,assets);upload_lock=threading.RLock()
    app.extensions['model_hub']={'downloads':dl,'tokens':tokens,'transport':transport,'uploads':uploads,'assets':assets}

    @app.before_request
    def qwen_hub_guard():
        scoped=request.path.startswith(('/api/model-hub/','/api/models/'))
        if not scoped or request.method not in ('POST','PUT','PATCH','DELETE'):return
        if not valid('model-hub',request.headers.get('X-CSRF-Token','')):
            return jsonify(error='Session token expired. Please retry.',code='csrf_failed'),403
        if request.path.startswith('/api/model-hub/uploads/') and request.path.endswith('/chunk'):return
        if not request.is_json or not isinstance(request.get_json(silent=True),dict):return jsonify(error='Expected a JSON object.'),400

    @app.errorhandler(QwenAssetError)
    def asset_error(exc):return jsonify(error=str(exc)),400
    @app.errorhandler(DownloadError)
    def download_error(exc):return jsonify(error=str(exc)),400

    def current():return cb['model_by_id']('qwen_image_21') or {}
    def inventory():return assets.inventory(current().get('config') or {})

    @app.get('/api/model-hub/status')
    def hub_status():
        model=current();health=worker_health()
        return jsonify(ok=True,edition='qwen21',csrf=token('model-hub'),models=[{**model,'validation':cb['validate_model'](model),'task_support':support_matrix(model)}],
            catalog=catalog(),policy_version=POLICY_VERSION,accounts=tokens.public(),jobs=dl.list(),
            local_catalog=inventory(),paths={'repository':str(MODELS_ROOT/'Qwen-Image-2.1'),
                'transformer':str(MODELS_ROOT/'Qwen-2.1/transformers'),'lora':str(MODELS_ROOT/'Qwen-2.1/loras'),'vae':str(MODELS_ROOT/'Qwen-2.1/vae')},
            runtime=health,disk_free=shutil.disk_usage(MODELS_ROOT).free)

    @app.get('/api/model-hub/local')
    def hub_local_list():return jsonify(ok=True,catalog=inventory())
    @app.post('/api/model-hub/local')
    def hub_local_add():
        d=request.get_json()
        raw=d.get('path');kind=d.get('kind');path=Path(str(raw or '')).expanduser()
        if kind=='lora' and path.is_dir():
            result=assets.add_lora_folder(raw, accept_unknown=d.get('accept_unknown') is True)
            return jsonify(ok=True,bulk=True,**result,catalog=inventory(),browse_directory=str(path.resolve()))
        result=assets.add(raw,kind,accept_unknown=d.get('accept_unknown') is True)
        return jsonify(ok=True,bulk=False,entry=result,catalog=inventory(),browse_directory=str(Path(result['path']).parent if Path(result['path']).is_file() else Path(result['path'])))
    @app.post('/api/model-hub/local/scan-folder')
    def hub_scan_folder():
        d=request.get_json() or {}
        return jsonify(ok=True,**scan_model_folder(d.get('path')))

    @app.post('/api/model-hub/local/link-folder')
    def hub_link_folder():
        d=request.get_json() or {};root=d.get('root');selected=d.get('entries')
        if not isinstance(selected,list) or not 1<=len(selected)<=250:
            raise QwenAssetError('Select between 1 and 250 entries per import.')
        # Re-scan on POST: do not trust arbitrary unchecked client paths.
        scan=scan_model_folder(root);allowed={item['path']:item for item in scan['candidates']}
        imported=[];errors=[];done=set()
        for item in selected:
            if not isinstance(item,dict):raise QwenAssetError('Invalid bulk selection.')
            path=item.get('path');kind=item.get('kind')
            if not isinstance(path,str) or path not in allowed or path in done:
                raise QwenAssetError('A selected path was not in this folder preview.')
            done.add(path)
            if kind not in (BULK_DIR_KINDS if allowed[path]['folder'] else BULK_FILE_KINDS):
                raise QwenAssetError('Unsupported component kind for '+str(path))
            try:
                entry=assets.add(path,kind,accept_unknown=d.get('accept_unknown') is True)
                imported.append(entry)
            except (QwenAssetError,ValueError,OSError) as exc:
                errors.append({'path':path,'kind':kind,'error':str(exc)[:500]})
        return jsonify(ok=True,linked=len(imported),failed=len(errors),entries=imported,errors=errors,
                       catalog=inventory(),note='Original model files are kept in place; no bulk load to RAM/VRAM.' )

    @app.delete('/api/model-hub/local/<rid>')
    def hub_local_remove(rid):return jsonify(ok=True,**assets.remove(rid),catalog=inventory())

    @app.post('/api/model-hub/select')
    def hub_select():
        d=request.get_json();entry=next((e for e in inventory() if e['id']==d.get('id')),None)
        if not entry or not entry['present'] or entry.get('selectable') is False:raise QwenAssetError(entry.get('issue') if entry else 'Choose an available catalog entry.')
        fields={'repository':'model_dir','transformer':'model_path','vae':'vae_path','text_encoder':'text_encoder_dir','text_encoder_file':'text_encoder_path','processor':'processor_dir','scheduler':'scheduler_dir'}
        if entry['kind'] not in fields:raise QwenAssetError('LoRA adapters are selected per generation.')
        field=fields[entry['kind']];patch={field:entry['path']}
        if entry['kind']=='text_encoder_file':patch['text_encoder_dir']=''
        elif entry['kind']=='text_encoder':patch['text_encoder_path']=''
        if entry['kind']=='vae':
            if Path(entry['path']).is_file():raise QwenAssetError('For single-file VAE select the file and its matching config.json in Components, then save.')
            patch['vae_config_path']=''
        model=cb['update_model']('qwen_image_21',{'config':patch})
        return jsonify(ok=True,model=model,validation=cb['validate_model'](model))

    @app.post('/api/model-hub/uploads')
    def hub_upload_create():
        d=request.get_json()
        with upload_lock:upload=uploads.create(d.get('filename'),d.get('size'),d.get('kind'))
        return jsonify(ok=True,upload=upload,chunk_size=8*1024**2)
    @app.post('/api/model-hub/uploads/<uid>/chunk')
    def hub_upload_chunk(uid):
        with upload_lock:upload=uploads.append(uid,request.form.get('offset'),request.files.get('chunk'))
        return jsonify(ok=True,upload=upload)
    @app.post('/api/model-hub/uploads/<uid>/finalize')
    def hub_upload_finalize(uid):
        with upload_lock:result=uploads.finalize(uid,request.get_json().get('accept_unknown') is True)
        return jsonify(ok=True,**result,catalog=inventory())
    @app.delete('/api/model-hub/uploads/<uid>')
    def hub_upload_cancel(uid):
        with upload_lock:result=uploads.cancel(uid)
        return jsonify(ok=True,**result)

    @app.get('/api/model-hub/downloads')
    def hub_downloads():return jsonify(ok=True,jobs=dl.list())
    @app.post('/api/model-hub/inspect')
    def hub_inspect():
        d=request.get_json();kind=d.get('kind','auto');filename='without-transformer' if kind=='repository' and d.get('exclude_transformer') is True else str(d.get('filename') or '')
        return jsonify(ok=True,preview=dl.inspect(str(d.get('url') or ''),kind,filename))
    @app.post('/api/model-hub/downloads')
    def hub_start():
        d=request.get_json();return jsonify(ok=True,job=dl.start(d.get('preview_id'),d.get('accept_terms') is True)),202
    @app.post('/api/model-hub/downloads/<jid>/cancel')
    def hub_cancel(jid):return jsonify(ok=True,job=dl.cancel(jid))
    @app.post('/api/model-hub/downloads/<jid>/install')
    def hub_install(jid):
        d=request.get_json();return jsonify(ok=True,job=dl.finalize(jid,d.get('kind'),d.get('accept_unknown') is True))

    @app.put('/api/model-hub/accounts/<provider>')
    def hub_account_save(provider):
        if provider not in tokens.PROVIDERS:raise DownloadError('Piattaforma non supportata.')
        d=request.get_json() or {};token=d.get('token')
        if not isinstance(token,str) or not 8<=len(token.strip())<=2048 or any(not 33<=ord(c)<=126 for c in token.strip()):
            raise DownloadError('Incolla un token valido, senza spazi.')
        token=token.strip()
        if provider=='huggingface' and not token.startswith('hf_'):raise DownloadError('Inserisci un token personale Hugging Face (hf_...).')
        if d.get('store_without_test') is True:
            account='Token salvato · non verificato'
        else:
            url='https://huggingface.co/api/whoami-v2' if provider=='huggingface' else 'https://civitai.com/api/v1/me'
            data=transport.json(url,token_override=token)
            if not isinstance(data,dict):raise DownloadError('Risposta account non valida.')
            account=str(data.get('name') or data.get('username') or 'Accesso verificato')[:120]
        tokens.put(provider,token,account)
        return jsonify(ok=True,accounts=tokens.public())


    @app.delete('/api/model-hub/accounts/<provider>')
    def hub_account_delete(provider):
        tokens.delete(provider)
        return jsonify(ok=True,accounts=tokens.public())


