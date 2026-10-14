"""Component profiles, VAE tests and re-decode. All routes use the existing
model-hub JSON/CSRF guard plus the application login and Origin guards.
"""
from __future__ import annotations
import json
from pathlib import Path
from flask import request,jsonify,Response
from core.runtime_env import SHARED_ROOT
from core.qwen_components import component_status,decode_options,EXTRA_DEFAULTS,DECODE_DEFAULTS
from core.qwen_profiles import ModelProfiles
from core.qwen_latents import LatentStore,identifier
from core.model_registry import normalize_config
from core.qwen_assets import read_object


def register(app,cb):
    profiles=ModelProfiles(SHARED_ROOT/'presets/model-profiles/qwen21')
    latents=LatentStore(SHARED_ROOT/'cache/qwen-latents')
    app.extensions['qwen_lab']={'profiles':profiles,'latents':latents}
    def current():return cb['model_by_id']('qwen_image_21') or {}
    def config():return dict(current().get('config') or {})
    def project(pid):
        identifier(pid)
        if not cb['one']('SELECT id FROM projects WHERE id=? AND COALESCE(trashed,0)=0',(pid,)):raise ValueError('Choose an active project.')
        return pid
    def snapshot(d,decode_only=False):
        cfg=normalize_config(config(),d.get('config') or {})
        if decode_only:
            check=component_status(cfg,decode_only=True)
            if not check['ready']:raise ValueError('; '.join(f['detail'] for f in check['fields'] if not f['exists']))
        return cfg
    def enqueue(pid,tool,cfg,params):
        params={**params,'_model_config_snapshot':cfg}
        jid=cb['create_job'](project(pid),tool,'qwen_image_21','VAE test' if tool=='qwen_vae_test' else 'Re-decode preserved Qwen latent',params)
        cb['start_job'](jid)
        return jsonify(ok=True,id=jid,tool=tool),202

    @app.get('/api/model-hub/components')
    def components_get():return jsonify(ok=True,config=config(),components=component_status(config()),defaults=EXTRA_DEFAULTS)
    @app.post('/api/model-hub/components/check')
    def components_check():
        cfg=snapshot(request.get_json())
        return jsonify(ok=True,components=component_status(cfg),validation=cb['validate_model']({'id':'qwen_image_21','provider':'qwen_image21','config':cfg}))
    @app.post('/api/model-hub/components/vae/test')
    def components_vae_test():
        d=request.get_json();cfg=snapshot(d,decode_only=True)
        return enqueue(d.get('project_id'),'qwen_vae_test',cfg,{})
    @app.get('/api/model-hub/components/tests/<jid>')
    def components_test_report(jid):
        identifier(jid);job=cb['one']('SELECT id,project_id,status,error,tool FROM jobs WHERE id=?',(jid,))
        if not job or job['tool']!='qwen_vae_test':return jsonify(error='VAE test not found.'),404
        p=SHARED_ROOT/'cache/qwen-component-tests'/(jid+'.json')
        return jsonify(ok=True,job=job,report=read_object(p) if p.is_file() else None)

    @app.get('/api/model-hub/profiles')
    def profiles_list():return jsonify(ok=True,profiles=profiles.list())
    @app.post('/api/model-hub/profiles')
    def profiles_save():
        d=request.get_json();data={**d,'config':normalize_config(config(),d.get('config') or {})}
        saved=profiles.save(data,rid=d.get('id'))
        return jsonify(ok=True,profile=saved,profiles=profiles.list())
    @app.post('/api/model-hub/profiles/import')
    def profiles_import():
        d=request.get_json()
        # Always import as a new ID; never overwrite an existing profile by file ID.
        saved=profiles.save(d)
        return jsonify(ok=True,profile=saved,profiles=profiles.list())
    @app.get('/api/model-hub/profiles/<rid>/download')
    def profiles_download(rid):
        d=profiles.get(rid)
        return Response(json.dumps(d,ensure_ascii=False,indent=2),mimetype='application/json',
            headers={'Content-Disposition':'attachment; filename="qwen-model-profile-'+rid+'.json"','Cache-Control':'no-store'})
    @app.post('/api/model-hub/profiles/<rid>/duplicate')
    def profiles_duplicate(rid):
        d=profiles.get(rid);d['name']=(d['name']+' (copy)')[:120]
        saved=profiles.save(d);return jsonify(ok=True,profile=saved,profiles=profiles.list())
    @app.delete('/api/model-hub/profiles/<rid>')
    def profiles_delete(rid):return jsonify(ok=True,**profiles.delete(rid),profiles=profiles.list())
    @app.post('/api/model-hub/profiles/<rid>/activate')
    def profiles_activate(rid):
        if request.get_json().get('confirm') is not True:raise ValueError('Confirm activation of this saved component profile.')
        d=profiles.get(rid);cfg={**EXTRA_DEFAULTS,**d['config']}
        model=cb['update_model']('qwen_image_21',{'config':cfg})
        # Existing jobs use their immutable configuration snapshots.
        return jsonify(ok=True,model=model,validation=cb['validate_model'](model),profile=d['name'],loras=d.get('loras',[]))

    @app.get('/api/model-hub/latents')
    def latents_list():
        pid=project(request.args.get('project_id'));items=latents.list(pid)
        # Resolve output paths to actual asset IDs, never expose filesystem reads
        # through an arbitrary user-supplied path parameter.
        rows=cb['rows']('SELECT id,path FROM artifacts WHERE project_id=?',(pid,));by_path={r['path']:r['id'] for r in rows}
        for item in items:
            for result in item.get('decode_results',[]):
                aid=by_path.get(result.get('output_path'))
                if aid:result['artifact_id']=aid;result['url']='/api/artifacts/'+aid+'/file'
        return jsonify(ok=True,latents=items,quota=latents.quota())
    @app.post('/api/model-hub/latents/<rid>/decode')
    def latent_decode(rid):
        with latents.locked():
            d=request.get_json();pid=project(d.get('project_id'));meta=latents.metadata(rid,pid)
            cfg=config();choice=d.get('vae_selection','configured')
            if choice=='original':cfg.update(vae_path='',vae_config_path='')
            elif choice!='configured':
                assets=app.extensions['model_hub']['assets'];entry=next((x for x in assets.inventory(cfg) if x['id']==choice and x['kind']=='vae' and x['present']),None)
                if not entry:raise ValueError('Select a compatible VAE from the catalog.')
                cfg.update(vae_path=entry['path'],vae_config_path='')
                if Path(entry['path']).is_file():
                    # Match a config next to the file only when explicitly present;
                    # otherwise require selection in Components, not a guessed model.
                    cp=Path(entry['path']).parent/'config.json'
                    if not cp.is_file():raise ValueError('For this single-file VAE configure its matching config.json in Models → Components, then choose Current configuration.')
                    cfg['vae_config_path']=str(cp)
            options=d.get('options') or {}
            if not isinstance(options,dict) or set(options)-set(DECODE_DEFAULTS):raise ValueError('Only decode options are accepted; resolution/seed/sampling cannot change in a re-decode.')
            options=decode_options(cfg,options)
            check=component_status(cfg,decode_only=True)
            if not check['ready']:raise ValueError('; '.join(f['detail'] for f in check['fields'] if not f['exists']))
            from core.qwen_components import assert_same_latent_space
            assert_same_latent_space(check['details']['vae']['config'],meta['contract'])
            return enqueue(pid,'qwen_decode',cfg,{'latent_id':rid,**options})
    @app.delete('/api/model-hub/latents/<rid>')
    def latent_delete(rid):
        d=request.get_json();pid=project(d.get('project_id'))
        if cb['one']("SELECT id FROM jobs WHERE status IN ('running','queued') LIMIT 1"):
            return jsonify(error='Wait until the queue is idle before deleting cached latents.'),409
        def require_idle():
            if cb['one']("SELECT id FROM jobs WHERE status IN ('running','queued') LIMIT 1"):
                raise ValueError('The queue became busy; cached latent was kept. Wait until idle.')
        return jsonify(ok=True,**latents.delete(rid,pid,before_delete=require_idle))
