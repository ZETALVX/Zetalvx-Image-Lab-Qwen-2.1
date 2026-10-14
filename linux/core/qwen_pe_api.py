"""Authenticated Prompt Enhancer controls. All writes inherit model-hub CSRF/Origin."""
import json
from pathlib import Path
from flask import request,jsonify
from core.runtime_env import SHARED_ROOT,worker_headers
from core.qwen_prompt_enhancer import PEStore,PE_KINDS,normalize_config,build_request
from core.qwen_external_runtime import ComfyRuntimeStore


def register(app,cb):
    store=PEStore(SHARED_ROOT);runtime_store=ComfyRuntimeStore(SHARED_ROOT);assets=app.extensions['model_hub']['assets']
    app.extensions['qwen_pe']={'store':store}
    base='/api/model-hub/pe'
    def body():
        d=request.get_json()
        if not isinstance(d,dict):raise ValueError('Expected a JSON object.')
        return d
    def project(pid):
        if not isinstance(pid,str) or not cb['one']('SELECT id FROM projects WHERE id=? AND COALESCE(trashed,0)=0',(pid,)):
            raise ValueError('Select an active project first.')
        return pid
    def check_job(jid,pid):
        store.result_path(jid)
        job=cb['one']("SELECT * FROM jobs WHERE id=? AND project_id=? AND tool='qwen_prompt_enhance'",(jid,project(pid)))
        if not job:raise ValueError('PE job not found in this project.')
        return job
    def with_runtime(cfg):
        rt=runtime_store.config()
        if not runtime_store.verified(rt):raise ValueError('Verify the global ComfyUI runtime in Settings → External runtimes before using Prompt Enhancer.')
        if not (rt.get('capabilities') or {}).get('prompt_enhancer_imports'):
            raise ValueError('The verified ComfyUI runtime did not pass the Prompt Enhancer import check. Re-verify it in Settings → External runtimes or choose a compatible ComfyUI environment.')
        out=dict(cfg);out.update(comfy_root=rt.get('comfy_root',''),comfy_python=rt.get('comfy_python',''),runtime_confirmed=True)
        return out
    def pe_only(cfg):
        out=dict(cfg);out.update(comfy_root='',comfy_python='',runtime_confirmed=False);return out
    @app.get(base+'/config')
    def pe_config_get():
        return jsonify(ok=True,config=store.config(),catalog=[x for x in assets.inventory() if x['kind'] in PE_KINDS.values()],runtime=runtime_store.public())
    @app.post(base+'/config')
    def pe_config_save():return jsonify(ok=True,config=store.save(body()))
    @app.post(base+'/probe')
    def pe_probe():
        from core.image_runtime import URL
        import requests
        cfg=with_runtime(normalize_config(body().get('config') or store.config(),store.config()))
        try:
            r=requests.post(URL+'/pe/probe',json={'config':cfg},headers=worker_headers(),timeout=70)
            data=r.json()
        except (requests.RequestException, ValueError) as exc:
            raise ValueError('Cannot check the PE runtime. Check that the Image Lab image worker is running.') from exc
        if not r.ok or not data.get('ok'):raise ValueError(data.get('error') or 'PE runtime check failed.')
        return jsonify(data)
    @app.post(base+'/jobs')
    def pe_enqueue():
        d=body();pid=project(d.get('project_id'))
        if d.get('confirm') is not True:raise ValueError('Explicitly request a PE preview before queueing.')
        cfg=normalize_config(d.get('config') or store.config(),store.config());exec_cfg=with_runtime(cfg)
        refs=d.get('reference_artifact_ids') or []
        snapshot=build_request(exec_cfg,assets.inventory(),d.get('prompt'),refs)
        for aid in refs:
            if not cb['one']("SELECT id FROM artifacts WHERE id=? AND project_id=? AND type='image'",(aid,pid)):
                raise ValueError('PE reference images must belong to the current project.')
        jid=cb['create_job'](pid,'qwen_prompt_enhance','qwen_prompt_enhancer',snapshot['original_prompt'],{'_pe_request':snapshot})
        store.save(pe_only(cfg));cb['start_job'](jid)
        return jsonify(ok=True,id=jid),202
    @app.get(base+'/jobs')
    def pe_history():
        pid=project(request.args.get('project_id'))
        jobs=cb['rows']("SELECT id,project_id,status,phase,error,created_at,duration_seconds FROM jobs WHERE project_id=? AND tool='qwen_prompt_enhance' ORDER BY created_at DESC,rowid DESC LIMIT 30",(pid,))
        return jsonify(ok=True,jobs=jobs)
    @app.get(base+'/jobs/<jid>')
    def pe_result(jid):
        job=check_job(jid,request.args.get('project_id'))
        return jsonify(ok=True,job={k:job.get(k) for k in ('id','project_id','status','phase','error','duration_seconds')},result=store.result(jid) if job.get('status')=='completed' else None)
    @app.post(base+'/jobs/<jid>/cancel')
    def pe_cancel(jid):
        job=check_job(jid,body().get('project_id'))
        if job['status'] in ('queued','running'):cb['request_cancel'](jid)
        return jsonify(ok=True)
