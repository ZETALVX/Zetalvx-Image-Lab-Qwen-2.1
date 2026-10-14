"""Authenticated unified download view and queue settings for Zetalvx Image Lab UI."""
from __future__ import annotations
import json
from flask import jsonify, request
from .download_manager import ACTIVE, COORDINATOR, normalize_job
from .runtime_env import SHARED_ROOT, atomic_json
from .csrf import valid

CONFIG=SHARED_ROOT/'config/download-manager.json'

def _load_config():
    try:d=json.loads(CONFIG.read_text(encoding='utf-8'))
    except (OSError,ValueError):d={}
    limit=int(d.get('parallel_downloads') or 1) if isinstance(d,dict) else 1
    if limit not in (1,2,3):limit=1
    return {'parallel_downloads':limit}

def register(app):
    cfg=_load_config();COORDINATOR.set_limit(cfg['parallel_downloads'])
    def all_jobs():
        out=[];hub=app.extensions.get('model_hub') or {};dl=hub.get('downloads')
        if dl:
            try:out.extend(normalize_job('models',x) for x in dl.list())
            except Exception:pass
        queued=sorted((j for j in out if j.get('status')=='queued'),key=lambda x:(x.get('created_at') or 0,x.get('id') or ''))
        for pos,j in enumerate(queued,1):j['queue_position']=pos
        out.sort(key=lambda x:(x.get('created_at') or 0),reverse=True);return out[:150]
    @app.get('/api/download-manager')
    def download_manager_list():
        jobs=all_jobs();snap=COORDINATOR.snapshot()
        return jsonify(ok=True,jobs=jobs,active=sum(j['status'] in ACTIVE for j in jobs),queued=sum(j['status']=='queued' for j in jobs),settings={'parallel_downloads':snap['limit'],'running_slots':snap['active'],'max_parallel_downloads':3})
    @app.post('/api/download-manager/settings')
    def download_manager_settings():
        if not valid('model-hub',request.headers.get('X-CSRF-Token','')):return jsonify(error='Session token expired. Please retry.',code='csrf_failed'),403
        d=request.get_json(silent=True) or {}
        try:limit=int(d.get('parallel_downloads'))
        except (TypeError,ValueError):return jsonify(error='Choose 1, 2 or 3 simultaneous downloads.'),400
        if limit not in (1,2,3):return jsonify(error='Choose 1, 2 or 3 simultaneous downloads.'),400
        COORDINATOR.set_limit(limit);atomic_json(CONFIG,{'schema':1,'parallel_downloads':limit})
        return jsonify(ok=True,settings={'parallel_downloads':limit,'max_parallel_downloads':3})
    @app.get('/api/download-manager/<source>/<jid>')
    def download_manager_detail(source,jid):
        if source=='models':obj=(app.extensions.get('model_hub') or {}).get('downloads')
        else:return jsonify(error='Unknown download source'),404
        if not obj:return jsonify(error='Download source unavailable'),404
        try:return jsonify(ok=True,job=normalize_job(source,obj.get(jid) if hasattr(obj,'get') else next(x for x in obj.list() if x.get('id')==jid)))
        except Exception:return jsonify(error='Download not found'),404
    return all_jobs
