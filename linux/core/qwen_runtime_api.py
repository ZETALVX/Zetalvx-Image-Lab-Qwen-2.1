"""Authenticated global external-runtime controls for Qwen Image Lab."""
from __future__ import annotations
import requests
from flask import request,jsonify
from core.runtime_env import SHARED_ROOT,worker_headers
from core.image_runtime import URL
from core.qwen_external_runtime import ComfyRuntimeStore,discover_runtimes


def register(app):
    store=ComfyRuntimeStore(SHARED_ROOT)
    app.extensions['qwen_comfy_runtime']={'store':store}
    base='/api/model-hub/runtime/comfyui'

    def body():
        d=request.get_json()
        if not isinstance(d,dict):raise ValueError('Expected a JSON object.')
        return d

    @app.get(base)
    def comfy_runtime_get():
        return jsonify(ok=True,config=store.public(),runtimes=discover_runtimes())

    @app.post(base)
    def comfy_runtime_save():
        d=body();cfg=store.save(d.get('config') if isinstance(d.get('config'),dict) else d)
        return jsonify(ok=True,config={**cfg,'verified':store.verified(cfg)})

    @app.post(base+'/probe')
    def comfy_runtime_probe():
        d=body();raw=d.get('config') if isinstance(d.get('config'),dict) else d
        # Validate and persist user intent first. Any path/trust change invalidates the old probe.
        cfg=store.invalidate(raw)
        try:
            r=requests.post(URL+'/comfy/probe',json={'config':cfg},headers=worker_headers(),timeout=100)
            data=r.json()
        except (requests.RequestException,ValueError) as exc:
            raise ValueError('Cannot verify the external ComfyUI runtime. Check that the Image Lab image worker is running.') from exc
        if not r.ok or not data.get('ok'):raise ValueError(data.get('error') or 'ComfyUI runtime verification failed.')
        verified=store.mark_verified(cfg,data.get('capabilities') or {})
        return jsonify(ok=True,config={**verified,'verified':store.verified(verified)},probe=data)
