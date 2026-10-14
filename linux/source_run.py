#!/usr/bin/env python3
"""Manual developer entry point: use `python source_run.py web|image`.

Two terminals, same explicitly activated venv. No package installation,
service registration, automatic updates, uninstallers or model downloads.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from core.runtime_env import initialize_environment, SHARED_ROOT, DATA_ROOT

def prepare():
    initialize_environment()
    from core.model_registry import ensure_config
    from core.db import init
    ensure_config()
    init()
    from core.local_certificate import generate_certificate
    folder=SHARED_ROOT/'config'
    if not (folder/'cert.pem').is_file() or not (folder/'key.pem').is_file():
        generate_certificate(folder)
    auth=folder/'creator_auth.json'
    if not auth.exists():
        from core.runtime_env import atomic_json
        atomic_json(auth, {'enabled':True,'username':'','password_hash':'','session_hours':12})

def main():
    parser=argparse.ArgumentParser(description='Run Qwen Image Lab from manually provisioned source')
    parser.add_argument('component',choices=['web','image'])
    args=parser.parse_args()
    if sys.prefix==sys.base_prefix:
        parser.error('Activate your own virtual environment first; system Python is not supported.')
    prepare()
    if args.component=='web':
        from app import main as start
        print('Manual source web server (TLS on 127.0.0.1:8298 by default).',flush=True)
    else:
        from core.runtime_security import require_reviewed_torch
        require_reviewed_torch()
        from services.image_worker import main as start
        print('Manual source image worker (loopback 127.0.0.1:8299).',flush=True)
    start()

if __name__=='__main__': main()
