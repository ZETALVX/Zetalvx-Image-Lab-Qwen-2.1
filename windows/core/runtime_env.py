"""Isolated, per-user paths for the Qwen 2.1 edition. No global environment is edited."""
from __future__ import annotations
import json, os, secrets, tempfile, sys
from pathlib import Path
from core.platform_support import default_data_root, venv_python

VERSION = '1.0.76'
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('QWEN_SOURCE_HOME') or str(default_data_root().with_name('ImageLabQwen21Source'))).expanduser().resolve()
SHARED_ROOT = DATA_ROOT/'shared'
MODELS_ROOT = DATA_ROOT/'models'
RUNTIME_ROOT = DATA_ROOT/'runtime'

def runtime_dir(name):
    """The environment is owned and provisioned by the source user."""
    if name not in ('app','qwen21'):
        raise ValueError('Unknown environment')
    return Path(sys.prefix).resolve() if sys.prefix != sys.base_prefix else APP_ROOT/'.venv'

SETTINGS_FILE = SHARED_ROOT/'config/settings.json'

DEFAULTS = {
    'host':'127.0.0.1','port':8298,'https':True,'image_port':8299,
}


def atomic_json(path: Path, data: object):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def settings():
    data=json.loads(SETTINGS_FILE.read_text(encoding='utf-8')) if SETTINGS_FILE.exists() else {}
    if not isinstance(data,dict): raise ValueError('Invalid settings.json (must be an object)')
    cfg={**DEFAULTS, **data}
    # One-launch CLI override. It is inherited by the supervisor/web process but never written to settings.json.
    override=os.environ.get('SDXL_STUDIO_HOST_OVERRIDE','').strip()
    if override in ('127.0.0.1','0.0.0.0'):cfg['host']=override
    return cfg

def initialize_environment():
    # Qwen 2.1 full-frame VAE decode benefits from an allocator that can grow segments
    # instead of leaving large unusable reservations. Preserve any explicit user setting.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    for d in [DATA_ROOT,SHARED_ROOT/'config',SHARED_ROOT/'logs',SHARED_ROOT/'run',
              MODELS_ROOT/'Qwen-Image-2.1',MODELS_ROOT/'Qwen-2.1/transformers',
              MODELS_ROOT/'Qwen-2.1/loras',MODELS_ROOT/'Qwen-2.1/vae']:
        d.mkdir(parents=True,exist_ok=True)
    cfg=settings()
    if not SETTINGS_FILE.exists():atomic_json(SETTINGS_FILE,cfg)
    token_file=SHARED_ROOT/'config/worker_key'
    if not token_file.exists():
        try:
            fd=os.open(token_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as f:f.write(secrets.token_hex(32))
        except FileExistsError:pass
    env={
        'CREATOR_SHARED_ROOT':str(SHARED_ROOT),
        'HF_HOME':str(DATA_ROOT/'cache/huggingface'),
        'HF_HUB_CACHE':str(DATA_ROOT/'cache/huggingface/hub'),
        'CREATOR_HOST':str(cfg['host']),'CREATOR_PORT':str(cfg['port']),
        'CREATOR_IMAGE_WORKER_URL':'http://127.0.0.1:'+str(cfg['image_port']),
        'CREATOR_IMAGE_WORKER_HOST':'127.0.0.1','CREATOR_IMAGE_WORKER_PORT':str(cfg['image_port']),
        'SDXL_WORKER_KEY':token_file.read_text().strip(),'HF_HUB_DISABLE_TELEMETRY':'1',
    }
    # Never inherit endpoints belonging to the separate SDXL/Creator installation.
    for k in list(os.environ):
        if k.startswith(('CREATOR_IDENTITY_','CREATOR_TRAINING_','CREATOR_LORA_')):os.environ.pop(k,None)
    for k,v in env.items():os.environ[k]=str(v)
    return cfg


def worker_headers():return {'X-SDXL-Worker-Key':os.environ.get('SDXL_WORKER_KEY','')}

def protect_worker(app):
    import hmac
    from flask import request, jsonify
    @app.before_request
    def _check_internal_request():
        if request.method in ('POST','PUT','PATCH','DELETE'):
            key=os.environ.get('SDXL_WORKER_KEY','')
            if not key or not hmac.compare_digest(request.headers.get('X-SDXL-Worker-Key',''), key):
                return jsonify(ok=False,error='Internal worker authentication required'),403
