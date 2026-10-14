import sqlite3
import os,json,uuid,re,shutil,threading,time,subprocess,signal,secrets,hmac,math,sys,tempfile
from pathlib import Path
from datetime import timedelta
from flask import Flask,render_template,jsonify,request,send_file,session,redirect,url_for,make_response
from werkzeug.exceptions import HTTPException, BadRequest, ClientDisconnected, RequestEntityTooLarge, UnsupportedMediaType
from functools import wraps
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash

from core.runtime_env import initialize_environment, DATA_ROOT, MODELS_ROOT, RUNTIME_ROOT, settings, atomic_json, worker_headers, VERSION
initialize_environment()
ROOT=Path(__file__).resolve().parent
from core.db import init,rows,one,execute,create_job,add_message
from core.paths import SHARED_ROOT,PROJECTS_DIR,ARTIFACTS_DIR,UPLOADS_DIR,THUMBNAILS_DIR,CACHE_DIR
from core.model_registry import load_registry,update_model,validate_model
from core.image_runtime import health as image_worker_health,probe as image_worker_probe
from core.media_tools import run_media_tool
from core.job_executor import start_job,recover_pending_jobs,read_progress,request_cancel,queue_snapshot
from core.job_management import require_project,change_prompt,move_queued,hide_terminal,clear_terminal,restore_history,delete_terminal_record,JobControlError
from core.bulk_actions import library_bulk,jobs_bulk
from core.task_adapters import support_matrix,task_support,task_from_tool

app=Flask(__name__)
@app.errorhandler(ValueError)
def user_value_error(error):
    return jsonify(ok=False,error=str(error)),400

from core.login_throttle import LoginThrottle
login_throttle=LoginThrottle(DATA_ROOT/"secrets/login-attempts.sqlite3")

# --- Simple local login gate (no sign-up) ---
AUTH_CONFIG_PATH=SHARED_ROOT/"config"/"creator_auth.json"
SESSION_SECRET_PATH=SHARED_ROOT/"config"/"creator_session_secret.txt"
from core.first_run_security import FirstRunCodes, SetupCodeError, local_setup_request, print_setup_code
first_run_codes = FirstRunCodes(AUTH_CONFIG_PATH, DATA_ROOT / "secrets")
_first_run_display_lock=threading.RLock()
_first_run_display_code=os.environ.get("SDXL_STUDIO_FIRST_RUN_DISPLAY_CODE","")
_SETUP_CONTROL_PURPOSE=b"zetalvx-first-run-code-control-v1"

def _setup_control_token():
    try:
        key=SESSION_SECRET_PATH.read_text(encoding="utf-8").strip().encode("utf-8")
    except OSError:
        return ""
    return hmac.new(key,_SETUP_CONTROL_PURPOSE,"sha256").hexdigest()

def _ensure_local_setup_code(*,rotate=False):
    """Return the live clear setup code kept only in the web process memory.

    Disk retains only the SHA-256 state managed by FirstRunCodes. A local page
    silently renews an expired code; an authenticated host-control request can
    rotate it immediately so SETUP-CODE.cmd and the local page stay in sync.
    """
    global _first_run_display_code
    with _first_run_display_lock:
        remaining=0 if rotate else first_run_codes.remaining_seconds(_first_run_display_code)
        if remaining<=0:
            code=first_run_codes.issue()
            _first_run_display_code=code or ""
            remaining=first_run_codes.remaining_seconds(_first_run_display_code) if code else 0
        return _first_run_display_code or None,remaining

def _ensure_auth_files():
    from core.private_files import private_directory, protect
    private_directory(AUTH_CONFIG_PATH.parent)
    if not AUTH_CONFIG_PATH.exists():
        AUTH_CONFIG_PATH.write_text(json.dumps({
            "enabled":True,
            "username":"",
            "password_hash":"",
            "session_hours":12
        },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        try:AUTH_CONFIG_PATH.chmod(0o600)
        except Exception:pass
    if not SESSION_SECRET_PATH.exists():
        try:
            fd=os.open(SESSION_SECRET_PATH, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(secrets.token_hex(48)+"\n")
        except FileExistsError:pass
    protect(AUTH_CONFIG_PATH)
    protect(SESSION_SECRET_PATH)
    private_directory(DATA_ROOT/"secrets")
    for protected in (AUTH_CONFIG_PATH.parent/"key.pem", DATA_ROOT/"secrets/first-run-code.json"):
        if protected.exists():protect(protected)

def _load_auth_config():
    _ensure_auth_files()
    try:
        cfg=json.loads(AUTH_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        cfg={}
    return {
        "enabled":bool(cfg.get("enabled",True)),
        "username":str(cfg.get("username") or ""),
        "password_hash":str(cfg.get("password_hash") or ""),
        "session_hours":max(1,min(int(cfg.get("session_hours",12) or 12),168)),
    }

def _auth_ok():
    cfg=_load_auth_config()
    if not cfg["enabled"]:
        return True
    return bool(session.get("creator_authenticated")) and hmac.compare_digest(str(session.get("creator_username") or "").encode("utf-8"),cfg["username"].encode("utf-8")) and session.get("auth_stamp")==cfg["password_hash"][-20:]


def _save_auth_config(username,password_hash,session_hours=12,enabled=True):
    username=str(username or "C").strip()[:80] or "C"
    atomic_json(AUTH_CONFIG_PATH,{"enabled":bool(enabled),"username":username,"password_hash":str(password_hash or ""),"session_hours":max(1,min(int(session_hours or 12),168))})
    return _load_auth_config()

def _login_session(cfg):
    session.clear()
    session["creator_authenticated"]=True
    session["creator_csrf_nonce"]=secrets.token_urlsafe(32)
    session["creator_username"]=cfg["username"]
    session["auth_stamp"]=cfg["password_hash"][-20:]
    session.permanent=True
    app.permanent_session_lifetime=timedelta(hours=cfg["session_hours"])

_ensure_auth_files()
app.secret_key=SESSION_SECRET_PATH.read_text(encoding="utf-8").strip()
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
    SESSION_COOKIE_NAME="imagelab_qwen21_session",
    MAX_CONTENT_LENGTH=128*1024*1024,
    SESSION_COOKIE_SECURE=bool(settings().get("https",True)),
)

@app.before_request
def _creator_login_gate():
    from core.http_security import same_origin
    if request.method in {"POST","PUT","PATCH","DELETE"}:
        if not same_origin(request.headers.get("Origin"), request.host_url):
            return jsonify(error="Cross-origin request refused"),403
    if request.endpoint in {"login","first_run","first_run_local_code","static","edition_health"} or request.path in {"/login","/first-run","/api/first-run/local-code"}:
        return None
    if _auth_ok():
        return None
    if request.path.startswith("/api/"):
        return jsonify(error="Authentication required",auth_required=True),401
    nxt=request.full_path if request.query_string else request.path
    return redirect(url_for("login",next=nxt))

@app.after_request
def _security_headers(response):
    # Log status/endpoint only: never request bodies, prompts, cookies or tokens.
    if request.endpoint in {'job_create'}:
        print(f"[REQUEST] {time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())} method={request.method} endpoint={request.endpoint} status={response.status_code}",flush=True)
    if request.path.endswith(('/download','/export')):
        print(f"[DOWNLOAD] method={request.method} endpoint={request.endpoint or 'unmatched'} status={response.status_code} bytes={response.headers.get('Content-Length','stream')}",flush=True)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response

@app.route("/first-run",methods=["GET","POST"])
def first_run():
    request.max_content_length=16*1024
    cfg=_load_auth_config()
    if cfg["password_hash"]:
        response=redirect(url_for("login"))
        response.headers["Cache-Control"]="no-store"
        return response
    local_client=local_setup_request(request.remote_addr,request.host_url,request.headers)
    # An explicitly proxied request is not treated as local. Direct LAN setup
    # requires HTTPS; forwarding headers do not grant transport trust.
    if not local_client and not request.is_secure:
        return "La prima configurazione da un altro dispositivo richiede HTTPS.",403,{"Cache-Control":"no-store"}
    error="";status=200;retry_after=0
    setup_code_display=None; setup_code_expires_in=0
    if local_client:
        try:
            setup_code_display,setup_code_expires_in=_ensure_local_setup_code()
        except (OSError,RuntimeError):
            setup_code_display=None;setup_code_expires_in=0
    username=(str(request.form.get("username") or "").strip() if request.method=="POST" else "")
    network=str(request.form.get("network") or ("local" if local_client else "lan"))
    csrf=session.get("first_run_csrf")
    if not csrf:
        csrf=secrets.token_urlsafe(32)
        session["first_run_csrf"]=csrf
    if request.method=="POST":
        supplied=request.form.get("first_run_csrf","")
        if not supplied.isascii() or not secrets.compare_digest(supplied,csrf):
            error="Sessione di configurazione scaduta. Ricarica la pagina e riprova.";status=403
        else:
            try:
                # Serializes parallel local/LAN submissions and code rotation.
                # Account existence is rechecked INSIDE the lock, not only above.
                with first_run_codes.locked():
                    if first_run_codes.account_exists_locked():
                        raise SetupCodeError("Account già configurato. Usa il login.",409)
                    if not local_client:
                        first_run_codes.verify_locked(request.form.get("setup_code",""))
                    password=str(request.form.get("password") or "")
                    confirm=str(request.form.get("confirm_password") or "")
                    if not 1<=len(username)<=80:
                        error="Il nome utente deve avere da 1 a 80 caratteri.";status=400
                    elif not 10<=len(password)<=1024:
                        error="La password deve avere da 10 a 1024 caratteri.";status=400
                    elif password!=confirm:
                        error="Le password non coincidono.";status=400
                    elif network not in {"local","lan"}:
                        error="Scegli accesso locale oppure LAN.";status=400
                    else:
                        password_hash=generate_password_hash(password)
                        from core.runtime_env import SETTINGS_FILE
                        app_cfg=settings()
                        old_host=app_cfg["host"]
                        new_host="0.0.0.0" if network=="lan" else "127.0.0.1"
                        app_cfg.update(host=new_host,https=True)
                        atomic_json(SETTINGS_FILE,app_cfg)
                        cfg=_save_auth_config(username,password_hash,12,True)
                        # Persisted account itself makes all bootstrap codes
                        # unusable even if cleanup is interrupted by a crash.
                        try:first_run_codes.revoke_locked()
                        except OSError:app.logger.warning("First-setup state cleanup failed; account is already configured.")
                        _login_session(cfg)
                        response=redirect('/?onboarding=1'+('&network_restart=1' if new_host!=old_host else ''))
                        response.headers["Cache-Control"]="no-store"
                        return response
            except SetupCodeError as exc:
                error=str(exc);status=exc.status;retry_after=exc.retry_after
            except (OSError,RuntimeError):
                error="Impossibile salvare la configurazione. Controlla i permessi sul computer host e riprova.";status=503
    response=make_response(render_template("first_run.html",username=username,error=error,
        network=network,code_required=not local_client,setup_code_display=setup_code_display,
        setup_code_expires_in=setup_code_expires_in,setup_code_command=('SETUP-CODE.cmd' if os.name=='nt' else 'imagelab-qwen setup-code'),first_run_csrf=csrf),status)
    response.headers["Cache-Control"]="no-store"
    response.headers["Referrer-Policy"]="no-referrer"
    if retry_after:response.headers["Retry-After"]=str(retry_after)
    return response

@app.route("/api/first-run/local-code",methods=["GET","POST"])
def first_run_local_code():
    if not local_setup_request(request.remote_addr,request.host_url,request.headers):
        return jsonify(error="Local host only"),403,{"Cache-Control":"no-store"}
    cfg=_load_auth_config()
    if cfg["password_hash"]:
        return jsonify(error="First setup already completed"),409,{"Cache-Control":"no-store"}
    rotate=request.method=="POST"
    if rotate:
        supplied=str(request.headers.get("X-Zetalvx-Setup-Control") or "")
        expected=_setup_control_token()
        if not expected or not supplied.isascii() or not hmac.compare_digest(supplied,expected):
            return jsonify(error="Host control authentication failed"),403,{"Cache-Control":"no-store"}
    try:
        code,remaining=_ensure_local_setup_code(rotate=rotate)
    except (OSError,RuntimeError):
        return jsonify(error="Setup code unavailable"),503,{"Cache-Control":"no-store"}
    if not code:
        return jsonify(error="First setup already completed"),409,{"Cache-Control":"no-store"}
    response=jsonify(code=code,expires_in=int(remaining))
    response.headers["Cache-Control"]="no-store"
    response.headers["Referrer-Policy"]="no-referrer"
    return response

@app.route("/login",methods=["GET","POST"])
def login():
    cfg=_load_auth_config()
    if not cfg["enabled"]:
        return redirect(url_for("home"))
    if not cfg["password_hash"]:
        return redirect(url_for("first_run"))
    if _auth_ok():
        return redirect(url_for("home"))
    error="";status=200;wait=0
    if request.method=="POST":
        request.max_content_length=16*1024
        try:wait=login_throttle.reserve(request.remote_addr)
        except (OSError,RuntimeError,sqlite3.Error):
            return render_template("login.html",username=cfg["username"],error="Accesso temporaneamente non disponibile. Riprova.",next_path="/"),503
        if wait:
            error="Troppi tentativi di accesso. Attendi un minuto e riprova.";status=429
        else:
            username=str(request.form.get("username") or "").strip()
            password=str(request.form.get("password") or "")
            correct=False
            if 1<=len(username)<=80 and 1<=len(password)<=1024:
                valid_hash=check_password_hash(cfg["password_hash"],password)
                correct=hmac.compare_digest(username.encode("utf-8"),cfg["username"].encode("utf-8")) and valid_hash
            if correct:
                login_throttle.success(request.remote_addr)
                _login_session(cfg)
                nxt=str(request.form.get("next") or request.args.get("next") or "/")
                if len(nxt)>2048 or not nxt.startswith("/") or nxt.startswith("//") or "\\" in nxt or any(ord(c)<32 for c in nxt):nxt="/"
                return redirect(nxt)
            error="Utente o password non corretti."
    response=make_response(render_template("login.html",username=cfg["username"],error=error,next_path=request.args.get("next") or "/"),status)
    response.headers["Cache-Control"]="no-store"
    if wait:response.headers["Retry-After"]=str(wait)
    return response

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

init()
if os.environ.get("SDXL_STUDIO_NO_BACKGROUND")!="1":recover_pending_jobs()

def models():return load_registry().get("models",[])
def model_by_id(mid):return next((m for m in models() if m.get("id")==mid),None)

def recipes():
    out=[]
    for r in rows("SELECT * FROM recipes ORDER BY name"):
        try:r["params"]=json.loads(r.pop("params_json") or "{}")
        except:r["params"]={}
        out.append(r)
    return out


def _download_attachment(path, download_name=None):
    p=Path(path).expanduser().resolve()
    if not p.is_file():raise FileNotFoundError(str(p))
    response=send_file(p,as_attachment=True,download_name=download_name or p.name,conditional=True,max_age=0,mimetype='application/octet-stream' if p.suffix.lower()=='.safetensors' else None)
    response.headers['Cache-Control']='private, no-store'
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['Accept-Ranges']='bytes'
    return response


# --- Persistent Image preset system (v0.1.70) ---
USER_IMAGE_PRESETS_DIR=SHARED_ROOT/"presets"/"user"/"image"
IMAGE_PRESET_BINDINGS_PATH=SHARED_ROOT/"config"/"image_preset_bindings.json"

def _ensure_image_preset_storage():
    USER_IMAGE_PRESETS_DIR.mkdir(parents=True,exist_ok=True)


def _safe_preset_id(raw):
    value=re.sub(r"[^a-zA-Z0-9_-]+","-",str(raw or "").strip()).strip("-").lower()
    return value[:80]

def _normalize_user_image_preset(payload,preset_id=None):
    if not isinstance(payload,dict):raise ValueError('Preset must be a JSON object.')
    from core.qwen_request import validate_params
    d=dict(payload);name=str(d.get('name') or 'User preset').strip()[:120]
    if d.get('edition') not in (None,'qwen21'):raise ValueError('This preset belongs to a different Image Lab edition.')
    p=validate_params(d.get('params') or {})
    # Presets store settings, never reference IDs or image/model bytes.
    for key in ('source_artifact_id','reference_artifact_ids'):p.pop(key,None)
    include=d.get('include_prompt',True)
    if type(include) is not bool:raise ValueError('include_prompt must be boolean.')
    return {'schema_version':3,'edition':'qwen21','id':_safe_preset_id(preset_id or d.get('id') or name) or uuid.uuid4().hex[:12],
        'name':name,'description':str(d.get('description') or '')[:500],'created_at':float(d.get('created_at') or time.time()),
        'updated_at':time.time(),'model_id':'qwen_image_21','params':p,'include_prompt':include,
        'prompt':str(d.get('prompt') or '')[:30000] if include else '',
        'negative_prompt':str(d.get('negative_prompt') or '')[:20000] if include else ''}

def _user_image_preset_path(pid):
    return USER_IMAGE_PRESETS_DIR/f"{_safe_preset_id(pid)}.json"

def _list_user_image_presets():
    _ensure_image_preset_storage()
    out=[]
    for path in USER_IMAGE_PRESETS_DIR.glob("*.json"):
        try:
            d=json.loads(path.read_text(encoding="utf-8"))
            if isinstance(d,dict):out.append(d)
        except Exception:pass
    return sorted(out,key=lambda x:(str(x.get("name") or "").lower(),str(x.get("id") or "")))

_ensure_image_preset_storage()

# --- Persistent Video preset system (v0.1.75) ---


def decode_artifact(a):
    if not a:return None
    try:a["metadata"]=json.loads(a.get("metadata_json") or "{}")
    except:a["metadata"]={}
    a["url"]=f"/api/artifacts/{a['id']}/file"
    return a

def project_payload(pid):
    p=one("SELECT * FROM projects WHERE id=?",(pid,))
    if not p:return None
    p["state"]=one("SELECT * FROM project_state WHERE project_id=?",(pid,)) or {"active_artifact_id":""}
    p["artifacts"]=[decode_artifact(x) for x in rows("SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at DESC LIMIT 400",(pid,))]
    jobs=rows("SELECT * FROM jobs WHERE project_id=? AND COALESCE(history_hidden,0)=0 ORDER BY CASE WHEN status IN ('queued','running','cancelling') THEN 0 ELSE 1 END,created_at DESC,id DESC LIMIT 400",(pid,))
    for j in jobs:
        live=read_progress(j["id"])
        if live and j.get("status") in ("queued","running","cancelling"):
            j.update({k:v for k,v in live.items() if k in ("phase","progress","step_current","step_total","elapsed_seconds","error")})
        if not j.get("elapsed_seconds"):
            j["elapsed_seconds"]=j.get("duration_seconds") or live.get("elapsed_seconds",0) if live else (j.get("duration_seconds") or 0)
        try:j["params"]=json.loads(j.get("params_json") or "{}")
        except:j["params"]={}
    p["jobs"]=jobs
    p["messages"]=rows("SELECT role,content,created_at FROM conversations WHERE project_id=? ORDER BY created_at LIMIT 400",(pid,))
    return p


def _creator_gpu_jobs_busy():
    active=rows("SELECT id,tool,status FROM jobs WHERE status IN ('queued','running','cancelling') AND NOT (status='queued' AND COALESCE(cancel_requested,0)=1) ORDER BY created_at LIMIT 1")
    return active[0] if active else None


def _gpu_guard_payload():
    j=_creator_gpu_jobs_busy()
    return {'busy':bool(j),'owner':'image' if j else '', 'job_id':j.get('id') if j else None}

def _release_creator_gpu():
    from core.image_runtime import unload
    return {'image':unload()}


@app.get("/")
def home():return render_template("index.html")



@app.get('/api/bootstrap')
def bootstrap():
    return jsonify(projects=rows('SELECT * FROM projects ORDER BY trashed ASC,sort_order ASC,created_at ASC'),
        models=[{**m,'validation':validate_model(m),'task_support':support_matrix(m)} for m in models()],
        user_image_presets=_list_user_image_presets(),shared_root=str(SHARED_ROOT),runtime=image_worker_health(),edition='qwen21',version=VERSION)


@app.get("/api/image/presets/user")
def image_user_presets_list():
    return jsonify(ok=True,presets=_list_user_image_presets())

@app.post("/api/image/presets/user")
def image_user_presets_save():
    d=request.get_json(force=True) or {}
    requested_id=_safe_preset_id(d.get("id"))
    preset=_normalize_user_image_preset(d,requested_id or None)
    path=_user_image_preset_path(preset["id"])
    if path.exists() and not requested_id:
        preset["id"]=_safe_preset_id(f"{preset['id']}-{uuid.uuid4().hex[:6]}")
        path=_user_image_preset_path(preset["id"])
    elif path.exists():
        try:
            old=json.loads(path.read_text(encoding="utf-8"))
            preset["created_at"]=float(old.get("created_at") or preset["created_at"])
        except Exception:pass
    path.write_text(json.dumps(preset,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return jsonify(ok=True,preset=preset,presets=_list_user_image_presets())

@app.post("/api/image/presets/user/import")
def image_user_presets_import():
    data=None
    if request.files.get("file"):
        try:data=json.loads(request.files["file"].read().decode("utf-8"))
        except Exception as e:return jsonify(error=f"Invalid preset JSON: {e}"),400
    else:
        data=request.get_json(force=True) or {}
    items=data if isinstance(data,list) else [data]
    saved=[]
    for item in items[:100]:
        if not isinstance(item,dict):continue
        preset=_normalize_user_image_preset(item,item.get("id"))
        path=_user_image_preset_path(preset["id"])
        if path.exists():
            preset["id"]=_safe_preset_id(f"{preset['id']}-{uuid.uuid4().hex[:6]}")
            path=_user_image_preset_path(preset["id"])
        path.write_text(json.dumps(preset,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        saved.append(preset)
    return jsonify(ok=True,imported=len(saved),presets=_list_user_image_presets())

@app.post("/api/image/presets/user/<pid>/duplicate")
def image_user_presets_duplicate(pid):
    path=_user_image_preset_path(pid)
    if not path.exists():return jsonify(error="Preset not found"),404
    try:d=json.loads(path.read_text(encoding="utf-8"))
    except Exception:return jsonify(error="Preset is invalid"),400
    d["id"]=_safe_preset_id(f"{d.get('id') or pid}-copy-{uuid.uuid4().hex[:5]}")
    d["name"]=f"{d.get('name') or 'User preset'} Copy"
    d["created_at"]=time.time()
    preset=_normalize_user_image_preset(d,d["id"])
    _user_image_preset_path(preset["id"]).write_text(json.dumps(preset,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return jsonify(ok=True,preset=preset,presets=_list_user_image_presets())

@app.delete("/api/image/presets/user/<pid>")
def image_user_presets_delete(pid):
    path=_user_image_preset_path(pid)
    if path.exists():path.unlink()
    return jsonify(ok=True,presets=_list_user_image_presets())

@app.get("/api/image/presets/user/<pid>/download")
def image_user_presets_download(pid):
    path=_user_image_preset_path(pid)
    if not path.exists():return jsonify(error="Preset not found"),404
    return _download_attachment(path,path.name)


@app.get("/api/projects/<pid>")
def project_get(pid):
    p=project_payload(pid)
    return jsonify(project=p) if p else (jsonify(error="Project not found"),404)

@app.post("/api/projects")
def project_create():
    d=request.get_json(force=True) or {}
    pid=uuid.uuid4().hex
    name=(d.get("name") or "Untitled Project").strip()[:120]
    description=(d.get("description") or "").strip()[:500]
    next_order=(one("SELECT COALESCE(MAX(sort_order),0)+10 AS n FROM projects WHERE trashed=0") or {}).get("n") or 10
    execute("INSERT INTO projects(id,name,description,sort_order) VALUES(?,?,?,?)",
            (pid,name,description,int(next_order)))
    execute("INSERT OR IGNORE INTO project_state(project_id) VALUES(?)",(pid,))
    return jsonify(ok=True,project=project_payload(pid))



@app.put("/api/projects/<pid>")
def project_update(pid):
    if not one("SELECT id FROM projects WHERE id=?",(pid,)):
        return jsonify(error="Project not found"),404
    d=request.get_json(force=True) or {}
    name=str(d.get("name") or "").strip()
    if not name:
        return jsonify(error="Project name cannot be empty"),400
    if len(name)>120:
        return jsonify(error="Project name is too long"),400
    description=str(d.get("description") or "").strip()[:500]
    execute("UPDATE projects SET name=?,description=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",(name,description,pid))
    return jsonify(ok=True,project=project_payload(pid))


def _normalize_project_order():
    current=rows("SELECT id FROM projects WHERE trashed=0 ORDER BY sort_order ASC, created_at ASC, id ASC")
    for idx,row in enumerate(current,1):
        execute("UPDATE projects SET sort_order=? WHERE id=?",(idx*10,row["id"]))
    return current

@app.post("/api/projects/<pid>/move")
def project_move(pid):
    d=request.get_json(force=True) or {}
    direction=str(d.get("direction") or "").lower()
    if direction not in ("up","down"):
        return jsonify(error="Choose up or down"),400
    current=_normalize_project_order()
    ids=[x["id"] for x in current]
    if pid not in ids:
        return jsonify(error="Project not found"),404
    i=ids.index(pid);j=i-1 if direction=="up" else i+1
    if j<0 or j>=len(ids):
        return jsonify(ok=True,changed=False)
    a=one("SELECT sort_order FROM projects WHERE id=?",(ids[i],)) or {}
    b=one("SELECT sort_order FROM projects WHERE id=?",(ids[j],)) or {}
    execute("UPDATE projects SET sort_order=? WHERE id=?",(int(b.get("sort_order") or ((j+1)*10)),ids[i]))
    execute("UPDATE projects SET sort_order=? WHERE id=?",(int(a.get("sort_order") or ((i+1)*10)),ids[j]))
    return jsonify(ok=True,changed=True)


@app.put("/api/projects/<pid>/active-artifact")
def active_artifact(pid):
    d=request.get_json(force=True) or {};aid=d.get("artifact_id") or ""
    execute("""INSERT INTO project_state(project_id,active_artifact_id,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)
               ON CONFLICT(project_id) DO UPDATE SET active_artifact_id=excluded.active_artifact_id,updated_at=CURRENT_TIMESTAMP""",(pid,aid))
    return jsonify(ok=True)

@app.post('/api/projects/<pid>/upload')
def upload(pid):
    from PIL import Image, ImageOps
    if not one('SELECT id FROM projects WHERE id=?',(pid,)):return jsonify(error='Project not found'),404
    f=request.files.get('file')
    if not f:return jsonify(error='Choose an image'),400
    aid=uuid.uuid4().hex;directory=UPLOADS_DIR/pid;directory.mkdir(parents=True,exist_ok=True)
    dest=directory/f'{aid}.png'
    try:
        with Image.open(f.stream) as im:
            if im.width*im.height>40000000:raise ValueError('Maximum input size: 40 megapixels.')
            ImageOps.exif_transpose(im).convert('RGB').save(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        return jsonify(error='Invalid/unsupported image: '+str(e)),400
    role=request.form.get('role') or 'reference'
    meta={'source':'upload','original_name':secure_filename(f.filename or 'image.png'),'role':role,'media_type':'image'}
    execute('INSERT INTO artifacts(id,project_id,type,path,parent_id,metadata_json) VALUES(?,?,?,?,?,?)',(aid,pid,'image',str(dest),None,json.dumps(meta)))
    return jsonify(ok=True,artifact=decode_artifact(one('SELECT * FROM artifacts WHERE id=?',(aid,))))


@app.post('/api/image-tools/extract-frame')
def edition_extract_frame():
    import tempfile
    from core.frame_extract import extract_frame
    pid=str(request.form.get('project_id') or '')
    if not re.fullmatch('[a-f0-9]{32}',pid) or not one('SELECT id FROM projects WHERE id=?',(pid,)):return jsonify(error='Apri prima un progetto.'),400
    f=request.files.get('file')
    if not f or Path(f.filename or '').suffix.lower() not in {'.mp4','.mkv','.webm','.mov','.avi'}:return jsonify(error='Scegli un video MP4, MOV, MKV, WebM o AVI.'),400
    aid=uuid.uuid4().hex;dest=ARTIFACTS_DIR/pid/(aid+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
    try:
        (SHARED_ROOT/'cache').mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='frame-',dir=SHARED_ROOT/'cache') as folder:
            src=Path(folder)/('input'+Path(f.filename).suffix.lower());total=0
            with src.open('wb') as out:
                while True:
                    chunk=f.stream.read(1024*1024)
                    if not chunk:break
                    total+=len(chunk)
                    if total>120*1024**2:raise ValueError('Per estrarre un frame usa un video fino a 120 MiB.')
                    out.write(chunk)
            info=extract_frame(src,dest,request.form.get('mode') or 'first',request.form.get('timestamp') or '0')
        meta={'source':'media_tools','operation':'extract_frame','original_name':secure_filename(f.filename),'extraction':info,'width':info['width'],'height':info['height']}
        execute('INSERT INTO artifacts(id,project_id,type,path,parent_id,metadata_json) VALUES(?,?,?,?,?,?)',(aid,pid,'image',str(dest),None,json.dumps(meta)))
        return jsonify(ok=True,artifact=decode_artifact(one('SELECT * FROM artifacts WHERE id=?',(aid,))))
    except (ValueError,OSError) as exc:
        dest.unlink(missing_ok=True)
        return jsonify(error=str(exc) if isinstance(exc,ValueError) else 'Impossibile salvare il frame. Controlla spazio e permessi.'),400

@app.post("/api/media-tools/run")
def api_media_tools_run():
    d=request.get_json(force=True) or {}
    project_id=(d.get("project_id") or "").strip()
    source_ids=[str(x).strip() for x in (d.get("source_artifact_ids") or []) if str(x).strip()]
    legacy=(d.get("source_artifact_id") or "").strip()
    if legacy and not source_ids:
        source_ids=[legacy]
    operation=(d.get("operation") or "").strip()
    params=d.get("params") or {}

    if not project_id:
        return jsonify(error="Select a project first"),400
    if not one("SELECT id FROM projects WHERE id=?",(project_id,)):
        return jsonify(error="Project not found"),404
    if not source_ids:
        return jsonify(error="Choose or upload one or more source assets"),400
    if not operation:
        return jsonify(error="Choose an operation"),400

    artifacts=[]
    results=[]
    errors=[]
    for source_id in source_ids:
        src=one("SELECT * FROM artifacts WHERE id=? AND project_id=?",(source_id,project_id))
        if not src:
            errors.append({"source_id":source_id,"error":"Source asset not found in the active project"})
            continue
        try:
            result=run_media_tool(project_id,src,operation,params)
            artifact=decode_artifact(one("SELECT * FROM artifacts WHERE id=?",(result["artifact_id"],)))
            results.append(result)
            artifacts.append(artifact)
        except Exception as e:
            errors.append({"source_id":source_id,"error":str(e)})

    if not artifacts:
        return jsonify(error="All media operations failed",errors=errors),400

    # Keep legacy single-result fields for older frontends.
    payload={"ok":True,"results":results,"artifacts":artifacts,"errors":errors}
    if len(artifacts)==1:
        payload["result"]=results[0]
        payload["artifact"]=artifacts[0]
    return jsonify(**payload)


@app.post("/api/projects/<pid>/trash")
def api_project_trash(pid):
    p=one("SELECT * FROM projects WHERE id=?",(pid,))
    if not p:
        return jsonify(error="Project not found"),404

    # Stop/mark pending work belonging to the project.
    for j in rows("SELECT id,status FROM jobs WHERE project_id=?",(pid,)):
        if j.get("status") in ("queued","running","cancelling"):
            try:
                request_cancel(j["id"])
            except Exception:
                pass

    execute(
        "UPDATE projects SET trashed=1,trashed_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (pid,)
    )
    return jsonify(ok=True)


@app.post("/api/projects/<pid>/restore")
def api_project_restore(pid):
    if not one("SELECT id FROM projects WHERE id=?",(pid,)):
        return jsonify(error="Project not found"),404
    execute(
        "UPDATE projects SET trashed=0,trashed_at=NULL,updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (pid,)
    )
    return jsonify(ok=True)


@app.post("/api/projects/<pid>/delete-forever")
def api_project_delete_forever(pid):
    p=one("SELECT * FROM projects WHERE id=?",(pid,))
    if not p:
        return jsonify(error="Project not found"),404
    if not int(p.get("trashed") or 0):
        return jsonify(error="Project must be moved to Trash first"),400

    jobs=rows("SELECT id,status FROM jobs WHERE project_id=?",(pid,))
    if any(j.get("status")=="running" for j in jobs):
        return jsonify(error="A project job is still stopping. Wait a moment and try Delete forever again."),409

    # Remove every concrete artifact path first.
    for art in rows("SELECT path FROM artifacts WHERE project_id=?",(pid,)):
        try:
            path=Path(art.get("path") or "")
            if path.exists() and path.is_file():
                path.unlink()
        except Exception:
            pass

    # Remove job progress/cancel files.
    for j in jobs:
        for pth in (
            SHARED_ROOT/"cache"/"job_progress"/f"{j['id']}.json",
            SHARED_ROOT/"cache"/"job_cancel"/f"{j['id']}.cancel",
        ):
            try:
                pth.unlink(missing_ok=True)
            except Exception:
                pass

    # Remove project-owned directories.
    for base in (PROJECTS_DIR,ARTIFACTS_DIR,UPLOADS_DIR,THUMBNAILS_DIR):
        try:
            shutil.rmtree(base/pid,ignore_errors=True)
        except Exception:
            pass

    # Latents are project data too; delete only cache records owned by this project.
    from core.qwen_latents import LatentStore
    cache=LatentStore(SHARED_ROOT/'cache/qwen-latents')
    for record in cache.list(pid):cache.delete(record['id'],pid)
    # Database dependants first, then project.
    execute("DELETE FROM conversations WHERE project_id=?",(pid,))
    execute("DELETE FROM project_state WHERE project_id=?",(pid,))
    execute("DELETE FROM artifacts WHERE project_id=?",(pid,))
    execute("DELETE FROM jobs WHERE project_id=?",(pid,))
    execute("DELETE FROM projects WHERE id=?",(pid,))
    return jsonify(ok=True)


@app.get("/api/artifacts/<aid>")
def artifact_get(aid):
    a=decode_artifact(one("SELECT * FROM artifacts WHERE id=?",(aid,)))
    return jsonify(artifact=a) if a else (jsonify(error="Artifact not found"),404)

@app.delete("/api/artifacts/<aid>")
def artifact_delete(aid):
    art=one("SELECT * FROM artifacts WHERE id=?",(aid,))
    if not art:return jsonify(error="Artifact not found"),404
    try:
        p=Path(art.get("path") or "")
        if p.exists() and p.is_file(): p.unlink()
    except Exception as e:
        return jsonify(error=f"Could not delete artifact file: {e}"),500
    execute("UPDATE jobs SET output_artifact_id=NULL WHERE output_artifact_id=?",(aid,))
    execute("UPDATE project_state SET active_artifact_id='' WHERE active_artifact_id=?",(aid,))
    execute("DELETE FROM artifacts WHERE id=?",(aid,))
    return jsonify(ok=True)

@app.post('/api/jobs/<jid>/cancel')
def job_cancel(jid):
    j=one('SELECT * FROM jobs WHERE id=?',(jid,))
    if not j:return jsonify(error='Job not found'),404
    request_cancel(jid)
    return jsonify(ok=True)



def _job_control_response(fn,*args):
    try:return jsonify(ok=True,**fn(*args))
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.post('/api/projects/<pid>/jobs/<jid>/cancel')
def project_job_cancel(pid,jid):
    try:
        require_project(pid)
        j=one('SELECT id FROM jobs WHERE id=? AND project_id=?',(jid,pid))
        if not j:raise JobControlError('Job not found in this project.',404)
        return jsonify(ok=True,changed=request_cancel(jid))
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.post('/api/projects/<pid>/jobs/<jid>/move')
def project_job_move(pid,jid):
    try:
        require_project(pid)
        return _job_control_response(move_queued,pid,jid,(request.get_json(silent=True) or {}).get('direction'))
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.patch('/api/projects/<pid>/jobs/<jid>')
def project_job_edit(pid,jid):
    try:
        require_project(pid)
        return _job_control_response(change_prompt,pid,jid,(request.get_json(silent=True) or {}).get('prompt'))
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.post('/api/projects/<pid>/jobs/<jid>/hide')
def project_job_hide(pid,jid):
    try:
        require_project(pid)
        return _job_control_response(hide_terminal,pid,jid)
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.delete('/api/projects/<pid>/jobs/<jid>')
def project_job_delete_record(pid,jid):
    try:
        require_project(pid)
        return _job_control_response(delete_terminal_record,pid,jid)
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.post('/api/projects/<pid>/jobs/bulk')
def project_jobs_bulk(pid):
    try:
        require_project(pid)
        data=request.get_json(silent=True) or {}
        return jsonify(ok=True,**jobs_bulk(pid,data.get('ids'),data.get('action'),request_cancel))
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.post('/api/projects/<pid>/artifacts/bulk')
def project_artifacts_bulk(pid):
    try:
        require_project(pid)
        data=request.get_json(silent=True) or {}
        return jsonify(ok=True,**library_bulk(pid,data.get('ids'),data.get('action')))
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.post('/api/projects/<pid>/jobs/clear-history')
def project_jobs_clear_history(pid):
    try:
        require_project(pid)
        return _job_control_response(clear_terminal,pid)
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.post('/api/projects/<pid>/jobs/restore-history')
def project_jobs_restore_history(pid):
    try:
        require_project(pid)
        return _job_control_response(restore_history,pid)
    except JobControlError as e:return jsonify(error=str(e)),e.code

@app.post('/api/jobs/cleanup')
def jobs_cleanup():
    """Never cancel work in a different project merely by clicking Jobs."""
    pid=(request.get_json(silent=True) or {}).get('project_id')
    try:require_project(pid)
    except JobControlError as e:return jsonify(error=str(e)),e.code
    pending=rows("SELECT id,status FROM jobs WHERE project_id=? AND status IN ('queued','running','cancelling')",(pid,))
    queued=sum(j.get('status')=='queued' for j in pending)
    running=sum(j.get('status') in ('running','cancelling') for j in pending)
    for j in pending:request_cancel(j['id'])
    message=(f"Cancelled {queued} queued job(s) in this project." +
             (f" Cancellation requested for {running} running job(s)." if running else ""))
    return jsonify(ok=True,cancelled=queued,cancelling=running,message=message,queue=queue_snapshot(pid))

@app.get('/api/queue')
def queue_get():
    pid=request.args.get('project_id','')
    try:require_project(pid)
    except JobControlError as e:return jsonify(error=str(e)),e.code
    return jsonify(queue=queue_snapshot(pid))

@app.get("/api/artifacts/<aid>/file")
def artifact_file(aid):
    a=one("SELECT * FROM artifacts WHERE id=?",(aid,))
    if not a:return jsonify(error="Artifact not found"),404
    p=Path(a["path"])
    if not p.exists():return jsonify(error="Artifact file missing"),404
    if request.args.get('download')=='1':return _download_attachment(p,p.name)
    return send_file(p)

def validate_image_request(pid,tool,params):
    from core.qwen_request import validate_params
    if tool not in {'generate_image','reference','multi_image','edit_image','img2img'}:raise ValueError('Use the unified Qwen create/edit form.')
    normalized=validate_params(params)
    for aid in normalized['reference_artifact_ids']:
        if not one("SELECT id FROM artifacts WHERE id=? AND project_id=? AND type='image'",(aid,pid)):
            raise ValueError('Input image not in this project.')
    return normalized

@app.post('/api/jobs')
def job_create():
    from core.qwen_request import auto_tool,effective_prompt
    d=request.get_json(force=True) or {}
    if not isinstance(d,dict):return jsonify(error='Expected a job object.'),400
    pid=d.get('project_id')
    if not one('SELECT id FROM projects WHERE id=? AND COALESCE(trashed,0)=0',(pid,)):return jsonify(error='Select an active project first.'),400
    if d.get('model_id','qwen_image_21') not in ('auto','qwen_image_21'):return jsonify(error='Only Qwen Image 2.1 is available.'),400
    try:
        params=validate_image_request(pid,d.get('tool','generate_image'),d.get('params') or {})
        original_prompt=d.get('prompt') or ''
        prompt=effective_prompt(original_prompt,params.get('transparent_output',False))
    except (ValueError,TypeError) as exc:return jsonify(error=str(exc)),400
    from core.qwen_studio import resolve_request
    studio=app.extensions['qwen_studio']
    cfg,model_loras,label=resolve_request(d.get('model_setup') or {},studio['library'],studio['assets'],dict(model_by_id('qwen_image_21').get('config') or {}))
    # Per-job component overrides are applied after the optional Studio preset so
    # the Create/Edit form remains authoritative for this one request.
    q21_mode=str(params.get('qwen21_transformer_mode') or 'configured')
    q21_path=str(params.get('qwen21_transformer_path') or '').strip()
    if q21_mode=='original':cfg['model_path']=''
    elif q21_mode in ('gguf','safetensors'):cfg['model_path']=q21_path
    if params.get('qwen21_text_encoder_path'):
        cfg['text_encoder_path']=params['qwen21_text_encoder_path'];cfg['text_encoder_dir']=''
    if params.get('qwen21_vae_path'):
        cfg['vae_path']=params['qwen21_vae_path']
    if str(cfg.get('model_path') or '').lower().endswith('.gguf'):
        if params.get('reference_artifact_ids'):
            raise ValueError('Native GGUF generation currently supports text-to-image only. Remove input/reference images for this job.')
        cfg['quantized_loading']='native_comfyui'
        required=(('text_encoder_path','Text Encoder'),('vae_path','VAE'))
        missing=[]
        for key,name in required:
            raw=str(cfg.get(key) or '').strip();cp=Path(raw).expanduser() if raw else None
            if not raw or not cp.is_absolute() or cp.suffix.lower()!='.safetensors' or not cp.is_file():missing.append(name)
        if missing:
            raise ValueError('GGUF generation needs explicit single-file .safetensors companions: '+', '.join(missing)+'. Select them in Advanced Qwen settings or in the active model configuration.')
    params['_model_config_snapshot']=cfg
    if (d.get('model_setup') or {}).get('use_profile_loras') is True:params['loras']=model_loras
    if params['loras'] and str(cfg.get('model_path','')).lower().endswith('.gguf'):raise ValueError('LoRA with GGUF is not supported.')
    params['_model_setup']=label
    if d.get('prompt_enhancer') is not None:
        from core.qwen_prompt_enhancer import provenance
        params['prompt_enhancer']=provenance(app.extensions['qwen_pe']['store'],d['prompt_enhancer'],pid,original_prompt,params.get('reference_artifact_ids') or [])
    params['user_prompt']=original_prompt;batch=params['batch'];ids=[];seed=params['seed'];tool=auto_tool(params)
    for i in range(batch):
        p=dict(params);p['batch']=1
        if seed>=0:p['seed']=seed+i
        jid=create_job(pid,tool,'qwen_image_21',prompt,p);ids.append(jid);start_job(jid)
    return jsonify(ok=True,id=ids[0],ids=ids,batch=batch)

@app.get("/api/jobs/<jid>")
def job_get(jid):
    j=one("SELECT * FROM jobs WHERE id=?",(jid,))
    return jsonify(job=j) if j else (jsonify(error="Job not found"),404)


# 0.1.0.27: one planner for preview and enqueue; no silent Cartesian truncation.


@app.get("/api/runtime/gpu/guard")
def gpu_guard():
    return jsonify(ok=True,**_gpu_guard_payload())

@app.post("/api/runtime/gpu/release")
def gpu_release():
    guard=_gpu_guard_payload()
    if guard.get("busy"):
        return jsonify(ok=False,error="Zetalvx Image Lab GPU is busy",**guard),409
    return jsonify(ok=True,released=_release_creator_gpu())

@app.get("/api/runtime/image/health")
def image_runtime_health():return jsonify(image_worker_health())

@app.post("/api/runtime/image/unload")
def image_runtime_unload():
    if _gpu_guard_payload().get("busy"):return jsonify(error="A GPU job is active"),409
    return jsonify(_release_creator_gpu())

@app.get("/api/models/<mid>")
def model_get(mid):
    m=model_by_id(mid)
    if not m:return jsonify(error="Model not found"),404
    return jsonify(model=m,validation=validate_model(m))

@app.put("/api/models/<mid>")
def model_put(mid):
    d=request.get_json(force=True) or {};m=update_model(mid,d)
    if not m:return jsonify(error="Model not found"),404
    return jsonify(ok=True,model=m,validation=validate_model(m))

@app.post("/api/models/<mid>/validate")
def model_validate(mid):
    m=model_by_id(mid)
    if not m:return jsonify(error="Model not found"),404
    return jsonify(validation=validate_model(m))

@app.post("/api/models/<mid>/probe")
def model_probe(mid):
    m=model_by_id(mid)
    if not m:return jsonify(error="Model not found"),404
    return jsonify(probe=image_worker_probe(m))


# 1.0.11: an upload failure is not an inference failure. Keep this boundary
# scoped to Identity, before any input file/job is committed. Never read the
# raw WSGI stream here: Werkzeug owns multipart parsing and enforces the same
# MAX_CONTENT_LENGTH / form-memory / part-count limits as before.


_IDENTITY_DISPATCHER_STARTED=False
_IDENTITY_DISPATCHER_LOCK=threading.Lock()


@app.get('/api/health')
def edition_health():return jsonify(ok=True,edition='qwen21',version=VERSION)

@app.before_request
def edition_input_guard():
    # Route IDs are identifiers, never filesystem paths.
    for k,v in (request.view_args or {}).items():
        if isinstance(v,str) and k in {'pid','jid','did','iid','aid','lid','mid','uid','key'} and not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',v):
            return jsonify(error='Invalid identifier'),400
    if request.is_json:
        try:
            data=request.get_json(silent=False)
            json.dumps(data,allow_nan=False)
        except (ValueError,TypeError):return jsonify(error='Invalid JSON or non-finite value'),400

@app.after_request
def edition_headers(response):
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['X-Frame-Options']='DENY'
    response.headers['Referrer-Policy']='same-origin'
    if request.path.startswith('/api/') or request.path=='/login':response.headers['Cache-Control']='no-store'
    return response

@app.get('/api/account')
def edition_account_get():
    cfg=_load_auth_config()
    return jsonify(ok=True,username=cfg["username"],session_hours=cfg["session_hours"],enabled=cfg["enabled"])

@app.put('/api/account')
def edition_account_put():
    cfg=_load_auth_config();d=request.get_json(force=True) or {}
    current=str(d.get('current_password') or '')
    username=str(d.get('username') or cfg['username']).strip()[:80] or cfg['username']
    new_password=str(d.get('new_password') or '')
    confirm=str(d.get('confirm_password') or '')
    if not current or not check_password_hash(cfg['password_hash'],current):return jsonify(error='Password attuale non corretta'),400
    password_hash=cfg['password_hash']
    if new_password or confirm:
        if len(new_password)<10:return jsonify(error='La nuova password deve avere almeno 10 caratteri'),400
        if new_password!=confirm:return jsonify(error='Le nuove password non coincidono'),400
        password_hash=generate_password_hash(new_password)
    new_cfg=_save_auth_config(username,password_hash,cfg['session_hours'],cfg['enabled'])
    _login_session(new_cfg)
    return jsonify(ok=True,username=new_cfg['username'],message='Account aggiornato.')



# Restricted weight downloads are deliberately unavailable in this distribution.
# Existing local models remain configurable; acknowledgement does not grant rights.


@app.get('/api/session/tokens')
def edition_session_tokens():
    from core.csrf import token
    return jsonify(tokens={name:token(name) for name in ('model-hub','vision')})

@app.get('/api/setup')
def edition_setup_get():
    cfg=settings(); effective=os.environ.get('SDXL_STUDIO_EFFECTIVE_HOST',cfg.get('host','127.0.0.1'))
    import socket
    lan_ip=''
    try:
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.connect(('1.1.1.1',80));lan_ip=sock.getsockname()[0];sock.close()
    except OSError:
        try: lan_ip=socket.gethostbyname(socket.gethostname())
        except OSError: pass
    return jsonify(ok=True,settings=cfg,effective_host=effective,lan_ip=lan_ip,platform=('windows' if os.name=='nt' else 'linux'),models=models(),models_root=str(MODELS_ROOT),data_root=str(DATA_ROOT),version=VERSION,
                   image_runtime=image_worker_health(),
                   source_edition=True)

@app.put('/api/setup')
def edition_setup_put():
    from core.source_util import locked
    from core.runtime_env import SETTINGS_FILE,DEFAULTS
    from core.csrf import valid
    if not valid('vision',request.headers.get('X-CSRF-Token','')):return jsonify(error='Session token expired. Please retry.',code='csrf_failed'),403
    d=request.get_json(force=True) or {}
    with locked(SHARED_ROOT/'config/settings.lock',blocking=True):
        saved=json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
        cfg={**DEFAULTS,**saved}
        if 'network_mode' in d:
            if d['network_mode'] not in ('lan','local'):return jsonify(error='Choose local or LAN.'),400
            cfg['host']='0.0.0.0' if d['network_mode']=='lan' else '127.0.0.1'
        atomic_json(SETTINGS_FILE,cfg)
    return jsonify(ok=True,restart_required=True,message='Network setting saved. Restart to apply it.')

@app.get('/api/setup/browse')
def edition_browse():
    path=Path(request.args.get('path') or MODELS_ROOT).expanduser().resolve()
    if path.is_relative_to((DATA_ROOT/'secrets').resolve()):return jsonify(error='Credential directory is private'),403
    from core.qwen_assets import browse_directory
    path=browse_directory(path)
    if not path.is_dir():return jsonify(error='Directory not found'),404
    try:
        entries=[]
        for p in sorted(path.iterdir(),key=lambda p:(not p.is_dir(),p.name.lower())):
            if p.name.startswith('.'):continue
            if p.is_dir() or p.suffix.lower() in {'.safetensors','.gguf','.json'}:
                entries.append({'name':p.name,'path':str(p),'directory':p.is_dir()})
            if len(entries)>=600:break
        return jsonify(path=str(path),parent=str(path.parent),entries=entries)
    except OSError as e:return jsonify(error=str(e)),400


from core.studio_extensions import register as register_studio_extensions
register_studio_extensions(app,one,execute,decode_artifact)



from core.model_hub_api import register_model_hub
register_model_hub(app,{'models':models,'model_by_id':model_by_id,'validate_model':validate_model,'update_model':update_model,'settings':settings})
from core.qwen_lab_api import register as register_qwen_lab
register_qwen_lab(app, {'one':one,'rows':rows,'create_job':create_job,'start_job':start_job,'model_by_id':model_by_id,'update_model':update_model,'validate_model':validate_model})
from core.qwen_studio_api import register as register_qwen_studio
register_qwen_studio(app,{'generation_list':_list_user_image_presets,'generation_normalize':_normalize_user_image_preset,
    'generation_path':_user_image_preset_path,'one':one,'rows':rows,'start_job':start_job,'request_cancel':request_cancel,
    'model_by_id':model_by_id,'validate_request':validate_image_request,'validate_model':validate_model})
from core.qwen_pe_api import register as register_qwen_pe
register_qwen_pe(app,{'one':one,'rows':rows,'create_job':create_job,'start_job':start_job,'request_cancel':request_cancel})
from core.qwen_runtime_api import register as register_qwen_runtime
register_qwen_runtime(app)
from core.download_manager_api import register as register_download_manager
register_download_manager(app)

# Manual source checkout: no installer, updater or uninstaller endpoints.

def main():
    cfg=settings()
    # Direct foreground `run.py web` also supports a host-issued code. The
    # supervisor passes a marker (never the code) to prevent logging a second one.
    if os.environ.get("SDXL_STUDIO_SETUP_CODE_MANAGED")!="1":
        print_setup_code(first_run_codes.issue())
    cert=SHARED_ROOT/'config/cert.pem';key=SHARED_ROOT/'config/key.pem'
    tls=(str(cert),str(key)) if cfg.get('https') else None
    os.environ['SDXL_STUDIO_EFFECTIVE_HOST']=str(cfg['host'])
    from core.web_server import serve
    serve(app, cfg['host'], int(cfg['port']), tls=tls)


if __name__ == "__main__":
    main()
