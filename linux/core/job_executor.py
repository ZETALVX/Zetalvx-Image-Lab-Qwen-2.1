import json,os,uuid,threading,queue,time
from pathlib import Path
from PIL import Image,ImageOps
from core.db import one,rows,execute,con
from core.paths import ARTIFACTS_DIR,SHARED_ROOT
from core.model_registry import model_by_id,validate_model
from core.task_adapters import task_from_tool,task_support,resolve_strategy
from core.image_runtime import execute_job,unload as unload_image_runtime

CAP_MAP={
 "generate_image":"generate","edit_image":"edit","img2img":"img2img","inpaint":"inpaint",
 "reference":"reference","multi_image":"multi_image","refine_image":"refine"
}

PROGRESS_DIR=SHARED_ROOT/"cache"/"job_progress"
PROGRESS_DIR.mkdir(parents=True,exist_ok=True)
CANCEL_DIR=SHARED_ROOT/"cache"/"job_cancel"
CANCEL_DIR.mkdir(parents=True,exist_ok=True)

_job_queue=queue.Queue()
_queued=set()
_lock=threading.Lock()
_worker_started=False

def _params(job):
    try:return json.loads(job.get("params_json") or "{}")
    except:return {}

def _resolve_model(job):
    task=task_from_tool(job.get("tool"))
    if job.get("model_id")!="auto":
        m=model_by_id(job.get("model_id"))
        if not m:raise RuntimeError("Selected model not found.")
        if not m.get("enabled"):raise RuntimeError("Selected model is disabled.")
        support=task_support(m,task)
        if support.get("level")=="unavailable":raise RuntimeError(support.get("reason") or "Task unavailable for selected model.")
        return m
    from core.model_registry import load_registry
    candidates=[]
    for m in load_registry().get("models",[]):
        if not m.get("enabled"):continue
        support=task_support(m,task)
        if support.get("rank",0)<=0 or not support.get("executable",True):continue
        if not validate_model(m).get("ready"):continue
        priority=int((m.get("config") or {}).get("auto_priority") or 0)
        candidates.append((support.get("rank",0),priority,m))
    if candidates:
        candidates.sort(key=lambda x:(x[0],x[1]),reverse=True)
        return candidates[0][2]
    raise RuntimeError(f"No ready image model can execute '{task}'.")


def _artifact_record(aid):
    if not aid:return None
    a=one("SELECT * FROM artifacts WHERE id=?",(aid,))
    if not a:return None
    try:a["metadata"]=json.loads(a.get("metadata_json") or "{}")
    except:a["metadata"]={}
    try:
        path=Path(a.get("path") or "")
        if path.exists():
            with Image.open(path) as im:
                a["width"],a["height"]=im.size
                a["format"]=im.format
    except Exception:
        pass
    return a

def _artifact_path(aid):
    a=_artifact_record(aid)
    return a.get("path") if a else None

def _progress_path(jid): return PROGRESS_DIR/f"{jid}.json"
def _cancel_path(jid): return CANCEL_DIR/f"{jid}.cancel"

def write_progress(jid,**data):
    p=_progress_path(jid)
    payload={"job_id":jid,"updated_at":time.time(),**data}
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    tmp.replace(p)

def read_progress(jid):
    p=_progress_path(jid)
    try:return json.loads(p.read_text(encoding="utf-8"))
    except:return {}

def request_cancel(jid):
    """Request cancellation without leaving queued jobs in a permanent busy state.

    Queued jobs have not started GPU work, so cancellation is terminal immediately.
    Running jobs keep ``status=running`` while the worker unwinds and expose
    ``phase=cancelling``; they remain maintenance-busy until the worker exits.
    """
    job=one("SELECT status,cancel_requested FROM jobs WHERE id=?",(jid,))
    if not job:return False
    status=str(job.get("status") or "")
    if status in ("completed","failed","cancelled","interrupted"):
        return False
    if status=="queued":
        execute("""UPDATE jobs SET cancel_requested=1,status='cancelled',phase='cancelled',
                   completed_at=COALESCE(completed_at,CURRENT_TIMESTAMP),updated_at=CURRENT_TIMESTAMP
                   WHERE id=? AND status='queued'""",(jid,))
        write_progress(jid,phase="cancelled",progress=0,error="")
    else:
        execute("""UPDATE jobs SET cancel_requested=1,phase='cancelling',updated_at=CURRENT_TIMESTAMP
                   WHERE id=?""",(jid,))
    _cancel_path(jid).write_text("1",encoding="utf-8")
    return True

def queue_snapshot(project_id=None):
    """Queue metadata for a specific project; internal callers may omit scope."""
    args=(project_id,) if project_id else ()
    extra=' AND project_id=?' if project_id else ''
    active=one("SELECT id,project_id,tool,model_id,status,phase,progress,step_current,step_total,queue_order FROM jobs WHERE status='running'"+extra+" ORDER BY started_at LIMIT 1",args)
    waiting=rows("SELECT id,project_id,tool,model_id,status,phase,queue_order FROM jobs WHERE status='queued' AND cancel_requested=0"+extra+" ORDER BY queue_order,created_at,id",args)
    return {"active":active,"queued":waiting}


def run_job(jid):
    job=one("SELECT * FROM jobs WHERE id=?",(jid,))
    if not job:return
    if int(job.get("cancel_requested") or 0):
        execute("UPDATE jobs SET status='cancelled',phase='cancelled',completed_at=CURRENT_TIMESTAMP WHERE id=?",(jid,))
        write_progress(jid,phase="cancelled",progress=0)
        return

    started=time.time()
    try:
        if job.get('tool')=='qwen_prompt_enhance':
            return run_pe_job(job,started)
        if job.get('tool') in ('qwen_decode','qwen_vae_test','qwen_roundtrip'):
            return run_lab_job(job,started)
        model=_resolve_model(job)
        p=_params(job)
        if isinstance(p.get('_model_config_snapshot'),dict):
            model={**model,'config':dict(p['_model_config_snapshot'])}
        selected_model=dict(model);selected_model['config']=dict(model.get('config') or {})
        if p.get('qwen21_transformer_mode')=='original':selected_model['config']['model_path']=''
        elif p.get('qwen21_transformer_mode') in ('gguf','safetensors'):
            selected_model['config']['model_path']=p.get('qwen21_transformer_path','')
        if p.get('qwen21_text_encoder_path'):
            selected_model['config']['text_encoder_path']=p['qwen21_text_encoder_path'];selected_model['config']['text_encoder_dir']=''
        if p.get('qwen21_vae_path'):selected_model['config']['vae_path']=p['qwen21_vae_path']
        if str(selected_model['config'].get('model_path') or '').lower().endswith('.gguf'):
            selected_model['config']['quantized_loading']='native_comfyui'
        validation=validate_model(selected_model)
        if not validation.get('ready'):
            raise RuntimeError("Qwen Image 2.1 files are incomplete: "+', '.join(validation.get('missing',[])))
        aid=uuid.uuid4().hex
        outdir=ARTIFACTS_DIR/job["project_id"];outdir.mkdir(parents=True,exist_ok=True)
        outpath=outdir/f"{aid}.png"
        source_id=p.get("source_artifact_id")
        mask_id=p.get("mask_artifact_id")
        refs=p.get("reference_artifact_ids") or []
        strategy=resolve_strategy(model,job.get("tool"),p)

        source_record=_artifact_record(source_id) if source_id else None
        reference_records=[x for x in (_artifact_record(r) for r in refs) if x]
        input_assets=[]
        if source_record:
            input_assets.append({"id":source_record.get("id"),"type":source_record.get("type"),"role":"source","path":source_record.get("path")})
        input_assets.extend({"id":r.get("id"),"type":r.get("type"),"role":"reference","path":r.get("path")} for r in reference_records)

        image_providers={"qwen_image21"}
        if model.get("provider") in image_providers:
            bad=[x for x in input_assets if x.get("type")!="image"]
            if bad:
                names=", ".join(Path(x.get("path") or "").name for x in bad)
                raise RuntimeError(
                    f"{model['name']} currently accepts image inputs only in Zetalvx Image Lab. "
                    f"Remove video input(s): {names}"
                )

        source_path=source_record.get("path") if source_record else None
        reference_paths=[r.get("path") for r in reference_records if r.get("type")=="image" and r.get("path")]

        requested_width=int(p.get("width") or 1024)
        requested_height=int(p.get("height") or 1024)
        source_width=source_height=None
        if source_path and Path(source_path).exists():
            try:
                with Image.open(source_path) as sim:
                    source_width,source_height=sim.size
            except Exception:
                pass
        preserve_source_size=job.get("tool") in ("edit_image","inpaint") and bool(source_width and source_height)
        worker_width=source_width if preserve_source_size else requested_width
        worker_height=source_height if preserve_source_size else requested_height
        steps=int(p.get("steps") or (model.get("config") or {}).get("default_steps") or 0)
        progress_path=str(_progress_path(jid))
        cancel_path=str(_cancel_path(jid))
        try: Path(cancel_path).unlink(missing_ok=True)
        except: pass

        execute("""UPDATE jobs
                   SET model_id=?,status='running',phase='loading_model',progress=0,
                       step_current=0,step_total=?,error='',started_at=CURRENT_TIMESTAMP,
                       completed_at=NULL,duration_seconds=0,updated_at=CURRENT_TIMESTAMP
                   WHERE id=?""",(model["id"],steps,jid))
        write_progress(jid,phase="loading_model",progress=0,step_current=0,step_total=steps,elapsed_seconds=0)

        effective_config=dict(model.get("config") or {})
        q21_mode=str(p.get('qwen21_transformer_mode') or 'configured')
        q21_path=str(p.get('qwen21_transformer_path') or '').strip()
        if q21_mode=='original':effective_config['model_path']=''
        elif q21_mode in ('gguf','safetensors'):effective_config['model_path']=q21_path
        if p.get('qwen21_text_encoder_path'):
            effective_config['text_encoder_path']=p['qwen21_text_encoder_path'];effective_config['text_encoder_dir']=''
        if p.get('qwen21_vae_path'):
            effective_config['vae_path']=p['qwen21_vae_path']
        if str(effective_config.get('model_path') or '').lower().endswith('.gguf'):effective_config['quantized_loading']='native_comfyui'

        payload={
          "provider":model.get("provider"),"config":effective_config,
          "job":{
            "job_id":jid,"project_id":job["project_id"],
            "task":strategy.get("worker_task") or CAP_MAP.get(job.get("tool"),job.get("tool")),
            "prompt":job.get("prompt") or "",
            "negative_prompt":p.get("negative_prompt") or "",
            "width":worker_width,"height":worker_height,
            "requested_width":requested_width,"requested_height":requested_height,
            "preserve_source_size":preserve_source_size,
            "steps":p.get("steps"),"cfg":p.get("cfg"),"seed":p.get("seed",-1),
            "strength":p.get("strength",.35),"loras":p.get("loras") or [],
            "source_path":source_path,
            "mask_path":_artifact_path(mask_id) if mask_id else None,
            "reference_paths":reference_paths,
            "transparent_output":p.get("transparent_output",False),
            "vae_decode_mode":p.get("qwen21_vae_decode_mode") or p.get("vae_decode_mode") or "auto",
            "input_assets":input_assets,
            "output_path":str(outpath),
            "progress_path":progress_path,
            "cancel_path":cancel_path,
            "started_epoch":started
          }
        }
        for key in ('decode_device','vae_dtype','tile_height','tile_width','tile_stride_height','tile_stride_width','retain_latents'):
            if key in p:payload['job'][key]=p[key]
        result=execute_job(payload)
        duration=time.time()-started
        output_width=output_height=None
        try:
            with Image.open(outpath) as oim:
                output_width,output_height=oim.size
        except Exception:
            pass
        source_snapshot=_artifact_record(source_id) if source_id else None
        if source_snapshot:
            source_snapshot={k:source_snapshot.get(k) for k in ("id","type","path","created_at","width","height","format","metadata")}

        meta={"model_id":model["id"],"model_name":model["name"],"provider":model.get("provider"),
              "tool":job["tool"],"prompt":job.get("prompt") or "","params":p,
              "runtime_result":result,"source_artifact_id":source_id,"mask_artifact_id":mask_id,
              "reference_artifact_ids":refs,"input_assets":input_assets,"duration_seconds":duration,
              "requested_width":requested_width,"requested_height":requested_height,
              "source_width":source_width,"source_height":source_height,
              "output_width":output_width,"output_height":output_height,
              "preserved_source_size":preserve_source_size,
              "source_snapshot":source_snapshot,
              "task_strategy":strategy}

        execute("INSERT INTO artifacts(id,project_id,type,path,parent_id,metadata_json) VALUES(?,?,?,?,?,?)",
                (aid,job["project_id"],"image",str(outpath),source_id,json.dumps(meta,ensure_ascii=False)))
        execute("""UPDATE jobs SET status='completed',phase='completed',progress=1,
                   step_current=CASE WHEN step_total>0 THEN step_total ELSE step_current END,
                   output_artifact_id=?,duration_seconds=?,completed_at=CURRENT_TIMESTAMP,
                   updated_at=CURRENT_TIMESTAMP WHERE id=?""",(aid,duration,jid))
        execute("""INSERT INTO project_state(project_id,active_artifact_id,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)
                   ON CONFLICT(project_id) DO UPDATE SET active_artifact_id=excluded.active_artifact_id,updated_at=CURRENT_TIMESTAMP""",
                (job["project_id"],aid))
        write_progress(jid,phase="completed",progress=1,step_current=steps,step_total=steps,elapsed_seconds=duration,duration_seconds=duration)
    except Exception as e:
        duration=time.time()-started
        current=one("SELECT cancel_requested FROM jobs WHERE id=?",(jid,)) or {}
        cancelled=int(current.get("cancel_requested") or 0)==1 or "cancel" in str(e).lower()
        status="cancelled" if cancelled else "failed"
        execute("""UPDATE jobs SET status=?,phase=?,error=?,duration_seconds=?,
                   completed_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?""",
                (status,status,str(e),duration,jid))
        write_progress(jid,phase=status,progress=0,elapsed_seconds=duration,error=str(e))

def run_pe_job(job,started):
    """Prompt-only job through the existing image queue; never creates an image."""
    from core.qwen_prompt_enhancer import PEStore
    jid=job['id'];p=_params(job);snapshot=p.get('_pe_request')
    if not isinstance(snapshot,dict):raise ValueError('Missing captured PE configuration.')
    store=PEStore(SHARED_ROOT)
    result=store.result(jid)
    if result is None:
        refs=[]
        for aid in snapshot['reference_artifact_ids']:
            image=one("SELECT path FROM artifacts WHERE id=? AND project_id=? AND type='image'",(aid,job['project_id']))
            if not image or not Path(image['path']).is_file():raise ValueError('A PE reference image is missing from this project.')
            refs.append(image['path'])
        execute("UPDATE jobs SET status='running',phase='loading_model',step_total=0,started_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?",(jid,))
        write_progress(jid,phase='loading_model',progress=0,step_total=0)
        work={'job_id':jid,'project_id':job['project_id'],'task':'qwen_prompt_enhance',
            'pe_config':snapshot['config'],'pe_model_path':snapshot['model_path'],
            'prompt':snapshot['original_prompt'],'reference_paths':refs,
            'progress_path':str(_progress_path(jid)),'cancel_path':str(_cancel_path(jid)),'started_epoch':started}
        result=execute_job({'provider':'qwen_image21','config':{},'job':work})
        current=one('SELECT cancel_requested FROM jobs WHERE id=?',(jid,)) or {}
        if current.get('cancel_requested'):raise RuntimeError('Job cancelled by user.')
        if not result.get('rewritten_prompt'):raise RuntimeError('PE returned no rewritten prompt.')
        cfg=snapshot['config']
        result.update(job_id=jid,project_id=job['project_id'],original_prompt=snapshot['original_prompt'],
            reference_artifact_ids=snapshot['reference_artifact_ids'],mode=cfg['mode'],model_name=snapshot['model_name'],
            options={k:cfg[k] for k in ('max_tokens','seed','thinking','sampling','temperature','top_p','template_mode','system_prompt')})
        # Parent-process metrics do not represent the separate PE process.
        result.pop('metrics',None)
        store.put_result(jid,result)
    duration=time.time()-started
    execute("UPDATE jobs SET status='completed',phase='completed',progress=1,output_artifact_id=NULL,duration_seconds=?,completed_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?",(duration,jid))
    write_progress(jid,phase='completed',progress=1,step_total=0,elapsed_seconds=duration)


def run_lab_job(job,started):
    """Uses the SAME serialized queue, worker token and GPU lock as sampling."""
    from core.runtime_env import atomic_json
    jid=job['id'];p=_params(job);cfg=p.get('_model_config_snapshot')
    if not isinstance(cfg,dict):raise ValueError('A component test/re-decode must include a captured model configuration.')
    outdir=ARTIFACTS_DIR/job['project_id'];outdir.mkdir(parents=True,exist_ok=True)
    aid=uuid.uuid4().hex;outpath=outdir/(aid+'.png')
    execute("UPDATE jobs SET status='running',phase='loading_vae',step_total=0,started_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?",(jid,))
    work={k:v for k,v in p.items() if k!='_model_config_snapshot'}
    work.update(job_id=jid,project_id=job['project_id'],task=job['tool'],output_path=str(outpath),
        progress_path=str(_progress_path(jid)),cancel_path=str(_cancel_path(jid)),started_epoch=started)
    if job['tool']=='qwen_roundtrip':
        source=one("SELECT path FROM artifacts WHERE id=? AND project_id=? AND type='image'",(p.get('input_artifact_id'),job['project_id']))
        if not source or not Path(source['path']).is_file():raise ValueError('Round-trip input image is missing.')
        work['source_path']=source['path']
    result=execute_job({'provider':'qwen_image21','config':cfg,'job':work})
    current=one('SELECT cancel_requested FROM jobs WHERE id=?',(jid,)) or {}
    if current.get('cancel_requested'):
        outpath.unlink(missing_ok=True);raise RuntimeError('Job cancelled by user.')
    duration=time.time()-started
    if job['tool']=='qwen_vae_test':
        report=result.get('test_report')
        if not isinstance(report,dict) or not report.get('ok'):raise RuntimeError('VAE test did not return a successful report.')
        atomic_json(SHARED_ROOT/'cache/qwen-component-tests'/(jid+'.json'),report)
        aid=None
    else:
        if not outpath.is_file():raise RuntimeError('VAE decode produced no PNG output.')
        meta={'model_id':'qwen_image_21','model_name':'Qwen Image 2.1 — VAE re-decode','provider':'qwen_image21',
            'tool':job['tool'],'prompt':job.get('prompt',''),'params':p,'runtime_result':result,
            'source_latent_id':p.get('latent_id'),'sampling_repeated':False,'duration_seconds':duration,
            'output_width':result.get('width'),'output_height':result.get('height')}
        execute('INSERT INTO artifacts(id,project_id,type,path,parent_id,metadata_json) VALUES(?,?,?,?,?,?)',
            (aid,job['project_id'],'image',str(outpath),None,json.dumps(meta,ensure_ascii=False)))
        execute("""INSERT INTO project_state(project_id,active_artifact_id,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)
            ON CONFLICT(project_id) DO UPDATE SET active_artifact_id=excluded.active_artifact_id,updated_at=CURRENT_TIMESTAMP""",(job['project_id'],aid))
    execute("UPDATE jobs SET status='completed',phase='completed',progress=1,output_artifact_id=?,duration_seconds=?,completed_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE id=?",(aid,duration,jid))
    write_progress(jid,phase='completed',progress=1,step_current=0,step_total=0,elapsed_seconds=duration)


def _claim_next_queued():
    """Atomically reserve the next persisted priority before accessing models.

    This closes the race where UI could still edit/reorder a job that the worker
    had already selected but had not yet marked as running.
    """
    with con() as c:
        c.execute('BEGIN IMMEDIATE')
        job=c.execute("SELECT id FROM jobs WHERE status='queued' AND COALESCE(cancel_requested,0)=0 ORDER BY queue_order,created_at,id LIMIT 1").fetchone()
        if not job:return None
        jid=job['id']
        c.execute("UPDATE jobs SET status='running',phase='starting',started_at=COALESCE(started_at,CURRENT_TIMESTAMP),updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='queued' AND COALESCE(cancel_requested,0)=0",(jid,))
        return jid


def _worker_loop():
    while True:
        # In-memory FIFO is a wake-up token, not dispatch order. Pending jobs
        # are claimed in priority order from the database under a write lock.
        wake_id=_job_queue.get()
        try:
            with _lock:_queued.discard(wake_id)
            jid=_claim_next_queued()
            if jid:run_job(jid)
        finally:
            _job_queue.task_done()

def ensure_worker():
    global _worker_started
    if _worker_started:return
    with _lock:
        if _worker_started:return
        threading.Thread(target=_worker_loop,daemon=True,name="creator-serial-job-queue").start()
        _worker_started=True

def start_job(jid):
    ensure_worker()
    with _lock:
        if jid in _queued:return
        _queued.add(jid)
    _job_queue.put(jid)

def recover_pending_jobs():
    """Reconcile persisted queue state at web-process startup.

    A row that is still ``running`` cannot belong to the new process, so it is
    terminally marked interrupted instead of silently re-running an expensive
    generation. Queued work remains resumable. Queued/running rows that already
    carried a cancellation request become cancelled and can never block
    maintenance after restart.
    """
    ensure_worker()
    cancelled=rows("""SELECT id FROM jobs
                       WHERE cancel_requested=1 AND status IN ('queued','running','cancelling')""")
    interrupted=rows("""SELECT id FROM jobs
                         WHERE COALESCE(cancel_requested,0)=0 AND status IN ('running','cancelling')""")
    execute("""UPDATE jobs SET status='cancelled',phase='cancelled',
               completed_at=COALESCE(completed_at,CURRENT_TIMESTAMP),updated_at=CURRENT_TIMESTAMP
               WHERE cancel_requested=1 AND status IN ('queued','running','cancelling')""")
    execute("""UPDATE jobs SET status='interrupted',phase='interrupted',
               error=CASE WHEN COALESCE(error,'')='' THEN 'Application restarted before this job completed.' ELSE error END,
               completed_at=COALESCE(completed_at,CURRENT_TIMESTAMP),updated_at=CURRENT_TIMESTAMP
               WHERE COALESCE(cancel_requested,0)=0 AND status IN ('running','cancelling')""")
    for j in cancelled:
        write_progress(j['id'],phase='cancelled',progress=0,error='')
    for j in interrupted:
        write_progress(j['id'],phase='interrupted',progress=0,error='Application restarted before this job completed.')
    for j in rows("SELECT id FROM jobs WHERE status='queued' AND COALESCE(cancel_requested,0)=0 ORDER BY queue_order,created_at"):
        start_job(j['id'])
    return {'cancelled':len(cancelled),'interrupted':len(interrupted)}

if os.environ.get("SDXL_STUDIO_NO_BACKGROUND")!="1":ensure_worker()
