"""Supervise one existing-ComfyUI process inside the serialized image worker."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from core.qwen_prompt_enhancer import normalize_config,runtime_paths,parse_output,inspect_pe_file,PE_KINDS,MAX_RESULT
from core.runtime_env import SHARED_ROOT,APP_ROOT as ROOT


def child_environment():
    env=dict(os.environ)
    for key in list(env):
        if key.startswith(('PYTHON','SDXL_STUDIO_')) or key in ('VIRTUAL_ENV','CONDA_PREFIX','LD_LIBRARY_PATH','HF_TOKEN','HUGGING_FACE_HUB_TOKEN'):
            env.pop(key,None)
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',DO_NOT_TRACK='1')
    return env


def stop_child(proc):
    if proc.poll() is not None:return
    try:
        if os.name=='nt':proc.terminate()
        else:os.killpg(proc.pid,signal.SIGTERM)
        proc.wait(timeout=5)
    except (OSError,subprocess.TimeoutExpired):
        try:
            if os.name=='nt':proc.kill()
            else:os.killpg(proc.pid,signal.SIGKILL)
        except OSError:pass
        proc.wait(timeout=5)


def run(worker,job,probe=False):
    cfg=normalize_config(job.get('pe_config') or {})
    root,exe=runtime_paths(cfg)
    request={'comfy_root':str(root),'probe':probe,'config':cfg,'supervisor_pid':os.getpid()}
    if not probe:
        inspection=inspect_pe_file(job.get('pe_model_path',''),PE_KINDS.get(cfg['mode']))
        refs=job.get('reference_paths') or []
        if len(refs)>10 or (cfg['mode']=='i2i' and not refs) or (cfg['mode']=='t2i' and refs):
            raise ValueError('PE task and input images do not match.')
        request.update(model_path=job['pe_model_path'],original_prompt=job['prompt'],reference_paths=refs)
    private=SHARED_ROOT/'cache/qwen-pe';private.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic();deadline=begin+(60 if probe else cfg['timeout_seconds'])
    proc=None
    with tempfile.TemporaryDirectory(prefix='request-',dir=private) as tmp:
        td=Path(tmp);rq=td/'request.json';result_path=td/'result.json';log=td/'process.log'
        rq.write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
        if os.name!='nt':rq.chmod(0o600)
        with log.open('wb') as output:
            options={'stdin':subprocess.DEVNULL,'stdout':output,'stderr':subprocess.STDOUT,
                     'env':child_environment(),'cwd':str(root),'shell':False}
            if os.name=='nt':options['creationflags']=getattr(subprocess,'CREATE_NO_WINDOW',0)
            else:options['start_new_session']=True
            try:
                proc=subprocess.Popen([str(exe),'-I','-B',str(ROOT/'scripts/pe_comfy_bridge.py'),
                    '--request',str(rq),'--result',str(result_path)],**options)
                while proc.poll() is None:
                    if not probe:worker._check_cancel(job)
                    if time.monotonic()>deadline:raise RuntimeError('Prompt Enhancer timed out. The separate process was stopped; the original prompt is unchanged.')
                    if log.stat().st_size>20*1024**2:raise RuntimeError('Prompt Enhancer stopped because its log exceeded the size limit.')
                    time.sleep(0.25)
            finally:
                if proc is not None:stop_child(proc)
        if not result_path.is_file() or result_path.stat().st_size>MAX_RESULT:
            tail=log.read_bytes()[-6000:].decode('utf-8',errors='replace')
            raise RuntimeError('Prompt Enhancer process returned no valid result. Check the selected ComfyUI runtime.\n'+tail)
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if not result.get('ok') or proc.returncode!=0:
            raise RuntimeError('Prompt Enhancer failed: '+str(result.get('error','process exited')))
        if probe:return result
        result.pop('ok',None)
        parsed=parse_output(result.pop('raw_text',''))
        result.update(parsed)
        result['timing']['end_to_end_seconds']=time.monotonic()-begin
        result['model_identity']={'filename':Path(job['pe_model_path']).name,'bytes':inspection['bytes'],
                                  'header_sha256':inspection['header_sha256'],'full_file_hash_computed':False}
        return result
