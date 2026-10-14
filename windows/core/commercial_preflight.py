"""Read-only release review helper (not a full CVE scan or legal certification)."""
from __future__ import annotations
import argparse, importlib, importlib.util, json, os, platform, re, subprocess, sys, tempfile
from pathlib import Path
from core.platform_support import venv_python
from core.platform_support import hidden_kwargs

def check_runtime(executable):
    if not Path(executable).is_file():
        return {'installed':False,'path':str(executable),'status':'missing'}
    code='''import sys,json,importlib.metadata as m
names=['torch','torchvision','diffusers','transformers','peft','Flask','Werkzeug','requests','easydict','certifi','tqdm']
r={}
for n in names:
 try:r[n]=m.version(n)
 except m.PackageNotFoundError:pass
print(json.dumps({'python':sys.version.split()[0],'packages':r}))'''
    try:
        p=subprocess.run([str(executable),'-I','-c',code],capture_output=True,text=True,timeout=30,check=True, **hidden_kwargs())
        d=json.loads(p.stdout);d.update(installed=True,path=str(executable));return d
    except Exception as e:return {'installed':True,'status':'error','error':str(e)[:500],'path':str(executable)}

def review(data):
    issues=[]
    for name,d in data.items():
        version=(d.get('packages') or {}).get('torch')
        if version:
            from core.runtime_security import reviewed_torch_version
            if not reviewed_torch_version(version):
                issues.append({'severity':'block-public-release','runtime':name,'package':'torch','version':version,
                  'reason':'Checkpoint-loader advisories; current candidate baseline is 2.14.0; advisory feed checks are still required.',
                  'source':'https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p'})
        if d.get('status') in ('error','missing'):issues.append({'severity':'review','runtime':name,'reason':d['status']})
    return issues

def selftest():
    """Exercise ABI import and non-GPU logic in a fresh process and private data root."""
    with tempfile.TemporaryDirectory(prefix='zetalvx-native-selftest-') as tmp:
        root=Path(__file__).resolve().parents[1]
        env={**os.environ,'QWEN_STUDIO_HOME':tmp,'SDXL_STUDIO_HOME':tmp,'SDXL_STUDIO_NO_BACKGROUND':'1','PYTHONPATH':str(root)}
        code='''import json
from pathlib import Path
from core.platform_support import venv_python
from core.runtime_env import initialize_environment,VERSION
initialize_environment()
from core import db, model_registry
# No model is loaded and no network request is made.
db.init();r=model_registry.load_registry()
assert r['models'][0]['provider']=='qwen_image21'
print(json.dumps({'ok':True,'version':VERSION,'native_module':__import__('core.db',fromlist=['db']).__file__}))'''
        # Only use stable symbols for preflight; caption symbol varies in legacy builds.
        code=code.replace('from core.training_captions import saved_training_caption\n','').replace('from core.dataset_exchange import trigger_count\n','').replace('from vision.registry import Registry\n','')
        p=subprocess.run([sys.executable,'-c',code],env=env,cwd=root,capture_output=True,text=True,timeout=30, **hidden_kwargs())
        if p.returncode:raise RuntimeError('Native self-test failed: '+(p.stderr or p.stdout)[-2000:])
        return json.loads(p.stdout.strip().splitlines()[-1])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');parser.add_argument('--report',default='');a=parser.parse_args()
    if a.self_test:print(json.dumps(selftest(),indent=2));return
    home=Path(os.environ.get('QWEN_STUDIO_HOME') or os.environ.get('SDXL_STUDIO_HOME') or str(Path.home()/'.local/share/ImageLabQwen21'))
    data={name:check_runtime(venv_python(home/'runtime'/name)) for name in ('app','qwen21')}
    d={'version':'0.1.0.21','purpose':'open-source release candidate review','platform':platform.platform(),
       'runtimes':data,'issues':review(data),'public_release_approved':False,
       'remaining_gates':['Rights/provenance confirmation','Seller terms and privacy review','Validated updated AI runtime','NVIDIA and supported Python tests'],
       'not_a_certificate':True}
    if a.report:Path(a.report).write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d,indent=2))
    if d['issues']:raise SystemExit(2)
