from core.runtime_env import worker_headers
import os,requests
URL=os.getenv("CREATOR_IMAGE_WORKER_URL","http://127.0.0.1:8299").rstrip("/")

def health():
    try:
        r=requests.get(URL+"/health",timeout=3); return r.json()
    except Exception as e:return {"ok":False,"error":str(e)}

def probe(model):
    try:
        r=requests.post(URL+"/probe",json={"provider":model.get("provider"),"config":model.get("config") or {}},headers=worker_headers(),timeout=20)
        return r.json()
    except Exception as e:return {"ok":False,"error":str(e)}

def execute_job(payload,timeout=7200):
    r=requests.post(URL+"/execute",json=payload,headers=worker_headers(),timeout=timeout)
    try:d=r.json()
    except Exception:d={"error":r.text[:2000]}
    if not r.ok or not d.get("ok"):
        raise RuntimeError((d.get("error") or f"Image worker HTTP {r.status_code}")+("\n"+d["traceback"] if d.get("traceback") else ""))
    return d

def unload():
    try:return requests.post(URL+"/unload",headers=worker_headers(),timeout=20).json()
    except Exception as e:return {"ok":False,"error":str(e)}
