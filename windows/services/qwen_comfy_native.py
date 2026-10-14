"""Native Qwen Image 2.1 quantized execution through an existing ComfyUI runtime.

This bridge is intentionally separate from Diffusers. It is used for supported
ConvRot safetensors and Qwen Image 2.1 GGUF transformers when the user explicitly
selects native_comfyui. No model is dequantized by Image Lab. ComfyUI and its
installed GGUF loader nodes perform loading and quantized execution.
"""
from __future__ import annotations
import json,os,shutil,socket,subprocess,tempfile,threading,time,urllib.request,urllib.error,uuid
from pathlib import Path
from core.runtime_env import SHARED_ROOT
from core.qwen_external_runtime import ComfyRuntimeStore,native_runtime_paths
from core.qwen_weight_format import inspect_weights


def _file(cfg,key,suffixes=('.safetensors',)):
    raw=str(cfg.get(key) or '').strip();p=Path(raw).expanduser()
    allowed=tuple(str(x).lower() for x in (suffixes or ()))
    if not raw or not p.is_absolute() or not p.is_file() or (allowed and p.suffix.lower() not in allowed):
        label='/'.join(allowed) if allowed else 'file'
        raise RuntimeError(f'Native ComfyUI execution needs an explicit {key} {label}.')
    return p.resolve()


def native_requested(cfg):
    if cfg.get('quantized_loading')!='native_comfyui':return False
    def suffix(key,ext):
        raw=str(cfg.get(key) or '').strip();return bool(raw and raw.lower().endswith(ext))
    # GGUF Qwen 2.1 files are ComfyUI-GGUF/stable-diffusion.cpp artifacts, not
    # Diffusers GGUF single-file checkpoints. Always route them to the explicit
    # native bridge so they never reach QwenImage21Transformer2DModel.from_single_file.
    if suffix('model_path','.gguf'):return True
    def sf(key):return suffix(key,'.safetensors')
    # When all three components are explicit single files, use the same ComfyUI
    # execution stack for BF16 and ConvRot so an A/B run does not silently change
    # sampler implementation. If a selected Transformer/encoder is quantized, route
    # to this bridge even when another prerequisite is missing: run() then reports
    # that job as unsupported instead of falling back to float reconstruction.
    if sf('model_path') and sf('text_encoder_path') and sf('vae_path'):return True
    for key in ('model_path','text_encoder_path'):
        if not sf(key):continue
        path=Path(str(cfg.get(key))).expanduser()
        if not path.is_file():continue
        try:
            if inspect_weights(path).get('quantized'):return True
        except (OSError,ValueError):continue
    return False


def _runtime():
    store=ComfyRuntimeStore(SHARED_ROOT);cfg=store.config()
    try:
        if not store.verified(cfg):raise ValueError('The global ComfyUI runtime has not been verified for these paths.')
        return native_runtime_paths(cfg)
    except Exception as exc:
        raise RuntimeError('Native ConvRot/GGUF uses the global ComfyUI runtime in Settings → External runtimes. Configure, trust and verify that runtime first. '+str(exc)) from exc


def _port():
    s=socket.socket();s.bind(('127.0.0.1',0));p=s.getsockname()[1];s.close();return p


def _http_json(url,data=None,timeout=5):
    body=None;headers={}
    if data is not None:
        body=json.dumps(data,separators=(',',':')).encode();headers['Content-Type']='application/json'
    req=urllib.request.Request(url,data=body,headers=headers,method='POST' if body is not None else 'GET')
    with urllib.request.urlopen(req,timeout=timeout) as r:
        raw=r.read(4*1024*1024)
    return json.loads(raw.decode())


def _wait_ready(base,proc,timeout,log):
    end=time.monotonic()+timeout
    last=''
    while time.monotonic()<end:
        if proc.poll() is not None:
            try:last=Path(log).read_text(encoding='utf-8',errors='replace')[-5000:]
            except Exception:pass
            raise RuntimeError('ComfyUI native process exited during startup. '+last)
        try:
            data=_http_json(base+'/system_stats',timeout=2)
            if isinstance(data,dict):return data
        except Exception as exc:last=str(exc)
        time.sleep(.25)
    raise RuntimeError('Timed out starting the configured ComfyUI runtime. '+last)




def _validate_object_info(info,*,gguf=False):
    required=('CLIPLoader','VAELoader','TextEncodeQwenImage21','KSampler','VAEDecode','SaveImage')
    required+=(('UnetLoaderGGUF',) if gguf else ('UNETLoader',))
    if not isinstance(info,dict):raise RuntimeError('Configured ComfyUI returned invalid node metadata.')
    missing=[name for name in required if name not in info]
    if missing:
        extra=' Install/enable a compatible ComfyUI-GGUF loader (UnetLoaderGGUF) in that ComfyUI runtime.' if gguf and 'UnetLoaderGGUF' in missing else ' Update that ComfyUI installation.'
        raise RuntimeError('Configured ComfyUI is missing Qwen 2.1 native nodes: '+', '.join(missing)+'.'+extra)
    # CLIPLoader must advertise Qwen Image support. Object-info structures vary
    # slightly between ComfyUI revisions, so inspect only this bounded node JSON.
    clip=json.dumps(info.get('CLIPLoader'),ensure_ascii=False)
    if 'qwen_image' not in clip:raise RuntimeError('Configured ComfyUI CLIPLoader does not advertise qwen_image. Update that ComfyUI installation before using Native ConvRot.')
    return True

def _workflow(job,transformer,encoder,vae):
    width=int(job.get('width') or 1024);height=int(job.get('height') or 1024)
    steps=int(job.get('steps') or 25);seed=int(job.get('seed',-1));seed=314159 if seed<0 else seed
    cfg=float(job.get('cfg') or 1.0);cfg=cfg if cfg>=1 else 1.0
    prompt=str(job.get('prompt') or '')
    neg=str(job.get('negative_prompt') or '')
    gguf=transformer.suffix.lower()=='.gguf'
    loader={'class_type':'UnetLoaderGGUF','inputs':{'unet_name':transformer.name}} if gguf else {'class_type':'UNETLoader','inputs':{'unet_name':transformer.name,'weight_dtype':'default'}}
    return {
      '1':loader,
      '2':{'class_type':'CLIPLoader','inputs':{'clip_name':encoder.name,'type':'qwen_image','device':'default'}},
      '3':{'class_type':'VAELoader','inputs':{'vae_name':vae.name}},
      '4':{'class_type':'TextEncodeQwenImage21','inputs':{'clip':['2',0],'prompt':prompt,'negative_prompt':neg,'resolution':max(width,height)}},
      # Qwen Image 2.1 conditioning returns the model-specific 64-channel latent.
      # Do not use the generic EmptyLatentImage (SD-style latent contract).
      '6':{'class_type':'KSampler','inputs':{'model':['1',0],'seed':seed,'steps':steps,'cfg':cfg,'sampler_name':'euler','scheduler':'simple','positive':['4',0],'negative':['4',1],'latent_image':['4',2],'denoise':1.0}},
      '7':{'class_type':'VAEDecode','inputs':{'samples':['6',0],'vae':['3',0]}},
      '8':{'class_type':'SaveImage','inputs':{'images':['7',0],'filename_prefix':'ImageLabNative'}},
    },seed



def _gguf_qwen21_support(root):
    """Read-only compatibility check for ComfyUI-GGUF Qwen Image 2.1 support."""
    root=Path(root)
    custom=root/'custom_nodes'
    if not custom.is_dir():return False,'ComfyUI custom_nodes folder is missing.'
    candidates=[]
    try:
        for d in custom.iterdir():
            if d.is_dir() and 'gguf' in d.name.lower():candidates.append(d)
    except OSError:return False,'Could not inspect ComfyUI custom_nodes.'
    for d in candidates:
        try:
            for py in d.rglob('*.py'):
                if py.stat().st_size>2_000_000:continue
                text=py.read_text(encoding='utf-8',errors='ignore')
                if 'qwen_image21' in text:return True,f'{d.name} advertises qwen_image21.'
        except OSError:continue
    return False,'ComfyUI-GGUF is installed but does not advertise qwen_image21. Update it or add qwen_image21 to its image architecture allow-list.'


def _gguf_arch(exe,path):
    code='''import sys
from gguf import GGUFReader
r=GGUFReader(sys.argv[1])
f=r.fields.get("general.architecture")
print("MISSING" if f is None else "ARCH:"+str(f.contents()))
'''
    try:
        out=subprocess.check_output([str(exe),'-c',code,str(path)],stderr=subprocess.STDOUT,text=True,timeout=60).strip().splitlines()
    except Exception as exc:
        raise RuntimeError('Could not inspect GGUF metadata with the configured ComfyUI Python. Ensure ComfyUI-GGUF requirements are installed. '+str(exc)) from exc
    return out[-1] if out else 'MISSING'


def _prepare_qwen21_gguf(exe,path,job=None,worker=None):
    """Create/cache a metadata-compatible copy only when architecture is missing."""
    path=Path(path).resolve();state=_gguf_arch(exe,path)
    if state=='ARCH:qwen_image21':return path
    if state.startswith('ARCH:'):
        raise RuntimeError('Selected GGUF declares an incompatible architecture: '+state[5:]+'. Expected qwen_image21.')
    if state!='MISSING':raise RuntimeError('Could not determine GGUF architecture metadata: '+state)
    cache=SHARED_ROOT/'cache/qwen21-gguf-compat';cache.mkdir(parents=True,exist_ok=True)
    stat=path.stat();key=__import__('hashlib').sha256((str(path)+'\0'+str(stat.st_size)+'\0'+str(stat.st_mtime_ns)).encode()).hexdigest()[:16]
    dst=cache/(path.stem+'.'+key+'.qwen_image21.gguf')
    if dst.is_file() and dst.stat().st_size>0 and _gguf_arch(exe,dst)=='ARCH:qwen_image21':return dst
    free=shutil.disk_usage(cache).free
    if free<stat.st_size+512*1024*1024:
        raise RuntimeError(f'Not enough free disk space to create the non-destructive Qwen 2.1 GGUF compatibility copy. Need about {stat.st_size/1024**3:.1f} GiB plus safety margin.')
    if worker is not None and job is not None:
        worker._write_job_progress(job,'preparing_gguf',0,int(job.get('steps') or 25),.01,compatibility_copy=True)
    tmp=dst.with_suffix(dst.suffix+'.tmp')
    try:tmp.unlink(missing_ok=True)
    except Exception:pass
    code='''import sys
from gguf import GGUFReader,GGUFWriter,Keys
src,dst=sys.argv[1],sys.argv[2]
r=GGUFReader(src)
w=GGUFWriter(dst,arch="qwen_image21",endianess=r.endianess)
for field in r.fields.values():
    if field.name==Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    vt=field.types[0];st=field.types[-1] if len(field.types)>1 else None
    try:w.add_key_value(field.name,field.contents(),vt,sub_type=st)
    except TypeError:w.add_key_value(field.name,field.contents(),vt)
for t in r.tensors:
    w.add_tensor_info(t.name,t.data.shape,t.data.dtype,t.data.nbytes,t.tensor_type)
w.write_header_to_file();w.write_kv_data_to_file();w.write_ti_data_to_file()
for t in r.tensors:w.write_tensor_data(t.data,tensor_endianess=r.endianess)
w.close()
'''
    try:
        subprocess.run([str(exe),'-c',code,str(path),str(tmp)],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=1800)
        if _gguf_arch(exe,tmp)!='ARCH:qwen_image21':raise RuntimeError('Compatibility copy was written but architecture verification failed.')
        os.replace(tmp,dst)
    except Exception as exc:
        try:tmp.unlink(missing_ok=True)
        except Exception:pass
        detail=getattr(exc,'stdout',None) or str(exc)
        raise RuntimeError('Could not create the Qwen 2.1 GGUF compatibility copy. '+str(detail)[-3000:]) from exc
    return dst


def probe_runtime(cfg):
    """Start the selected ComfyUI in an isolated temporary server and inspect Qwen nodes."""
    root,exe=native_runtime_paths(cfg);work=Path(tempfile.mkdtemp(prefix='imagelab-comfy-probe-'));proc=None;lf=None
    try:
        outdir=work/'output';tmpdir=work/'temp';userdir=work/'user'
        for d in (outdir,tmpdir,userdir):d.mkdir(parents=True,exist_ok=True)
        port=_port();log=work/'comfy-probe.log';lf=open(log,'w',encoding='utf-8')
        cmd=[str(exe),'-B',str(root/'main.py'),'--listen','127.0.0.1','--port',str(port),'--disable-auto-launch',
             '--output-directory',str(outdir),'--temp-directory',str(tmpdir),'--user-directory',str(userdir)]
        proc=subprocess.Popen(cmd,cwd=str(root),stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
        base=f'http://127.0.0.1:{port}';stats=_wait_ready(base,proc,90,log)
        node_info={}
        for node_name in ('CLIPLoader','VAELoader','TextEncodeQwenImage21','KSampler','VAEDecode','SaveImage','UNETLoader','UnetLoaderGGUF'):
            try:one=_http_json(base+'/object_info/'+node_name,timeout=5)
            except urllib.error.HTTPError as exc:
                if exc.code==404:continue
                raise
            if isinstance(one,dict):node_info.update(one)
        required_common=('CLIPLoader','VAELoader','TextEncodeQwenImage21','KSampler','VAEDecode','SaveImage')
        common_ok=all(name in node_info for name in required_common)
        clip_blob=json.dumps(node_info.get('CLIPLoader',{}),ensure_ascii=False)
        qwen_nodes=bool(common_ok and 'qwen_image' in clip_blob)
        devices=(stats.get('devices') if isinstance(stats,dict) else None) or []
        gpu=''
        cuda=False
        for item in devices if isinstance(devices,list) else []:
            if not isinstance(item,dict):continue
            name=str(item.get('name') or item.get('type') or '')
            if name and not gpu:gpu=name
            blob=json.dumps(item,ensure_ascii=False).lower()
            if 'cuda' in blob or 'nvidia' in blob:cuda=True
        gguf_qwen21,gguf_detail=_gguf_qwen21_support(root)
        return {'comfyui_ok':True,'cuda_available':cuda,'qwen_image_nodes':qwen_nodes,'native_qwen_loader':'UNETLoader' in node_info,
                'gguf_loader':'UnetLoaderGGUF' in node_info,'gguf_qwen21':gguf_qwen21,'gguf_qwen21_detail':gguf_detail,'gpu':gpu,'system_stats':stats}
    finally:
        if proc is not None and proc.poll() is None:
            try:proc.terminate();proc.wait(10)
            except Exception:
                try:proc.kill()
                except Exception:pass
        if lf is not None:
            try:lf.close()
            except Exception:pass
        shutil.rmtree(work,ignore_errors=True)

def _monitor(proc,stop,out):
    peak_rss=0;peak_vram=0
    try:import psutil
    except Exception:psutil=None
    while not stop.wait(.1):
        pids={proc.pid}
        if psutil:
            try:
                pp=psutil.Process(proc.pid);children=pp.children(recursive=True);pids.update(x.pid for x in children)
                rss=pp.memory_info().rss+sum(x.memory_info().rss for x in children if x.is_running())
                peak_rss=max(peak_rss,rss)
            except Exception:pass
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=2)
            total=0
            for line in raw.splitlines():
                parts=[x.strip() for x in line.split(',')]
                if len(parts)>=2 and parts[0].isdigit() and int(parts[0]) in pids:
                    try:total+=int(parts[1])*1024*1024
                    except ValueError:pass
            peak_vram=max(peak_vram,total)
        except Exception:pass
        # Publish running peaks so a successful result never snapshots an empty
        # metrics dict before the monitor thread is joined.
        out.update(peak_rss_bytes=peak_rss or None,peak_gpu_process_bytes=peak_vram or None)
    out.update(peak_rss_bytes=peak_rss or None,peak_gpu_process_bytes=peak_vram or None)


def decode_qwen21_vae_file(worker,job,vae_path,tensor):
    """Decode an already-sampled Qwen 2.1 latent with a ComfyUI-layout VAE.

    Sampling remains in Image Lab/Diffusers. The tensor captured at the exact
    upstream vae.decode input boundary is decoded by ComfyUI's own VAE loader
    in the user's verified external runtime.
    """
    from safetensors.torch import save_file
    from PIL import Image
    path=Path(str(vae_path or '')).expanduser()
    if not path.is_absolute() or not path.is_file() or path.suffix.lower()!='.safetensors':
        raise RuntimeError('Hybrid ComfyUI VAE decode needs an explicit .safetensors VAE file.')
    root,exe=_runtime();work=Path(tempfile.mkdtemp(prefix='imagelab-qwen-vae-decode-'))
    latent=work/'latent.safetensors';out=work/'decoded.png';script=work/'decode_vae.py'
    log=SHARED_ROOT/'logs/qwen-native-comfy'/f"{job.get('job_id') or uuid.uuid4().hex}-vae.log"
    log.parent.mkdir(parents=True,exist_ok=True)
    try:
        samples=tensor.detach().to('cpu',copy=True).contiguous()
        save_file({'samples':samples},str(latent))
        code="""import json,sys,time
from pathlib import Path
root=Path(sys.argv[1]);vae_path=Path(sys.argv[2]);latent_path=Path(sys.argv[3]);out_path=Path(sys.argv[4])
sys.path.insert(0,str(root))
import torch
from safetensors.torch import load_file
from PIL import Image
import comfy.utils,comfy.sd
started=time.perf_counter()
sd=comfy.utils.load_torch_file(str(vae_path),safe_load=True)
vae=comfy.sd.VAE(sd=sd)
samples=load_file(str(latent_path),device='cpu')['samples']
if int(getattr(vae,'latent_channels',-1))!=int(samples.shape[1]):
    raise RuntimeError(f'VAE latent channels {getattr(vae,\"latent_channels\",None)} do not match sampled latent {samples.shape[1]}.')
if int(getattr(vae,'output_channels',4) or 4) not in (3,4):
    raise RuntimeError(f'Unexpected VAE output channels: {getattr(vae,\"output_channels\",None)}')
image=vae.decode(samples)
if not torch.is_tensor(image):raise RuntimeError('ComfyUI VAE decode did not return a tensor.')
x=image.detach().float().cpu()
if x.ndim==5:
    if x.shape[-1] in (3,4):
        if x.shape[1]==1:x=x[:,0]
        elif x.shape[2]==1:x=x[:,:,0]
        else:raise RuntimeError('Hybrid benchmark expects a single decoded image frame.')
    elif x.shape[1] in (3,4):
        if x.shape[2]!=1:raise RuntimeError('Hybrid benchmark expects a single decoded image frame.')
        x=x[:,:,0].permute(0,2,3,1)
    else:raise RuntimeError(f'Unexpected 5D decoded VAE tensor shape: {tuple(x.shape)}')
if x.ndim!=4:raise RuntimeError(f'Unexpected decoded VAE tensor rank/shape: {tuple(x.shape)}')
if x.shape[-1] in (3,4):frame=x[0]
elif x.shape[1] in (3,4):frame=x[0].permute(1,2,0)
else:raise RuntimeError(f'Unexpected decoded VAE channel layout: {tuple(x.shape)}')
if not bool(torch.isfinite(frame).all()):raise RuntimeError('ComfyUI VAE decode produced NaN/Inf.')
frame=(frame.clamp(0,1)*255.0).round().to(torch.uint8).numpy()
mode='RGBA' if frame.shape[-1]==4 else 'RGB'
Image.fromarray(frame,mode=mode).save(out_path,format='PNG')
print('__IMAGELAB_JSON__'+json.dumps({'decode_seconds':time.perf_counter()-started,'latent_channels':int(getattr(vae,'latent_channels',samples.shape[1])),'output_channels':int(frame.shape[-1]),'width':int(frame.shape[1]),'height':int(frame.shape[0]),'dtype':str(samples.dtype)}))
"""
        script.write_text(code,encoding='utf-8')
        cmd=[str(exe),'-B',str(script),str(root),str(path.resolve()),str(latent),str(out)]
        with open(log,'w',encoding='utf-8') as lf:
            proc=subprocess.Popen(cmd,cwd=str(root),stdout=subprocess.PIPE,stderr=lf,text=True)
            deadline=time.monotonic()+900
            while proc.poll() is None:
                worker._check_cancel(job)
                if time.monotonic()>deadline:
                    proc.terminate();raise RuntimeError('Timed out decoding the benchmark latent with the configured ComfyUI VAE.')
                time.sleep(.2)
            stdout=proc.stdout.read() if proc.stdout else ''
        if proc.returncode!=0:
            tail=''
            try:tail=log.read_text(encoding='utf-8',errors='replace')[-5000:]
            except OSError:pass
            raise RuntimeError('ComfyUI VAE-only decode failed. See '+str(log)+'. '+tail)
        if not out.is_file():raise RuntimeError('ComfyUI VAE-only decode completed without an output image. See '+str(log))
        meta={}
        for line in stdout.splitlines()[::-1]:
            if line.startswith('__IMAGELAB_JSON__'):
                try:meta=json.loads(line[len('__IMAGELAB_JSON__'):])
                except ValueError:meta={}
                break
        with Image.open(out) as im:img=im.copy()
        info={'decode_device':'configured ComfyUI','vae_dtype':'ComfyUI native','vae_decode_mode':'comfyui_hybrid','decode_seconds':meta.get('decode_seconds'),
              'automatic_tiled_fallback':False,'vae_path':str(path.resolve()),'custom_tiles':None,'execution_backend':'ComfyUI VAE-only decode',
              'hybrid_sampling_backend':'Diffusers baseline','hybrid_vae_backend':'ComfyUI','comfy_log':str(log)}
        return img,info
    finally:
        shutil.rmtree(work,ignore_errors=True)

def run(worker,job,cfg):
    if job.get('task')!='generate' or job.get('source_path') or job.get('reference_paths'):
        raise RuntimeError('Native ComfyUI quantized execution currently supports text-to-image benchmark jobs only. This job is marked failed/unsupported; the remaining batch continues.')
    if job.get('loras'):
        raise RuntimeError('Native ComfyUI quantized bridge does not apply Image Lab LoRAs yet. Disable LoRAs for this benchmark job.')
    try:
        transformer=_file(cfg,'model_path',('.safetensors','.gguf'));encoder=_file(cfg,'text_encoder_path');vae=_file(cfg,'vae_path')
    except RuntimeError as exc:
        raise RuntimeError('Native ComfyUI quantized execution requires an explicit .safetensors or .gguf Transformer plus explicit single-file .safetensors Text Encoder and VAE. Repository folders are not silently converted or reconstructed. '+str(exc)) from exc
    # At least one selected component must actually be quantized; native_comfyui is
    # not a hidden alternative backend for arbitrary user files.
    encoder_info=inspect_weights(encoder)
    transformer_info=None if transformer.suffix.lower()=='.gguf' else inspect_weights(transformer)
    native_int8=bool((transformer_info or {}).get('quantized') or encoder_info.get('quantized'))
    native_quantized=bool(transformer.suffix.lower()=='.gguf' or native_int8)
    root,exe=_runtime();jid=job.get('job_id') or uuid.uuid4().hex
    if transformer.suffix.lower()=='.gguf':
        supported,detail=_gguf_qwen21_support(root)
        if not supported:
            raise RuntimeError('Configured ComfyUI-GGUF does not support Qwen Image 2.1 architecture qwen_image21. '+detail)
        transformer=_prepare_qwen21_gguf(exe,transformer,job,worker)
    work=Path(tempfile.mkdtemp(prefix='imagelab-qwen-native-'))
    proc=None;stop=threading.Event();measure={};thread=None;started=time.perf_counter()
    try:
        models=work/'models';outdir=work/'output';tmpdir=work/'temp';userdir=work/'user'
        for d in (models/'diffusion_models',models/'text_encoders',models/'vae',outdir,tmpdir,userdir):d.mkdir(parents=True,exist_ok=True)
        for src,dst in ((transformer,models/'diffusion_models'/transformer.name),(encoder,models/'text_encoders'/encoder.name),(vae,models/'vae'/vae.name)):
            os.symlink(src,dst)
        yaml=work/'extra_model_paths.yaml'
        # Paths are controlled absolute temp paths; quote JSON-style for YAML safety.
        yaml.write_text('imagelab:\n  base_path: '+json.dumps(str(models))+'\n  diffusion_models: diffusion_models\n  unet: diffusion_models\n  text_encoders: text_encoders\n  clip: text_encoders\n  vae: vae\n',encoding='utf-8')
        port=_port();log=SHARED_ROOT/'logs/qwen-native-comfy'/f'{jid}.log';log.parent.mkdir(parents=True,exist_ok=True)
        lf=open(log,'w',encoding='utf-8')
        cmd=[str(exe),'-B',str(root/'main.py'),'--listen','127.0.0.1','--port',str(port),'--disable-auto-launch','--extra-model-paths-config',str(yaml),'--output-directory',str(outdir),'--temp-directory',str(tmpdir),'--user-directory',str(userdir)]
        proc=subprocess.Popen(cmd,cwd=str(root),stdout=lf,stderr=subprocess.STDOUT,start_new_session=True)
        thread=threading.Thread(target=_monitor,args=(proc,stop,measure),daemon=True);thread.start()
        base=f'http://127.0.0.1:{port}';t=time.perf_counter();stats=_wait_ready(base,proc,90,log);server_start=time.perf_counter()-t
        gguf=transformer.suffix.lower()=='.gguf'
        node_info={}
        names=('CLIPLoader','VAELoader','TextEncodeQwenImage21','KSampler','VAEDecode','SaveImage',('UnetLoaderGGUF' if gguf else 'UNETLoader'))
        for node_name in names:
            try:one=_http_json(base+'/object_info/'+node_name,timeout=5)
            except urllib.error.HTTPError as exc:
                if exc.code==404:continue
                raise
            if isinstance(one,dict):node_info.update(one)
        _validate_object_info(node_info,gguf=gguf)
        workflow,seed=_workflow(job,transformer,encoder,vae)
        worker._write_job_progress(job,'native_comfyui',0,int(job.get('steps') or 25),.02,execution_mode='comfyui_native_quantized')
        t=time.perf_counter()
        try:reply=_http_json(base+'/prompt',{'prompt':workflow,'client_id':'imagelab-'+jid},timeout=20)
        except urllib.error.HTTPError as exc:
            detail=exc.read(65536).decode('utf-8','replace')
            raise RuntimeError('ComfyUI rejected the native Qwen workflow: '+detail[-4000:]) from exc
        pid=reply.get('prompt_id')
        if not pid:raise RuntimeError('ComfyUI returned no prompt identifier.')
        history=None
        while time.monotonic()-t<1800:
            worker._check_cancel(job)
            if proc.poll() is not None:raise RuntimeError('ComfyUI native process exited before the prompt completed.')
            try:
                h=_http_json(base+'/history/'+pid,timeout=3)
                if isinstance(h,dict) and pid in h:
                    history=h[pid];break
            except Exception:pass
            time.sleep(.3)
        if history is None:raise RuntimeError('Native ComfyUI Qwen job timed out.')
        execution=time.perf_counter()-t
        status=history.get('status') or {}
        if status.get('status_str')=='error':raise RuntimeError('ComfyUI native Qwen execution failed. See '+str(log))
        outputs=history.get('outputs') or {};images=[]
        for node in outputs.values():
            if isinstance(node,dict):images.extend(node.get('images') or [])
        if not images:raise RuntimeError('Native ComfyUI Qwen job completed without an output image. See '+str(log))
        item=images[0];src=outdir/str(item.get('subfolder') or '')/str(item.get('filename') or '')
        if not src.is_file():raise RuntimeError('Native ComfyUI output file is missing: '+str(src))
        dst=Path(job['output_path']);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        worker._write_job_progress(job,'saving',int(job.get('steps') or 25),int(job.get('steps') or 25),1)
        # Freeze the monitor after inference/output so the returned metrics are the
        # final observed peaks, not a race against the daemon thread.
        stop.set()
        if thread:thread.join(2);thread=None
        ext={'schema':1,'status':'completed','native_int8_benchmark':native_int8,'native_quantized_benchmark':native_quantized,'execution_mode':'comfyui_native_quantized',
             'memory_scope':'nvidia-smi process memory for isolated ComfyUI process; RSS includes its child processes',
             'peak_rss_bytes':measure.get('peak_rss_bytes'),'peak_gpu_process_bytes':measure.get('peak_gpu_process_bytes'),
             'peak_cuda_allocated_bytes':None,'peak_cuda_reserved_bytes':None,'worker_seconds':execution,'gpu':None}
        return {'output_path':str(dst),'seed':seed,'steps':int(job.get('steps') or 25),'task':'generate','pipeline':'ComfyUI/QwenImage21',
                'width':int(job.get('width') or 1024),'height':int(job.get('height') or 1024),'input_images':0,'input_paths':[],
                'load_seconds':server_start,'conditioning_and_sampling_seconds':execution,'decode_seconds':None,'offload_seconds':None,
                'quantized_loading':'native_comfyui','precision_note':('Native ComfyUI-GGUF loader; no Diffusers GGUF reinterpretation.' if transformer.suffix.lower()=='.gguf' else 'Native ComfyUI loader; no Image Lab float reconstruction.'),
                'execution_mode':'comfyui_native_quantized','execution_backend':'ComfyUI local isolated process','sampler_name':'euler','scheduler_name':'simple',
                'native_int8_benchmark':native_int8,'native_quantized_benchmark':native_quantized,'_external_metrics':ext,
                'native_runtime':{'comfy_root':str(root),'python':str(exe),'system_stats':stats}}
    finally:
        stop.set()
        if thread:thread.join(2)
        if proc is not None and proc.poll() is None:
            try:proc.terminate();proc.wait(10)
            except Exception:
                try:proc.kill()
                except Exception:pass
        try:
            if 'lf' in locals():lf.close()
        except Exception:pass
        shutil.rmtree(work,ignore_errors=True)
