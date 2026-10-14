"""Qwen Image Lab worker. Original model loading / inputs / CFG call contract
retained; 1.0.22 separates the exact upstream VAE-input boundary, preserves latents,
and adds explicitly selected compatible components. No SDXL worker or tiled fallback.
"""
from core.runtime_env import initialize_environment,protect_worker,SHARED_ROOT,MODELS_ROOT
initialize_environment()
from core.gpu_session import gpu_session
import os,sys,gc,traceback,inspect,json,time,threading
from pathlib import Path
from flask import Flask,request,jsonify
from PIL import Image,ImageFilter
app=Flask(__name__)
protect_worker(app)
_execution_lock=threading.Lock()
loaded={"key":None,"pipe":None,"lora_signature":""}
_last_options=None

def versions():
    out={"python":sys.version.split()[0]}
    for name in ("torch","diffusers","transformers","safetensors","accelerate","peft"):
        try:
            mod=__import__(name); out[name]=getattr(mod,"__version__","OK")
        except Exception as e: out[name]="MISSING: "+str(e)
    try:
        import torch
        out["cuda_available"]=torch.cuda.is_available()
        out["cuda_device"]=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info()
            out["cuda_free_gib"]=round(free/1024**3,2)
            out["cuda_total_gib"]=round(total/1024**3,2)
    except Exception: pass
    return out


def dtype_from(cfg):
    import torch
    return {"float16":torch.float16,"bfloat16":torch.bfloat16,"float32":torch.float32}.get(cfg.get("dtype"),torch.bfloat16)


def clear():
    loaded["pipe"]=None; loaded["key"]=None; loaded["lora_signature"]=""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception: pass


def _dtype_kwargs(dtype):
    # Current Diffusers prefers dtype; older builds use torch_dtype.
    return {"dtype":dtype}


def _load_pretrained(cls, model_dir, cfg):
    dtype=dtype_from(cfg)
    base={"local_files_only":True,"low_cpu_mem_usage":True}
    mode=cfg.get("memory_mode") or ("balanced" if cfg.get("cpu_offload") else "full_gpu")
    device=cfg.get("device","cuda")
    gpu_mem=cfg.get("gpu_memory") or "21GiB"
    cpu_mem=cfg.get("cpu_memory") or "44GiB"

    if mode=="balanced" and device=="cuda":
        kw=dict(base)
        kw.update(_dtype_kwargs(dtype))
        kw.update({"device_map":"balanced","max_memory":{0:gpu_mem,"cpu":cpu_mem}})
        try:
            return cls.from_pretrained(model_dir,**kw), "balanced"
        except TypeError:
            kw.pop("dtype",None);kw["torch_dtype"]=dtype
            return cls.from_pretrained(model_dir,**kw), "balanced"

    kw=dict(base);kw.update(_dtype_kwargs(dtype))
    try:
        pipe=cls.from_pretrained(model_dir,**kw)
    except TypeError:
        kw.pop("dtype",None);kw["torch_dtype"]=dtype
        pipe=cls.from_pretrained(model_dir,**kw)

    if mode in ("cpu_offload","model_cpu_offload") and device=="cuda":
        pipe.enable_model_cpu_offload()
        return pipe,"cpu_offload"

    pipe.to(device)
    return pipe,"full_gpu"


def _load_qwen21_with_transformer_override(pipe_cls, transformer_cls, model_dir, transformer_path, cfg):
    dtype=dtype_from(cfg)
    base={"local_files_only":True,"low_cpu_mem_usage":True}
    mode=cfg.get("memory_mode") or ("balanced" if cfg.get("cpu_offload") else "full_gpu")
    device=cfg.get("device","cuda")
    transformer_path=Path(transformer_path).expanduser()

    # Diffusers GGUF loading must happen on the model class (not the pipeline)
    # and requires GGUFQuantizationConfig so weights stay quantized in memory.
    if transformer_path.suffix.lower()=='.safetensors':
        from core.qwen_weight_format import require_supported_weights
        require_supported_weights(transformer_path,cfg.get('quantized_loading','reject'))
    transform_kw={"config":str(model_dir),"subfolder":"transformer"}
    transform_kw.update(_dtype_kwargs(dtype))
    if transformer_path.suffix.lower()==".gguf":
        try:
            import gguf  # noqa: F401
            from diffusers import GGUFQuantizationConfig
        except Exception as e:
            raise RuntimeError(
                "Qwen 2.1 GGUF requires the 'gguf' package in this app's private runtime. "
                "Update this app runtime before selecting GGUF. "
                f"Details: {e}"
            ) from e
        transform_kw["quantization_config"]=GGUFQuantizationConfig(compute_dtype=dtype)

    try:
        transformer=transformer_cls.from_single_file(str(transformer_path),**transform_kw)
    except TypeError:
        transform_kw.pop("dtype",None);transform_kw["torch_dtype"]=dtype
        transformer=transformer_cls.from_single_file(str(transformer_path),**transform_kw)
    except Exception as e:
        raise RuntimeError(f"Could not load Qwen 2.1 transformer override from {transformer_path}: {e}") from e

    kw=dict(base)
    kw["transformer"]=transformer
    kw.update(_dtype_kwargs(dtype))
    try:
        pipe=pipe_cls.from_pretrained(model_dir,**kw)
    except TypeError:
        kw.pop("dtype",None);kw["torch_dtype"]=dtype
        pipe=pipe_cls.from_pretrained(model_dir,**kw)
    except Exception as e:
        raise RuntimeError(f"Could not assemble Qwen 2.1 pipeline with transformer override {transformer_path}: {e}") from e

    # With a manually injected transformer, balanced device_map is less predictable.
    # Prefer model CPU offload on CUDA for the 24 GB class of GPUs.
    if mode in ("balanced","cpu_offload","model_cpu_offload") and device=="cuda":
        pipe.enable_model_cpu_offload()
        return pipe,"cpu_offload"

    pipe.to(device)
    return pipe,"full_gpu"


def _enable_vae_memory(pipe):
    try:
        if hasattr(pipe,"enable_vae_tiling"): pipe.enable_vae_tiling()
        elif getattr(pipe,"vae",None) is not None and hasattr(pipe.vae,"enable_tiling"): pipe.vae.enable_tiling()
    except Exception: pass
    try:
        if hasattr(pipe,"enable_vae_slicing"): pipe.enable_vae_slicing()
    except Exception: pass


def _seed(job):
    import torch
    seed=int(job.get("seed",-1))
    if seed<0: seed=int(torch.randint(0,2**31-1,(1,)).item())
    return seed


def _generator(seed):
    import torch
    # CPU generator is friendlier to mixed CPU/GPU device maps.
    return torch.Generator(device="cpu").manual_seed(seed)


class JobCancelled(RuntimeError):
    pass


def _write_job_progress(job,phase,step_current=0,step_total=0,progress=None,**extra):
    path=job.get("progress_path")
    if not path:return
    try:
        started=float(job.get("started_epoch") or time.time())
        if progress is None:
            progress=(float(step_current)/float(step_total)) if step_total else 0.0
        payload={
            "job_id":job.get("job_id"),
            "phase":phase,
            "step_current":int(step_current or 0),
            "step_total":int(step_total or 0),
            "progress":max(0.0,min(1.0,float(progress))),
            "elapsed_seconds":max(0.0,time.time()-started),
            "updated_at":time.time(),
            **extra
        }
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


def _check_cancel(job):
    p=job.get("cancel_path")
    if p and Path(p).exists():
        raise JobCancelled("Job cancelled by user.")


def _progress_args(pipe,job,total_steps):
    sig=inspect.signature(pipe.__call__).parameters
    if "callback_on_step_end" in sig:
        def cb(_pipe,step,timestep,callback_kwargs):
            _check_cancel(job)
            cur=int(step)+1
            # 1.0.59: only inspect latent health for LoRA jobs. Qwen 2.1
            # may expose latents through callback_on_step_end_tensor_inputs;
            # never alter, clamp or resample the tensor.
            latents=callback_kwargs.get('latents') if job.get('loras') else None
            if latents is not None:
                import torch
                if not bool(torch.isfinite(latents).all()):
                    invalid=int((~torch.isfinite(latents)).sum().item())
                    raise RuntimeError(f'LoRA sampling became non-finite at step {cur}/{total_steps}: '
                                       f'{invalid}/{latents.numel()} invalid latent values. '
                                       'Generation stopped before VAE decode. Check LoRA adapter validation and CPU-first loading diagnostics.')
            _write_job_progress(job,"generating",cur,total_steps,cur/max(1,total_steps))
            return callback_kwargs
        output={"callback_on_step_end":cb}
        if job.get('loras') and 'callback_on_step_end_tensor_inputs' in sig:
            output['callback_on_step_end_tensor_inputs']=['latents']
        return output
    if "callback" in sig:
        def cb(step,timestep,latents):
            _check_cancel(job)
            cur=int(step)+1
            if job.get('loras') and latents is not None:
                import torch
                if not bool(torch.isfinite(latents).all()):
                    invalid=int((~torch.isfinite(latents)).sum().item())
                    raise RuntimeError(f'LoRA sampling became non-finite at step {cur}/{total_steps}: '
                                       f'{invalid}/{latents.numel()} invalid latent values.')
            _write_job_progress(job,"generating",cur,total_steps,cur/max(1,total_steps))
        out={"callback":cb}
        if "callback_steps" in sig: out["callback_steps"]=1
        return out
    return {}


def load_qwen21(cfg):
    try:
        from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel
    except Exception as e:
        raise RuntimeError(
            "QwenImage21Pipeline unavailable. "
            "The private runtime must provide QwenImage21Pipeline. Check the Qwen edition runtime diagnostics. "
            f"Details: {e}"
        )
    model_dir=str(cfg.get("model_dir") or "").strip()
    if not model_dir or not Path(model_dir).exists():
        raise RuntimeError(
            "Qwen Image 2.1 repository not found. Expected model_dir: "
            f"{model_dir or str(MODELS_ROOT/'Qwen-Image-2.1')}"
        )
    index_path=Path(model_dir)/"model_index.json"
    if not index_path.exists():
        raise RuntimeError(f"Qwen Image 2.1 model_index.json not found: {index_path}")
    try:
        idx=json.loads(index_path.read_text(encoding="utf-8"))
        cls=idx.get("_class_name")
        if cls and cls!="QwenImage21Pipeline":
            raise RuntimeError(f"Unexpected Qwen 2.1 pipeline class in model_index.json: {cls}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Could not read Qwen Image 2.1 model_index.json: {e}")

    transformer_override=str(cfg.get("model_path") or "").strip()
    use_transformer_override=False
    if transformer_override:
        tpath=Path(transformer_override).expanduser()
        if not tpath.exists():
            raise RuntimeError(f"Qwen Image 2.1 transformer override not found: {tpath}")
        use_transformer_override=True
        transformer_override=str(tpath)

    key=("qwen_image_21",model_dir,transformer_override,cfg.get("memory_mode","balanced"),cfg.get("dtype","bfloat16"))
    if loaded["key"]==key and loaded["pipe"] is not None:
        return loaded["pipe"]

    clear()
    if use_transformer_override:
        pipe,mode=_load_qwen21_with_transformer_override(QwenImage21Pipeline,QwenImage21Transformer2DModel,model_dir,transformer_override,cfg)
    else:
        pipe,mode=_load_pretrained(QwenImage21Pipeline,model_dir,cfg)
    _enable_vae_memory(pipe)
    try:
        pipe.set_progress_bar_config(disable=None)
    except Exception:
        pass
    loaded.update(key=key,pipe=pipe)
    return pipe


def _set_qwen21_vae_decode_mode(pipe, mode):
    mode=str(mode or "auto").strip().lower()
    if mode not in ("auto","tiled","full_frame"):
        mode="auto"

    if mode=="tiled":
        # Manual emergency/low-VRAM mode only. Never selected automatically:
        # tiled Qwen 2.1 VAE decode previously produced purple-line/tile artifacts.
        _enable_vae_memory(pipe)
        return "tiled"

    # Auto and Full frame deliberately use one full-frame VAE decode. Memory
    # headroom is created immediately before decode, after diffusion has ended.
    try:
        if hasattr(pipe,"disable_vae_tiling"): pipe.disable_vae_tiling()
        elif getattr(pipe,"vae",None) is not None and hasattr(pipe.vae,"disable_tiling"): pipe.vae.disable_tiling()
    except Exception: pass
    try:
        if hasattr(pipe,"disable_vae_slicing"): pipe.disable_vae_slicing()
        elif getattr(pipe,"vae",None) is not None and hasattr(pipe.vae,"disable_slicing"): pipe.vae.disable_slicing()
    except Exception: pass
    return "full_frame" if mode=="full_frame" else "auto_full_frame"


def _cuda_decode_cleanup():
    """Release only unused CUDA reservations; never change inference tensors."""
    import torch
    gc.collect()
    if not torch.cuda.is_available():
        return {"cuda":False}
    try: torch.cuda.synchronize()
    except Exception: pass
    try: torch.cuda.empty_cache()
    except Exception: pass
    try: torch.cuda.ipc_collect()
    except Exception: pass
    try:
        free,total=torch.cuda.mem_get_info()
        return {"cuda":True,"free_gib":round(free/1024**3,2),"total_gib":round(total/1024**3,2),
                "allocated_gib":round(torch.cuda.memory_allocated()/1024**3,2),
                "reserved_gib":round(torch.cuda.memory_reserved()/1024**3,2)}
    except Exception:
        return {"cuda":True}


def _offload_qwen21_for_vae(pipe, latent_device):
    """Create VAE headroom only after denoising has completed.

    A balanced device_map is reset with Diffusers' public reset_device_map();
    otherwise only no-longer-needed text/transformer modules are moved to CPU.
    The VAE is kept/moved on the latent device. The caller invalidates the cached
    pipeline if this path was needed, so no modified placement is reused silently.
    """
    actions=[];errors=[]
    vae=getattr(pipe,"vae",None)
    device_map=getattr(pipe,"hf_device_map",None)
    if device_map and callable(getattr(pipe,"reset_device_map",None)):
        try:
            pipe.reset_device_map()
            if vae is not None and callable(getattr(vae,"to",None)):
                vae.to(latent_device)
            actions.append("reset_device_map_then_vae_to_"+str(latent_device))
            return {"actions":actions,"errors":errors}
        except Exception as exc:
            errors.append("reset_device_map: "+str(exc))

    for name in ("transformer","text_encoder","text_encoder_2"):
        module=getattr(pipe,name,None)
        if module is None or not callable(getattr(module,"to",None)):
            continue
        try:
            module.to("cpu");actions.append(name+"->cpu")
        except Exception as exc:
            errors.append(name+": "+str(exc))
    if vae is not None and callable(getattr(vae,"to",None)):
        try:
            vae.to(latent_device);actions.append("vae->"+str(latent_device))
        except Exception as exc:
            errors.append("vae: "+str(exc))
    return {"actions":actions,"errors":errors}


def _qwen21_call_with_full_frame_guard(pipe, callargs, job, effective_vae_mode):
    """Run Qwen while intercepting only the final full-frame VAE decode.

    Auto/full-frame never fall back to tiling. First release unused CUDA cache;
    only if the exact full-frame decode still OOMs do we offload finished
    transformer/text-encoder components and retry the SAME latent tensor once.
    """
    import torch
    vae=getattr(pipe,"vae",None)
    original_decode=getattr(vae,"decode",None) if vae is not None else None
    if effective_vae_mode=="tiled" or not callable(original_decode):
        with torch.inference_mode():
            return pipe(**callargs),False

    offloaded={"value":False}
    def guarded_decode(z,*args,**kwargs):
        total=int(job.get("steps") or 40)
        before=_cuda_decode_cleanup()
        _write_job_progress(job,"vae_full_frame",total,total,0.99,memory_before_decode=before)
        first_oom_text=""
        try:
            return original_decode(z,*args,**kwargs)
        except torch.OutOfMemoryError as exc:
            # Keep only text. Leaving the except block releases the traceback frames
            # and decoder intermediates before the offload/retry allocates again.
            first_oom_text=str(exc)
        placement=_offload_qwen21_for_vae(pipe,getattr(z,"device","cuda"))
        offloaded["value"]=bool(placement.get("actions"))
        after=_cuda_decode_cleanup()
        _write_job_progress(job,"vae_full_frame_retry",total,total,0.995,
                            final_stage_offload=placement,memory_after_offload=after)
        try:
            return original_decode(z,*args,**kwargs)
        except torch.OutOfMemoryError as second_oom:
            raise RuntimeError(
                "Qwen Image 2.1 full-frame VAE decode ran out of VRAM even after final-stage "
                "transformer/text-encoder offload. Tiled fallback is intentionally disabled "
                "because it can reintroduce purple-line/tile artifacts. Reduce output size or "
                "free other GPU users, then retry. First OOM: "+first_oom_text+
                " | Retry OOM: "+str(second_oom)
            ) from second_oom

    vae.decode=guarded_decode
    try:
        with torch.inference_mode():
            return pipe(**callargs),offloaded["value"]
    finally:
        vae.decode=original_decode


def _qwen21_inputs(job):
    task=job.get("task","generate")
    source=job.get("source_path")
    refs=[p for p in (job.get("reference_paths") or []) if p and Path(p).exists()]
    paths=[]
    if source and Path(source).exists():
        paths.append(source)
    for p in refs:
        if p not in paths:
            paths.append(p)
    if task=="generate" and not paths:
        return task,[],[]
    if not paths:
        raise RuntimeError(
            "Qwen Image 2.1 requires at least one source/reference image for Edit, Reference, Multi-image or Inpaint. "
            "Use Generate for pure text-to-image."
        )
    images=[Image.open(p).convert("RGB") for p in paths]
    return task,images,paths


def run_qwen21(job,cfg):
    from services.qwen_decode_lab import run_split
    return run_split(sys.modules[__name__],job,cfg)



def qwen_capabilities():
    import importlib.util
    try:
        from diffusers import QwenImage21Pipeline,QwenImage21Transformer2DModel
        return {'qwen_pipeline':True,'transformer_single_file':hasattr(QwenImage21Transformer2DModel,'from_single_file'),
                'gguf':importlib.util.find_spec('gguf') is not None,
                'lora':all(hasattr(QwenImage21Pipeline,n) for n in ('load_lora_weights','set_adapters')),
                'components':True,'decode_lab':True,'latent_representation':'qwen21.vae-input.v1'}
    except Exception as exc:return {'qwen_pipeline':False,'lora':False,'error':str(exc)}

@app.get('/health')
def health():
    return jsonify(ok=True,runtime='native_worker',edition='qwen21',busy=_execution_lock.locked(),
        python_executable=sys.executable,loaded=str(loaded['key']) if loaded['key'] else None,versions=versions(),capabilities=qwen_capabilities())

@app.post('/probe')
def probe():
    from core.model_registry import validate_model
    data=request.get_json(force=True) or {}
    if data.get('provider')!='qwen_image21':return jsonify(ok=False,error='Qwen Image 2.1 only'),400
    check=validate_model({'id':'qwen_image_21','provider':'qwen_image21','config':data.get('config') or {}})
    return jsonify(ok=True,provider='qwen_image21',validation=check,capabilities=qwen_capabilities(),versions=versions())

def apply_optional_loras(pipe,loras,verify=False):
    if not loras:return []
    from core.qwen_assets import inspect_component
    if not all(callable(getattr(pipe,n,None)) for n in ('load_lora_weights','set_adapters')):
        raise ValueError('This QwenImage21Pipeline runtime does not support LoRA loading. No adapter was ignored or applied.')
    import math
    from services.qwen_lora_compat import load_compatible_lora, apply_additive_plans
    names=[];weights=[];additive_plans=[]
    for index,item in enumerate(loras):
        path=Path(item['path']).expanduser();inspect_component(path,'lora');name='qwen_user_'+str(index)
        strength=float(item.get('strength',1))
        if not math.isfinite(strength):raise ValueError('LoRA strength must be finite: '+path.name)
        # Eager injection on undispatched CPU weights; special Turbo layouts
        # retain their extra normalization deltas and compatible PEFT metadata.
        try:
            delta_plan=load_compatible_lora(pipe,path,name,eager=verify)
            additive_plans.append((delta_plan,strength))
        except Exception as exc:
            raise RuntimeError(f'LoRA loading failed for {path.name} (adapter #{index+1}, strength {strength}): {exc}') from exc
        names.append(name);weights.append(strength)
    try:pipe.set_adapters(names,adapter_weights=weights)
    except Exception as exc:
        raise RuntimeError('LoRA adapter activation failed for '+', '.join(names)+': '+str(exc)) from exc
    try:
        # Additive deltas share the adapter strength. A LoRA job creates a
        # fresh CPU pipeline, discarded on failure and after sampling.
        apply_additive_plans(additive_plans)
    except Exception as exc:
        raise RuntimeError('Qwen Turbo additive-parameter application failed: '+str(exc)) from exc
    return names

def execute_qwen_job(job,cfg):
    global _last_options
    if job.get('task')=='qwen_prompt_enhance':
        clear();_last_options=None
        from services.qwen_pe_process import run as run_pe
        return run_pe(sys.modules[__name__],job)
    if job.get('task')=='qwen_roundtrip':
        from services.qwen_roundtrip import run
        return run(sys.modules[__name__],job,cfg)
    if job.get('task') in ('qwen_decode','qwen_vae_test'):
        from services.qwen_decode_lab import redecode,test_vae
        if job['task']=='qwen_vae_test':return test_vae(sys.modules[__name__],job,cfg)
        result=redecode(sys.modules[__name__],job,cfg,job['latent_id'])
        with Image.open(result['output_path']) as image:
            alpha=image.getchannel('A').getextrema() if 'A' in image.getbands() else None
        result.update(alpha_present=alpha is not None,has_transparency=bool(alpha and alpha[0]<255),alpha_extrema=list(alpha) if alpha else None)
        return result
    from services.qwen_comfy_native import native_requested,run as run_native_comfy
    if native_requested(cfg):
        clear();_last_options=None
        return run_native_comfy(sys.modules[__name__],job,cfg)
    refs=job.get('reference_paths') or [];paths=([job['source_path']] if job.get('source_path') else [])+refs
    if not isinstance(refs,list) or len(set(paths))>10:raise ValueError('At most 10 Qwen input images are supported.')
    if any(not isinstance(p,str) or not Path(p).is_file() for p in paths):raise ValueError('An input image file is missing.')
    if job.get('task') not in ('generate','reference','multi_image','edit','inpaint'):raise ValueError('Unsupported Qwen task.')
    loras=job.get('loras') or []
    if loras and str(cfg.get('model_path','')).lower().endswith('.gguf'):
        raise ValueError('LoRA with a GGUF transformer is not enabled. Select Diffusers or a safetensors transformer.')
    # Include ALL relevant options, not just the legacy cache key, so switching
    # CUDA/CPU or adapter stacks cannot reuse an incompatible cached pipeline.
    signature=json.dumps({'config':cfg,'loras':loras,'vae_decode_mode':job.get('vae_decode_mode','auto')},sort_keys=True)
    if signature!=_last_options:clear();_last_options=signature
    result=run_qwen21(job,cfg)
    with Image.open(result['output_path']) as image:
        alpha=image.getchannel('A').getextrema() if 'A' in image.getbands() else None
    result.update(alpha_present=alpha is not None,has_transparency=bool(alpha and alpha[0]<255),
                  alpha_extrema=list(alpha) if alpha else None)
    if job.get('transparent_output') and not result['has_transparency']:
        result['warning']='Transparency was requested, but this result has no transparent pixels. No artificial alpha channel or background removal was added.'
    return result

@app.post('/unload')
def unload():
    global _last_options
    if not _execution_lock.acquire(False):return jsonify(ok=False,error='A generation is active; cancel or wait before unloading.'),409
    try:clear();_last_options=None;return jsonify(ok=True)
    finally:_execution_lock.release()

@app.post('/comfy/probe')
def comfy_probe():
    if not _execution_lock.acquire(blocking=False):return jsonify(ok=False,error='Image worker busy. Wait for the current job.'),409
    try:
        from services.qwen_comfy_native import probe_runtime
        from services.qwen_pe_process import run as run_pe
        d=request.get_json(silent=True) or {};cfg=d.get('config') or {}
        native=probe_runtime(cfg)
        pe_error='';pe={}
        try:
            pe=run_pe(sys.modules[__name__],{'pe_config':{
                'comfy_root':cfg.get('comfy_root',''),'comfy_python':cfg.get('comfy_python',''),
                'runtime_confirmed':bool(cfg.get('runtime_confirmed'))}},probe=True)
        except Exception as exc:
            pe_error=str(exc)[:2000]
        runtime=pe.get('runtime') or {}
        caps={**native,'prompt_enhancer_imports':bool(pe.get('ok')),
              'cuda_available':bool(native.get('cuda_available') or pe.get('cuda_available')),
              'gpu':native.get('gpu') or runtime.get('gpu') or '',
              'python_version':runtime.get('python') or '',
              'torch_version':runtime.get('torch') or ''}
        return jsonify(ok=True,capabilities=caps,native=native,prompt_enhancer=pe,prompt_enhancer_error=pe_error)
    except Exception as exc:return jsonify(ok=False,error=str(exc)),400
    finally:_execution_lock.release()

@app.post('/pe/probe')
def pe_probe():
    if not _execution_lock.acquire(blocking=False):return jsonify(ok=False,error='Image worker busy. Wait for the current job.'),409
    try:
        from services.qwen_pe_process import run as run_pe
        d=request.get_json(silent=True) or {}
        return jsonify(run_pe(sys.modules[__name__],{'pe_config':d.get('config')},probe=True))
    except Exception as exc:return jsonify(ok=False,error=str(exc)),400
    finally:_execution_lock.release()

@app.post('/execute')
def execute():
    data=request.get_json(force=True) or {};cfg=data.get('config') or {};job=data.get('job') or {}
    if data.get('provider')!='qwen_image21':return jsonify(ok=False,error='Qwen Image 2.1 only'),400
    if not _execution_lock.acquire(False):return jsonify(ok=False,error='Qwen worker busy'),409
    try:
        _check_cancel(job)
        display_steps=0 if job.get('task')=='qwen_prompt_enhance' else int(job.get('steps') or 40)
        _write_job_progress(job,'waiting_gpu',0,display_steps,0)
        with gpu_session('image'):
            _write_job_progress(job,'loading_model',0,display_steps,0)
            from services.qwen_metrics import measure_job
            with measure_job(job,cfg) as metrics:
                result=execute_qwen_job(job,cfg)
            external=result.pop('_external_metrics',None)
            result['metrics']=external or metrics
        return jsonify(ok=True,provider='qwen_image21',**result)
    except Exception as exc:
        clear()
        return jsonify(ok=False,error=str(exc),traceback=traceback.format_exc()[-8000:]),500
    finally:_execution_lock.release()

def main():
    from core.runtime_security import require_reviewed_torch
    require_reviewed_torch()
    app.run(host=os.getenv('CREATOR_IMAGE_WORKER_HOST','127.0.0.1'),port=int(os.getenv('CREATOR_IMAGE_WORKER_PORT','8299')),threaded=True)

if __name__=='__main__':main()
