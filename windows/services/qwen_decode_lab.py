"""Sampling/decoding separation at the exact upstream vae.decode input boundary.

No rewrite of the scheduler loop, latent unpacking, normalisation, attention or
prompt processing. The temporary *instance-local* decode boundary aborts only
with a private control signal, after saving an immutable CPU tensor. Unwinding
the pipeline releases sampling/KV-cache locals before loading decoder outputs.
"""
from __future__ import annotations
import copy,gc,inspect,time
from pathlib import Path
from core.qwen_components import decode_options,latent_contract,assert_same_latent_space
from core.qwen_latents import LatentStore
from core.runtime_env import SHARED_ROOT
from services.qwen_components_runtime import (has_custom_components,build_custom_pipeline,
    detach_for_decode,decode_tensor,load_vae,miniature_vae_test)

class _SamplingComplete(BaseException):pass

def store():return LatentStore(SHARED_ROOT/'cache/qwen-latents')

def check_finite_sampling_latent(tensor,job,cfg):
    """Record observable NaN/Inf failure, without guessing or changing output."""
    import torch
    finite=torch.isfinite(tensor)
    if bool(finite.all()):return
    nan=int(torch.isnan(tensor).sum().item())
    posinf=int(torch.isposinf(tensor).sum().item())
    neginf=int(torch.isneginf(tensor).sum().item())
    finite_count=int(finite.sum().item());total=tensor.numel()
    valid=tensor[finite]
    bounds=(f'finite min={float(valid.min()):.6g} max={float(valid.max()):.6g}'
            if finite_count else 'no finite values')
    loras=job.get('loras') or []
    entries=', '.join(f"{Path(str(l.get('path',''))).name} (strength={l.get('strength',1)})" for l in loras[:8]) or 'none'
    details=(f'Non-finite latent after sampling, before VAE decode: '
             f'shape={tuple(tensor.shape)}, dtype={tensor.dtype}, '
             f'NaN={nan}, +Inf={posinf}, -Inf={neginf}, finite={finite_count}/{total}; '
             f'{bounds}. Transformer={Path(str(cfg.get("model_path") or cfg.get("model_dir") or "baseline")).name}; '
             f'LoRA={entries}; CFG={job.get("cfg")}, steps={job.get("steps")}, '
             f'precision={cfg.get("dtype")}, memory_mode={cfg.get("memory_mode")}. '
             f'No automatic repair or PNG write attempted.')
    raise ValueError(details)


def run_split(worker,job,cfg):
    import torch
    options=decode_options(cfg,job);cache=store();rid=job['job_id'];keep=options['retain_latents']
    # A recovered job whose sampler had finished must not spend 40 steps again.
    if keep and cache.exists(rid):return redecode(worker,job,cfg,rid)
    if keep:cache.preflight()
    pipe=None;vae=None;holder={};saved=None;info={};processor=None;lora_audit=None
    hybrid_vae=cfg.get('_benchmark_vae_backend')=='comfyui_hybrid' and bool(cfg.get('vae_path'))
    sampling_cfg=cfg
    if hybrid_vae:
        sampling_cfg=copy.deepcopy(cfg)
        sampling_cfg['vae_path']='';sampling_cfg['vae_config_path']='';sampling_cfg['_benchmark_vae_backend']=''
        sampling_cfg['quantized_loading']='reject'
    run_started=time.perf_counter();timings={}
    try:
        worker._check_cancel(job)
        loras=job.get('loras') or []
        if loras:
            from services.qwen_lora_runtime import prepare_lora_pipeline
            worker._write_job_progress(job,'lora_loading',0,0,0)
            pipe,lora_audit=prepare_lora_pipeline(worker,sampling_cfg,loras)
            worker._write_job_progress(job,'lora_validated',0,0,0,
                                       adapter_layers=lora_audit['adapter_layers'],
                                       adapter_placement=lora_audit['placement'])
        elif has_custom_components(sampling_cfg):
            worker.clear();pipe=build_custom_pipeline(sampling_cfg)
        else:pipe=worker.load_qwen21(sampling_cfg)
        timings['load_seconds']=time.perf_counter()-run_started
        # Full-frame remains the encode mode too, matching the clean 1.0.21 path.
        # Custom tiles only alter final decoding, never silently alter the sampler.
        worker._set_qwen21_vae_decode_mode(pipe,'tiled' if options['vae_decode_mode']=='tiled' else 'full_frame')
        task,images,input_paths=worker._qwen21_inputs(job)
        seed=worker._seed(job);steps=int(job.get('steps') or cfg.get('default_steps') or 40)
        true_cfg=float(job.get('cfg') if job.get('cfg') is not None else cfg.get('default_cfg',1.0));neg=(job.get('negative_prompt') or '').strip()
        sig=inspect.signature(pipe.__call__).parameters
        args={'prompt':job.get('prompt',''),'num_inference_steps':steps,'num_images_per_prompt':1,'generator':worker._generator(seed)}
        if neg and 'negative_prompt' in sig:args['negative_prompt']=neg
        if 'true_cfg_scale' in sig and (neg or true_cfg>1):args['true_cfg_scale']=true_cfg
        elif 'guidance_scale' in sig and (neg or true_cfg>1):args['guidance_scale']=true_cfg
        if images:args['image']=images[0] if len(images)==1 else images
        if task in ('edit','inpaint') and images:w,h=images[0].size
        else:w,h=job.get('width'),job.get('height')
        for k,v in [('width',w),('height',h)]:
            if k in sig and v:args[k]=int(v)
        if 'strength' in sig and images:args['strength']=float(job.get('strength',.35))
        args={k:v for k,v in args.items() if k in sig};args.update(worker._progress_args(pipe,job,steps))
        vae=pipe.vae;processor=pipe.image_processor;contract=latent_contract(dict(vae.config))
        original=vae.__dict__.get('decode');had_override='decode' in vae.__dict__
        def capture(latents,*decode_args,**decode_kwargs):
            worker._check_cancel(job)
            # Copy prevents reuse/mutation of a view and drops CUDA ownership.
            holder['tensor']=latents.detach().to('cpu',copy=True).contiguous()
            raise _SamplingComplete()
        vae.decode=capture
        worker._write_job_progress(job,'generating',0,steps,0)
        sample_started=time.perf_counter()
        try:
            try:
                with torch.inference_mode():pipe(**args)
            except _SamplingComplete:pass
        finally:
            if had_override:vae.decode=original
            else:delattr(vae,'decode')
        if 'tensor' not in holder:raise RuntimeError('The installed pipeline did not reach the supported Qwen VAE decode boundary. No guessed latent conversion was used.')
        timings['conditioning_and_sampling_seconds']=time.perf_counter()-sample_started
        tensor=holder.pop('tensor')
        del args,images
        check_finite_sampling_latent(tensor,job,cfg)
        worker._write_job_progress(job,'saving_latent',steps,steps,.96)
        metadata={'id':rid,'project_id':job['project_id'],'job_id':rid,'seed':seed,'steps':steps,
            'prompt':job.get('prompt',''),'transparent_output':bool(job.get('transparent_output')),
            'contract':contract,'width':int(tensor.shape[-1])*16,'height':int(tensor.shape[-2])*16,
            'source_config':dict(cfg),'source_parameters':{k:job.get(k) for k in ('task','cfg','negative_prompt','width','height','loras')},
            'input_images':len(input_paths),'input_paths':input_paths}
        if keep:saved=cache.save(tensor,metadata)
        worker._write_job_progress(job,'offloading_for_decode',steps,steps,.97,latent_id=rid if keep else None)
        offload_started=time.perf_counter()
        vae_needs_reload=detach_for_decode(pipe)
        # The discarded DiT/encoder may contain Accelerate meta placeholders;
        # NEVER call .to(cpu) / reset_device_map() on them. Their references
        # and hooks go away with the pipeline, not with a forced copy.
        if vae_needs_reload:
            vae=None
        worker.clear();pipe=None;gc.collect()
        if vae_needs_reload and not hybrid_vae:
            worker._write_job_progress(job,'reloading_vae_after_offload',steps,steps,.975)
            vae,vae_info=load_vae(sampling_cfg,dtype=torch.float32 if options['decode_device']=='cpu' else None)
            assert_same_latent_space(vae_info['config'],contract)
        timings['offload_seconds']=time.perf_counter()-offload_started
        worker._write_job_progress(job,'vae_decoding',steps,steps,.98,latent_id=rid if keep else None,decode_device=options['decode_device'])
        if hybrid_vae:
            from services.qwen_comfy_native import decode_qwen21_vae_file
            img,info=decode_qwen21_vae_file(worker,job,cfg.get('vae_path'),tensor)
        else:
            try:
                img,info=decode_tensor(vae,tensor,cfg,options,image_processor=processor,cancel=lambda:worker._check_cancel(job))
            except torch.OutOfMemoryError as first_oom:
                if not bool(job.get('oom_cpu_decode_fallback')) or options['decode_device']=='cpu':
                    raise
                # The CUDA failure occurred AFTER sampling and the exact latent is safe on CPU.
                # No scheduler steps are repeated and tiled decoding is not silently enabled.
                first_message=str(first_oom)
                del first_oom
                vae.to('cpu');gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
                worker._write_job_progress(job,'vae_cpu_oom_retry',steps,steps,.985)
                fallback={**options,'decode_device':'cpu','vae_dtype':'float32'}
                try:
                    img,info=decode_tensor(vae,tensor,cfg,fallback,image_processor=processor,cancel=lambda:worker._check_cancel(job))
                except Exception as cpu_error:
                    raise RuntimeError('GPU VAE decode OOM followed by unsuccessful CPU full-frame decode. The latent is preserved when retain_latents is enabled. GPU: '+first_message+' | CPU: '+str(cpu_error)) from cpu_error
                info['oom_recovered_on_cpu']=True
                info['oom_initial_cuda_error']=first_message[:400]
        # Unified UI never submits inpaint. Preserve the existing programmatic
        # composite for source compatibility, without claiming native masking.
        if task=='inpaint':
            from PIL import Image,ImageFilter
            with Image.open(input_paths[0]) as src:original_img=src.convert('RGB')
            with Image.open(job['mask_path']) as src:mask=src.convert('L').resize(original_img.size,Image.Resampling.LANCZOS)
            mask=mask.filter(ImageFilter.GaussianBlur(max(1,min(original_img.size)//300)))
            img=Image.composite(img.convert('RGB').resize(original_img.size),original_img,mask)
        worker._check_cancel(job);worker._write_job_progress(job,'saving',steps,steps,1)
        out=Path(job['output_path']);out.parent.mkdir(parents=True,exist_ok=True);img.save(out,format='PNG')
        result={**timings,**info,'output_path':str(out),'seed':seed,'steps':steps,'task':task,'pipeline':('QwenImage21Pipeline + ComfyUI VAE decode' if hybrid_vae else 'QwenImage21Pipeline'),
            'input_images':len(input_paths),'input_paths':input_paths,'width':img.width,'height':img.height,
            'rgba':img.mode=='RGBA','latent_id':rid if keep else None,'sampling_repeated':False,
            'final_stage_offload':True,'transformer_override':cfg.get('model_path',''),'runtime_memory_mode':cfg.get('memory_mode','balanced'),
            'true_cfg_scale':true_cfg if (neg or true_cfg>1) else 1.0,
            'quantized_loading':cfg.get('quantized_loading','reject'),'benchmark_vae_backend':('comfyui_hybrid' if hybrid_vae else None),
            'lora_application':lora_audit,
            'precision_note':('Hybrid benchmark: baseline Diffusers sampling with selected VAE-only decode in verified ComfyUI.' if hybrid_vae else ('Reference dequantization: floating reconstruction, not native INT8 execution.' if cfg.get('quantized_loading')=='reference_dequantize' else ''))}
        if keep:cache.result(rid,{'ok':True,'job_id':job['job_id'],'output_path':str(out),'decode':info})
        return result
    except torch.OutOfMemoryError as exc:
        # Tensor is already on disk. Do NOT retry tiled or resample invisibly.
        if saved:
            cache.result(rid,{'ok':False,'job_id':job['job_id'],'error':'CUDA OOM during full decode','decode':options})
            raise RuntimeError('Qwen sampling completed; latent '+rid+' is preserved. Decode still exceeded GPU memory. Open Decode Lab and select CPU full-frame, or another compatible VAE. No tiled fallback and no repeated steps.') from exc
        raise RuntimeError('CUDA OOM before completing Qwen sampling or before a recoverable latent was saved. A CPU decode fallback cannot recover sampling-stage OOM. Reduce VRAM pressure or use CPU offload; no hidden resolution/step changes were made.') from exc
    except Exception as exc:
        if saved:
            try:cache.result(rid,{'ok':False,'job_id':job['job_id'],'error':str(exc)[:2000]})
            except (ValueError,OSError):pass
        raise
    finally:
        # Errors/cancellation restore the bound method in the inner finally.
        if pipe is not None:
            try:detach_for_decode(pipe)
            except Exception:pass
        if vae is not None:
            try:vae.to('cpu')
            except Exception:pass
        pipe=None;vae=None;holder.clear();worker.clear();gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def redecode(worker,job,cfg,rid):
    import torch
    cache=store();vae=None
    try:
        worker._check_cancel(job);worker.clear()
        tensor,meta=cache.load(rid,job['project_id'])
        worker._write_job_progress(job,'loading_vae',0,0,.1,latent_id=rid)
        # Load ONLY the VAE. No transformer, processor, TE or scheduler model.
        if cfg.get('_benchmark_vae_backend')=='comfyui_hybrid' and cfg.get('vae_path'):
            from services.qwen_comfy_native import decode_qwen21_vae_file
            worker._write_job_progress(job,'vae_decoding',0,0,.5,latent_id=rid)
            img,info=decode_qwen21_vae_file(worker,job,cfg['vae_path'],tensor)
            out=Path(job['output_path']);out.parent.mkdir(parents=True,exist_ok=True)
            worker._check_cancel(job);img.save(out,format='PNG')
            cache.result(rid,{'ok':True,'job_id':job['job_id'],'output_path':str(out),'decode':info})
            return {**info,'output_path':str(out),'seed':meta['seed'],'steps':0,'original_steps':meta['steps'],
                'pipeline':'QwenImage21 + ComfyUI VAE re-decode','latent_id':rid,'latent_sha256':meta['sha256'],
                'sampling_repeated':False,'width':img.width,'height':img.height,'rgba':img.mode=='RGBA'}
        vae,loaded_info=load_vae(cfg,dtype=torch.float32 if decode_options(cfg,job)['decode_device']=='cpu' else None)
        assert_same_latent_space(loaded_info['config'],meta['contract'])
        if cfg.get('vae_path'):
            # Custom weights must pass the same small structural test for a
            # re-decode as for first use in sampling.
            original_dtype=vae.dtype;vae.to(device='cpu',dtype=torch.float32)
            miniature_vae_test(vae);vae.to(dtype=original_dtype)
        options=decode_options(cfg,job)
        worker._write_job_progress(job,'vae_decoding',0,0,.5,latent_id=rid)
        img,info=decode_tensor(vae,tensor,cfg,options,cancel=lambda:worker._check_cancel(job))
        out=Path(job['output_path']);out.parent.mkdir(parents=True,exist_ok=True)
        worker._check_cancel(job);img.save(out,format='PNG')
        cache.result(rid,{'ok':True,'job_id':job['job_id'],'output_path':str(out),'decode':info})
        return {**info,'output_path':str(out),'seed':meta['seed'],'steps':0,'original_steps':meta['steps'],
            'pipeline':'QwenImage21 VAE re-decode','latent_id':rid,'latent_sha256':meta['sha256'],
            'sampling_repeated':False,'width':img.width,'height':img.height,'rgba':img.mode=='RGBA'}
    except torch.OutOfMemoryError as exc:
        cache.result(rid,{'ok':False,'job_id':job['job_id'],'error':'CUDA OOM during re-decode'})
        raise RuntimeError('VAE re-decode exceeded GPU memory; the original latent is intact. Select CPU full-frame explicitly. No resampling/tiled fallback.') from exc
    finally:
        if vae is not None:
            try:vae.to('cpu')
            except Exception:pass
        vae=None;worker.clear();gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()


def test_vae(worker,job,cfg):
    import torch
    vae=None
    try:
        worker.clear();worker._check_cancel(job)
        worker._write_job_progress(job,'loading_vae_test',0,0,.1)
        vae,info=load_vae(cfg,dtype=torch.float32)
        # Diagnostic is deliberately CPU, small full-frame; no GPU claim.
        vae.to(device='cpu',dtype=torch.float32);worker._check_cancel(job)
        result=miniature_vae_test(vae);worker._check_cancel(job)
        return {'test_report':{**result,'vae_path':info['path'],'contract_hash':info['contract_hash'],'class':info['class']}}
    finally:
        vae=None;worker.clear();gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
