"""Fixed-class local Qwen component loading and full-frame decode support.
No training code, no pickle, no remote Python. This module is worker-only.
"""
from __future__ import annotations
import gc,inspect,json,time
from pathlib import Path
from core.qwen_components import (inspect_vae,inspect_folder,component_location,
    latent_contract,assert_same_latent_space,decode_options,weight_files)
from core.qwen_assets import read_object


def dtype_value(name,default=None):
    import torch
    return {'float16':torch.float16,'bfloat16':torch.bfloat16,'float32':torch.float32}.get(name,default or torch.bfloat16)


def strict_loading_info(info):
    errors={k:info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs') if info.get(k)}
    if errors:raise ValueError('VAE weights are not a strict match for the selected configuration: '+str(errors)[:3000])


def inspect_vae_for_config(cfg):
    reference=None
    if cfg.get('vae_path'):
        try:reference=read_object(Path(cfg.get('model_dir',''))/'vae/config.json')
        except (ValueError,OSError):pass
    return inspect_vae(component_location(cfg,'vae'),cfg.get('vae_config_path',''),reference)


def load_vae(cfg,*,dtype=None):
    import torch
    from diffusers import AutoencoderKLQwenImage21
    info=inspect_vae_for_config(cfg);p=Path(info['path']);dt=dtype or dtype_value(cfg.get('dtype','bfloat16'))
    if p.is_dir():
        # Native Diffusers format is preferred (also used by Texture-Fix).
        vae,loading=AutoencoderKLQwenImage21.from_pretrained(str(p),torch_dtype=dt,
            local_files_only=True,use_safetensors=True,output_loading_info=True,low_cpu_mem_usage=False)
        strict_loading_info(loading)
    else:
        from safetensors import safe_open
        from safetensors.torch import load_file
        vae=AutoencoderKLQwenImage21.from_config(info['config']).to(dtype=dt)
        # Reject a different/comfy-converted key layout, rather than accepting
        # randomly initialised missing layers. No filename-based conversion.
        expected=vae.state_dict()
        with safe_open(str(p),framework='pt',device='cpu') as sf:
            if set(sf.keys())!=set(expected):raise ValueError('Single-file VAE keys differ from native Diffusers Qwen 2.1. Use the Diffusers folder/config + weights format.')
            for key,value in expected.items():
                if tuple(sf.get_slice(key).get_shape())!=tuple(value.shape):raise ValueError('VAE tensor shape mismatch: '+key)
        state=load_file(str(p),device='cpu');vae.load_state_dict(state,strict=True);del state,expected
    vae.eval();vae.requires_grad_(False)
    return vae,info


def miniature_vae_test(vae):
    """Deterministic 64x64 RGBA encode->mode->decode, no sampler / generator use."""
    import torch
    from contextlib import nullcontext
    if hasattr(vae,'disable_tiling'):vae.disable_tiling()
    if hasattr(vae,'disable_slicing'):vae.disable_slicing()
    device=next(vae.parameters()).device;dtype=next(vae.parameters()).dtype
    x=torch.linspace(-.8,.8,4*64*64,device=device,dtype=dtype).reshape(1,4,1,64,64)
    started=time.monotonic()
    with torch.inference_mode():
        encoded=vae.encode(x)
        posterior=getattr(encoded,'latent_dist',None)
        if posterior is None:raise ValueError('VAE encode did not return a latent distribution.')
        z=posterior.mode()
        if tuple(z.shape)!=(1,64,1,4,4):raise ValueError('VAE test latent shape differs from Qwen 2.1: '+str(tuple(z.shape)))
        result=vae.decode(z,return_dict=False)[0]
        if tuple(result.shape)!=tuple(x.shape):raise ValueError('VAE round-trip returned an unexpected RGBA shape: '+str(tuple(result.shape)))
        if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(result).all()):raise ValueError('VAE test produced NaN/Inf.')
        mse=float(torch.mean((result.float()-x.float())**2).item())
    return {'ok':True,'test':'miniature_encode_decode','input_shape':[1,4,1,64,64],
        'latent_shape':[1,64,1,4,4],'output_shape':list(result.shape),'finite':True,'mse':mse,
        'duration_seconds':time.monotonic()-started,'device':str(device),'dtype':str(dtype),
        'quality_guarantee':False,'note':'Structural/numerical test, not a judgement of image quality or 2048 VRAM capacity.'}


def has_custom_components(cfg):
    return any(cfg.get(k) for k in ('vae_path','text_encoder_dir','processor_dir','scheduler_dir','text_encoder_path')) or cfg.get('quantized_loading')=='reference_dequantize' or cfg.get('text_encoder_dtype','auto')!='auto' or cfg.get('scheduler_shift') is not None


def build_custom_pipeline(cfg):
    """Assemble explicitly selected components, retaining the original memory modes."""
    import torch
    from diffusers import QwenImage21Pipeline,QwenImage21Transformer2DModel,FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3VLForConditionalGeneration,Qwen3VLProcessor
    root=Path(cfg['model_dir']);dt=dtype_value(cfg.get('dtype'));kwargs={}
    if cfg.get('vae_path'):
        vae,info=load_vae(cfg,dtype=dt)
        # Test on CPU float32 to keep setup independent of available VRAM.
        # Cast back before the pipeline applies its configured placement.
        vae.to(device='cpu',dtype=torch.float32);miniature_vae_test(vae);vae.to(dtype=dt)
        kwargs['vae']=vae
    if cfg.get('text_encoder_path'):
        from services.qwen_single_file import load_single_encoder
        kwargs['text_encoder']=load_single_encoder(cfg,dtype_value(cfg.get('text_encoder_dtype'),dt))
    elif cfg.get('text_encoder_dir') or cfg.get('text_encoder_dtype','auto')!='auto':
        path=component_location(cfg,'text_encoder')
        reference=read_object(root/'text_encoder/config.json') if (root/'text_encoder/config.json').is_file() else None
        inspect_folder(path,'text_encoder',reference,require_weights=not bool(cfg.get('text_encoder_path')))
        kwargs['text_encoder']=Qwen3VLForConditionalGeneration.from_pretrained(str(path),
            torch_dtype=dtype_value(cfg.get('text_encoder_dtype'),dt),local_files_only=True,
            trust_remote_code=False,use_safetensors=True)
    if cfg.get('processor_dir'):
        path=component_location(cfg,'processor');inspect_folder(path,'processor')
        kwargs['processor']=Qwen3VLProcessor.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
    if cfg.get('scheduler_dir') or cfg.get('scheduler_shift') is not None:
        path=component_location(cfg,'scheduler');info=inspect_folder(path,'scheduler');sc=dict(info['config'])
        if cfg.get('scheduler_shift') is not None:
            # Explicit static shift disables dynamic shift so it has a defined
            # effect; the UI labels this override as an advanced sampler change.
            sc.update(shift=cfg['scheduler_shift'],use_dynamic_shifting=False)
        kwargs['scheduler']=FlowMatchEulerDiscreteScheduler.from_config(sc)
    override=str(cfg.get('model_path') or '')
    if override:
        path=Path(override);kw={'config':str(root),'subfolder':'transformer','torch_dtype':dt,'local_files_only':True}
        if path.suffix.lower()=='.gguf':
            raise ValueError('Qwen Image 2.1 GGUF overrides must use Native ComfyUI GGUF execution. These ComfyUI-GGUF files are not loaded through Diffusers QwenImage21Transformer2DModel.')
        from core.qwen_weight_format import require_supported_weights
        quant=path.suffix.lower()=='.safetensors' and require_supported_weights(path,cfg.get('quantized_loading','reject'))['quantized']
        if quant:
            from services.qwen_single_file import load_reference_transformer
            kwargs['transformer']=load_reference_transformer(cfg,dt)
        else:
            # The pinned Qwen 2.1 Diffusers revision exposes the model class but
            # predates its upstream single-file registry entry. Backport only
            # that reviewed key mapping; do not rebuild the multi-GB runtime.
            from services.qwen_diffusers_compat import install_qwen21_single_file_compat
            install_qwen21_single_file_compat()
            kwargs['transformer']=QwenImage21Transformer2DModel.from_single_file(str(path),**kw)
    mode=cfg.get('memory_mode','balanced');device=cfg.get('device','cuda')
    base={'local_files_only':True,'low_cpu_mem_usage':True,'torch_dtype':dt,'use_safetensors':True,**kwargs}
    # Accelerate device_map does not redistribute already-instantiated modules
    # passed through kwargs.  With an explicit single-file encoder/VAE that left
    # weights on CPU while pipeline inputs were created on CUDA.  Use the normal
    # pipeline + model_cpu_offload path whenever any custom module is injected.
    if mode=='balanced' and device=='cuda' and not override and not kwargs:
        base.update(device_map='balanced',max_memory={0:cfg.get('gpu_memory') or '21GiB','cpu':cfg.get('cpu_memory') or '44GiB'})
        pipe=QwenImage21Pipeline.from_pretrained(str(root),**base)
    else:
        pipe=QwenImage21Pipeline.from_pretrained(str(root),**base)
        if mode in ('balanced','cpu_offload','model_cpu_offload') and device=='cuda':pipe.enable_model_cpu_offload()
        else:pipe.to(device)
    return pipe


def _has_meta_tensors(module):
    """Check parameters and buffers without materialising or moving model weights."""
    if module is None:return False
    return any(getattr(t,'is_meta',False) for t in module.parameters(recurse=True)) or any(
        getattr(t,'is_meta',False) for t in module.buffers(recurse=True))


def detach_for_decode(pipe,device='cpu'):
    """Release sampling pipeline safely, retaining only a usable VAE for decode.

    A balanced Accelerate device-map may leave *meta* placeholders in large
    transformer/encoder blocks. reset_device_map() calls .to(cpu) on those and
    raises NotImplementedError. Those blocks are discarded, not reconstructed.
    Return True only if the VAE itself must be reloaded from its verified source.
    """
    import torch
    if torch.cuda.is_available():torch.cuda.synchronize()
    vae=getattr(pipe,'vae',None)
    if vae is None:return False
    # Do not reset the whole pipeline's device map and do not move the
    # discarded transformer/text encoder. Their hooks and meta placeholders
    # will be reclaimed when the worker releases the pipeline.
    has_hooks=any(hasattr(part,'_hf_hook') for part in vae.modules())
    try:
        if has_hooks:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(vae,recurse=True)
    except (ImportError,ValueError,RuntimeError,NotImplementedError):
        # Some Accelerate revisions cannot restore a partially dispatched
        # VAE after hook removal. Reload only this verified VAE if necessary.
        return True
    if _has_meta_tensors(vae):return True
    vae.to(device)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return False


def configure_decode(vae,options):
    mode=options['vae_decode_mode']
    if mode in ('auto','full_frame'):
        if not callable(getattr(vae,'disable_tiling',None)):raise RuntimeError('This VAE cannot confirm full-frame decoding; no tiled fallback was used.')
        vae.disable_tiling()
        if callable(getattr(vae,'disable_slicing',None)):vae.disable_slicing()
        return 'full_frame'
    if not callable(getattr(vae,'enable_tiling',None)):raise ValueError('VAE tiling is not supported by this runtime.')
    if mode=='tiled_custom':
        args={f'tile_sample_{"min" if k in ("height","width") else "stride"}_{k.split("_")[-1]}':options['tile_'+k] for k in ('height','width','stride_height','stride_width')}
        params=inspect.signature(vae.enable_tiling).parameters
        if not set(args).issubset(params):raise ValueError('The installed Qwen VAE does not expose custom tiling dimensions/stride. No settings were ignored.')
        vae.enable_tiling(**args)
    else:vae.enable_tiling()
    return mode


def decode_tensor(vae,tensor,cfg,options,*,image_processor=None,cancel=None):
    import torch
    from diffusers.image_processor import VaeImageProcessor
    device=cfg.get('device','cuda') if options['decode_device']=='configured' else options['decode_device']
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA decode requested but unavailable. Choose CPU explicitly; no hidden CPU fallback.')
    chosen=options['vae_dtype']
    if device=='cpu':dt=torch.float32  # explicit CPU path prioritises compatibility
    else:dt=dtype_value(chosen, getattr(vae,'dtype',tensor.dtype)) if chosen!='auto' else getattr(vae,'dtype',tensor.dtype)
    mode=configure_decode(vae,options)
    vae.to(device=device,dtype=dt)
    if cancel:cancel()
    started=time.monotonic()
    with torch.inference_mode():
        z=tensor.to(device=device,dtype=dt)
        decoded=vae.decode(z,return_dict=False)[0]
        if decoded.ndim!=5 or decoded.shape[2]!=1 or decoded.shape[1] not in (3,4):raise ValueError('Unexpected Qwen VAE output dimensions.')
        # Chunk finiteness check prevents a second float32 image copy on CUDA.
        for chunk in decoded.split(1,dim=1):
            if not bool(torch.isfinite(chunk).all()):raise RuntimeError('VAE decode contains NaN/Inf. No broken PNG was saved.')
        if cancel:cancel()
        processor=image_processor or VaeImageProcessor(vae_scale_factor=16,vae_latent_channels=64)
        img=processor.postprocess(decoded[:,:,0].cpu().float(),output_type='pil')[0]
    return img,{'decode_device':device,'vae_dtype':str(dt),'vae_decode_mode':mode,
        'decode_seconds':time.monotonic()-started,'automatic_tiled_fallback':False,
        'vae_path':str(component_location(cfg,'vae')),
        'custom_tiles':{k:v for k,v in options.items() if k.startswith('tile_')} if mode=='tiled_custom' else None}
