"""Qwen Image 2.1 LoRA setup with a safe, observable placement boundary.

1.0.59: install/verify PEFT layers while the transformer is *real on CPU*,
before any Accelerate offload hook or balanced device map is installed.
Non-LoRA jobs never use this module. No adapter fusion, model surgery, weight
repair or runtime dependency upgrade is performed.
"""
from __future__ import annotations

import math
from pathlib import Path


def inspect_lora_header(path):
    """Read only safetensors header; do not copy the multi-GB base model."""
    from safetensors import safe_open
    from core.qwen_assets import inspect_component
    path=Path(path).expanduser()
    inspect_component(path,'lora')
    with safe_open(str(path),framework='pt',device='cpu') as src:
        keys=list(src.keys())
        metadata=src.metadata() or {}
    if not keys:
        raise ValueError(f'Empty LoRA file: {path.name}')
    lower=[key.lower() for key in keys]
    has_a=any('lora_a' in key or 'lora_down' in key for key in lower)
    has_b=any('lora_b' in key or 'lora_up' in key for key in lower)
    if not (has_a and has_b):
        raise ValueError(f'No recognizable LoRA A/B weight pairs in {path.name}. The file cannot be used as a Qwen 2.1 adapter.')
    return {'name':path.name,'tensors':len(keys),
            'metadata_base':str(metadata.get('ss_base_model_version') or metadata.get('base_model') or
                                metadata.get('modelspec.architecture') or '')[:160]}


def audit_active_loras(pipe, names):
    """Ensure that *real, finite* adapters were attached to the transformer.

    A successful load_lora_weights() alone is not proof of an adapter working:
    PEFT low-memory initialization and Accelerate offload can leave meta
    placeholders. Only scan actual adapter tensors, not the base weights.
    """
    import torch
    transformer=getattr(pipe,'transformer',None)
    if transformer is None:raise RuntimeError('LoRA cannot be verified: Qwen transformer is missing.')
    counts={name:0 for name in names}
    failures=[]
    for module in transformer.modules():
        a=getattr(module,'lora_A',None)
        b=getattr(module,'lora_B',None)
        if a is None and b is None:continue
        for name in names:
            if a is None or b is None or name not in a or name not in b:continue
            left=a[name].weight;right=b[name].weight
            if left.is_meta or right.is_meta:
                failures.append(f'{name}: adapter weights on meta device')
                continue
            if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
                failures.append(f'{name}: adapter weights contain NaN/Inf')
                continue
            counts[name]+=1
    if failures:
        raise RuntimeError('LoRA validation failed: '+', '.join(failures[:6]))
    missing=[name for name in names if not counts[name]]
    if missing:
        raise RuntimeError('LoRA loader returned without installing usable linear layers in Qwen 2.1 transformer: '
                           +', '.join(missing)+'. Check model family, tensor key mapping and Diffusers/PEFT compatibility.')
    getter=getattr(pipe,'get_active_adapters',None)
    if callable(getter):
        active=set(getter() or [])
        if not set(names).issubset(active):
            raise RuntimeError(f'PEFT LoRA activation mismatch: expected {names}, active={sorted(active)}')
    return {'adapter_layers':counts,'verified_before_offload':True}


def prepare_lora_pipeline(worker,cfg,loras):
    """Install LoRA on undispatched CPU weights and only then place/offload.

    Important: ``device_map='balanced'`` in Diffusers dispatches whole pipeline
    components before the current image_worker used load_lora_weights(). This
    was the only shared failure path across all of the user's unrelated LoRAs.
    For these jobs use Accelerate model CPU offload *after* adapter injection.
    Base generations retain the exact pre-existing balanced behavior.
    """
    from services.qwen_components_runtime import has_custom_components,build_custom_pipeline
    worker.clear()
    cpu_cfg={**cfg,'device':'cpu','memory_mode':'full_gpu','cpu_offload':False}
    headers=[]
    for entry in loras:
        strength=float(entry.get('strength',1.0))
        if not math.isfinite(strength):raise ValueError('LoRA strength must be finite.')
        headers.append(inspect_lora_header(entry['path']))
    pipe=None
    try:
        if has_custom_components(cpu_cfg):
            pipe=build_custom_pipeline(cpu_cfg)
        else:
            pipe=worker.load_qwen21(cpu_cfg)
        # A fresh, undispatched CPU pipeline must have materialised weights.
        # Reject an incomplete CPU load rather than sampling with placeholders.
        from services.qwen_components_runtime import _has_meta_tensors
        for component in ('transformer','text_encoder'):
            if _has_meta_tensors(getattr(pipe,component,None)):
                raise RuntimeError(f'Qwen {component} still contains meta tensors before LoRA setup; '
                                   'the CPU-first model load is incomplete. No generation started.')
        # No Accelerate hooks, no device map, no meta-only weights at injection.
        names=worker.apply_optional_loras(pipe,loras,verify=True)
        audit=audit_active_loras(pipe,names)
        target=str(cfg.get('device','cuda'))
        mode=str(cfg.get('memory_mode') or ('balanced' if cfg.get('cpu_offload') else 'full_gpu'))
        if target=='cuda' and mode in ('balanced','cpu_offload','model_cpu_offload'):
            # Less aggressive VRAM residency, but equivalent sampling precision.
            pipe.enable_model_cpu_offload()
            placement='model_cpu_offload_after_lora'
        elif target=='cuda':
            pipe.to('cuda')
            placement='full_gpu_after_lora'
        else:
            pipe.to(target)
            placement=target+'_after_lora'
        return pipe,{'header':headers,**audit,'placement':placement}
    except BaseException:
        pipe=None
        worker.clear()
        raise
