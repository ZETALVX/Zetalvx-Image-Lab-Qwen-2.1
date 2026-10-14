"""Opt-in single-file components and quantization reconstruction reference path.

No ComfyUI code is imported/bundled. The ConvRot storage contract (int8 weights,
row scales and regular Hadamard basis) is documented by Comfy Org comfy-kitchen.
This deliberately reconstructs floating weights, NOT INT8 activation/GEMM kernels.
All key/shape mappings must be complete; mismatches fail, never random-init.
"""
from __future__ import annotations
import json
from pathlib import Path
from core.qwen_weight_format import require_supported_weights


def inverse_regular_hadamard(x,group_size):
    """Apply the normalized symmetric H4 Kronecker basis on the last dimension.
    Uses four-way sum/difference butterflies, not a dense group matrix allocation.
    """
    import torch
    g=int(group_size);n=g
    while n>1 and n%4==0:n//=4
    if n!=1 or g<4 or g>4096 or x.shape[-1]%g:raise ValueError('Unsupported regular-Hadamard group size.')
    shape=x.shape;y=x.reshape(-1,g)
    stride=1
    while stride<g:
        z=y.reshape(-1,g//(4*stride),4,stride)
        total=z.sum(dim=2,keepdim=True)
        # H4 has its negative entries on the anti-diagonal.
        y=((total-2*z.flip(2))*.5).reshape(-1,g)
        stride*=4
    return y.reshape(shape)


def iter_float_weights(path,dtype,policy):
    import torch
    from safetensors import safe_open
    info=require_supported_weights(path,policy)
    aux={p+k for p in info['layers'] for k in ('comfy_quant','weight_scale')}
    with safe_open(str(path),framework='pt',device='cpu') as sf:
        for name in sf.keys():
            if name in aux:continue
            t=sf.get_tensor(name)
            conf=info['layers'].get(name[:-len('weight')]) if name.endswith('weight') else None
            if conf:
                if t.dtype!=torch.int8 or t.ndim!=2:raise ValueError('Expected a 2D INT8 weight: '+name)
                prefix=name[:-len('weight')];scale=sf.get_tensor(prefix+'weight_scale').float()
                if scale.numel() not in (1,t.shape[0]) or not bool(torch.isfinite(scale).all()) or bool((scale<0).any()):raise ValueError('Invalid INT8 scales: '+name)
                scale=scale.reshape(-1,1);params=conf.get('params') or {}
                use_rotation=conf.get('convrot',params.get('convrot',False))
                group=conf.get('convrot_groupsize',params.get('convrot_groupsize',256))
                # Bounded transient float32 blocks for large embedding tables.
                out=torch.empty(t.shape,dtype=dtype)
                for start in range(0,t.shape[0],256):
                    stop=min(start+256,t.shape[0]);s=scale if scale.numel()==1 else scale[start:stop]
                    block=t[start:stop].float()*s
                    if use_rotation:block=inverse_regular_hadamard(block,group)
                    out[start:stop]=block.to(dtype)
                t=out
            elif t.dtype in (torch.int8,torch.uint8) or 'float8' in str(t.dtype):
                raise ValueError('Undescribed quantized tensor refused: '+name)
            elif t.is_floating_point():t=t.to(dtype)
            yield name,t


def strict_state_keys(state,expected,kind):
    """Only known wrapper prefixes and Qwen3VL language/vision namespace variants."""
    result={}
    for key,value in state:
        names=[key]
        for prefix in ('model.diffusion_model.','diffusion_model.','transformer.','text_encoder.'):
            if key.startswith(prefix):names.append(key[len(prefix):])
        if kind=='text_encoder':
            for k in list(names):
                if k.startswith('model.') and not k.startswith(('model.language_model.','model.visual.')):
                    names.append('model.language_model.'+k[len('model.'):])
                if k.startswith('visual.'):names.append('model.'+k)
                if k.startswith('language_model.'):names.append('model.'+k)
        matches=[n for n in dict.fromkeys(names) if n in expected]
        if len(matches)!=1:raise ValueError('Unmapped or ambiguous '+kind+' tensor: '+key)
        dest=matches[0]
        if dest in result:raise ValueError('Duplicate tensor mapping: '+dest)
        if tuple(value.shape)!=tuple(expected[dest].shape):raise ValueError('Tensor shape mismatch: '+dest+'; selected file '+str(tuple(value.shape))+' vs Qwen 2.1 baseline interface '+str(tuple(expected[dest].shape)))
        result[dest]=value
    missing=set(expected)-set(result)
    if missing:raise ValueError('Missing '+kind+' tensors: '+', '.join(sorted(missing)[:8]))
    return result


def load_single_encoder(cfg,dtype):
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from accelerate import init_empty_weights
    from core.qwen_components import component_location,inspect_folder
    root=component_location(cfg,'text_encoder')
    inspect_folder(root,'text_encoder',require_weights=False)
    config=Qwen3VLConfig.from_pretrained(str(root),local_files_only=True)
    with init_empty_weights():model=Qwen3VLForConditionalGeneration(config)
    model.tie_weights()
    state=strict_state_keys(iter_float_weights(cfg['text_encoder_path'],dtype,cfg.get('quantized_loading','reject')),model.state_dict(),'text_encoder')
    model.load_state_dict(state,strict=True,assign=True);del state
    model.eval();model.requires_grad_(False)
    return model


def load_reference_transformer(cfg,dtype):
    from diffusers import QwenImage21Transformer2DModel
    from accelerate import init_empty_weights
    root=Path(cfg['model_dir'])/'transformer'
    config=QwenImage21Transformer2DModel.load_config(str(root),local_files_only=True)
    with init_empty_weights():model=QwenImage21Transformer2DModel.from_config(config)
    state=strict_state_keys(iter_float_weights(cfg['model_path'],dtype,'reference_dequantize'),model.state_dict(),'transformer')
    model.load_state_dict(state,strict=True,assign=True);del state
    model.eval();model.requires_grad_(False)
    return model
