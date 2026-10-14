"""Compatibility shim for the pinned Diffusers Qwen Image 2.1 source build.

The application pins the first upstream Qwen 2.1-capable Diffusers revision for
runtime stability. That revision exposes QwenImage21Transformer2DModel but its
single-file registry predates the Qwen 2.1 entry. Newer upstream Diffusers adds
exactly this registry entry plus a small ComfyUI-key conversion. Backport only
that data-only mapping at runtime so BF16 safetensors and GGUF can use the
existing, reviewed FromSingleFileMixin without replacing the AI runtime.
"""
from __future__ import annotations


def convert_qwen_image21_transformer_checkpoint_to_diffusers(checkpoint, **kwargs):
    """Backport of the upstream Qwen Image 2.1 single-file key conversion.

    The conversion removes ComfyUI's optional model.diffusion_model prefix and
    splits the fused SwiGLU gate/up matrix used by ComfyUI checkpoints.
    """
    converted={}
    for key in list(checkpoint.keys()):
        new_key=key.replace('model.diffusion_model.','')
        value=checkpoint.pop(key)
        if new_key.endswith('img_mlp.gate_up.weight'):
            gate,up=value.chunk(2,dim=0)
            converted[new_key.replace('gate_up','gate_layer')]=gate
            converted[new_key.replace('gate_up','proj')]=up
        else:
            converted[new_key]=value
    return converted


def install_qwen21_single_file_compat(module=None):
    """Register QwenImage21Transformer2DModel in pinned Diffusers if missing.

    Returns True when the registry is usable. Fails closed if the pinned loader
    layout is no longer the reviewed one instead of mutating an unknown object.
    """
    if module is None:
        import diffusers.loaders.single_file_model as module
    mapping=None
    for name in ('SINGLE_FILE_LOADABLE_CLASSES','SINGLE_FILE_LOADABLE_MODEL_CLASSES'):
        candidate=getattr(module,name,None)
        if isinstance(candidate,dict):
            mapping=candidate;break
    if mapping is None:
        raise RuntimeError('Pinned Diffusers single-file registry was not found; Qwen 2.1 single-file compatibility cannot be installed safely.')
    current=mapping.get('QwenImage21Transformer2DModel')
    if current is None:
        mapping['QwenImage21Transformer2DModel']={
            'checkpoint_mapping_fn':convert_qwen_image21_transformer_checkpoint_to_diffusers,
            'default_subfolder':'transformer',
        }
    elif not isinstance(current,dict) or not callable(current.get('checkpoint_mapping_fn')):
        raise RuntimeError('Unexpected Qwen Image 2.1 single-file registry entry in Diffusers.')
    return True
