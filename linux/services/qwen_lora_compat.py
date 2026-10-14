"""Compatibility bridge for selected Qwen-Image 2.1 LoRA safetensors layouts.

No changes to Diffusers/PEFT installations or to checkpoint files. Only the
CPU-first, per-job Qwen pipeline uses this bridge. Unknown non-null adapter
options and unknown additive-weight targets are rejected, not discarded.
"""
from __future__ import annotations

import inspect
import json
import math
import re
from pathlib import Path

_DIFF_BLOCK = re.compile(r"^(?:diffusion_model\.)?transformer_blocks\.(\d+)\.attn\.norm_([qk])\.diff$")
_DIFF_TEXT = {"diffusion_model.txt_in.text_norm.diff", "txt_in.text_norm.diff"}
_TEXT_TARGET_CANDIDATES = ("txt_in.text_norm.weight", "txt_norm.weight")


def _candidate_targets(key: str) -> tuple[str, ...]:
    """Resolve supported additive tensors to one or more transformer targets."""
    match = _DIFF_BLOCK.fullmatch(key)
    if match:
        return (f"transformer_blocks.{int(match.group(1))}.attn.norm_{match.group(2)}.weight",)
    if key in _DIFF_TEXT:
        return _TEXT_TARGET_CANDIDATES
    raise ValueError(f"Unsupported Qwen 2.1 additive parameter: {key}")


def _diff_target(key: str) -> str:
    """Backwards-compatible primary target name used by tests and diagnostics."""
    return _candidate_targets(key)[0]


def _safe_peft_metadata(metadata):
    """Remove only unknown *null* PEFT options; preserve all real settings."""
    if metadata is None:
        return None, ()
    if not isinstance(metadata, dict):
        raise ValueError("Invalid LoRA adapter metadata: expected an object")
    from peft import LoraConfig
    supported = set(inspect.signature(LoraConfig).parameters)
    sanitized = {}
    removed = []
    for fullkey, value in metadata.items():
        if not isinstance(fullkey, str):
            raise ValueError("Invalid LoRA metadata key")
        if fullkey.startswith("transformer."):
            key = fullkey.removeprefix("transformer.")
            if key not in supported:
                if value is not None:
                    raise ValueError(f"Unsupported PEFT adapter option has a non-null value: {fullkey}")
                removed.append(fullkey)
                continue
        sanitized[fullkey] = value
    return sanitized, tuple(removed)


def _read_compatible_header(path):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = handle.keys()
        additive = [key for key in keys if key.endswith(".diff")]
        metadata = handle.metadata() or {}
    raw = metadata.get("lora_adapter_metadata")
    if raw is not None:
        try:
            metadata = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid lora_adapter_metadata JSON") from exc
    else:
        metadata = None
    cleaned, removed = _safe_peft_metadata(metadata)
    # Fail before injecting any adapter if an additive key has no known target syntax.
    for key in additive:
        _candidate_targets(key)
    if len(additive) != len(set(additive)):
        raise ValueError("Duplicate additive keys")
    return additive, cleaned, removed


def _read_additive_values(path, keys, transformer):
    """Verify all additive deltas, *before* changing a single model tensor."""
    import torch
    from safetensors import safe_open
    params = dict(transformer.named_parameters())
    planned = []
    seen = set()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in keys:
            candidates = _candidate_targets(key)
            resolved = None
            for target in candidates:
                param = params.get(target)
                if param is not None:
                    resolved = (target, param)
                    break
            if resolved is None:
                raise ValueError(
                    "Qwen 2.1 transformer is missing additive target "
                    + " / ".join(candidates)
                )
            target, param = resolved
            if target in seen:
                raise ValueError(f"Duplicate additive target: {target}")
            seen.add(target)
            if param.is_meta:
                raise ValueError(f"Qwen 2.1 additive target is meta: {target}")
            delta = handle.get_tensor(key)
            if tuple(delta.shape) != tuple(param.shape):
                raise ValueError(f"Shape mismatch for {target}: {tuple(delta.shape)} != {tuple(param.shape)}")
            if not torch.is_floating_point(delta) or not bool(torch.isfinite(delta).all()):
                raise ValueError(f"Additive tensor contains invalid values: {key}")
            planned.append((target, param, delta))
    return planned


def load_compatible_lora(pipe, path, name: str, *, eager: bool):
    """Attach the normal LoRA layers, preserving Diffusers' built-in path.

    The official Comfy Turbo file adds normalization deltas not recognized by
    the Qwen converter. PEFT 0.18 cannot parse some null fields in adapters
    exported by PEFT 0.21. No files are written or rewritten on disk.

    Returns a *deferred* additive plan. Apply after set_adapters() succeeds so
    adapter strengths are known and rollback uses the fresh per-job pipeline.
    """
    from safetensors.torch import load_file
    path = Path(path).expanduser()
    additive, clean_metadata, removed = _read_compatible_header(path)
    deltas = _read_additive_values(path, additive, pipe.transformer) if additive else []
    if additive:
        state_dict = load_file(str(path), device="cpu")
        state_dict = {key: value for key, value in state_dict.items() if key not in additive}
        if not state_dict:
            raise ValueError("Adapter has no LoRA matrices after separating normalization deltas")
        if clean_metadata is not None and removed:
            # Unusual combined-format file: keep metadata and conversions.
            state_dict, _ = pipe.lora_state_dict(state_dict, return_lora_metadata=True)
            pipe.load_lora_into_transformer(
                state_dict, transformer=pipe.transformer, adapter_name=name,
                metadata=clean_metadata, _pipeline=pipe, low_cpu_mem_usage=False)
        else:
            kwargs = {"low_cpu_mem_usage": False} if eager else {}
            pipe.load_lora_weights(state_dict, adapter_name=name, **kwargs)
    elif removed:
        # Diffusers' normal path tries constructing LoraConfig with unsupported
        # PEFT 0.21 null fields. Reuse its decoder and injection APIs while
        # handing the transformer only the compatible metadata dictionary.
        state_dict, original_metadata = pipe.lora_state_dict(str(path), return_lora_metadata=True, local_files_only=True)
        if original_metadata is None:
            raise ValueError("LoRA metadata vanished while reading the file")
        clean_metadata, _ = _safe_peft_metadata(original_metadata)
        pipe.load_lora_into_transformer(
            state_dict, transformer=pipe.transformer, adapter_name=name,
            metadata=clean_metadata, _pipeline=pipe, low_cpu_mem_usage=False)
    else:
        kwargs = {"low_cpu_mem_usage": False} if eager else {}
        pipe.load_lora_weights(str(path), adapter_name=name, local_files_only=True, **kwargs)
    return deltas


def apply_additive_plans(plans):
    """Apply previously checked (target, parameter, delta, strength) tuples.

    All-or-nothing for the local pipeline: if an add fails, restore the exact
    previously cloned CPU parameters. The enclosing worker also drops the
    entire new pipeline on any error, including cancellation.
    """
    import torch
    flattened = []
    for items, strength in plans:
        strength = float(strength)
        if not math.isfinite(strength):
            raise ValueError("Invalid additive LoRA strength")
        flattened.extend((key, param, delta, strength) for key, param, delta in items)
    if not flattened:
        return 0
    originals = {}
    try:
        with torch.no_grad():
            for key, param, delta, strength in flattened:
                parameter_id = id(param)
                if parameter_id not in originals:
                    originals[parameter_id] = (param, param.detach().clone())
                param.add_(delta.to(device=param.device, dtype=param.dtype), alpha=strength)
    except BaseException:
        with torch.no_grad():
            for param, backup in originals.values():
                param.copy_(backup)
        raise
    return len(flattened)
