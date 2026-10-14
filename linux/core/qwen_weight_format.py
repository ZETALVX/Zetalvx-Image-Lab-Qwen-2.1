"""Bounded data-only safetensors inspection. Never import model code on web host."""
from __future__ import annotations
import json,struct,hashlib,math
from pathlib import Path
from functools import lru_cache


def inspect_weights(path):
    p=Path(path);s=p.stat()
    return _inspect(str(p),s.st_size,s.st_mtime_ns)


@lru_cache(maxsize=64)
def _inspect(path,size,mtime):
    with open(path,'rb') as f:
        prefix=f.read(8)
        if len(prefix)!=8:raise ValueError('Incomplete safetensors file.')
        length=struct.unpack('<Q',prefix)[0]
        if not 2<=length<=32*1024**2 or length+8>size:raise ValueError('Invalid safetensors header length.')
        raw=f.read(length)
        def pairs(items):
            out={}
            for k,v in items:
                if k in out:raise ValueError('Duplicate safetensors metadata key.')
                out[k]=v
            return out
        header=json.loads(raw,object_pairs_hook=pairs)
        if not isinstance(header,dict):raise ValueError('Invalid safetensors header.')
        keys=[];formats={};dtypes={};intervals=[]
        sizes={'BOOL':1,'U8':1,'I8':1,'I16':2,'U16':2,'I32':4,'U32':4,'I64':8,'U64':8,'F16':2,'BF16':2,'F32':4,'F64':8,'F8_E4M3':1,'F8_E5M2':1,'F8_E4M3FN':1}
        if len(header)>100001:raise ValueError('Too many tensors.')
        for key,v in header.items():
            if key=='__metadata__':continue
            if not isinstance(v,dict) or not isinstance(v.get('shape'),list):raise ValueError('Invalid tensor entry.')
            offsets=v.get('data_offsets')
            if not isinstance(offsets,list) or len(offsets)!=2 or any(type(n)!=int for n in offsets) or not 0<=offsets[0]<=offsets[1]<=size-8-length:raise ValueError('Invalid tensor byte range.')
            shape=v['shape'];dt=v.get('dtype','unknown')
            if dt not in sizes or len(shape)>16 or any(type(n)!=int or n<0 or n>2**32 for n in shape):raise ValueError('Invalid tensor dtype/shape.')
            if math.prod(shape)*sizes[dt]!=offsets[1]-offsets[0]:raise ValueError('Tensor size does not match byte range.')
            intervals.append(tuple(offsets));keys.append(key);dtypes[dt]=dtypes.get(dt,0)+1
            if key.endswith('.comfy_quant'):
                if v.get('dtype')!='U8' or offsets[1]-offsets[0]>8192:raise ValueError('Invalid quantization descriptor.')
                f.seek(8+length+offsets[0]);conf=json.loads(f.read(offsets[1]-offsets[0]))
                if not isinstance(conf,dict):raise ValueError('Invalid quantization configuration.')
                formats[key[:-len('comfy_quant')]]=conf
        position=0
        for lo,hi in sorted(intervals):
            if lo!=position:raise ValueError('Overlapping or discontinuous tensor ranges.')
            position=hi
        if not keys or position!=size-length-8:raise ValueError('Incomplete or trailing tensor data.')
        quant=bool(formats) or any(k in ('I8','U8') or k.startswith('F8') for k in dtypes)
        return {'keys':keys,'quantized':quant,'layers':formats,'dtypes':dtypes,
            'header_sha256':hashlib.sha256(raw).hexdigest(),'bytes':size}


def require_supported_weights(path,policy='reject'):
    info=inspect_weights(path)
    if not info['quantized']:return info
    if policy=='native_comfyui':
        # Accepted for planning/validation only. The Diffusers loader must never
        # consume this path; image_worker dispatches the full job to ComfyUI.
        return info
    if policy!='reference_dequantize':
        raise ValueError('Quantized safetensors / ConvRot need Native ComfyUI execution or explicit Reference dequantization. Reference reconstruction is not native INT8 and cannot measure native INT8 speed/VRAM.')
    if not info['layers'] or any(v.get('format')!='int8_tensorwise' for v in info['layers'].values()):
        raise ValueError('Only explicitly described int8_tensorwise / ConvRot weights can be reconstructed. FP8/W4A8/unknown layouts are not guessed.')
    return info
