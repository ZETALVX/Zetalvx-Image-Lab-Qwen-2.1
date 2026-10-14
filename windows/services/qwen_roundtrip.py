"""VAE-only image reconstruction. Explicit resize + RGBA8 between cycles; no diffusion."""
from __future__ import annotations
import gc,time,math
from pathlib import Path
from PIL import Image,ImageOps
from services.qwen_components_runtime import load_vae,decode_tensor,configure_decode
from core.qwen_components import decode_options


def image_metrics(original,result):
    import numpy as np
    a=np.asarray(original.convert('RGBA'),dtype=np.float32)/255.;b=np.asarray(result.convert('RGBA'),dtype=np.float32)/255.
    if a.shape!=b.shape:raise ValueError('Reconstruction dimensions differ.')
    out={}
    for label,slice_ in [('rgb',slice(0,3)),('alpha',slice(3,4)),('rgba',slice(0,4))]:
        mse=float(np.mean((a[...,slice_]-b[...,slice_])**2));out[label+'_mse']=mse
        out[label+'_psnr_db']=(-10*math.log10(mse)) if mse>0 else None
        out[label+'_identical']=mse==0
    return out


def run(worker,job,cfg):
    import numpy as np
    import torch
    cycles=int(job.get('cycles',1));w=int(job.get('width',1024));h=int(job.get('height',1024))
    if cycles not in (1,5,10) or w%16 or h%16 or not (64<=w<=4096 and 64<=h<=4096):raise ValueError('Invalid VAE round-trip dimensions/cycles.')
    options=decode_options(cfg,job);device=cfg.get('device','cuda') if options['decode_device']=='configured' else options['decode_device']
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; select CPU explicitly.')
    from services.qwen_components_runtime import dtype_value
    dtype=torch.float32 if device=='cpu' else dtype_value(options['vae_dtype'],dtype_value(cfg.get('dtype')))
    vae=None;started=time.perf_counter();stages=[]
    try:
        worker.clear();worker._check_cancel(job)
        with Image.open(job['source_path']) as im:original=ImageOps.exif_transpose(im).convert('RGBA');original_size=original.size;original=original.resize((w,h),Image.Resampling.LANCZOS)
        current=original.copy();vae,info=load_vae(cfg,dtype=dtype);vae.to(device=device,dtype=dtype)
        configure_decode(vae,options);load_seconds=time.perf_counter()-started
        for cycle in range(1,cycles+1):
            worker._check_cancel(job);worker._write_job_progress(job,'vae_roundtrip',cycle-1,cycles,(cycle-1)/cycles)
            x=torch.from_numpy(np.asarray(current).copy()).permute(2,0,1).unsqueeze(0).unsqueeze(2).to(device=device,dtype=dtype)/127.5-1
            tick=time.perf_counter()
            with torch.inference_mode():
                posterior=vae.encode(x).latent_dist;z=posterior.mode();del posterior,x
                if not bool(torch.isfinite(z).all()):raise RuntimeError('Non-finite VAE encoding.')
                current,di=decode_tensor(vae,z,cfg,options,cancel=lambda:worker._check_cancel(job));del z
            stages.append({'cycle':cycle,'seconds':time.perf_counter()-tick,**image_metrics(original,current)})
        worker._check_cancel(job);out=Path(job['output_path']);out.parent.mkdir(parents=True,exist_ok=True);current.save(out,format='PNG')
        return {'output_path':str(out),'width':w,'height':h,'pipeline':'QwenImage21 VAE round-trip',
            'steps':0,'cycles':cycles,'input_original_size':list(original_size),'resize_to':[w,h],
            'posterior':'mode','between_cycles':'RGBA PNG-equivalent uint8 quantization','sampling_repeated':False,
            'vae_path':info['path'],'load_seconds':load_seconds,'roundtrip':stages,**di}
    finally:
        vae=None;worker.clear();gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
