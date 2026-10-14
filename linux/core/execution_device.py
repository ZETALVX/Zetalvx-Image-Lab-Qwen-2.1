"""Explicit device normalization; CPU offload is not CPU-only execution."""
def normalized_config(cfg, cuda_available):
    cfg=dict(cfg)
    device=str(cfg.get('device','cuda')).lower()
    if device=='cpu':
        cfg.update(device='cpu',dtype='float32',memory_mode='full_gpu',cpu_offload=False)
    elif device=='cuda' or device.startswith('cuda:'):
        if not cuda_available:
            raise RuntimeError('CUDA is not available in this runtime. Select CPU (float32, device memory), or install a CUDA runtime and a compatible NVIDIA driver.')
    else:raise RuntimeError('Unsupported device: '+device)
    return cfg
