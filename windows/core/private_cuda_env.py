"""Private CUDA/cuDNN loader paths for the isolated Qwen 2.1 runtime.

No system PATH, LD_LIBRARY_PATH, driver or CUDA installation is modified.
The returned mapping is intended only for child processes launched by Zetalvx Image Lab.
"""
from __future__ import annotations
import os
from pathlib import Path


def _runtime_root(runtime_python: Path) -> Path:
    p=Path(runtime_python).expanduser().absolute()
    # Linux: <venv>/bin/python ; Windows: <venv>/Scripts/python.exe
    return p.parent.parent


def private_cuda_library_dirs(runtime_python: Path, *, platform_name: str | None = None) -> list[Path]:
    root=_runtime_root(runtime_python)
    is_windows=(os.name=='nt') if platform_name is None else (platform_name=='nt')
    if is_windows:
        sites=[root/'Lib'/'site-packages']
    else:
        sites=sorted(root.glob('lib/python*/site-packages'))
    found=[]
    def add(p:Path):
        if p.is_dir() and p not in found: found.append(p)
    for site in sites:
        nvidia=site/'nvidia'
        if nvidia.is_dir():
            # Prefer NVIDIA wheel directories first so ORT resolves cuDNN/CUDA from
            # the same private environment as PyTorch, never from the host system.
            for component in sorted(nvidia.iterdir()):
                if component.is_dir():
                    if is_windows:
                        add(component/'bin');add(component/'lib')
                    else:
                        add(component/'lib')
        add(site/'torch'/'lib')
    return found


def private_cuda_env(runtime_python: Path, base_env: dict | None = None, *, platform_name: str | None = None) -> dict:
    env=dict(os.environ if base_env is None else base_env)
    is_windows=(os.name=='nt') if platform_name is None else (platform_name=='nt')
    dirs=private_cuda_library_dirs(runtime_python,platform_name=platform_name)
    key='PATH' if is_windows else 'LD_LIBRARY_PATH'
    sep=';' if is_windows else ':'
    current=[p for p in env.get(key,'').split(sep) if p]
    prefix=[str(p) for p in dirs]
    seen=set();ordered=[]
    for item in prefix+current:
        norm=item.lower() if is_windows else item
        if norm in seen: continue
        seen.add(norm);ordered.append(item)
    if ordered: env[key]=sep.join(ordered)
    env['ZETALVX_PRIVATE_CUDA_LIBRARY_PATHS']=sep.join(prefix)
    return env
