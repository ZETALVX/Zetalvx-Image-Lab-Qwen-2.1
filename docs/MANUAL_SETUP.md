# Manual developer setup (no installer)

This source edition contains neither a bootstrap installer nor an automatic updater. Users are responsible for their interpreter, venv, compatible CUDA/PyTorch environment, model files and backups.

## 1. Choose a platform

Use `linux/` **or** `windows/`, never a mixture of both. Python 3.11 is recommended. Ensure a compatible NVIDIA driver and enough disk/VRAM for the chosen models. A CUDA Toolkit on your PATH is not automatically required by pip wheels, but your driver must support the selected CUDA build.

## 2. Create your own venv and install dependencies

From your chosen platform directory:

**Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0 torchvision==0.29.0
python -m pip install -r requirements-app.txt -r requirements-qwen21.txt
python -m pip install --no-deps --no-build-isolation https://github.com/huggingface/diffusers/archive/80c7ed262aeffbeb43ef13ae04baeb9b84515a69.tar.gz
python -m pip check
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0 torchvision==0.29.0
python -m pip install -r requirements-app.txt -r requirements-qwen21.txt
python -m pip install --no-deps --no-build-isolation https://github.com/huggingface/diffusers/archive/80c7ed262aeffbeb43ef13ae04baeb9b84515a69.tar.gz
python -m pip check
```

These pins/revision match the original separate Qwen runtime's reviewed versions, but are **not a compatibility guarantee**. Inspect the project requirements and model-library compatibility for your machine. The source worker checks for a reviewed final `torch >= 2.14.0` before starting. The PyTorch `cu126` index is an example supported by the original installer; alternative platforms require manually selected compatible wheels. Do not run `pip install` in system Python.

## 3. Confirm your environment

```bash
python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available())"
python -c "from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel; print('Diffusers OK')"
python -m unittest discover -s tests -v
```

If CUDA is False, stop here and repair the driver/wheel/environment. A model download does not repair CUDA.

## 4. Run web + worker manually

Start from your chosen platform folder in **two separate terminals**, both with the same venv activated:

```bash
python source_run.py image
# In the other terminal:
python source_run.py web
```

The source entry point creates only necessary **runtime data**, such as a model-registry default, database schema, internal worker token, a per-user self-signed HTTPS certificate and first-account setup data. It **does not** create or install Python environments, download weights, install shell commands, configure a system service or upgrade your application.

Open `https://127.0.0.1:8298`. The first-run account code is displayed in the **web terminal**. Keep the worker on `127.0.0.1:8299` only. Stop by pressing `Ctrl+C` in both terminals.

## 5. Configure your models

Link an existing, complete Qwen Image 2.1 repository via **Models → My models**, or use the model manager to acquire permitted components. Models, VAE, tokenizer/text-encoder configurations, optional LoRA and GGUF components must form a supported combination. Models are **not** present in this repository. Users download them independently and accept their licenses.

An optional external ComfyUI runtime must be separately installed, trusted and configured through its own settings. Image Lab does not maintain or update ComfyUI.

## 6. Isolated storage

By default, source data is stored separately from any Patreon-installed edition:

- Windows: `%LOCALAPPDATA%\ImageLabQwen21Source`
- Linux: `~/.local/share/ImageLabQwen21Source`

You can override it with `QWEN_SOURCE_HOME` pointing to an absolute host directory **before** starting both terminals. Example:

```powershell
$env:QWEN_SOURCE_HOME = "D:\ImageLabQwen21SourceData"
```

```bash
export QWEN_SOURCE_HOME="$HOME/ImageLabQwen21SourceData"
```

Use the same data root for web and worker. Back it up before changing code. Do not commit this directory, model weights, certificates, secrets, or personal projects.

## 7. LAN (advanced)

The settings page can choose Local (`127.0.0.1`) or LAN (`0.0.0.0`) for the **web** process. Restart the web terminal to apply the change. Only connect on a trusted private LAN, review firewall/VPN settings and consider the self-signed certificate warning. The image worker must remain loopback-only. **Never expose these services directly to the Internet.**

## 8. Manual updates and removal

1. Stop both processes.
2. Back up `QWEN_SOURCE_HOME` (or the default source data directory).
3. Review any upstream changes, then use `git pull` for code changes or replace files manually.
4. Revisit requirements and rebuild/upgrade your own venv as needed; do not assume a new source checkout is automatically compatible with an old environment.
5. Start both processes and check the model/runtime status.

To remove the manual source edition, stop its processes and delete your checkout and the user-created `.venv`. Delete your **separate source data directory only if you explicitly want to discard your projects, model files, credentials and generated images**. No uninstall automation is supplied.

## 9. Local account password recovery

From the **same** platform venv (on the host), run this Python one-liner; it prompts interactively without putting your new password in shell history:

```bash
python -c "import getpass; from core.runtime_env import SHARED_ROOT,DATA_ROOT; from core.local_password_reset import reset_password; p=getpass.getpass('New password: '); q=getpass.getpass('Confirm password: '); print('Updated:',reset_password(SHARED_ROOT/'config/creator_auth.json',DATA_ROOT/'secrets',p,q))"
```

If no account exists, use the first-run setup page and code printed by `python source_run.py web` instead.

## Limitations

This source variant is intentionally independent of the managed Patreon installer. In-app bundled multilingual help may describe concepts shared by both editions; **for source setup and updates, this document takes precedence**. Runtime dependency installation, Windows UAC/firewall actions, CUDA inference and platform-specific behaviour must be validated on the user's own machine.
