# Source distribution policy

This public GitHub source tree is maintained separately from the installer distributions:

| Package | Contents | Environment management | App updates / uninstall |
|---|---|---|---|
| GitHub source (this repository) | Windows/Linux Python source, web assets, requirement pins, tests and documentation | User creates and manages venv manually | Manual Git/file update; removal by user only |
| Patreon Windows 1.0.76 | Platform-specific installer and maintained runtime | Managed by Windows installer | Installer/updater/uninstaller included |
| Patreon Linux 1.0.75 | Platform-specific installer and maintained runtime | Managed by Linux installer | Installer/updater/uninstaller included |

The difference is packaging and lifecycle management. The core generation, benchmark, scene-library and image UI logic remains derived from the corresponding release source.

This source tree **does not ship** the installers or their upgrade/uninstall modules, UI endpoints, Windows command entry points, bash maintenance scripts, release manifests, version pointers, runtime bootstrapper, automatic app-update interface or model weights. Its small `source_run.py` entry point only starts the selected foreground component after initializing application data.

The source is not an auto-configuring binary distribution, and the PyTorch/CUDA/Diffusers pins do not guarantee every system can run them. Users should prepare a fresh venv and test their configurations. Model and dependency licensing is independent from the application Apache-2.0 license.
