# Third-party software and model licences

**Zetalvx Image Lab — Qwen 2.1 Edition**: the application source in this archive is distributed under **Apache License 2.0**. Read `LICENSE` and `NOTICE`; the separate third-party software, services, downloaded weights and user datasets are **not** relicensed by this notice.

## Contents of the distribution

This ZIP contains the application source, UI assets, configuration, platform-specific installation tools and documentation. It **does not** bundle Qwen weights, LoRA/checkpoints, user data, private Python environments, pip wheels, ComfyUI, NVIDIA drivers/CUDA Toolkit, FFmpeg binaries or operating-system libraries. A user-initiated installer creates a private runtime and downloads dependencies for the chosen platform.

## Application and attribution

- Application source: **Apache-2.0**, see exact original `LICENSE` and copyright/attribution `NOTICE` (also displayed unchanged in the in-app Legal page).
- Inherited contributor notices relevant to the current code are retained in `NOTICE` and source-file headers. Such notices do not mean that other applications are bundled.
- Brand/trademark permissions and rights to third-party model weights are not granted by the application licence.

## Python dependencies installed separately

Direct dependencies are declared in `requirements-app.txt`, `requirements-common.txt` and `requirements-qwen21.txt`. They include Flask, Cheroot, cryptography, requests, Pillow, torch/torchvision, transformers, huggingface-hub, accelerate, peft, safetensors, tokenizers, gguf and their transitive dependencies. The installer also obtains a pinned external Diffusers source revision and creates private Python/uv tooling.

Each downloaded wheel and its transitive/native libraries retain their **own** copyright and licence terms, which may differ by revision, platform and build. The requirements manifest is **not** a resolved Software Bill of Materials and the ZIP does **not** assert that every future downloaded dependency is Apache-2.0. Consult the actual installed wheel metadata (`*.dist-info/METADATA`, `LICENSE*`, `NOTICE*`) and the packages' upstream licences before **redistributing an installed Python runtime**. The application ZIP itself does not redistribute those third-party binaries.

## Model weights, components and services

- **Qwen Image 2.1:** model weights are external and subject to the **exact revision's published model licence**. The official Qwen-Image-2.1 model card lists `qwen-research` (Qwen Research License Agreement), distinct from the application source licence; separate permission can be required for commercial model use. See <https://huggingface.co/Qwen/Qwen-Image-2.1> and the model's `LICENSE` at the selected revision.
- Optional **Texture-Fix VAE** and other model derivatives have independent terms; see <https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1>. Do not assume that replacing a VAE changes the original model's licence.
- LoRAs, VAEs, text encoders, GGUF conversions, weights linked from local folders and downloads from Hugging Face, Civitai or direct URLs: verify the author/provider's applicable revision, usage and redistribution permissions. Compatibility alone implies no licence rights.
- **ComfyUI and its nodes** are optional *external* runtimes launched via a user-configured installation. ComfyUI source, nodes and Python environment are not included in the ZIP; their applicable licences remain separate.
- **FFmpeg/ffprobe**: separately installed host tools (not bundled). The actual binary configuration determines applicable LGPL/GPL/nonfree terms. See <https://ffmpeg.org/legal.html>.

No licence notice here grants commercial usage rights in a model where the actual model licence does not. Software distribution, use of model weights and rights in generated outputs are distinct questions. The package has no publisher signature; SHA-256/checksum checks attest content integrity, not the identity of the publisher.

## Transparency / limitations

This is a source-install distribution and a cross-platform installation acceptance candidate. All bundled original `LICENSE`/`NOTICE` texts are preserved. A **final legal sign-off of source provenance, each resolved runtime dependency, model weights and actual redistributed binaries cannot be inferred from static inspection alone**. No third-party runtime SBOM or commercial-clearance certification is claimed. No personal data or provider credentials are bundled.
