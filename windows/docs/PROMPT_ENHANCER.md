# Prompt Enhancer — Qwen 2.1 Edition

Prompt Enhancer (PE) rewrites an image prompt or editing instruction. It is separate from the Qwen3-VL encoder used by diffusion. It is **Off by default** and never runs when you simply press Generate.

## Existing software and files

This optional feature uses an **existing, trusted local ComfyUI installation and its Python environment**, with Qwen3.5 and TextGenerate support. Image Lab invokes its model loader in a short-lived separate process: no ComfyUI web server is started and no custom nodes are loaded by this adapter. Image Lab does not download ComfyUI, models or dependencies, and does not modify that environment.

Link the single-file Qwen3.5 PE weights under **Models → Add → PE-I2I / PE-T2I**. Keep diffusion encoders in Text encoder; PE weights cannot substitute for them. File inspection checks bounded safetensors structure and Qwen3.5 keys, not the provenance or accuracy of a fine-tuned model. Review the exact model source and its terms.

## Use

1. Open a project and enter the main prompt. For PE-I2I add 1–10 ordered image references. PE-T2I requires no references and its matching PE-T2I weights.
2. Open **Create / Edit → Prompt Enhancer**, select the task and PE model.
3. Expand **PE runtime and settings**. Select the existing ComfyUI directory and its Python (the interpreter inside its venv, not system Python). Review and explicitly approve the local runtime. The discovery button only checks conventional folders; paths can be entered manually.
4. Start with 512 output tokens, thinking off and text sampling off. The seed and sampling settings here belong to PE, not image diffusion. Check existing runtime tests imports and CUDA visibility only; it does not load model weights or prove generation works.
5. Press **Preview rewritten prompt**. This queues a prompt-only job on the same queue and GPU lock as image jobs. The Image Lab diffusion cache is released before PE loading. Wait or cancel from the panel/Jobs.
6. Review and optionally edit the rewritten text, then press **Use this prompt**. This only changes the prompt field, not reference images, image seed, dimensions, sampling or VAE settings. Generate separately when ready.

The external process exits before the next image job. If another application, including a separately running ComfyUI server, already occupies the GPU, its allocations are not automatically removed.

## Templates and output

Image Lab defaults are original, concise task instructions requesting `rewritten_prompt` JSON. **They are not a copy of Qwen's official `system_prompt.txt`.** Custom system instruction allows pasting a reviewed instruction, including the exact upstream one for a controlled comparison. ComfyUI model template uses that installed tokenizer's normal template, without claiming it embeds the PE system prompt.

Reasoning blocks are excluded from the usable prompt. PE JSON is parsed for `rewritten_prompt`; `wh_ratio` and `ratio_follow`, when returned, are shown as advice and never silently change dimensions or references. Incomplete JSON or unfinished reasoning is rejected; the original prompt is preserved. Plain-text output is also supported. A token-limit warning requires reviewing the output rather than assuming it completed.

References are EXIF-oriented, composited on white if transparent and reduced to fit 1024×1024 before PE vision processing. Their order is preserved. The originals sent to image generation are not modified.

## Persistence and comparisons

Settings are stored under `shared/config/qwen-prompt-enhancer.json`; prompt results under `shared/prompt-enhancer/results`. Recent PE requests survive page refresh and remain associated with their project. Selecting a completed request restores its preview, not the main prompt automatically. Changing project or reference images clears the currently applied PE association.

A generation launched with an applied PE result records the original, rewritten and effective user-edited prompt, PE options, model filename/header identity, external runtime code hashes, timing and actual PE child-process CUDA counters. This is provenance, not a signature or full model checksum. Parent worker memory counters are not mislabeled as PE metrics. Images already generated are never overwritten.

For a manual PE Off/On comparison, first generate with the original prompt and fixed image parameters, then create/apply a PE preview and repeat with exactly those image parameters. The existing Benchmark Lab remains unchanged; automatic PE matrices are not added by this release.

## Limits and security

The feature is optional: normal generation, model presets, Decode Lab and updates do not require ComfyUI. Supported local interface compatibility is probed and errors are explicit; no alternate backend or CPU fallback is used to hide failure. Existing ComfyUI handles the native quantized loader; Image Lab's experimental float-reconstruction path for diffusion ConvRot is not used for PE. This does not by itself establish hardware INT8 acceleration: exact kernels and performance depend on the installed runtime/GPU.

ComfyUI and PE models keep their own licenses. No upstream ComfyUI code, model weights, Python environment or CUDA libraries are included. Private prompts, reference paths and model paths stay on the host; technical errors can include local paths. Review logs/results before sharing.

This RC needs a real PE request on the target GPU and exact files. Synthetic tests and runtime imports are not native model acceptance.
