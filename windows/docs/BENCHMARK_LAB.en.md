# Presets and Benchmark Lab — updated for 1.0.57

## Start with the existing configuration

The current Models configuration remains the default. Existing user model profiles and generation presets are preserved. No model is downloaded by applying a preset. Generation runs only after pressing Generate; benchmark jobs run only after reviewing and confirming a plan.

In Models → Add, link the existing Qwen 2.1 repository and any compatible transformer, VAE, encoder file, encoder folder, processor or scheduler. Single-file encoder weights have their own **Encoder weights** category. Keep the matching encoder config and processor: a safetensors filename does not establish compatibility. Downloaded or uploaded components enter the same catalog.

## User presets

**Model** stores the repository, component paths, loading/precision/offload settings and optional LoRAs. **Generation** stores dimensions, steps, guidance, seed and decode options, with an optional prompt, but no input images or model paths. **Full** snapshots both so subsequent edits to the original model preset do not silently change the full preset.

Choose a model and a generation preset independently in Create / Edit, or select a full preset. A selection changes the form, not the global Models default. Under **Temporary model components**, checking a component enables an override for the next job; leaving it unchecked inherits the preset. It does not remove a mandatory module. The path shown under each selector indicates the selected override or original component. LoRAs saved with a model are used only with the explicit checkbox; a full preset enables its own stack.

Save the current workspace in User presets. Existing user items can be edited, duplicated, renamed, replaced (with confirmation) or deleted. Built-in starter settings are read-only but can be duplicated. They are starting points, not certified quality or RTX 3090 performance profiles.

Portable JSON exports replace automatic filesystem paths with component kind/name descriptors. Import resolves unique local matches; missing/ambiguous items require explicit catalog bindings. Import always creates a new preset. Manual prompts and names are exported intentionally; review them before sharing. The previous generation-only tools remain in a collapsed section for compatibility.

## Comparison workflows

Open a project and configure Create / Edit first. Benchmark Lab uses those prompt, reference order, resolution, seed, step and guidance settings. A random seed is resolved **once per plan**, including repetitions. Each job has its own immutable configuration snapshot.

- **Compare model presets:** select profiles. VAE and scheduler are locked to the first profile by default to avoid accidental extra variables; disable a lock only when that is the intended comparison.
- **Transformer × encoder:** choose the baseline model profile, transformer variants and single-file encoder variants. A 2×2 selection creates four configurations. Folder-based encoders can instead be compared using model profiles.
- **Full automatic benchmark:** choose registered Transformer, encoder and VAE variants plus one or more saved scenes. **Full Cartesian** queues every selected Transformer × encoder × VAE × prompt combination. **Screening** queues one baseline and then changes one component family at a time. The plan is shown before anything is queued.
- **Same latent / VAE comparison:** choose an existing project latent and model profiles with compatible VAEs. No diffusion or text encoder runs. The original latent is never overwritten.
- **VAE round-trip:** choose a project image and 1, 5 and/or 10 cycles. The image is explicitly resized to the selected dimensions. Each cycle uses deterministic posterior mode and an RGBA8 image between cycles. This tests cumulative reconstruction plus 8-bit image conversion, not latent-only precision.

**Scene library:** In Full automatic benchmark, tick only the scenes you want to include (up to 8 per plan). Use **+ New scene** to create a named prompt and save it; **Edit scene** to update a built-in or user scene; **Restore original** to revert a modified built-in; or **Delete** for a user-created scene. All five original example scenes remain available by default. Custom scenes and built-in edits persist across application updates in `shared/benchmarks/qwen21-scenes/scenes.json`, under the Qwen edition data directory. Changes to a scene affect only new plans; existing queued/saved benchmark plans, prompts and reports are unchanged. No model, queue or inference setting is altered.

Press **Prepare comparison**, review the variants, then **Confirm and queue all tests**. Limits: up to 12 variants per component family, 24 focused matrix combinations, 3 repetitions and 32 jobs for focused suites. Full automatic mode allows up to 128 component combinations and 512 queued jobs after prompts/repetitions; large plans show an explicit warning before confirmation. Jobs use the existing serialized GPU queue. Repeated confirmation does not duplicate jobs. Keep other GPU applications idle for meaningful comparisons.

History and results persist on the server. Failed jobs remain visible. Cancel affects only that comparison; existing images and model files are kept. Export a report as JSON or CSV. No automatic aesthetic, text-recognition or identity score is claimed; inspect outputs and crops yourself.

## What is measured

Reports record total job time, component loading, combined conditioning/sampling, offload and VAE decode where available. CUDA peaks are PyTorch allocator **allocated / reserved** bytes for the worker, not total GPU-board memory. Process RAM is sampled every 50 ms, not total system RAM. Jobs include loading overhead; these are not warm-cache kernel microbenchmarks.

Round-trip reports contain RGB, alpha and RGBA MSE/PSNR relative to the resized original. For identical pixels, the PSNR field is null and the corresponding `identical` flag is true instead of writing invalid JSON infinity.

## Formats and important limitations

The existing Diffusers and GGUF generation paths are retained. Single-file encoder loading requires a compatible Qwen3-VL configuration and a complete strict tensor-name/shape match. VAE files must follow the native Diffusers Qwen 2.1 layout; a matching Diffusers folder is preferred. A generic/Comfy-converted safetensors file is not accepted merely because its extension or filename looks correct.

**INT8 ConvRot has two explicit execution modes.** `Native ComfyUI ConvRot` uses the already configured/verified local ComfyUI runtime in an isolated localhost process. Image Lab maps the selected single-file Transformer, Text Encoder and VAE into a temporary ComfyUI model tree, executes the stock Qwen Image 2.1 loaders/conditioning/sampler, copies the result back, then terminates the process. No Image Lab dequantization is performed in this mode. Peak GPU process memory is sampled through `nvidia-smi`; those figures are not PyTorch allocator peaks from the Image Lab worker.

`Reference float reconstruction` is retained for controlled quality/debug checks only. It reconstructs supported `int8_tensorwise`/ConvRot weights into floating-point tensors before Diffusers inference, so its speed, load time and VRAM must **not** be described as native INT8 performance. Unsupported metadata/layouts are rejected instead of guessed.

Native ConvRot in this RC is intentionally limited to T2I jobs that can supply explicit single-file safetensors Transformer, Text Encoder and VAE components. GGUF continues through the existing GGUF path. Unsupported native combinations fail that individual benchmark job; the serialized queue continues. Run one native job first before a large matrix.

No native acceptance test with the user's large model files, GPU or 2048 output was performed while preparing this archive. Successful synthetic tests and a Ready file check do not establish model quality or memory capacity. Confirm each profile with a single real generation before launching a large matrix.

Auto/Full-frame decoding never silently switches to tiling or CPU. For a full-frame GPU OOM, use the retained latent in Decode Lab and explicitly select CPU, or a compatible alternative VAE. The round-trip path likewise requires an explicit device choice.


## Comparison gallery and saved results (1.0.57)

The Comparison Gallery can read saved benchmark reports across projects. It does not regenerate images. It groups examples by prompt, offers filters for model components, resolution, CFG, steps, seed and status, and allows A/B or side-by-side viewing. The archive selector sits in a collapsible section above the results. Deleting a complete benchmark is destructive and requires confirmation; hiding a result is non-destructive.

Open a thumbnail to view the full image **fitted within the viewer**, then use the Zoom slider, plus/minus controls, mouse drag or touch pinch/drag. **Fit image** restores the entire picture. Information remains underneath the image. The actual source image is not cropped or modified by zoom. The A/B panel supports the same controls; compare like-for-like aspect ratios for aligned crops.

Runtime and model-specific features, including Turbo LoRA, require real inference smoke tests before they can be advertised as working with every backend. Static package tests do not establish CUDA compatibility.
