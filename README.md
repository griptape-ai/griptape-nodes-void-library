# Griptape Nodes VOID Library

A [Griptape Nodes](https://www.griptapenodes.com/) library for physics-aware video object removal using [VOID](https://github.com/Netflix/void-model).

## Overview

This library wraps the VOID (Video Inpainting with Object-aware Diffusion) model from Netflix Research. VOID removes objects from videos along with all physical interactions they induce on the scene -- not just secondary effects like shadows and reflections, but physics-level interactions such as objects falling when a supporting person is removed.

The library exposes a single `VOID` node that takes a source video and one or two binary mask videos, generates the VOID quadmask internally, runs Pass 1 diffusion inpainting, and optionally runs Pass 2 warped-noise refinement for improved temporal consistency.

## Requirements

- **GPU**: CUDA (NVIDIA) required
- **Griptape Nodes Engine**: Version 0.77.5 or later

## Input constraints

- **Resolution**: `width` and `height` must be multiples of **16**. The node snaps non-conforming values down to the nearest multiple. (CogVideoX's VAE compresses spatially by 8x and the transformer applies 2x2 patches, so latent dimensions must be even.)
- **Frame count**: VOID's VAE requires a frame count of the form **`4k + 1`** (1, 5, 9, ..., 193, 197) and caps total frames at **197**. You must pre-process your input video so it has a valid frame count before feeding it into this node. If the frame count is invalid you will see a `conv3d` kernel-size error from PyTorch; if the frame count exceeds 197 VOID will fail or silently truncate.

## Node: VOID

Removes a masked object from a video, with optional refinement pass.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_model_id` | HuggingFace repo | Base CogVideoX-Fun model (`alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`) |
| `void_checkpoint_repo` | HuggingFace repo | VOID checkpoint repo (`netflix/void-model`) |
| `input_video` | VideoUrlArtifact | Source video. Must have a frame count of `4k + 1` (max 197). |
| `primary_mask_video` | VideoUrlArtifact | Binary mask video of the object to remove (white=remove, black=keep). Must have the same frame count as `input_video`. |
| `affected_mask_video` | VideoUrlArtifact (optional) | Optional binary mask of regions physically affected by the removal (e.g. shadows, reflections, cushions, water displacement). When omitted, only the primary mask is used. |
| `prompt` | str | Text description of the scene background after the object is removed |
| `negative_prompt` | str | Text prompt describing what to avoid in the output |
| `primary_threshold` | int | Pixel intensity cutoff (0-255) for binarizing the primary mask (default: 20) |
| `affected_threshold` | int | Pixel intensity cutoff (0-255) for binarizing the affected mask (default: 20) |
| `width` | int | Output video width in pixels. Snapped down to a multiple of 16. (default: 672) |
| `height` | int | Output video height in pixels. Snapped down to a multiple of 16. (default: 384) |
| `temporal_window_size` | int | Number of frames to process per temporal window (default: 85) |
| `pass1_num_inference_steps` | int | Number of Pass 1 diffusion denoising steps (default: 30) |
| `pass1_guidance_scale` | float | Pass 1 classifier-free guidance scale (default: 1.0) |
| `enable_pass2_refinement` | bool | When enabled, runs VOID Pass 2 warped-noise refinement for improved temporal consistency. Significantly increases runtime. (default: false) |
| `pass2_num_inference_steps` | int | Number of Pass 2 denoising steps. Only used when refinement is enabled. (default: 50) |
| `pass2_guidance_scale` | float | Pass 2 classifier-free guidance scale. Only used when refinement is enabled. (default: 6.0) |
| `seed` | int | Random seed for reproducible generation |
| `output_video` | VideoUrlArtifact | Inpainted video. Frame count and fps match `input_video`. |

## Quadmask semantics

The node generates VOID's quadmask internally, so customers do not need to produce one themselves. The quadmask encodes four pixel-level intents:

| Value | Region | Meaning |
|---|---|---|
| `0` | primary only | remove and inpaint |
| `63` | primary AND affected | overlap (remove, use affected context) |
| `127` | affected only | regenerate as if the removed object were never there |
| `255` | neither | keep untouched |

Without an `affected_mask_video`, only values `0` and `255` are used (plain inpainting without interaction modeling).

## Available Models

| Model | Description |
|-------|-------------|
| `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP` | Base inpainting model (CogVideoX-Fun 5B) used as the backbone for both passes |
| `netflix/void-model` | VOID checkpoint repo containing `void_pass1.safetensors` and `void_pass2.safetensors` |

Models are downloaded automatically on first use and cached for subsequent runs.

## Installation

### Prerequisites

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running
- A CUDA-capable NVIDIA GPU

### Install the Library

1. **Clone the repository** to your Griptape Nodes workspace directory:

   ```bash
   cd `gtn config show workspace_directory`
   git clone --recurse-submodules https://github.com/griptape-ai/griptape-nodes-void-library.git
   ```

2. **Add the library** in the Griptape Nodes Editor:

   - Open the Settings menu and navigate to the *Libraries* settings
   - Click on *+ Add Library* at the bottom of the settings panel
   - Enter the path to the library JSON file:
     ```
     <workspace_directory>/griptape-nodes-void-library/griptape_nodes_void_library/griptape-nodes-library.json
     ```
   - You can check your workspace directory with `gtn config show workspace_directory`
   - Close the Settings Panel
   - Click on *Refresh Libraries*

3. **Verify installation** by checking that the `VOID` node appears in the node palette under the "VOID Inpainting" category.

## Usage

1. Ensure your input video satisfies the [frame-count constraint](#input-constraints): `4k + 1` frames, max 197. Pre-process with `ffmpeg` or a trimming node if needed.
2. Add a **VOID** node to your workflow.
3. Connect your source video to `input_video`.
4. Connect a binary mask video (white=remove, black=keep) to `primary_mask_video`. It must have the same frame count as the input video.
5. *(Optional)* Connect a second binary mask to `affected_mask_video` covering regions that should change because the object is gone (shadows, reflections, objects that rested on the removed object). Same frame count as the input.
6. Set `prompt` to describe the scene background after removal.
7. *(Optional)* Enable `enable_pass2_refinement` for improved temporal consistency at the cost of additional runtime.
8. Connect `output_video` to your next node or a display node.

The node builds the VOID quadmask from your masks internally and preserves the input's frame count and fps on the output.

## Troubleshooting

### Library Not Loading

- Ensure the git submodule is initialized. If you cloned without `--recurse-submodules`, run:
  ```bash
  git submodule update --init --recursive
  ```

### CUDA Not Available

- Verify your GPU drivers are up to date
- Ensure CUDA is properly installed and `nvidia-smi` reports your GPU correctly

### Out of Memory Errors

- Reduce `height` and `width` (default 384x672 is already a conservative resolution)
- Reduce `temporal_window_size` to process fewer frames per window
- Close other GPU-intensive applications before running inference

### conv3d Kernel Size Errors

A `RuntimeError: ... Kernel size can't be greater than actual input size` from `conv3d` means your input video's frame count is not of the form `4k + 1`. Re-encode the video so it has a valid frame count (1, 5, 9, ..., 193, 197) before feeding it into the node. For example, with `ffmpeg`:

```bash
ffmpeg -i input.mp4 -frames:v 193 -c:v libx264 trimmed.mp4
```

### Input Longer Than 197 Frames

VOID's `max_video_length` is 197. Trim your input (and masks) to 197 frames or fewer before feeding them in. Longer inputs will either fail or be silently truncated by VOID.

### Windows: "The paging file is too small for this operation to complete" (os error 1455)

On Windows, loading the CogVideoX transformer memory-maps a multi-gigabyte safetensors file. If your system page file is undersized, the OS rejects the mapping with `OSError: ... (os error 1455)` (`WinError 1455`) and Pass 1 fails before any inference runs.

The node detects this signature and re-raises with a guidance message, but the fix is at the OS level: increase the Windows page file size.

1. Open **Settings > System > About > Advanced system settings**.
2. Under *Performance*, click **Settings... > Advanced > Virtual memory > Change...**.
3. Either select **System managed size** for the drive that holds the page file, or set a custom size of roughly **2-3x your installed RAM** (e.g. 64-96 GB initial/maximum on a 32 GB machine).
4. Click *Set*, *OK*, and reboot.

After rebooting, re-run the workflow. The same fix applies to Pass 2.

## Additional Resources

- [VOID Model GitHub](https://github.com/Netflix/void-model)
- [Griptape Nodes Documentation](https://docs.griptapenodes.com/)
- [Griptape Discord](https://discord.gg/griptape)

## License

This library is provided under the Apache License 2.0. The bundled VOID submodule is subject to its own license: see [Netflix/void-model](https://github.com/Netflix/void-model) for details.
