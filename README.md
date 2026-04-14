# Griptape Nodes VOID Library

A [Griptape Nodes](https://www.griptapenodes.com/) library for physics-aware video object removal using [VOID](https://github.com/Netflix/void-model).

## Overview

This library wraps the VOID (Video Inpainting with Object-aware Diffusion) model from Netflix Research in two Griptape Nodes nodes. VOID removes objects from videos along with all physical interactions they induce on the scene -- not just secondary effects like shadows and reflections, but physics-level interactions such as objects falling when a supporting person is removed. Pass 1 generates a plausible inpainted video via diffusion; Pass 2 (optional) refines temporal consistency using optical-flow warped noise derived from the Pass 1 output.

## Requirements

- **GPU**: CUDA (NVIDIA) required
- **Griptape Nodes Engine**: Version 0.77.5 or later

## Nodes

### VOID Pass 1 Inpainting

Removes a masked object from a video using VOID Pass 1 inference. Takes an input video, a quadmask video encoding which regions to remove, overlap, be affected, or keep, and a text prompt describing the background after removal. Outputs an inpainted video with the object and its physical interactions removed.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_model_id` | HuggingFace repo | Base CogVideoX-Fun model (e.g. `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`) |
| `void_checkpoint_repo` | HuggingFace repo | VOID model checkpoint repo (e.g. `netflix/void-model`) |
| `input_video` | VideoUrlArtifact | Source video from which the object will be removed |
| `quadmask_video` | VideoUrlArtifact | Quadmask video encoding removal regions: 0=remove, 63=overlap, 127=affected, 255=keep |
| `prompt` | str | Text description of the scene background after the object is removed |
| `negative_prompt` | str | Text prompt describing what to avoid in the output |
| `height` | int | Output video height in pixels (default: 384) |
| `width` | int | Output video width in pixels (default: 672) |
| `temporal_window_size` | int | Number of frames to process per temporal window (default: 85) |
| `num_inference_steps` | int | Number of diffusion denoising steps (default: 50) |
| `guidance_scale` | float | Classifier-free guidance scale (default: 1.0) |
| `seed` | int | Random seed for reproducible generation (default: 42) |
| `output_video` | VideoUrlArtifact | Inpainted video with the masked object and its physical interactions removed |

### VOID Pass 2 Refinement

Refines the temporal consistency of a VOID Pass 1 output using Pass 2 warped-noise refinement. Takes the original input video, the quadmask, the Pass 1 output video (used to generate warped optical-flow noise), and a text prompt. Outputs a temporally consistent refined inpainted video.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_model_id` | HuggingFace repo | Base CogVideoX-Fun model (e.g. `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`) |
| `void_checkpoint_repo` | HuggingFace repo | VOID model checkpoint repo (e.g. `netflix/void-model`) |
| `input_video` | VideoUrlArtifact | Original source video (same as used in Pass 1) |
| `quadmask_video` | VideoUrlArtifact | Quadmask video used in Pass 1 |
| `pass1_output_video` | VideoUrlArtifact | Output video from VOID Pass 1, used to generate warped optical-flow noise |
| `prompt` | str | Text description of the background after removal (same prompt as Pass 1) |
| `negative_prompt` | str | Text prompt describing what to avoid in the output |
| `height` | int | Output video height in pixels (default: 384) |
| `width` | int | Output video width in pixels (default: 672) |
| `temporal_window_size` | int | Number of frames to process per temporal window (default: 85) |
| `num_inference_steps` | int | Number of diffusion denoising steps (default: 50) |
| `guidance_scale` | float | Classifier-free guidance scale (default: 6.0) |
| `seed` | int | Random seed for reproducible generation (default: 42) |
| `output_video` | VideoUrlArtifact | Temporally refined inpainted video with improved consistency compared to Pass 1 |

## Available Models

The following models are available from HuggingFace:

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

3. **Verify installation** by checking that the nodes appear in the node palette under the "VOID Inpainting" category.

## Usage

### VOID Pass 1 Inpainting

1. Add a **VOID Pass 1 Inpainting** node to your workflow
2. Connect your source video to the `input_video` input
3. Connect your quadmask video to the `quadmask_video` input -- the quadmask encodes per-pixel removal intent using four grayscale levels (0=remove, 63=overlap, 127=affected, 255=keep)
4. Set the `prompt` to describe what the background should look like after the object is removed
5. Connect the `output_video` to your next node or a display node

### VOID Pass 2 Refinement

1. Run **VOID Pass 1 Inpainting** first and keep the Pass 1 `output_video` wired up
2. Add a **VOID Pass 2 Refinement** node to your workflow
3. Connect the same `input_video` and `quadmask_video` used in Pass 1
4. Connect the Pass 1 `output_video` to the `pass1_output_video` input
5. Use the same `prompt` as Pass 1
6. Connect the `output_video` to your next node or a display node

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

### make_warped_noise.py Errors (Pass 2)

- Confirm the submodule is fully initialized and the `inference/cogvideox_fun/make_warped_noise.py` script exists inside `griptape_nodes_void_library/void-model/`
- Ensure `cv2` (OpenCV) is available in the library's virtual environment

## Additional Resources

- [VOID Model GitHub](https://github.com/Netflix/void-model)
- [Griptape Nodes Documentation](https://docs.griptapenodes.com/)
- [Griptape Discord](https://discord.gg/griptape)

## License

This library is provided under the Apache License 2.0. The bundled VOID submodule is subject to its own license: see [Netflix/void-model](https://github.com/Netflix/void-model) for details.
