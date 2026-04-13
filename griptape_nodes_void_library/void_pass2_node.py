import logging
import os
import subprocess
import sys
import tempfile
import uuid
from typing import Any

import torch
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("void_library")

BASE_MODEL_REPO_IDS = [
    "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP",
]

VOID_CHECKPOINT_REPO_IDS = [
    "netflix/void-model",
]


class VoidPass2Node(SuccessFailureNode):
    """Refines temporal consistency of a VOID Pass 1 output using Pass 2 warped-noise refinement.

    Takes the original input video, the quadmask, the Pass 1 output video (used to generate
    warped optical-flow noise), and a text prompt. Outputs a temporally consistent refined
    inpainted video.
    """

    # Class-level pipeline cache keyed on (base_model_id, void_checkpoint_repo)
    _pipeline_cache: dict[tuple[str, str], Any] = {}
    _vae_cache: dict[tuple[str, str], Any] = {}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # HuggingFace model selection for base model
        self._base_model_param = HuggingFaceRepoParameter(
            self,
            repo_ids=BASE_MODEL_REPO_IDS,
            parameter_name="base_model_id",
        )
        self._base_model_param.add_input_parameters()

        # HuggingFace model selection for VOID checkpoint
        self._void_checkpoint_param = HuggingFaceRepoParameter(
            self,
            repo_ids=VOID_CHECKPOINT_REPO_IDS,
            parameter_name="void_checkpoint_repo",
        )
        self._void_checkpoint_param.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="input_video",
                allowed_modes={ParameterMode.INPUT},
                type="VideoUrlArtifact",
                input_types=["VideoUrlArtifact"],
                default_value=None,
                tooltip="Original source video (same as used in Pass 1).",
            )
        )

        self.add_parameter(
            Parameter(
                name="quadmask_video",
                allowed_modes={ParameterMode.INPUT},
                type="VideoUrlArtifact",
                input_types=["VideoUrlArtifact"],
                default_value=None,
                tooltip="Quadmask video used in Pass 1.",
            )
        )

        self.add_parameter(
            Parameter(
                name="pass1_output_video",
                allowed_modes={ParameterMode.INPUT},
                type="VideoUrlArtifact",
                input_types=["VideoUrlArtifact"],
                default_value=None,
                tooltip="Output video from VOID Pass 1 -- used to generate warped optical-flow noise for Pass 2.",
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="Text description of the background after removal (same prompt as Pass 1).",
            )
        )

        self.add_parameter(
            Parameter(
                name="negative_prompt",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="The video is not of a high quality, it has a low resolution. Watermark present in each frame. The background is solid. Strange body and strange trajectory. Distortion.",
                tooltip="Text prompt describing what to avoid in the output.",
            )
        )

        self.add_parameter(
            Parameter(
                name="height",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=384,
                tooltip="Output video height in pixels.",
            )
        )

        self.add_parameter(
            Parameter(
                name="width",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=672,
                tooltip="Output video width in pixels.",
            )
        )

        self.add_parameter(
            Parameter(
                name="temporal_window_size",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=85,
                tooltip="Number of frames to process per temporal window.",
            )
        )

        self.add_parameter(
            Parameter(
                name="num_inference_steps",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=50,
                tooltip="Number of diffusion denoising steps.",
            )
        )

        self.add_parameter(
            Parameter(
                name="guidance_scale",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=6.0,
                tooltip="Classifier-free guidance scale (Pass 2 typically uses higher guidance than Pass 1).",
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=42,
                tooltip="Random seed for reproducible generation.",
            )
        )

        self.add_parameter(
            Parameter(
                name="output_video",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="VideoUrlArtifact",
                default_value=None,
                tooltip="Temporally refined inpainted video with improved consistency compared to Pass 1.",
            )
        )

        self._create_status_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that required inputs are present."""
        errors: list[Exception] = []

        base_errors = self._base_model_param.validate_before_node_run()
        if base_errors:
            errors.extend(base_errors)

        void_errors = self._void_checkpoint_param.validate_before_node_run()
        if void_errors:
            errors.extend(void_errors)

        if not self.parameter_values.get("input_video"):
            errors.append(ValueError("input_video is required"))

        if not self.parameter_values.get("quadmask_video"):
            errors.append(ValueError("quadmask_video is required"))

        if not self.parameter_values.get("pass1_output_video"):
            errors.append(ValueError("pass1_output_video is required"))

        if not self.parameter_values.get("prompt"):
            errors.append(ValueError("prompt is required"))

        return errors if errors else None

    def _get_submodule_root(self) -> str:
        """Return the absolute path to the void-model submodule."""
        assert __file__ is not None
        return os.path.join(os.path.dirname(__file__), "void-model")

    def _load_pipeline(self, base_model_path: str, void_checkpoint_path: str, cache_key: tuple[str, str]) -> None:
        """Load and cache the VOID Pass 2 pipeline."""
        # DEFERRED IMPORTS: these are from the submodule and pip deps that only
        # exist after the advanced library has initialized the environment.
        submodule_root = self._get_submodule_root()
        if submodule_root not in sys.path:
            sys.path.insert(0, submodule_root)

        from diffusers import CogVideoXDDIMScheduler
        from safetensors.torch import load_file
        from videox_fun.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, T5EncoderModel, T5Tokenizer
        from videox_fun.pipeline import CogVideoXFunInpaintPipeline

        logger.info(f"Loading VOID Pass 2 pipeline from {base_model_path}")

        transformer = CogVideoXTransformer3DModel.from_pretrained(
            base_model_path,
            subfolder="transformer",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_vae_mask=True,
            stack_mask=False,
        ).to(torch.bfloat16)

        # Load VOID Pass 2 checkpoint weights on top of the base transformer
        logger.info(f"Loading VOID Pass 2 checkpoint: {void_checkpoint_path}")
        state_dict = load_file(void_checkpoint_path)
        state_dict = state_dict.get("state_dict", state_dict)

        param_name = "patch_embed.proj.weight"
        if param_name in state_dict and state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1):
            logger.info(f"Adapting {param_name} for channel mismatch")
            latent_ch = 16
            feat_scale = 8
            feat_dim = int(latent_ch * feat_scale)
            new_weight = transformer.state_dict()[param_name].clone()
            new_weight[:, :feat_dim] = state_dict[param_name][:, :feat_dim]
            new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
            state_dict[param_name] = new_weight

        transformer.load_state_dict(state_dict, strict=False)

        vae = AutoencoderKLCogVideoX.from_pretrained(
            base_model_path,
            subfolder="vae",
        ).to(torch.bfloat16)

        tokenizer = T5Tokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        scheduler = CogVideoXDDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

        pipe = CogVideoXFunInpaintPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        pipe.enable_model_cpu_offload(device="cuda")

        VoidPass2Node._pipeline_cache[cache_key] = pipe
        VoidPass2Node._vae_cache[cache_key] = vae
        logger.info("VOID Pass 2 pipeline loaded successfully")

    def _video_artifact_to_bytes(self, artifact: VideoUrlArtifact) -> bytes:
        """Download a VideoUrlArtifact and return raw video bytes."""
        import urllib.request

        with urllib.request.urlopen(artifact.value) as resp:
            return resp.read()

    def _decode_video_to_tensor(self, video_bytes: bytes, height: int, width: int, num_frames: int) -> torch.Tensor:
        """Decode video bytes into a float tensor of shape (1, C, T, H, W) in [0, 1]."""
        import imageio
        import numpy as np
        import torch.nn.functional as F

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            reader = imageio.get_reader(tmp_path)
            frames = []
            for i, frame in enumerate(reader):
                if i >= num_frames:
                    break
                frames.append(frame)
            reader.close()
        finally:
            os.unlink(tmp_path)

        if not frames:
            raise ValueError("No frames decoded from video")

        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)

        if tensor.shape[2] != height or tensor.shape[3] != width:
            tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)

        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        return tensor

    def _build_mask_tensor(self, quadmask_bytes: bytes, height: int, width: int, num_frames: int) -> torch.Tensor:
        """Build a binary mask tensor (1, 3, T, H, W) from quadmask video bytes.

        Values below 200 in the grayscale quadmask indicate the mask region.
        """
        import imageio
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(quadmask_bytes)
            tmp_path = tmp.name

        try:
            reader = imageio.get_reader(tmp_path)
            frames = []
            for i, frame in enumerate(reader):
                if i >= num_frames:
                    break
                if frame.ndim == 3 and frame.shape[2] >= 3:
                    gray = frame[:, :, 0].astype(np.float32)
                else:
                    gray = frame.astype(np.float32)
                frames.append(gray)
            reader.close()
        finally:
            os.unlink(tmp_path)

        if not frames:
            raise ValueError("No frames decoded from quadmask video")

        frames_np = np.stack(frames, axis=0)  # (T, H, W)
        mask_np = (frames_np < 200).astype(np.float32)
        tensor = torch.from_numpy(mask_np).unsqueeze(1)  # (T, 1, H, W)

        if tensor.shape[2] != height or tensor.shape[3] != width:
            tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="nearest")

        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, T, H, W)
        tensor = tensor.expand(-1, 3, -1, -1, -1)  # (1, 3, T, H, W)
        return tensor

    def _generate_warped_noise(
        self,
        pass1_video_bytes: bytes,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        latent_c: int,
        device: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate warped noise from Pass 1 video using make_warped_noise.py subprocess.

        Returns a warped noise tensor of shape (1, T, C, H, W).
        """
        import cv2
        import numpy as np

        submodule_root = self._get_submodule_root()
        make_warped_noise_script = os.path.join(submodule_root, "inference", "cogvideox_fun", "make_warped_noise.py")

        if not os.path.exists(make_warped_noise_script):
            raise FileNotFoundError(f"make_warped_noise.py not found at: {make_warped_noise_script}")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            tmp_vid.write(pass1_video_bytes)
            tmp_vid_path = tmp_vid.name

        with tempfile.TemporaryDirectory() as noise_output_dir:
            noise_subdir = os.path.join(noise_output_dir, "warped_noise")
            cmd = [
                sys.executable,
                make_warped_noise_script,
                os.path.abspath(tmp_vid_path),
                os.path.abspath(noise_subdir),
            ]

            logger.info("Running make_warped_noise.py (this may take several minutes)...")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"make_warped_noise.py failed (exit {result.returncode}):\n"
                        f"stdout: {result.stdout}\nstderr: {result.stderr}"
                    )
            finally:
                os.unlink(tmp_vid_path)

            noise_file = os.path.join(noise_subdir, "noises.npy")
            if not os.path.exists(noise_file):
                raise FileNotFoundError(f"Warped noise file not created at: {noise_file}")

            target_shape = (latent_t, latent_h, latent_w, latent_c)

            warped_noise_np = np.load(noise_file)
            logger.info(f"Loaded warped noise: shape={warped_noise_np.shape}, dtype={warped_noise_np.dtype}")

            if warped_noise_np.dtype == np.float16:
                warped_noise_np = warped_noise_np.astype(np.float32)

            # Convert TCHW to THWC if needed
            if warped_noise_np.ndim == 4 and warped_noise_np.shape[1] == 16:
                warped_noise_np = warped_noise_np.transpose(0, 2, 3, 1)

            # Resize to target shape
            if warped_noise_np.shape != target_shape:
                logger.info(f"Resizing noise from {warped_noise_np.shape} to {target_shape}")

                if warped_noise_np.shape[0] != latent_t:
                    indices = np.linspace(0, warped_noise_np.shape[0] - 1, latent_t).astype(int)
                    warped_noise_np = warped_noise_np[indices]

                resized_frames = []
                for t in range(latent_t):
                    frame = warped_noise_np[t]
                    channels_resized = []
                    for c in range(frame.shape[2]):
                        channel = frame[:, :, c]
                        channel_resized = cv2.resize(channel, (latent_w, latent_h), interpolation=cv2.INTER_LINEAR)
                        channels_resized.append(channel_resized)
                    frame_resized = np.stack(channels_resized, axis=2)
                    resized_frames.append(frame_resized)

                warped_noise_np = np.stack(resized_frames, axis=0)

            # (T, H, W, C) -> (1, T, C, H, W)
            warped_noise_np = warped_noise_np.transpose(0, 3, 1, 2)
            warped_noise = torch.from_numpy(warped_noise_np).float().unsqueeze(0)
            warped_noise = warped_noise.to(device, dtype=dtype)

            logger.info(f"Warped noise ready: shape={warped_noise.shape}")
            return warped_noise

    def _tensor_to_mp4_bytes(self, tensor: torch.Tensor, fps: int = 12) -> bytes:
        """Encode a video tensor (1, C, T, H, W) or (T, H, W, C) in [0,1] to MP4 bytes."""
        import io

        import imageio
        import numpy as np

        if tensor.dim() == 5:
            tensor = tensor[0].permute(1, 2, 3, 0)

        video_np = tensor.clamp(0, 1).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)

        buf = io.BytesIO()
        writer = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
        for frame in video_np:
            writer.append_data(frame)
        writer.close()
        buf.seek(0)
        return buf.read()

    def process(self) -> AsyncResult[None]:
        """Kick off async inference."""
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        """Run VOID Pass 2 inference in a background thread."""
        from huggingface_hub import hf_hub_download, snapshot_download

        base_model_id, _ = self._base_model_param.get_repo_revision()
        void_checkpoint_repo, _ = self._void_checkpoint_param.get_repo_revision()

        cache_key = (base_model_id, void_checkpoint_repo)

        if cache_key not in VoidPass2Node._pipeline_cache:
            logger.info(f"Downloading base model: {base_model_id}")
            base_model_path = snapshot_download(base_model_id)

            logger.info(f"Downloading VOID Pass 2 checkpoint from: {void_checkpoint_repo}")
            void_checkpoint_path = hf_hub_download(void_checkpoint_repo, "void_pass2.safetensors")

            self._load_pipeline(base_model_path, void_checkpoint_path, cache_key)

        pipe = VoidPass2Node._pipeline_cache[cache_key]
        vae = VoidPass2Node._vae_cache[cache_key]

        prompt: str = self.parameter_values.get("prompt") or ""
        negative_prompt: str = self.parameter_values.get("negative_prompt") or ""
        height: int = self.parameter_values.get("height") or 384
        width: int = self.parameter_values.get("width") or 672
        temporal_window_size: int = self.parameter_values.get("temporal_window_size") or 85
        num_inference_steps: int = self.parameter_values.get("num_inference_steps") or 50
        guidance_scale: float = self.parameter_values.get("guidance_scale") or 6.0
        seed: int = self.parameter_values.get("seed") or 42

        input_video_artifact = self.parameter_values.get("input_video")
        quadmask_artifact = self.parameter_values.get("quadmask_video")
        pass1_artifact = self.parameter_values.get("pass1_output_video")

        if not isinstance(input_video_artifact, VideoUrlArtifact):
            raise ValueError("input_video is required")
        if not isinstance(quadmask_artifact, VideoUrlArtifact):
            raise ValueError("quadmask_video is required")
        if not isinstance(pass1_artifact, VideoUrlArtifact):
            raise ValueError("pass1_output_video is required")

        input_video_bytes = self._video_artifact_to_bytes(input_video_artifact)
        quadmask_bytes = self._video_artifact_to_bytes(quadmask_artifact)
        pass1_video_bytes = self._video_artifact_to_bytes(pass1_artifact)

        # Align video length to VAE temporal compression
        max_video_length = temporal_window_size
        video_length = (
            int((max_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio)
            + 1
        )
        logger.info(f"Aligned video length: {video_length}")

        input_video_tensor = self._decode_video_to_tensor(input_video_bytes, height, width, video_length)
        mask_tensor = self._build_mask_tensor(quadmask_bytes, height, width, video_length)

        # Calculate latent dimensions for warped noise
        latent_t = (temporal_window_size - 1) // 4 + 1
        latent_h = height // 8
        latent_w = width // 8
        latent_c = 16

        warped_noise = self._generate_warped_noise(
            pass1_video_bytes,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            latent_c=latent_c,
            device="cuda",
            dtype=torch.bfloat16,
        )

        logger.info(
            f"Running VOID Pass 2 inference: {height}x{width}, {temporal_window_size} frames, "
            f"{num_inference_steps} steps"
        )

        with torch.no_grad():
            output = pipe(
                prompt,
                num_frames=temporal_window_size,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                video=input_video_tensor,
                mask_video=mask_tensor,
                strength=1.0,
                use_trimask=False,
                zero_out_mask_region=False,
                use_vae_mask=True,
                stack_mask=False,
                latents=warped_noise,
            ).videos

        mp4_bytes = self._tensor_to_mp4_bytes(output, fps=12)

        filename = f"void_pass2_{uuid.uuid4().hex[:8]}.mp4"
        url = GriptapeNodes.StaticFilesManager().save_static_file(mp4_bytes, filename)
        self.parameter_output_values["output_video"] = VideoUrlArtifact(url)
        logger.info("VOID Pass 2 inference complete")
