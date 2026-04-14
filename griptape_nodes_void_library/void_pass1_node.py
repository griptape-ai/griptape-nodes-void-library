import logging
import os
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


class VoidPass1Node(SuccessFailureNode):
    """Removes a masked object from a video using VOID Pass 1 inference.

    Takes an input video, a quadmask video (encoding which regions to remove,
    overlap, affected, or keep), and a text prompt describing the background
    after removal. Outputs an inpainted video with the object and its physical
    interactions removed.
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
                tooltip="Source video from which the object will be removed.",
            )
        )

        self.add_parameter(
            Parameter(
                name="quadmask_video",
                allowed_modes={ParameterMode.INPUT},
                type="VideoUrlArtifact",
                input_types=["VideoUrlArtifact"],
                default_value=None,
                tooltip="Quadmask video encoding removal regions: 0=remove, 63=overlap, 127=affected, 255=keep.",
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="Text description of the scene background after the object is removed.",
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
                default_value=1.0,
                tooltip="Classifier-free guidance scale.",
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
                tooltip="Inpainted video with the masked object and its physical interactions removed.",
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

        if not self.parameter_values.get("prompt"):
            errors.append(ValueError("prompt is required"))

        return errors if errors else None

    def _get_submodule_root(self) -> str:
        """Return the absolute path to the void-model submodule."""
        assert __file__ is not None
        return os.path.join(os.path.dirname(__file__), "void-model")

    def _load_pipeline(self, base_model_path: str, void_checkpoint_path: str, cache_key: tuple[str, str]) -> None:
        """Load and cache the VOID Pass 1 pipeline."""
        # DEFERRED IMPORTS: these are from the submodule and pip deps that only
        # exist after the advanced library has initialized the environment.
        import sys

        submodule_root = self._get_submodule_root()
        if submodule_root not in sys.path:
            sys.path.insert(0, submodule_root)

        from diffusers import DDIMScheduler
        from safetensors.torch import load_file
        from videox_fun.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, T5EncoderModel, T5Tokenizer
        from videox_fun.pipeline import CogVideoXFunInpaintPipeline
        from videox_fun.utils.fp8_optimization import convert_weight_dtype_wrapper

        logger.info(f"Loading VOID Pass 1 pipeline from {base_model_path}")

        transformer = CogVideoXTransformer3DModel.from_pretrained(
            base_model_path,
            subfolder="transformer",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_vae_mask=True,
        ).to(torch.bfloat16)

        # Load VOID Pass 1 checkpoint weights on top of the base transformer
        logger.info(f"Loading VOID Pass 1 checkpoint: {void_checkpoint_path}")
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
        scheduler = DDIMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

        pipe = CogVideoXFunInpaintPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        convert_weight_dtype_wrapper(pipe.transformer, torch.bfloat16)
        pipe.enable_model_cpu_offload(device="cuda")

        VoidPass1Node._pipeline_cache[cache_key] = pipe
        VoidPass1Node._vae_cache[cache_key] = vae
        logger.info("VOID Pass 1 pipeline loaded successfully")

    def _video_artifact_to_bytes(self, artifact: VideoUrlArtifact) -> bytes:
        """Read a VideoUrlArtifact and return raw video bytes."""
        from griptape_nodes.files.file import File

        return File(artifact.value).read_bytes()

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
            try:
                frames = []
                for i, frame in enumerate(reader):
                    if i >= num_frames:
                        break
                    frames.append(frame)
            finally:
                reader.close()
        finally:
            os.unlink(tmp_path)

        if not frames:
            raise ValueError("No frames decoded from video")

        # Pad to num_frames by repeating last frame if video is shorter
        while len(frames) < num_frames:
            frames.append(frames[-1])

        # Stack frames: (T, H, W, C) -> (T, C, H, W)
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # (T, C, H, W)

        # Resize spatially if needed
        if tensor.shape[2] != height or tensor.shape[3] != width:
            tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)

        # Add batch dim: (1, C, T, H, W)
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
        return tensor

    def _build_mask_tensor(self, quadmask_bytes: bytes, height: int, width: int, num_frames: int) -> torch.Tensor:
        """Build a binary mask tensor (1, 1, T, H, W) from quadmask video bytes.

        Values below 200 in the grayscale quadmask indicate the mask region.
        """

        import imageio
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(quadmask_bytes)
            tmp_path = tmp.name

        try:
            reader = imageio.get_reader(tmp_path)
            try:
                frames = []
                for i, frame in enumerate(reader):
                    if i >= num_frames:
                        break
                    # Convert to grayscale if RGB
                    if frame.ndim == 3 and frame.shape[2] >= 3:
                        gray = frame[:, :, 0].astype(np.float32)
                    else:
                        gray = frame.astype(np.float32)
                    frames.append(gray)
            finally:
                reader.close()
        finally:
            os.unlink(tmp_path)

        if not frames:
            raise ValueError("No frames decoded from quadmask video")

        # Pad to num_frames by repeating last frame if video is shorter
        while len(frames) < num_frames:
            frames.append(frames[-1])

        frames_np = np.stack(frames, axis=0)  # (T, H, W)
        # Mask region: quadmask < 200 means remove/overlap/affected
        mask_np = (frames_np < 200).astype(np.float32)
        tensor = torch.from_numpy(mask_np).unsqueeze(1)  # (T, 1, H, W)

        # Resize spatially if needed
        if tensor.shape[2] != height or tensor.shape[3] != width:
            tensor = torch.nn.functional.interpolate(tensor, size=(height, width), mode="nearest")

        # (T, 1, H, W) -> (1, 1, T, H, W) then expand channels to match video
        # Pipeline expects mask_video in same shape as input_video
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 1, T, H, W)
        tensor = tensor.expand(-1, 3, -1, -1, -1)  # (1, 3, T, H, W)
        return tensor

    def _tensor_to_mp4_bytes(self, tensor: torch.Tensor, fps: int = 12) -> bytes:
        """Encode a video tensor (T, H, W, C) or (1, C, T, H, W) in [0,1] to MP4 bytes."""
        import io

        import imageio
        import numpy as np

        # Accept (1, C, T, H, W) or (T, H, W, C) shaped tensors
        if tensor.dim() == 5:
            # (1, C, T, H, W) -> (T, H, W, C)
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
        """Run VOID Pass 1 inference in a background thread."""
        from huggingface_hub import hf_hub_download, snapshot_download

        base_model_id, _ = self._base_model_param.get_repo_revision()
        void_checkpoint_repo, _ = self._void_checkpoint_param.get_repo_revision()

        cache_key = (base_model_id, void_checkpoint_repo)

        if cache_key not in VoidPass1Node._pipeline_cache:
            logger.info(f"Downloading base model: {base_model_id}")
            base_model_path = snapshot_download(base_model_id)

            logger.info(f"Downloading VOID Pass 1 checkpoint from: {void_checkpoint_repo}")
            void_checkpoint_path = hf_hub_download(void_checkpoint_repo, "void_pass1.safetensors")

            self._load_pipeline(base_model_path, void_checkpoint_path, cache_key)

        pipe = VoidPass1Node._pipeline_cache[cache_key]
        vae = VoidPass1Node._vae_cache[cache_key]

        prompt: str = self.parameter_values.get("prompt") or ""
        negative_prompt: str = self.parameter_values.get("negative_prompt") or ""
        height: int = self.parameter_values.get("height") or 384
        width: int = self.parameter_values.get("width") or 672
        temporal_window_size: int = self.parameter_values.get("temporal_window_size") or 85
        num_inference_steps: int = self.parameter_values.get("num_inference_steps") or 50
        guidance_scale: float = self.parameter_values.get("guidance_scale") or 1.0
        raw_seed = self.parameter_values.get("seed")
        seed: int = raw_seed if raw_seed is not None else 42

        input_video_artifact = self.parameter_values.get("input_video")
        quadmask_artifact = self.parameter_values.get("quadmask_video")

        if not isinstance(input_video_artifact, VideoUrlArtifact):
            raise ValueError("input_video is required")
        if not isinstance(quadmask_artifact, VideoUrlArtifact):
            raise ValueError("quadmask_video is required")

        input_video_bytes = self._video_artifact_to_bytes(input_video_artifact)
        quadmask_bytes = self._video_artifact_to_bytes(quadmask_artifact)

        # Align video length to VAE temporal compression
        max_video_length = temporal_window_size
        video_length = (
            int((max_video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio)
            + 1
        )
        logger.info(f"Aligned video length: {video_length}")

        input_video_tensor = self._decode_video_to_tensor(input_video_bytes, height, width, video_length)
        mask_tensor = self._build_mask_tensor(quadmask_bytes, height, width, video_length)

        logger.info(
            f"Running VOID Pass 1 inference: {height}x{width}, {temporal_window_size} frames, "
            f"{num_inference_steps} steps"
        )

        with torch.no_grad():
            sample = pipe(
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
                use_trimask=True,
                use_vae_mask=True,
            ).videos

        mp4_bytes = self._tensor_to_mp4_bytes(sample, fps=12)

        filename = f"void_pass1_{uuid.uuid4().hex[:8]}.mp4"
        url = GriptapeNodes.StaticFilesManager().save_static_file(mp4_bytes, filename)
        self.parameter_output_values["output_video"] = VideoUrlArtifact(url)
        logger.info("VOID Pass 1 inference complete")
