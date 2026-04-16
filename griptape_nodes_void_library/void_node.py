import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("void_library")

BASE_MODEL_REPO_IDS = [
    "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP",
]

VOID_CHECKPOINT_REPO_IDS = [
    "netflix/void-model",
]

DEFAULT_NEGATIVE_PROMPT = (
    "The video is not of a high quality, it has a low resolution. Watermark present in each frame. "
    "The background is solid. Strange body and strange trajectory. Distortion."
)

# Script executed inside the library .venv to build a VOID quadmask from a
# binary primary mask and an optional affected mask. Matches the semantics
# of _build_quadmask_video in the reference minimax-remover node.
#
# Quadmask encoding: 0=remove, 63=overlap, 127=affected, 255=keep
_QUADMASK_BUILDER_SCRIPT = r"""
import sys
import numpy as np
import imageio.v2 as imageio

primary_path, affected_path, out_path, primary_threshold, affected_threshold = sys.argv[1:6]
primary_threshold = int(primary_threshold)
affected_threshold = int(affected_threshold)

primary_reader = imageio.get_reader(primary_path)
try:
    fps = float(primary_reader.get_meta_data().get("fps", 24.0) or 24.0)
    affected_frames = []
    if affected_path:
        affected_reader = imageio.get_reader(affected_path)
        try:
            affected_frames = [np.asarray(f) for f in affected_reader]
        finally:
            affected_reader.close()
    frames = []
    for idx, frame in enumerate(primary_reader):
        primary_np = np.asarray(frame)
        primary_gray = primary_np if primary_np.ndim == 2 else primary_np[:, :, 0]
        primary_region = primary_gray > primary_threshold
        if affected_frames and idx < len(affected_frames):
            affected_np = affected_frames[idx]
            affected_gray = affected_np if affected_np.ndim == 2 else affected_np[:, :, 0]
            h = min(primary_region.shape[0], affected_gray.shape[0])
            w = min(primary_region.shape[1], affected_gray.shape[1])
            primary_cut = primary_region[:h, :w]
            affected_cut = affected_gray[:h, :w] > affected_threshold
            quad = np.full((h, w), 255, dtype=np.uint8)
            quad[np.logical_and(affected_cut, np.logical_not(primary_cut))] = 127
            quad[np.logical_and(primary_cut, affected_cut)] = 63
            quad[np.logical_and(primary_cut, np.logical_not(affected_cut))] = 0
        else:
            quad = np.where(primary_region, 0, 255).astype(np.uint8)
        frames.append(np.stack([quad, quad, quad], axis=-1))
finally:
    primary_reader.close()

if not frames:
    raise SystemExit("Primary mask video has no readable frames.")

imageio.mimsave(out_path, frames, fps=max(1.0, fps))
"""


class VoidNode(SuccessFailureNode):
    """Removes a masked object from a video using Netflix's VOID model.

    Takes a binary primary mask video (object to remove) and an optional affected mask
    video (regions physically impacted by the removal), builds the VOID quadmask internally,
    runs Pass 1 inpainting, and optionally runs Pass 2 warped-noise refinement for improved
    temporal consistency.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Must be created before any add_parameter calls because after_value_set
        # can fire during parameter initialization
        self._seed_param = SeedParameter(self)

        self._base_model_param = HuggingFaceRepoParameter(
            self,
            repo_ids=BASE_MODEL_REPO_IDS,
            parameter_name="base_model_id",
        )
        self._base_model_param.add_input_parameters()

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
                name="primary_mask_video",
                allowed_modes={ParameterMode.INPUT},
                type="VideoUrlArtifact",
                input_types=["VideoUrlArtifact"],
                default_value=None,
                tooltip="Binary mask video of the object to remove (white=remove, black=keep).",
            )
        )

        self.add_parameter(
            Parameter(
                name="affected_mask_video",
                allowed_modes={ParameterMode.INPUT},
                type="VideoUrlArtifact",
                input_types=["VideoUrlArtifact"],
                default_value=None,
                tooltip=(
                    "Optional binary mask video of regions physically affected by the removal "
                    "(e.g., shadows, reflections, or objects that interact with the removed object). "
                    "When omitted, only the primary mask is used."
                ),
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
                default_value=DEFAULT_NEGATIVE_PROMPT,
                tooltip="Text prompt describing what to avoid in the output.",
            )
        )

        self.add_parameter(
            Parameter(
                name="primary_threshold",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=20,
                tooltip="Pixel intensity cutoff (0-255) for binarizing the primary mask.",
            )
        )

        self.add_parameter(
            Parameter(
                name="affected_threshold",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=20,
                tooltip="Pixel intensity cutoff (0-255) for binarizing the affected mask.",
            )
        )

        self.add_parameter(
            Parameter(
                name="height",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=384,
                tooltip="Output video height in pixels. Snapped down to the nearest multiple of 16.",
            )
        )

        self.add_parameter(
            Parameter(
                name="width",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=672,
                tooltip="Output video width in pixels. Snapped down to the nearest multiple of 16.",
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
                name="pass1_num_inference_steps",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=30,
                tooltip="Number of Pass 1 diffusion denoising steps.",
            )
        )

        self.add_parameter(
            Parameter(
                name="pass1_guidance_scale",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=1.0,
                tooltip="Pass 1 classifier-free guidance scale.",
            )
        )

        self.add_parameter(
            Parameter(
                name="enable_pass2_refinement",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="bool",
                default_value=False,
                tooltip=(
                    "When enabled, runs VOID Pass 2 warped-noise refinement on the Pass 1 output "
                    "for improved temporal consistency. Significantly increases runtime."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="pass2_num_inference_steps",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="int",
                default_value=50,
                tooltip="Number of Pass 2 diffusion denoising steps. Only used when refinement is enabled.",
            )
        )

        self.add_parameter(
            Parameter(
                name="pass2_guidance_scale",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=6.0,
                tooltip="Pass 2 classifier-free guidance scale. Only used when refinement is enabled.",
            )
        )

        self._seed_param.add_input_parameters()

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

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        self._seed_param.after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []

        base_errors = self._base_model_param.validate_before_node_run()
        if base_errors:
            errors.extend(base_errors)

        void_errors = self._void_checkpoint_param.validate_before_node_run()
        if void_errors:
            errors.extend(void_errors)

        if not self.parameter_values.get("input_video"):
            errors.append(ValueError("input_video is required"))

        if not self.parameter_values.get("primary_mask_video"):
            errors.append(ValueError("primary_mask_video is required"))

        if not self.parameter_values.get("prompt"):
            errors.append(ValueError("prompt is required"))

        return errors if errors else None

    def _get_library_root(self) -> str:
        assert __file__ is not None
        return os.path.dirname(__file__)

    def _get_submodule_root(self) -> str:
        return os.path.join(self._get_library_root(), "void-model")

    def _get_venv_python(self) -> str:
        library_root = self._get_library_root()
        if sys.platform == "win32":
            return os.path.join(library_root, ".venv", "Scripts", "python.exe")
        return os.path.join(library_root, ".venv", "bin", "python")

    def _probe_video_fps(self, video_path: str, default_fps: float = 24.0) -> float:
        """Probe a video's frame rate using imageio inside the library .venv.

        VOID's pass 1 reads every frame of the input and writes out at config.data.fps
        (default 12). If the source is 24 fps and we leave the default, an 8s clip
        becomes a 16s clip. We pass this fps back into both passes so frames-in
        equals frames-out at the correct timebase.
        """
        script = (
            "import sys, imageio.v2 as imageio\n"
            "r = imageio.get_reader(sys.argv[1])\n"
            "try:\n"
            "    fps = float(r.get_meta_data().get('fps', 0.0) or 0.0)\n"
            "finally:\n"
            "    r.close()\n"
            "print(fps)\n"
        )
        try:
            result = subprocess.run(
                [self._get_venv_python(), "-c", script, video_path],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            fps = float(result.stdout.strip() or 0.0)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
            logger.warning(f"Failed to probe video fps, falling back to {default_fps}: {e}")
            return default_fps
        if fps <= 0.0:
            logger.warning(f"Probed fps was {fps}, falling back to {default_fps}")
            return default_fps
        return fps

    def _rewrite_video_fps_in_venv(self, input_path: str, output_path: str, fps: float) -> str:
        """Re-encode a video at the target fps by reading every frame and rewriting.

        VOID's pass 2 script hardcodes fps=12 when writing its output. Running this
        helper reads all frames back (via imageio in the library .venv) and writes
        them at the correct fps -- every original frame is preserved, only the
        playback timebase changes. Returns the output path on success, or the
        input path on any failure.
        """
        script = (
            "import sys\n"
            "import numpy as np\n"
            "import imageio.v2 as imageio\n"
            "input_path, output_path, fps = sys.argv[1], sys.argv[2], float(sys.argv[3])\n"
            "reader = imageio.get_reader(input_path)\n"
            "try:\n"
            "    frames = [np.asarray(frame) for frame in reader]\n"
            "finally:\n"
            "    reader.close()\n"
            "if not frames:\n"
            "    raise SystemExit('input video has no frames')\n"
            "imageio.mimsave(output_path, frames, fps=max(1.0, fps))\n"
        )
        cmd = [self._get_venv_python(), "-c", script, input_path, output_path, str(fps)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            stderr = getattr(e, "stderr", "") or ""
            logger.warning(f"fps rewrite failed, using original: {stderr[-1000:] or e}")
            return input_path
        return output_path

    def _path_for_cli(self, path: str, repo_dir: str) -> str:
        r"""Convert path to POSIX-style to avoid Windows drive-letter parsing issues.

        ml_collections and other CLI tools can misparse Windows paths like C:\...
        as having a relative component "C". Using forward slashes and relative
        paths where possible avoids this.
        """
        try:
            path_resolved = Path(path).resolve()
            repo_resolved = Path(repo_dir).resolve()
            rel = path_resolved.relative_to(repo_resolved)
            return rel.as_posix()
        except ValueError:
            return str(path).replace("\\", "/")

    def _build_quadmask_in_venv(
        self,
        primary_mask_path: str,
        affected_mask_path: str | None,
        output_path: str,
        primary_threshold: int,
        affected_threshold: int,
    ) -> None:
        """Build the VOID quadmask video by running an inline script in the library .venv.

        The main Griptape process does not have imageio/numpy installed; the library's
        .venv does (via requirements.txt installed by void_library_advanced.py).
        """
        cmd = [
            self._get_venv_python(),
            "-c",
            _QUADMASK_BUILDER_SCRIPT,
            primary_mask_path,
            affected_mask_path or "",
            output_path,
            str(primary_threshold),
            str(affected_threshold),
        ]
        logger.info("Building VOID quadmask (affected mask: %s)", bool(affected_mask_path))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.stdout:
            logger.info(result.stdout[-4000:])
        if result.stderr:
            logger.warning(result.stderr[-4000:])
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-3000:]
            raise RuntimeError(f"Quadmask generation failed (exit {result.returncode}):\n{tail}")

    def _ffmpeg_env(self) -> dict[str, str]:
        """Build a subprocess env with submodule on PYTHONPATH and ffmpeg on PATH."""
        submodule_root = self._get_submodule_root()
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = submodule_root + os.pathsep + existing_pythonpath if existing_pythonpath else submodule_root
        try:
            import static_ffmpeg

            ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
        except (ImportError, FileNotFoundError, OSError) as e:
            logger.warning(f"static_ffmpeg not available, ffmpeg may not be found: {e}")
        return env

    def process(self) -> AsyncResult[None]:
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        from griptape_nodes.files.file import File
        from huggingface_hub import hf_hub_download, snapshot_download

        self._seed_param.preprocess()
        seed: int = self._seed_param.get_seed()

        base_model_id, _ = self._base_model_param.get_repo_revision()
        void_checkpoint_repo, _ = self._void_checkpoint_param.get_repo_revision()

        prompt: str = self.parameter_values.get("prompt") or ""
        negative_prompt: str = self.parameter_values.get("negative_prompt") or ""
        primary_threshold: int = self.parameter_values.get("primary_threshold") or 20
        affected_threshold: int = self.parameter_values.get("affected_threshold") or 20
        height: int = self.parameter_values.get("height") or 384
        width: int = self.parameter_values.get("width") or 672
        # CogVideoX: VAE compresses 8x spatially, then transformer uses 2x2 spatial
        # patches, so input h/w must be divisible by 16 (not 8) for an even latent dim.
        height = (height // 16) * 16
        width = (width // 16) * 16
        temporal_window_size: int = self.parameter_values.get("temporal_window_size") or 85
        pass1_num_inference_steps: int = self.parameter_values.get("pass1_num_inference_steps") or 30
        pass1_guidance_scale: float = self.parameter_values.get("pass1_guidance_scale") or 1.0
        enable_pass2: bool = bool(self.parameter_values.get("enable_pass2_refinement"))
        pass2_num_inference_steps: int = self.parameter_values.get("pass2_num_inference_steps") or 50
        pass2_guidance_scale: float = self.parameter_values.get("pass2_guidance_scale") or 6.0

        input_video_artifact = self.parameter_values.get("input_video")
        primary_mask_artifact = self.parameter_values.get("primary_mask_video")
        affected_mask_artifact = self.parameter_values.get("affected_mask_video")

        if not isinstance(input_video_artifact, VideoUrlArtifact):
            raise ValueError("input_video is required")
        if not isinstance(primary_mask_artifact, VideoUrlArtifact):
            raise ValueError("primary_mask_video is required")

        input_video_bytes = File(input_video_artifact.value).read_bytes()
        primary_mask_bytes = File(primary_mask_artifact.value).read_bytes()
        affected_mask_bytes: bytes | None = None
        if isinstance(affected_mask_artifact, VideoUrlArtifact):
            affected_mask_bytes = File(affected_mask_artifact.value).read_bytes()

        logger.info(f"Downloading base model: {base_model_id}")
        base_model_path = snapshot_download(base_model_id)

        logger.info(f"Downloading VOID Pass 1 checkpoint from: {void_checkpoint_repo}")
        void_pass1_checkpoint_path = hf_hub_download(void_checkpoint_repo, "void_pass1.safetensors")

        void_pass2_checkpoint_path: str | None = None
        if enable_pass2:
            logger.info(f"Downloading VOID Pass 2 checkpoint from: {void_checkpoint_repo}")
            void_pass2_checkpoint_path = hf_hub_download(void_checkpoint_repo, "void_pass2.safetensors")

        submodule_root = self._get_submodule_root()
        pass1_script = os.path.join(submodule_root, "inference", "cogvideox_fun", "predict_v2v.py")
        pass2_script = os.path.join(
            submodule_root, "inference", "cogvideox_fun", "inference_with_pass1_warped_noise.py"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            seq_name = "job0"
            data_root = os.path.join(tmp_dir, "void_data")
            seq_dir = os.path.join(data_root, seq_name)
            save_dir = os.path.join(tmp_dir, "void_outputs")
            os.makedirs(seq_dir, exist_ok=True)
            os.makedirs(save_dir, exist_ok=True)

            input_video_path = os.path.join(seq_dir, "input_video.mp4")
            quadmask_path = os.path.join(seq_dir, "quadmask_0.mp4")
            primary_mask_path = os.path.join(seq_dir, "_primary_mask.mp4")
            affected_mask_path = os.path.join(seq_dir, "_affected_mask.mp4") if affected_mask_bytes else None

            with open(input_video_path, "wb") as f:
                f.write(input_video_bytes)
            with open(primary_mask_path, "wb") as f:
                f.write(primary_mask_bytes)
            if affected_mask_path and affected_mask_bytes:
                with open(affected_mask_path, "wb") as f:
                    f.write(affected_mask_bytes)
            with open(os.path.join(seq_dir, "prompt.json"), "w", encoding="utf-8") as f:
                json.dump({"bg": prompt or "clean background"}, f)

            self._build_quadmask_in_venv(
                primary_mask_path=primary_mask_path,
                affected_mask_path=affected_mask_path,
                output_path=quadmask_path,
                primary_threshold=primary_threshold,
                affected_threshold=affected_threshold,
            )

            # --- VOID Pass 1 ---
            config_cli = self._path_for_cli(
                os.path.join(submodule_root, "config", "quadmask_cogvideox.py"), submodule_root
            )
            base_model_cli = self._path_for_cli(base_model_path, submodule_root)
            pass1_checkpoint_cli = self._path_for_cli(void_pass1_checkpoint_path, submodule_root)
            data_root_cli = self._path_for_cli(data_root, submodule_root)
            save_dir_cli = self._path_for_cli(save_dir, submodule_root)

            # Probe the input video's fps so both passes run at the source rate.
            # VOID's pass 1 config defaults to fps=12; pass 2 hardcodes fps=12 when
            # writing output. Without overriding, a 24 fps / 8s input becomes 16s.
            source_fps = self._probe_video_fps(input_video_path, default_fps=24.0)
            logger.info(f"Detected source fps: {source_fps}")

            pass1_cmd = [
                self._get_venv_python(),
                pass1_script,
                "--config",
                config_cli,
                f"--config.video_model.model_name={base_model_cli}",
                f"--config.video_model.transformer_path={pass1_checkpoint_cli}",
                f"--config.data.data_rootdir={data_root_cli}",
                f"--config.experiment.run_seqs={seq_name}",
                f"--config.experiment.save_path={save_dir_cli}",
                f"--config.data.sample_size={height}x{width}",
                f"--config.data.fps={source_fps}",
                f"--config.video_model.temporal_window_size={temporal_window_size}",
                f"--config.video_model.num_inference_steps={pass1_num_inference_steps}",
                f"--config.video_model.guidance_scale={pass1_guidance_scale}",
                f"--config.system.seed={seed}",
            ]
            if negative_prompt:
                pass1_cmd.append(f"--config.video_model.negative_prompt={negative_prompt}")

            env = self._ffmpeg_env()

            logger.info(f"Running VOID Pass 1: {height}x{width}, {temporal_window_size} frames")
            logger.info(f"Command: {' '.join(pass1_cmd)}")

            result = subprocess.run(
                pass1_cmd,
                cwd=submodule_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            if result.stdout:
                logger.info(result.stdout[-10000:])
            if result.stderr:
                logger.warning(result.stderr[-10000:])

            if result.returncode != 0:
                tail = (result.stderr or result.stdout or "")[-3000:]
                raise RuntimeError(f"VOID Pass 1 failed (exit {result.returncode}):\n{tail}")

            pass1_candidates = [p for p in glob.glob(os.path.join(save_dir, "*.mp4")) if not p.endswith("_tuple.mp4")]
            if not pass1_candidates:
                raise RuntimeError(f"VOID Pass 1 produced no output video in: {save_dir}")
            pass1_candidates.sort(key=os.path.getmtime, reverse=True)
            pass1_output_path = pass1_candidates[0]

            final_output_path = pass1_output_path

            # --- VOID Pass 2 (optional) ---
            if enable_pass2 and void_pass2_checkpoint_path is not None:
                pass1_dir = os.path.join(tmp_dir, "pass1_outputs")
                save_dir_pass2 = os.path.join(tmp_dir, "void_outputs_pass2")
                noise_cache_dir = os.path.join(tmp_dir, "noise_cache")
                for d in [pass1_dir, save_dir_pass2, noise_cache_dir]:
                    os.makedirs(d, exist_ok=True)

                # inference_with_pass1_warped_noise.py looks for {video_name}-fg=-1-*.mp4
                pass1_staged_path = os.path.join(pass1_dir, f"{seq_name}-fg=-1-auto.mp4")
                shutil.copyfile(pass1_output_path, pass1_staged_path)

                pass1_dir_cli = self._path_for_cli(pass1_dir, submodule_root)
                save_dir_pass2_cli = self._path_for_cli(save_dir_pass2, submodule_root)
                pass2_checkpoint_cli = self._path_for_cli(void_pass2_checkpoint_path, submodule_root)
                noise_cache_cli = self._path_for_cli(noise_cache_dir, submodule_root)

                pass2_cmd = [
                    self._get_venv_python(),
                    pass2_script,
                    "--video_name",
                    seq_name,
                    "--data_rootdir",
                    data_root_cli,
                    "--pass1_dir",
                    pass1_dir_cli,
                    "--output_dir",
                    save_dir_pass2_cli,
                    "--model_name",
                    base_model_cli,
                    "--model_checkpoint",
                    pass2_checkpoint_cli,
                    "--height",
                    str(height),
                    "--width",
                    str(width),
                    "--temporal_window_size",
                    str(temporal_window_size),
                    "--num_inference_steps",
                    str(pass2_num_inference_steps),
                    "--guidance_scale",
                    str(pass2_guidance_scale),
                    "--seed",
                    str(seed),
                    "--warped_noise_cache_dir",
                    noise_cache_cli,
                    "--use_quadmask",
                ]

                logger.info(f"Running VOID Pass 2: {height}x{width}, {temporal_window_size} frames")
                logger.info(f"Command: {' '.join(pass2_cmd)}")

                result = subprocess.run(
                    pass2_cmd,
                    cwd=submodule_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=7200,
                )

                if result.stdout:
                    logger.info(result.stdout[-10000:])
                if result.stderr:
                    logger.warning(result.stderr[-10000:])

                if result.returncode != 0:
                    tail = (result.stderr or result.stdout or "")[-3000:]
                    raise RuntimeError(f"VOID Pass 2 failed (exit {result.returncode}):\n{tail}")

                pass2_candidates = [
                    p
                    for p in glob.glob(os.path.join(save_dir_pass2, "**", "*.mp4"), recursive=True)
                    if not p.endswith("_tuple.mp4")
                ]
                if not pass2_candidates:
                    raise RuntimeError(f"VOID Pass 2 produced no output video in: {save_dir_pass2}")
                pass2_candidates.sort(key=os.path.getmtime, reverse=True)
                pass2_output_path = pass2_candidates[0]

                # Pass 2 writes output at a hardcoded fps=12. Rewrite to the source
                # fps so the output duration matches the input (all frames preserved).
                final_output_path = self._rewrite_video_fps_in_venv(
                    input_path=pass2_output_path,
                    output_path=os.path.join(tmp_dir, "pass2_final.mp4"),
                    fps=source_fps,
                )

            with open(final_output_path, "rb") as f:
                mp4_bytes = f.read()

        filename = f"void_{uuid.uuid4().hex[:8]}.mp4"
        url = GriptapeNodes.StaticFilesManager().save_static_file(mp4_bytes, filename)
        self.parameter_output_values["output_video"] = VideoUrlArtifact(url)
        logger.info("VOID inference complete (pass2=%s)", enable_pass2)
