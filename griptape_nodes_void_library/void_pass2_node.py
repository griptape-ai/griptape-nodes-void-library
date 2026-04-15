import glob
import json
import logging
import os
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


class VoidPass2Node(SuccessFailureNode):
    """Refines temporal consistency of a VOID Pass 1 output using Pass 2 warped-noise refinement.

    Takes the original input video, the quadmask, and the Pass 1 output video (used to generate
    warped optical-flow noise). Outputs a temporally consistent refined inpainted video.
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

        self._seed_param.add_input_parameters()

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

        if not self.parameter_values.get("quadmask_video"):
            errors.append(ValueError("quadmask_video is required"))

        if not self.parameter_values.get("pass1_output_video"):
            errors.append(ValueError("pass1_output_video is required"))

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

    def _path_for_cli(self, path: str, repo_dir: str) -> str:
        """Convert path to POSIX-style to avoid Windows drive-letter parsing issues."""
        try:
            path_resolved = Path(path).resolve()
            repo_resolved = Path(repo_dir).resolve()
            rel = path_resolved.relative_to(repo_resolved)
            return rel.as_posix()
        except ValueError:
            # Path is not relative to repo_dir, use absolute with forward slashes
            return str(path).replace("\\", "/")

    def process(self) -> AsyncResult[None]:
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        from griptape_nodes.files.file import File
        from huggingface_hub import hf_hub_download, snapshot_download

        self._seed_param.preprocess()
        seed: int = self._seed_param.get_seed()

        base_model_id, _ = self._base_model_param.get_repo_revision()
        void_checkpoint_repo, _ = self._void_checkpoint_param.get_repo_revision()

        logger.info(f"Downloading base model: {base_model_id}")
        base_model_path = snapshot_download(base_model_id)

        logger.info(f"Downloading VOID Pass 2 checkpoint from: {void_checkpoint_repo}")
        void_checkpoint_path = hf_hub_download(void_checkpoint_repo, "void_pass2.safetensors")

        prompt: str = self.parameter_values.get("prompt") or ""
        height: int = self.parameter_values.get("height") or 384
        width: int = self.parameter_values.get("width") or 672
        height = (height // 8) * 8
        width = (width // 8) * 8
        temporal_window_size: int = self.parameter_values.get("temporal_window_size") or 85
        num_inference_steps: int = self.parameter_values.get("num_inference_steps") or 50
        guidance_scale: float = self.parameter_values.get("guidance_scale") or 6.0

        input_video_artifact = self.parameter_values.get("input_video")
        quadmask_artifact = self.parameter_values.get("quadmask_video")
        pass1_artifact = self.parameter_values.get("pass1_output_video")

        if not isinstance(input_video_artifact, VideoUrlArtifact):
            raise ValueError("input_video is required")
        if not isinstance(quadmask_artifact, VideoUrlArtifact):
            raise ValueError("quadmask_video is required")
        if not isinstance(pass1_artifact, VideoUrlArtifact):
            raise ValueError("pass1_output_video is required")

        input_video_bytes = File(input_video_artifact.value).read_bytes()
        quadmask_bytes = File(quadmask_artifact.value).read_bytes()
        pass1_video_bytes = File(pass1_artifact.value).read_bytes()

        submodule_root = self._get_submodule_root()
        script_path = os.path.join(
            submodule_root, "inference", "cogvideox_fun", "inference_with_pass1_warped_noise.py"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            seq_name = "job0"
            data_root = os.path.join(tmp_dir, "void_data")
            seq_dir = os.path.join(data_root, seq_name)
            pass1_dir = os.path.join(tmp_dir, "pass1_outputs")
            save_dir = os.path.join(tmp_dir, "void_outputs_pass2")
            noise_cache_dir = os.path.join(tmp_dir, "noise_cache")

            for d in [seq_dir, pass1_dir, save_dir, noise_cache_dir]:
                os.makedirs(d, exist_ok=True)

            with open(os.path.join(seq_dir, "input_video.mp4"), "wb") as f:
                f.write(input_video_bytes)

            with open(os.path.join(seq_dir, "quadmask_0.mp4"), "wb") as f:
                f.write(quadmask_bytes)

            with open(os.path.join(seq_dir, "prompt.json"), "w", encoding="utf-8") as f:
                json.dump({"bg": prompt or "clean background"}, f)

            # inference_with_pass1_warped_noise.py looks for {video_name}-fg=-1-*.mp4
            pass1_video_path = os.path.join(pass1_dir, f"{seq_name}-fg=-1-auto.mp4")
            with open(pass1_video_path, "wb") as f:
                f.write(pass1_video_bytes)

            # Convert all paths to POSIX-style to avoid Windows parsing issues
            data_root_cli = self._path_for_cli(data_root, submodule_root)
            pass1_dir_cli = self._path_for_cli(pass1_dir, submodule_root)
            save_dir_cli = self._path_for_cli(save_dir, submodule_root)
            base_model_cli = self._path_for_cli(base_model_path, submodule_root)
            checkpoint_cli = self._path_for_cli(void_checkpoint_path, submodule_root)
            noise_cache_cli = self._path_for_cli(noise_cache_dir, submodule_root)

            cmd = [
                self._get_venv_python(),
                script_path,
                "--video_name", seq_name,
                "--data_rootdir", data_root_cli,
                "--pass1_dir", pass1_dir_cli,
                "--output_dir", save_dir_cli,
                "--model_name", base_model_cli,
                "--model_checkpoint", checkpoint_cli,
                "--height", str(height),
                "--width", str(width),
                "--temporal_window_size", str(temporal_window_size),
                "--num_inference_steps", str(num_inference_steps),
                "--guidance_scale", str(guidance_scale),
                "--seed", str(seed),
                "--warped_noise_cache_dir", noise_cache_cli,
                "--use_quadmask",
            ]

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                submodule_root + os.pathsep + existing_pythonpath
                if existing_pythonpath
                else submodule_root
            )

            # Add ffmpeg to PATH for mediapy (uses static_ffmpeg bundled binary)
            try:
                import static_ffmpeg
                ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
            except (ImportError, FileNotFoundError, OSError) as e:
                logger.warning(f"static_ffmpeg not available, ffmpeg may not be found: {e}")

            logger.info(f"Running VOID Pass 2: {height}x{width}, {temporal_window_size} frames")
            logger.info(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
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

            output_candidates = [
                p for p in glob.glob(os.path.join(save_dir, "**", "*.mp4"), recursive=True)
                if not p.endswith("_tuple.mp4")
            ]

            if not output_candidates:
                raise RuntimeError(f"VOID Pass 2 produced no output video in: {save_dir}")

            output_candidates.sort(key=os.path.getmtime, reverse=True)
            output_path = output_candidates[0]

            with open(output_path, "rb") as f:
                mp4_bytes = f.read()

        filename = f"void_pass2_{uuid.uuid4().hex[:8]}.mp4"
        url = GriptapeNodes.StaticFilesManager().save_static_file(mp4_bytes, filename)
        self.parameter_output_values["output_video"] = VideoUrlArtifact(url)
        logger.info("VOID Pass 2 inference complete")
