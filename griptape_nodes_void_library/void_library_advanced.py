import logging
import subprocess
import sys
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("void_library")


class VoidLibraryAdvanced(AdvancedNodeLibrary):
    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Loading '{library_data.name}' library...")
        # The work below populates the execution environment, which only the worker
        # imports, so the orchestrator must not run it.
        if not GriptapeNodes.LibraryManager().is_worker:
            return
        self._init_submodule()
        self._install_commonsource()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        logger.info(f"Finished loading '{library_data.name}' library")

    def _get_library_root(self) -> Path:
        return Path(__file__).parent

    def _get_venv_python_path(self) -> Path:
        """Interpreter of the execution environment the engine builds beside the manifest.

        `rp` is an execution dependency, so the CommonSource clone below has to be placed in
        `.venv-exec` rather than in the edit-time environment, which never holds it.
        """
        root = self._get_library_root()
        if sys.platform == "win32":
            return root / ".venv-exec" / "Scripts" / "python.exe"
        return root / ".venv-exec" / "bin" / "python"

    def _init_submodule(self) -> Path:
        library_root = self._get_library_root()
        submodule_dir = library_root / "void-model"
        if submodule_dir.exists() and any(submodule_dir.iterdir()):
            logger.info("Submodule already initialized")
            return submodule_dir
        # The git CLI rather than pygit2: the engine dropped pygit2 (its bundled TLS trust
        # store breaks on some platforms) and requires git on PATH, so it is the one tool
        # guaranteed to be here.
        subprocess.check_call(["git", "-C", str(library_root.parent), "submodule", "update", "--init", "--recursive"])
        if not submodule_dir.exists() or not any(submodule_dir.iterdir()):
            raise RuntimeError(f"Submodule init failed: {submodule_dir}")
        logger.info("Submodule initialized successfully")
        return submodule_dir

    def _install_commonsource(self) -> None:
        """Clone CommonSource into rp's git directory for Pass 2 warped noise generation.

        The rp package's git_import fails on Windows due to path issues, so we pre-clone
        the repo to the location rp expects: <rp_package_dir>/git/CommonSource
        """
        venv_python = self._get_venv_python_path()
        # Get rp's install location
        result = subprocess.run(
            [str(venv_python), "-c", "import rp; print(rp.__path__[0])"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning("Could not find rp package location, skipping CommonSource install")
            return
        rp_path = Path(result.stdout.strip())
        commonsource_path = rp_path / "git" / "CommonSource"
        if commonsource_path.exists() and any(commonsource_path.iterdir()):
            logger.info("CommonSource already installed")
            self._patch_commonsource_for_windows(commonsource_path)
            return
        # Create parent directory and clone
        commonsource_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cloning CommonSource to {commonsource_path}...")
        subprocess.check_call(["git", "clone", "https://github.com/RyannDaGreat/CommonSource", str(commonsource_path)])
        self._patch_commonsource_for_windows(commonsource_path)
        logger.info("CommonSource installed successfully")

    def _patch_commonsource_for_windows(self, commonsource_path: Path) -> None:
        """Patch noise_warp.py to fix Windows-specific issues.

        1. rp.select_torch_device() crashes on Windows because print_gpu_summary() tries
           to query process usernames which fails with AccessDenied for system processes.
           We patch it to use torch's native CUDA detection instead.

        2. rp.save_video_mp4() with video_bitrate='max' overflows on Windows (32-bit C long).
           We patch it to use the imageio backend instead.
        """
        if sys.platform != "win32":
            return
        noise_warp_path = commonsource_path / "noise_warp.py"
        if not noise_warp_path.exists():
            return
        content = noise_warp_path.read_text(encoding="utf-8")
        modified = False

        # Patch 1: GPU detection
        old_patterns = [
            "device = rp.select_torch_device(prefer_used=True)",
            "device = rp.select_torch_device(prefer_used=False)  # patched for Windows",
        ]
        new_line = "device = 'cuda' if torch.cuda.is_available() else 'cpu'  # patched for Windows"
        for old_line in old_patterns:
            if old_line in content:
                content = content.replace(old_line, new_line)
                modified = True
                break

        # Patch 2: Video encoding - use imageio backend to avoid bitrate overflow
        # The original has 'video_bitrate="max",' with a trailing comma
        old_bitrate = 'video_bitrate="max",'
        new_bitrate = 'video_bitrate="max", backend="imageio",  # patched for Windows'
        if old_bitrate in content and 'backend="imageio"' not in content:
            content = content.replace(old_bitrate, new_bitrate)
            modified = True

        if modified:
            noise_warp_path.write_text(content, encoding="utf-8")
            logger.info("Patched noise_warp.py for Windows compatibility")

        # Also patch make_warped_noise.py in the submodule
        self._patch_make_warped_noise_for_windows()

    def _patch_make_warped_noise_for_windows(self) -> None:
        """Patch make_warped_noise.py in void-model submodule to fix video encoding on Windows."""
        if sys.platform != "win32":
            return
        make_warped_noise_path = (
            self._get_library_root() / "void-model" / "inference" / "cogvideox_fun" / "make_warped_noise.py"
        )
        if not make_warped_noise_path.exists():
            return
        content = make_warped_noise_path.read_text(encoding="utf-8")
        # Patch video_bitrate='max' to add backend='imageio'
        old_line = (
            "rp.save_video_mp4(video, rp.path_join(output_folder, 'input.mp4'), framerate=12, video_bitrate='max')"
        )
        new_line = "rp.save_video_mp4(video, rp.path_join(output_folder, 'input.mp4'), framerate=12, video_bitrate='max', backend='imageio')  # patched for Windows"
        if old_line in content and new_line not in content:
            content = content.replace(old_line, new_line)
            make_warped_noise_path.write_text(content, encoding="utf-8")
            logger.info("Patched make_warped_noise.py for Windows compatibility")
