"""HECTOR public package API.

On macOS, an available MLX installation selects the MLX backend. Otherwise,
TensorFlow is imported during package initialization. Other dependencies are
loaded when their corresponding APIs are first accessed.

The TensorFlow backend enables GPU memory growth by default; opt out with
``HECTOR_TF_MEMORY_GROWTH=false``. An explicitly set
``TF_FORCE_GPU_ALLOW_GROWTH`` value is preserved.

TF32 matrix multiplication is disabled by default to avoid its reduced
precision. Set ``HECTOR_TF32=1`` to enable it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

__all__ = [
    "HECTOR",
    "InferenceConfig",
    "predict_from_anndata",
    "load_cellranger_data",
    "detect_doublets",
    "diagnose_cellranger_directory",
    "filter_cells",
    "map_gene_symbols",
    "HUMAN_MITO_GENE_IDS",
    "MOUSE_MITO_GENE_IDS",
    "HierarchicalTrajectoryAnalyzer",
    "TrajectoryAnalysisConfig",
    "create_trajectory_analysis",
    "create_trajectory_comparison",
    "create_pseudotime_analysis",
    "plot_pseudotime",
    "create_gene_trend_analysis",
    "plot_gene_trends",
    "simplify_ontology_tree",
    "plot_simplified_tree",
    "discover_inferred_paths",
    "VisualizationCache",
    "compute_elastic_depths",
    "extract_visualization_data",
    "download_model",
    "get_cache_dir",
    "get_model_path",
    "read_checkpoint_version",
    "check_for_update",
    "import_model_file",
    "clear_cache",
    "plot_obs_label_density",
]

DEFAULT_CACHE_ENV_VAR = "HECTOR_MODEL_CACHE"

MODEL_REPO_ID = "polligator/HECTOR_Pro"
MODEL_FILENAME = "multi_species_model.h5"
MODEL_REVISION = "main"
MODEL_SHA256: str | None = None  # set to enforce a download checksum


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Create and return the model-cache directory.

    Use cache_dir when supplied, otherwise HECTOR_MODEL_CACHE or the default
    directory under the user's home directory.
    """
    if cache_dir is not None:
        resolved_cache_dir = Path(cache_dir).expanduser()
    else:
        resolved_cache_dir = Path(
            os.environ.get(DEFAULT_CACHE_ENV_VAR, Path.home() / ".cache" / "hector_pro" / "models")
        ).expanduser()

    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    return resolved_cache_dir.resolve()


def compute_sha256(path: str | Path) -> str:
    """Compute a file's SHA-256 digest without loading the full file into memory."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_path(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".json")


def _resolve_cached_model_path(cache_dir: str | Path | None = None) -> Path:
    return (get_cache_dir(cache_dir) / MODEL_FILENAME).resolve()


def _write_download_metadata(
    model_path: Path,
    *,
    commit_sha: str | None = None,
    checkpoint_version: str | None = None,
) -> None:
    metadata = {
        "repo_id": MODEL_REPO_ID,
        "filename": MODEL_FILENAME,
        "revision": MODEL_REVISION,
        "commit_sha": commit_sha,
        "checkpoint_version": checkpoint_version,
    }
    _metadata_path(model_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _validate_checksum(model_path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return

    observed_sha256 = compute_sha256(model_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Checksum mismatch for '{model_path}'. "
            f"Expected {expected_sha256}, observed {observed_sha256}."
        )


def read_checkpoint_version(model_path: str | Path) -> str | None:
    """Read checkpoint_version without loading the model; return None if missing or unreadable."""
    import h5py
    try:
        with h5py.File(str(model_path), "r") as f:
            version = f.attrs.get("checkpoint_version", None)
            if isinstance(version, bytes):
                version = version.decode("utf-8")
            return version
    except Exception:
        return None


def _get_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "huggingface_hub is required for HECTOR model auto-download. "
            "Install the package dependencies or `pip install huggingface-hub`."
        ) from exc
    return hf_hub_download


def download_model(
    *,
    force_download: bool = False,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    """Download the default checkpoint into the model cache and return its path.

    force_download requests a fresh download; token is forwarded to the model
    host. The configured checksum is validated when set, and checkpoint
    metadata is written alongside the model.
    """
    cached_model_path = _resolve_cached_model_path(cache_dir=cache_dir)
    hf_hub_download = _get_hf_hub_download()

    hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        revision=MODEL_REVISION,
        repo_type="model",
        local_dir=str(cached_model_path.parent),
        force_download=force_download,
        token=token,
    )

    if not cached_model_path.exists():
        raise FileNotFoundError(
            f"Downloaded HECTOR Pro model was not found at {cached_model_path}."
        )

    _validate_checksum(cached_model_path, MODEL_SHA256)

    commit_sha = None
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(MODEL_REPO_ID)
        commit_sha = info.sha
    except Exception:
        pass

    ckpt_version = read_checkpoint_version(cached_model_path)

    _write_download_metadata(
        cached_model_path,
        commit_sha=commit_sha,
        checkpoint_version=ckpt_version,
    )
    return cached_model_path


_UPDATE_CHECK_DONE = False


def _background_update_check(*, cache_dir: str | Path | None = None) -> None:
    """Schedule at most one background update check unless disabled by the environment."""
    global _UPDATE_CHECK_DONE
    if _UPDATE_CHECK_DONE:
        return
    _UPDATE_CHECK_DONE = True

    if os.environ.get("HECTOR_NO_UPDATE_CHECK", "").lower() in ("1", "true", "yes"):
        return

    import threading

    def _check() -> None:
        try:
            result = check_for_update(cache_dir=cache_dir, timeout=5.0)
            if result["update_available"]:
                local = result.get("local_version") or "unknown"
                print(
                    f"\n  ❇️  A newer HECTOR Pro model checkpoint is available "
                    f"(local version: {local}).\n"
                    f"     Run:  hector_pro.download_model(force_download=True)\n"
                )
        except Exception:
            pass

    threading.Thread(target=_check, daemon=True).start()


def get_model_path(
    model_path: str | Path | None = None,
    *,
    auto_download: bool = True,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> Path:
    """Resolve an explicit checkpoint path or the default cached checkpoint.

    An explicit path must exist. For the default model, auto_download controls
    whether a missing checkpoint may be downloaded. Reusing the default cache
    also schedules a background update check; set HECTOR_NO_UPDATE_CHECK=1 to disable it.
    """
    if model_path is not None:
        candidate_path = Path(model_path).expanduser().resolve()
        if not candidate_path.exists():
            raise FileNotFoundError(f"Model checkpoint was not found at {candidate_path}.")
        return candidate_path

    cached_model_path = _resolve_cached_model_path(cache_dir=cache_dir)
    if cached_model_path.exists():
        _validate_checksum(cached_model_path, MODEL_SHA256)
        _background_update_check(cache_dir=cache_dir)
        return cached_model_path

    if not auto_download:
        raise FileNotFoundError(
            f"The HECTOR Pro model is not in the local cache at {cached_model_path}. "
            f"Run hector_pro.download_model() first, or pass auto_download=True."
        )
    return download_model(force_download=False, cache_dir=cache_dir, token=token)


def clear_cache(*, cache_dir: str | Path | None = None) -> None:
    """Delete the locally cached HECTOR Pro model and its metadata."""
    cached_model_path = _resolve_cached_model_path(cache_dir=cache_dir)
    for path in (cached_model_path, _metadata_path(cached_model_path)):
        if path.exists():
            path.unlink()


def check_for_update(
    *,
    cache_dir: str | Path | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """Compare the cached checkpoint commit with the model host's current commit.

    update_available is True when both commit identifiers are known and differ;
    this comparison does not order checkpoint version strings.

    Returns a dict with keys:
        update_available (bool)
        local_version (str | None) — checkpoint_version of the cached model
        local_commit_sha (str | None)
        remote_commit_sha (str | None)
    """
    cached_model_path = _resolve_cached_model_path(cache_dir=cache_dir)
    meta_path = _metadata_path(cached_model_path)

    local_commit_sha = None
    local_version = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            local_commit_sha = meta.get("commit_sha")
            local_version = meta.get("checkpoint_version")
        except Exception:
            pass

    if not local_version and cached_model_path.exists():
        local_version = read_checkpoint_version(cached_model_path)

    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(MODEL_REPO_ID, timeout=timeout)
    remote_commit_sha = info.sha

    update_available = (
        remote_commit_sha is not None
        and local_commit_sha is not None
        and remote_commit_sha != local_commit_sha
    )

    return {
        "update_available": update_available,
        "local_version": local_version,
        "local_commit_sha": local_commit_sha,
        "remote_commit_sha": remote_commit_sha,
    }


def import_model_file(
    source_path: str | Path,
    *,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Import a user-provided .h5 model file into the local cache.

    Returns a dict with:
        checkpoint_version (str | None)
        cache_path (str)
    """
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source model file not found: {source_path}")

    cached_model_path = _resolve_cached_model_path(cache_dir=cache_dir)

    if source_path != cached_model_path:
        shutil.copy2(source_path, cached_model_path)

    ckpt_version = read_checkpoint_version(cached_model_path)

    _write_download_metadata(
        cached_model_path,
        commit_sha=None,
        checkpoint_version=ckpt_version,
    )

    return {
        "checkpoint_version": ckpt_version,
        "cache_path": str(cached_model_path),
    }


# Select the available backend before initializing TensorFlow.
import sys as _sys
if _sys.platform == 'darwin':
    try:
        import mlx.core  # noqa: F401
        _HECTOR_USE_MLX = True
    except ImportError:
        _HECTOR_USE_MLX = False
else:
    _HECTOR_USE_MLX = False

def _resolve_tf_gpu_allocator():
    """Return the allocator to set, or None to leave TF's default.

    Use the stream-ordered CUDA allocator unless HECTOR_TF_ALLOCATOR=default
    or TF_GPU_ALLOCATOR is already set by the caller.
    """
    import os
    if os.environ.get("HECTOR_TF_ALLOCATOR", "").lower() == "default":
        return None
    if os.environ.get("TF_GPU_ALLOCATOR"):
        return None
    return "cuda_malloc_async"


if not _HECTOR_USE_MLX:
    if os.environ.get("HECTOR_TF_MEMORY_GROWTH", "true").lower() != "false":
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    _alloc = _resolve_tf_gpu_allocator()
    if _alloc is not None:
        os.environ.setdefault("TF_GPU_ALLOCATOR", _alloc)
    import tensorflow as tf  # noqa: F401

    # Disable reduced-precision TF32 matrix multiplication by default.
    # Set HECTOR_TF32=1 to enable it.
    if os.environ.get("HECTOR_TF32", "0").lower() not in ("1", "true"):
        try:
            tf.config.experimental.enable_tensor_float_32_execution(False)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lazy-import machinery
# ---------------------------------------------------------------------------

def __getattr__(name: str):
    """Module-level __getattr__ for deferred imports (PEP 562)."""

    # -----------------------------------------------------------------------
    # Predictor API; TensorFlow/Keras are used on the TensorFlow backend.
    # -----------------------------------------------------------------------
    if name in ("HECTOR", "InferenceConfig", "predict_from_anndata"):
        from .predictor import HECTOR, InferenceConfig, predict_from_anndata
        _inject_globals(
            HECTOR=HECTOR,
            InferenceConfig=InferenceConfig,
            predict_from_anndata=predict_from_anndata,
        )
        return globals()[name]

    # -----------------------------------------------------------------------
    # predictor_support.py — Cell Ranger I/O helpers, cell QC filter
    # -----------------------------------------------------------------------
    if name in ("detect_doublets", "load_cellranger_data",
                "diagnose_cellranger_directory",
                "filter_cells", "map_gene_symbols",
                "HUMAN_MITO_GENE_IDS", "MOUSE_MITO_GENE_IDS"):
        from .predictor_support import (
            HUMAN_MITO_GENE_IDS,
            MOUSE_MITO_GENE_IDS,
            detect_doublets,
            diagnose_cellranger_directory,
            filter_cells,
            load_cellranger_data,
            map_gene_symbols,
        )
        _inject_globals(
            HUMAN_MITO_GENE_IDS=HUMAN_MITO_GENE_IDS,
            MOUSE_MITO_GENE_IDS=MOUSE_MITO_GENE_IDS,
            detect_doublets=detect_doublets,
            diagnose_cellranger_directory=diagnose_cellranger_directory,
            filter_cells=filter_cells,
            load_cellranger_data=load_cellranger_data,
            map_gene_symbols=map_gene_symbols,
        )
        return globals()[name]

    # -----------------------------------------------------------------------
    # trajectory.py — public API orchestration
    # -----------------------------------------------------------------------
    if name in (
        "HierarchicalTrajectoryAnalyzer",
        "create_trajectory_analysis",
        "create_trajectory_comparison",
        "create_pseudotime_analysis",
        "plot_pseudotime",
        "create_gene_trend_analysis",
        "plot_gene_trends",
    ):
        from .trajectory import (
            HierarchicalTrajectoryAnalyzer,
            create_trajectory_analysis,
            create_trajectory_comparison,
            create_pseudotime_analysis,
            plot_pseudotime,
            create_gene_trend_analysis,
            plot_gene_trends,
        )
        _inject_globals(
            HierarchicalTrajectoryAnalyzer=HierarchicalTrajectoryAnalyzer,
            create_trajectory_analysis=create_trajectory_analysis,
            create_trajectory_comparison=create_trajectory_comparison,
            create_pseudotime_analysis=create_pseudotime_analysis,
            plot_pseudotime=plot_pseudotime,
            create_gene_trend_analysis=create_gene_trend_analysis,
            plot_gene_trends=plot_gene_trends,
        )
        return globals()[name]

    # -----------------------------------------------------------------------
    # trajectory_ontology.py — config, layout engines, color, simplification
    # -----------------------------------------------------------------------
    if name in (
        "TrajectoryAnalysisConfig",
        "VisualizationCache",
        "discover_inferred_paths",
        "compute_elastic_depths",
        "extract_visualization_data",
        "simplify_ontology_tree",
        "plot_simplified_tree",
    ):
        from .trajectory_ontology import (
            TrajectoryAnalysisConfig,
            VisualizationCache,
            compute_elastic_depths,
            discover_inferred_paths,
            extract_visualization_data,
            plot_simplified_tree,
            simplify_ontology_tree,
        )
        _inject_globals(
            TrajectoryAnalysisConfig=TrajectoryAnalysisConfig,
            VisualizationCache=VisualizationCache,
            compute_elastic_depths=compute_elastic_depths,
            discover_inferred_paths=discover_inferred_paths,
            extract_visualization_data=extract_visualization_data,
            plot_simplified_tree=plot_simplified_tree,
            simplify_ontology_tree=simplify_ontology_tree,
        )
        return globals()[name]

    # -----------------------------------------------------------------------
    # trajectory_render.py — standalone diagnostic plot helpers
    # -----------------------------------------------------------------------
    if name == "plot_obs_label_density":
        from .trajectory_render import plot_obs_label_density
        _inject_globals(
            plot_obs_label_density=plot_obs_label_density,
        )
        return globals()[name]

    raise AttributeError(f"module 'hector_pro' has no attribute {name!r}")


def _inject_globals(**kwargs) -> None:
    """Cache resolved names into the module's global namespace.

    This ensures subsequent accesses (e.g. a second ``hector_pro.HECTOR`` call)
    hit the module dict directly and bypass ``__getattr__`` entirely.
    """
    import sys
    module = sys.modules[__name__]
    for attr, value in kwargs.items():
        setattr(module, attr, value)
