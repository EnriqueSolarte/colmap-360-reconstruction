"""pycolmap boundary helpers for panorama reconstruction."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from colmap.panorama.keyframe_matching import KeyframeSelectionOptions
from colmap.panorama.types import FeaturePipelineOptions, PipelineExecutionOptions

try:  # Keep argument parsing and tests usable before pycolmap is installed.
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required for panorama reconstruction. Install the "
            "`colmap` extra or add `pycolmap` to the environment."
        )
    return pycolmap


def configure_ssl_cert_file(ssl_cert_file: Path | None = None) -> None:
    """Configure the CA bundle used by COLMAP/libcurl for optional downloads."""
    if ssl_cert_file is not None:
        if not ssl_cert_file.is_file():
            raise FileNotFoundError(f"SSL certificate bundle not found: {ssl_cert_file}")
        os.environ["SSL_CERT_FILE"] = str(ssl_cert_file)
        return

    if os.environ.get("SSL_CERT_FILE"):
        return

    default_cert_file = Path("/etc/ssl/certs/ca-certificates.crt")
    if default_cert_file.is_file():
        os.environ["SSL_CERT_FILE"] = str(default_cert_file)


def write_best_model(best_reconstruction: Any, paths: dict[str, Path]) -> None:
    if paths["best_sparse"].exists():
        shutil.rmtree(paths["best_sparse"])
    paths["best_sparse"].mkdir(parents=True, exist_ok=True)
    best_reconstruction.write(paths["best_sparse"])
    best_reconstruction.export_PLY(paths["raw_ply"])


def create_feature_extraction_options(
    pycolmap_module: Any,
    options: FeaturePipelineOptions,
    *,
    use_gpu: bool = False,
    gpu_index: str = "-1",
) -> Any:
    if options.extractor != "sift":
        raise ValueError(f"Unsupported feature extractor: {options.extractor}")
    extraction_options = pycolmap_module.FeatureExtractionOptions()
    extraction_options.type = pycolmap_module.FeatureExtractorType.SIFT
    extraction_options.use_gpu = use_gpu
    extraction_options.gpu_index = gpu_index
    extraction_options.max_image_size = options.max_image_size
    extraction_options.num_threads = options.num_threads
    extraction_options.sift.max_num_features = options.max_num_features
    return extraction_options


def create_feature_matching_options(
    pycolmap_module: Any,
    options: FeaturePipelineOptions,
    *,
    use_gpu: bool = False,
    gpu_index: str = "-1",
) -> Any:
    if options.matcher != "sift-bruteforce":
        raise ValueError(f"Unsupported feature matcher: {options.matcher}")
    matching_options = pycolmap_module.FeatureMatchingOptions()
    matching_options.type = pycolmap_module.FeatureMatcherType.SIFT_BRUTEFORCE
    matching_options.use_gpu = use_gpu
    matching_options.gpu_index = gpu_index
    matching_options.guided_matching = options.guided_matching
    matching_options.rig_verification = False
    matching_options.skip_image_pairs_in_same_frame = False
    return matching_options


def create_mapper_options(
    pycolmap_module: Any,
    execution: dict[str, Any],
) -> Any:
    mapper_options = pycolmap_module.IncrementalPipelineOptions()
    mapper_options.ba_refine_sensor_from_rig = False
    mapper_options.ba_refine_focal_length = False
    mapper_options.ba_refine_principal_point = False
    mapper_options.ba_refine_extra_params = False
    mapper_options.ba_use_gpu = execution["gpu_bundle_adjustment"]
    mapper_options.ba_gpu_index = execution["bundle_adjustment_gpu_index"]
    return mapper_options


def validate_vocab_tree_path(vocab_tree_path: Path | None) -> None:
    if vocab_tree_path is not None and not vocab_tree_path.is_file():
        raise FileNotFoundError(f"Vocabulary tree not found: {vocab_tree_path}")


def resolve_pipeline_execution(
    pycolmap_module: Any,
    options: PipelineExecutionOptions,
) -> dict[str, Any]:
    if options.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    validate_gpu_index(options.gpu_index)

    cuda_available = bool(pycolmap_module.has_cuda)
    if options.device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but this pycolmap build has no CUDA support. "
            "Replace the CPU wheel with the CUDA wheel using "
            "`python -m pip uninstall -y pycolmap` followed by "
            "`python -m pip install -e '.[colmap-cuda12]'`, or install another "
            "CUDA-enabled pycolmap build."
        )
    use_gpu = cuda_available and options.device != "cpu"
    resolved_device_name = "cuda" if use_gpu else "cpu"
    return {
        "requested_device": options.device,
        "resolved_device": resolved_device_name,
        "device": getattr(pycolmap_module.Device, resolved_device_name),
        "cuda_available": cuda_available,
        "gpu_index": options.gpu_index,
        "gpu_bundle_adjustment": use_gpu and options.gpu_bundle_adjustment,
        "bundle_adjustment_gpu_index": first_gpu_index(options.gpu_index),
    }


def validate_gpu_index(value: str) -> None:
    if value == "-1":
        return
    parts = value.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(
            "gpu_index must be -1 or comma-separated non-negative indices"
        )


def parse_gpu_index(value: str) -> str:
    try:
        validate_gpu_index(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if value == "-1":
        return value
    return ",".join(str(int(part)) for part in value.split(","))


def first_gpu_index(value: str) -> str:
    return value if value == "-1" else value.split(",", maxsplit=1)[0]


def match_features(
    pycolmap_module: Any,
    database_path: Path,
    *,
    matcher: str,
    matching_options: Any,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    image_pairs_path: Path | None = None,
    device: Any | None = None,
) -> None:
    validate_vocab_tree_path(vocab_tree_path)
    if device is None:
        device = pycolmap_module.Device.cpu
    if image_pairs_path is not None:
        pairing_options = pycolmap_module.ImportedPairingOptions()
        pairing_options.match_list_path = image_pairs_path
        pycolmap_module.match_image_pairs(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
        return
    if matcher == "sequential":
        pairing_options = pycolmap_module.SequentialPairingOptions()
        pairing_options.overlap = overlap
        pairing_options.expand_rig_images = False
        pairing_options.loop_detection = loop_detection
        if vocab_tree_path is not None:
            pairing_options.vocab_tree_path = vocab_tree_path
        pycolmap_module.match_sequential(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    elif matcher == "exhaustive":
        pycolmap_module.match_exhaustive(
            database_path,
            matching_options=matching_options,
            device=device,
        )
    elif matcher == "vocabtree":
        pairing_options = pycolmap_module.VocabTreePairingOptions()
        if vocab_tree_path is not None:
            pairing_options.vocab_tree_path = vocab_tree_path
        pycolmap_module.match_vocabtree(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    else:
        raise ValueError("matcher must be 'sequential', 'exhaustive', or 'vocabtree'")


def build_vocabtree_candidate_pairs(
    pycolmap_module: Any,
    database_path: Path,
    image_ids: list[int],
    options: KeyframeSelectionOptions,
    *,
    vocab_tree_path: Path | None,
) -> tuple[tuple[int, int], ...]:
    validate_vocab_tree_path(vocab_tree_path)
    pairing_options = pycolmap_module.VocabTreePairingOptions()
    pairing_options.num_nearest_neighbors = options.vocabtree_neighbors
    pairing_options.num_checks = options.vocabtree_checks
    if vocab_tree_path is not None:
        pairing_options.vocab_tree_path = vocab_tree_path

    database = pycolmap_module.Database.open(database_path)
    try:
        try:
            pair_generator = pycolmap_module.VocabTreePairGenerator(
                pairing_options,
                database,
                image_ids,
            )
        except ValueError as error:
            if "Failed to read faiss index" not in str(error):
                raise
            raise ValueError(
                "COLMAP could not read the vocabulary tree as a FAISS index. "
                "Your installed pycolmap/COLMAP expects the newer FAISS "
                "vocabulary-tree format, but the provided tree appears to be "
                "a legacy FLANN-format tree. "
                f"vocab_tree_path={vocab_tree_path}. "
                "Use a FAISS-compatible tree for this COLMAP build, upgrade "
                "the legacy tree with COLMAP's vocab_tree_upgrader, or run "
                "with --keyframe-selection interval until a compatible tree "
                "is available."
            ) from error
        return tuple(
            (int(image_id1), int(image_id2))
            for image_id1, image_id2 in pair_generator.all_pairs()
            if int(image_id1) != int(image_id2)
        )
    finally:
        close = getattr(database, "close", None)
        if close is not None:
            close()


def run_incremental_mapping(
    pycolmap_module: Any,
    paths: dict[str, Path],
    *,
    image_dir: Path,
    mapper_options: Any,
    input_path: Path | None = None,
) -> dict[int, Any]:
    kwargs: dict[str, Any] = {}
    if input_path is not None:
        kwargs["input_path"] = input_path
    return pycolmap_module.incremental_mapping(
        paths["database"],
        image_dir,
        paths["sparse"],
        options=mapper_options,
        **kwargs,
    )
