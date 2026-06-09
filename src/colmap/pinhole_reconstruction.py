"""Run COLMAP sparse reconstruction for pinhole image sequences."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colmap.camera import (
    ColmapCamera,
    camera_from_intrinsics,
    camera_from_resolution_guess,
)
from colmap.input_inspection import image_paths, inspect_input, limited_image_paths
from colmap.panorama_reconstruct import (
    PipelineExecutionOptions,
    configure_ssl_cert_file,
    parse_gpu_index,
    resolve_pipeline_execution,
    validate_vocab_tree_path,
)
from colmap.pipeline import write_best_model
from colmap.reporting import read_database_stats, reconstruction_stats
from colmap.workspace import (
    fail_if_workspace_has_outputs,
    prepare_limited_image_dir,
    prepare_workspace,
)

try:  # Keep parsing and inspection usable before pycolmap is installed.
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "pinhole"
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-pinhole"


@dataclass(frozen=True)
class PinholeFeatureOptions:
    extractor: str = "sift"
    matcher: str = "sift-bruteforce"
    max_num_features: int = 8192
    guided_matching: bool = False


@dataclass(frozen=True)
class CameraInitialization:
    camera: ColmapCamera
    source: str
    calibration_path: Path | None
    calibration: dict[str, Any] | None
    optimize_intrinsics: bool


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required for pinhole reconstruction. Install the "
            "`colmap` extra or add `pycolmap` to the environment."
        )
    return pycolmap


def camera_initialization_from_args(args: argparse.Namespace) -> CameraInitialization:
    if args.calibration is None:
        width, height = first_limited_image_resolution(args.image_dir, args.image_limit)
        camera = camera_from_resolution_guess(width, height)
        optimize_intrinsics = (
            True if args.optimize_intrinsics is None else args.optimize_intrinsics
        )
        return CameraInitialization(
            camera=camera,
            source="image_resolution_initial_guess",
            calibration_path=None,
            calibration=None,
            optimize_intrinsics=optimize_intrinsics,
        )

    calibration = read_calibration_yaml(args.calibration)
    camera = camera_from_intrinsics(
        calibration["width"],
        calibration["height"],
        calibration["fx"],
        calibration["fy"],
        calibration["cx"],
        calibration["cy"],
    )
    optimize_intrinsics = (
        False if args.optimize_intrinsics is None else args.optimize_intrinsics
    )
    return CameraInitialization(
        camera=camera,
        source="calibration_yaml",
        calibration_path=args.calibration,
        calibration=calibration,
        optimize_intrinsics=optimize_intrinsics,
    )


def first_limited_image_resolution(
    image_dir: Path,
    image_limit: int | None,
) -> tuple[int, int]:
    from colmap.input_inspection import first_image_resolution

    return first_image_resolution(image_dir, image_limit)


def read_calibration_yaml(path: Path) -> dict[str, Any]:
    """Read common pinhole calibration YAML without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    width = find_scalar(text, ("image_width", "width", "camera_width"))
    height = find_scalar(text, ("image_height", "height", "camera_height"))
    matrix = find_camera_matrix(text)
    fx, fy, cx, cy = matrix[0], matrix[4], matrix[2], matrix[5]
    return {
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "camera_matrix": [float(value) for value in matrix],
    }


def find_scalar(text: str, names: tuple[str, ...]) -> float:
    for name in names:
        match = re.search(
            rf"(?m)^\s*{re.escape(name)}\s*:\s*([-+0-9.eE]+)\s*$",
            text,
        )
        if match:
            return float(match.group(1))
    raise ValueError(f"Calibration YAML is missing one of: {', '.join(names)}")


def find_camera_matrix(text: str) -> list[float]:
    matrix_match = re.search(
        r"(?ms)^\s*camera_matrix\s*:.*?(?:^\s*data\s*:\s*\[([^\]]+)\]|"
        r"^\s*rows\s*:\s*3\s*$.*?^\s*cols\s*:\s*3\s*$.*?^\s*data\s*:\s*\[([^\]]+)\])",
        text,
    )
    if matrix_match:
        values = parse_number_list(matrix_match.group(1) or matrix_match.group(2))
        if len(values) == 9:
            return values

    inline_match = re.search(r"(?m)^\s*K\s*:\s*\[([^\]]+)\]\s*$", text)
    if inline_match:
        values = parse_number_list(inline_match.group(1))
        if len(values) == 9:
            return values

    keys = {
        name: find_optional_scalar(text, (name,))
        for name in ("fx", "fy", "cx", "cy")
    }
    if all(value is not None for value in keys.values()):
        return [
            float(keys["fx"]),
            0.0,
            float(keys["cx"]),
            0.0,
            float(keys["fy"]),
            float(keys["cy"]),
            0.0,
            0.0,
            1.0,
        ]
    raise ValueError(
        "Calibration YAML must contain camera_matrix.data, K, or fx/fy/cx/cy."
    )


def find_optional_scalar(text: str, names: tuple[str, ...]) -> float | None:
    try:
        return find_scalar(text, names)
    except ValueError:
        return None


def parse_number_list(value: str) -> list[float]:
    return [float(part) for part in re.split(r"[,\s]+", value.strip()) if part]


def create_feature_extraction_options(
    pycolmap_module: Any,
    options: PinholeFeatureOptions,
    *,
    use_gpu: bool,
    gpu_index: str,
) -> Any:
    if options.extractor != "sift":
        raise ValueError(f"Unsupported feature extractor: {options.extractor}")
    extraction_options = pycolmap_module.FeatureExtractionOptions()
    extraction_options.type = pycolmap_module.FeatureExtractorType.SIFT
    extraction_options.use_gpu = use_gpu
    extraction_options.gpu_index = gpu_index
    extraction_options.sift.max_num_features = options.max_num_features
    return extraction_options


def create_feature_matching_options(
    pycolmap_module: Any,
    options: PinholeFeatureOptions,
    *,
    use_gpu: bool,
    gpu_index: str,
) -> Any:
    if options.matcher != "sift-bruteforce":
        raise ValueError(f"Unsupported feature matcher: {options.matcher}")
    matching_options = pycolmap_module.FeatureMatchingOptions()
    matching_options.type = pycolmap_module.FeatureMatcherType.SIFT_BRUTEFORCE
    matching_options.use_gpu = use_gpu
    matching_options.gpu_index = gpu_index
    matching_options.guided_matching = options.guided_matching
    return matching_options


def create_mapper_options(
    pycolmap_module: Any,
    execution: dict[str, Any],
    *,
    optimize_intrinsics: bool,
) -> Any:
    mapper_options = pycolmap_module.IncrementalPipelineOptions()
    mapper_options.ba_refine_focal_length = optimize_intrinsics
    mapper_options.ba_refine_principal_point = optimize_intrinsics
    mapper_options.ba_refine_extra_params = False
    mapper_options.ba_use_gpu = execution["gpu_bundle_adjustment"]
    mapper_options.ba_gpu_index = execution["bundle_adjustment_gpu_index"]
    mapper_options.extract_colors = True
    mapper_options.mapper.abs_pose_refine_focal_length = optimize_intrinsics
    mapper_options.mapper.abs_pose_refine_extra_params = False
    return mapper_options


def match_features(
    pycolmap_module: Any,
    database_path: Path,
    *,
    matcher: str,
    matching_options: Any,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    device: Any,
) -> None:
    validate_vocab_tree_path(vocab_tree_path)
    if matcher == "sequential":
        pairing_options = pycolmap_module.SequentialPairingOptions()
        pairing_options.overlap = overlap
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


def run_pinhole_reconstruction(
    image_dir: Path,
    workspace: Path,
    *,
    camera_init: CameraInitialization,
    matcher: str = "sequential",
    overlap: int = 15,
    loop_detection: bool = True,
    vocab_tree_path: Path | None = None,
    ssl_cert_file: Path | None = None,
    feature_options: PinholeFeatureOptions = PinholeFeatureOptions(),
    execution_options: PipelineExecutionOptions = PipelineExecutionOptions(),
    image_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    pycolmap_module = require_pycolmap()
    pycolmap_module.set_random_seed(0)
    configure_ssl_cert_file(ssl_cert_file)
    execution = resolve_pipeline_execution(pycolmap_module, execution_options)

    all_images = image_paths(image_dir)
    images = limited_image_paths(all_images, image_limit)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    paths = prepare_workspace(workspace, overwrite)
    if not overwrite:
        fail_if_workspace_has_outputs(paths)
    inspect_input(
        image_dir=image_dir,
        workspace=workspace,
        camera=camera_init.camera,
        image_limit=image_limit,
    )
    reconstruction_image_dir = image_dir
    if image_limit is not None:
        reconstruction_image_dir = prepare_limited_image_dir(
            images,
            paths["limited_images"],
        )

    reader_options = pycolmap_module.ImageReaderOptions()
    reader_options.camera_model = camera_init.camera.model
    reader_options.camera_params = camera_init.camera.params_string
    extraction_options = create_feature_extraction_options(
        pycolmap_module,
        feature_options,
        use_gpu=execution["resolved_device"] == "cuda",
        gpu_index=execution["gpu_index"],
    )
    pycolmap_module.extract_features(
        paths["database"],
        reconstruction_image_dir,
        camera_mode=pycolmap_module.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=execution["device"],
    )

    matching_options = create_feature_matching_options(
        pycolmap_module,
        feature_options,
        use_gpu=execution["resolved_device"] == "cuda",
        gpu_index=execution["gpu_index"],
    )
    match_features(
        pycolmap_module,
        paths["database"],
        matcher=matcher,
        matching_options=matching_options,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        device=execution["device"],
    )

    mapper_options = create_mapper_options(
        pycolmap_module,
        execution,
        optimize_intrinsics=camera_init.optimize_intrinsics,
    )
    reconstructions = pycolmap_module.incremental_mapping(
        paths["database"],
        reconstruction_image_dir,
        paths["sparse"],
        options=mapper_options,
    )
    if not reconstructions:
        raise RuntimeError("COLMAP did not produce any sparse reconstruction")

    best_model_id, best_reconstruction = max(
        reconstructions.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
    )
    write_best_model(best_reconstruction, paths)
    report = build_pinhole_report(
        pycolmap_version=pycolmap_module.__version__,
        image_dir=image_dir,
        reconstruction_image_dir=reconstruction_image_dir,
        paths=paths,
        camera_init=camera_init,
        image_count=len(images),
        total_available_image_count=len(all_images),
        matcher=matcher,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        ssl_cert_file=ssl_cert_file,
        feature_options=feature_options,
        execution=execution,
        image_limit=image_limit,
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
    )
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_pinhole_report(
    *,
    pycolmap_version: str,
    image_dir: Path,
    reconstruction_image_dir: Path,
    paths: dict[str, Path],
    camera_init: CameraInitialization,
    image_count: int,
    total_available_image_count: int,
    matcher: str,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    ssl_cert_file: Path | None,
    feature_options: PinholeFeatureOptions,
    execution: dict[str, Any],
    image_limit: int | None,
    reconstructions: dict[int, Any],
    best_model_id: int,
    best_reconstruction: Any,
) -> dict[str, Any]:
    model_stats = {
        str(model_id): reconstruction_stats(reconstruction)
        for model_id, reconstruction in sorted(reconstructions.items())
    }
    camera_report = camera_init.camera.report()
    camera_report["intrinsics_refined"] = camera_init.optimize_intrinsics
    return {
        "status": "raw_colmap_pinhole_unscaled",
        "warning": (
            "This is a monocular COLMAP reconstruction in arbitrary COLMAP "
            "scale/frame. The image sequence uses one shared PINHOLE camera, "
            "but the resulting world is not metric without an external scale."
        ),
        "pycolmap_version": pycolmap_version,
        "image_dir": str(image_dir),
        "reconstruction_image_dir": str(reconstruction_image_dir),
        "workspace": str(paths["workspace"]),
        "database_path": str(paths["database"]),
        "sparse_path": str(paths["sparse"]),
        "best_sparse_path": str(paths["best_sparse"]),
        "raw_ply_path": str(paths["raw_ply"]),
        "input_report_path": str(paths["input_report"]),
        "camera": camera_report,
        "camera_initialization": {
            "source": camera_init.source,
            "calibration_path": None
            if camera_init.calibration_path is None
            else str(camera_init.calibration_path),
            "calibration": camera_init.calibration,
        },
        "options": {
            "matcher": matcher,
            "sequential_overlap": overlap,
            "loop_detection": loop_detection,
            "vocab_tree_path": None
            if vocab_tree_path is None
            else str(vocab_tree_path),
            "ssl_cert_file": (
                os.environ.get("SSL_CERT_FILE")
                if ssl_cert_file is None
                else str(ssl_cert_file)
            ),
            "feature_extractor": feature_options.extractor,
            "feature_matcher": feature_options.matcher,
            "max_num_features": feature_options.max_num_features,
            "guided_matching": feature_options.guided_matching,
            "image_limit": image_limit,
            "requested_device": execution["requested_device"],
            "resolved_device": execution["resolved_device"],
            "cuda_available": execution["cuda_available"],
            "gpu_index": execution["gpu_index"],
            "gpu_bundle_adjustment": execution["gpu_bundle_adjustment"],
            "bundle_adjustment_gpu_index": execution["bundle_adjustment_gpu_index"],
            "camera_mode": "SINGLE",
            "ba_refine_focal_length": camera_init.optimize_intrinsics,
            "ba_refine_principal_point": camera_init.optimize_intrinsics,
            "ba_refine_extra_params": False,
            "shared_single_camera": True,
        },
        "input_image_count": image_count,
        "total_available_image_count": total_available_image_count,
        "database": read_database_stats(paths["database"]),
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": model_stats,
        "best_model": reconstruction_stats(best_reconstruction),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run COLMAP reconstruction for pinhole images with one shared "
            "PINHOLE camera. A calibration YAML can fix or initialize "
            "intrinsics; without one, COLMAP refines a generic initial guess."
        )
    )
    parser.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Pinhole image directory (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"COLMAP pinhole workspace (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "-c",
        "--calibration",
        type=Path,
        default=None,
        help=(
            "Optional camera calibration YAML with image width/height and "
            "camera_matrix.data or fx/fy/cx/cy."
        ),
    )
    parser.add_argument(
        "--optimize-intrinsics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Refine focal length and principal point. Defaults to true without "
            "calibration and false with calibration."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect inputs and write colmap_input_report.json without COLMAP.",
    )
    parser.add_argument(
        "--matcher",
        choices=("sequential", "exhaustive", "vocabtree"),
        default="sequential",
        help="Feature matching strategy.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=15,
        help="Sequential matcher overlap in neighboring frames.",
    )
    parser.add_argument(
        "--loop-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vocabulary-tree loop detection during sequential matching.",
    )
    parser.add_argument(
        "--vocab-tree-path",
        type=Path,
        default=None,
        help=(
            "Local COLMAP vocabulary tree for loop detection or vocabtree "
            "matching. If omitted, COLMAP may download its default tree."
        ),
    )
    parser.add_argument(
        "--ssl-cert-file",
        type=Path,
        default=None,
        help=(
            "CA certificate bundle for COLMAP downloads. Defaults to "
            "$SSL_CERT_FILE or /etc/ssl/certs/ca-certificates.crt."
        ),
    )
    parser.add_argument(
        "--feature-extractor",
        choices=("sift",),
        default="sift",
        help="Local feature extractor. The pipeline is structured for future backends.",
    )
    parser.add_argument(
        "--feature-matcher",
        choices=("sift-bruteforce",),
        default="sift-bruteforce",
        help="Local feature matcher. The pipeline is structured for future backends.",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="Maximum SIFT features per image.",
    )
    parser.add_argument(
        "--guided-matching",
        action="store_true",
        help="Enable guided feature matching after geometric verification.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="COLMAP compute device. Auto uses CUDA when pycolmap supports it.",
    )
    parser.add_argument(
        "--gpu-index",
        type=parse_gpu_index,
        default="-1",
        help=(
            "CUDA GPU index, or comma-separated indices for multi-GPU feature "
            "extraction and matching (default: -1, all GPUs)."
        ),
    )
    parser.add_argument(
        "--gpu-bundle-adjustment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use GPU bundle adjustment when CUDA is selected. Bundle adjustment "
            "uses the first selected GPU."
        ),
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=None,
        help="Use only the first N sorted images for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the COLMAP workspace.",
    )
    args = parser.parse_args(argv)
    if not args.image_dir.is_dir():
        parser.error(f"--image-dir must be an existing directory: {args.image_dir}")
    if args.calibration is not None and not args.calibration.is_file():
        parser.error(f"--calibration must be an existing file: {args.calibration}")
    if args.overlap <= 0:
        parser.error("--overlap must be positive")
    if args.max_num_features <= 0:
        parser.error("--max-num-features must be positive")
    if args.image_limit is not None and args.image_limit <= 0:
        parser.error("--image-limit must be positive")
    if args.vocab_tree_path is not None and not args.vocab_tree_path.is_file():
        parser.error(f"--vocab-tree-path does not exist: {args.vocab_tree_path}")
    if args.ssl_cert_file is not None and not args.ssl_cert_file.is_file():
        parser.error(f"--ssl-cert-file does not exist: {args.ssl_cert_file}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    camera_init = camera_initialization_from_args(args)

    if args.inspect_only:
        report = inspect_input(
            image_dir=args.image_dir,
            workspace=args.workspace,
            camera=camera_init.camera,
            image_limit=args.image_limit,
        )
        print(f"workspace: {report['workspace']}")
        print(f"input report: {args.workspace / 'colmap_input_report.json'}")
        print(
            "images: "
            f"{report['input_image_count']} at "
            f"{report['image_resolution']['width']}x"
            f"{report['image_resolution']['height']}"
        )
        print(f"camera source: {camera_init.source}")
        print(f"camera params: {camera_init.camera.params_string}")
        print(f"optimize intrinsics: {camera_init.optimize_intrinsics}")
        return 0

    report = run_pinhole_reconstruction(
        image_dir=args.image_dir,
        workspace=args.workspace,
        camera_init=camera_init,
        matcher=args.matcher,
        overlap=args.overlap,
        loop_detection=args.loop_detection,
        vocab_tree_path=args.vocab_tree_path,
        ssl_cert_file=args.ssl_cert_file,
        feature_options=PinholeFeatureOptions(
            extractor=args.feature_extractor,
            matcher=args.feature_matcher,
            max_num_features=args.max_num_features,
            guided_matching=args.guided_matching,
        ),
        execution_options=PipelineExecutionOptions(
            device=args.device,
            gpu_index=args.gpu_index,
            gpu_bundle_adjustment=args.gpu_bundle_adjustment,
        ),
        image_limit=args.image_limit,
        overwrite=args.overwrite,
    )
    best = report["best_model"]
    print(f"images: {report['input_image_count']}")
    print(f"compute device: {report['options']['resolved_device']}")
    print(f"camera source: {report['camera_initialization']['source']}")
    print(f"optimize intrinsics: {report['options']['ba_refine_focal_length']}")
    print(f"best registered images: {best['num_registered_images']}")
    print(f"best sparse points: {best['num_points3D']}")
    print(f"raw sparse cloud: {report['raw_ply_path']}")
    print(f"report: {report['workspace']}/validation_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
