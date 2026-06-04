"""Run raw COLMAP sparse reconstruction for generated pinhole images.

The output model is COLMAP's arbitrary monocular world. It is not metric until
it is aligned with an external scale source such as poses, LiDAR, or markers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from math import pi, tan
from pathlib import Path
from typing import Any

from PIL import Image

from ep_pinhole.tools.render_pinhole_directory import (
    DEFAULT_FOV_X,
    DEFAULT_FOV_Y,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
)


try:  # Keep inspection usable before pycolmap is installed.
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = Path(
    "/data/itri-3d-reconstruction/itri-3d-reconstruction/pinhole_1780034936312"
)
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class ColmapCamera:
    model: str
    width: int
    height: int
    params: tuple[float, float, float, float]
    fov_x: float | None = None
    fov_y: float | None = None

    @property
    def params_string(self) -> str:
        return ",".join(f"{value:.12g}" for value in self.params)

    def report(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "params": list(self.params),
            "params_order": ["fx", "fy", "cx", "cy"],
            "fov_x": self.fov_x,
            "fov_y": self.fov_y,
            "shared_single_camera": True,
            "intrinsics_refined": False,
        }


def camera_from_fov(width: int, height: int, fov_x: float, fov_y: float) -> ColmapCamera:
    validate_positive_int(width, "width")
    validate_positive_int(height, "height")
    validate_fov(fov_x, "fov_x")
    validate_fov(fov_y, "fov_y")

    fx = (width / 2.0) / tan(fov_x / 2.0)
    fy = (height / 2.0) / tan(fov_y / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return ColmapCamera(
        model="PINHOLE",
        width=width,
        height=height,
        params=(fx, fy, cx, cy),
        fov_x=fov_x,
        fov_y=fov_y,
    )


def camera_from_resolution_guess(width: int, height: int) -> ColmapCamera:
    """Build a generic initial PINHOLE guess for BA to refine."""
    validate_positive_int(width, "width")
    validate_positive_int(height, "height")

    focal = 1.2 * max(width, height)
    return ColmapCamera(
        model="PINHOLE",
        width=width,
        height=height,
        params=(focal, focal, width / 2.0, height / 2.0),
    )


def first_image_resolution(image_dir: Path, image_limit: int | None = None) -> tuple[int, int]:
    images = limited_image_paths(image_paths(image_dir), image_limit)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    with Image.open(images[0]) as image:
        return image.size


def camera_from_args(args: argparse.Namespace) -> ColmapCamera:
    if args.optimize_intrinsics:
        width, height = optimized_camera_resolution(args)
        if args.fov_x is not None or args.fov_y is not None:
            fov_x = DEFAULT_FOV_X if args.fov_x is None else args.fov_x
            fov_y = DEFAULT_FOV_Y if args.fov_y is None else args.fov_y
            return camera_from_fov(width, height, fov_x, fov_y)
        return camera_from_resolution_guess(width, height)

    width = DEFAULT_WIDTH if args.width is None else args.width
    height = DEFAULT_HEIGHT if args.height is None else args.height
    fov_x = DEFAULT_FOV_X if args.fov_x is None else args.fov_x
    fov_y = DEFAULT_FOV_Y if args.fov_y is None else args.fov_y
    return camera_from_fov(width, height, fov_x, fov_y)


def optimized_camera_resolution(args: argparse.Namespace) -> tuple[int, int]:
    if args.width is not None and args.height is not None:
        return args.width, args.height
    if args.width is not None or args.height is not None:
        raise ValueError(
            "--optimize-intrinsics requires both --width and --height when "
            "overriding image-derived resolution"
        )
    return first_image_resolution(args.image_dir, args.image_limit)


def image_paths(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in IMAGE_EXTENSIONS:
        paths.extend(image_dir.glob(f"*{suffix}"))
        paths.extend(image_dir.glob(f"*{suffix.upper()}"))
    return sorted(set(paths))


def inspect_input(
    image_dir: Path,
    workspace: Path,
    camera: ColmapCamera,
    *,
    image_limit: int | None = None,
    verify_all_resolutions: bool = True,
) -> dict[str, Any]:
    images = limited_image_paths(image_paths(image_dir), image_limit)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    resolutions = image_resolutions(
        images if verify_all_resolutions else images[:20])
    expected_resolution = (camera.width, camera.height)
    mismatches = {
        str(path): list(resolution)
        for path, resolution in resolutions.items()
        if resolution != expected_resolution
    }
    if mismatches:
        examples = list(mismatches.items())[:5]
        raise ValueError(
            "Image resolution does not match camera intrinsics. "
            f"Expected {expected_resolution[0]}x{expected_resolution[1]}, "
            f"examples: {examples}"
        )

    report = {
        "status": "colmap_input_inspected",
        "warning": (
            "This report assumes all pinhole images share one virtual PINHOLE "
            "camera. Confirm the original render FOV before trusting COLMAP "
            "intrinsics."
        ),
        "image_dir": str(image_dir),
        "workspace": str(workspace),
        "input_image_count": len(images),
        "image_limit": image_limit,
        "first_image": images[0].name,
        "last_image": images[-1].name,
        "sample_images": [path.name for path in images[:5]],
        "image_resolution": {
            "width": expected_resolution[0],
            "height": expected_resolution[1],
        },
        "camera": camera.report(),
    }
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "colmap_input_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def image_resolutions(paths: list[Path]) -> dict[Path, tuple[int, int]]:
    resolutions: dict[Path, tuple[int, int]] = {}
    for path in paths:
        with Image.open(path) as image:
            resolutions[path] = image.size
    return resolutions


def limited_image_paths(paths: list[Path], image_limit: int | None) -> list[Path]:
    if image_limit is None:
        return paths
    if image_limit <= 0:
        raise ValueError("image_limit must be positive")
    return paths[:image_limit]


def prepare_workspace(workspace: Path, overwrite: bool) -> dict[str, Path]:
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    paths = {
        "workspace": workspace,
        "database": workspace / "database.db",
        "sparse": workspace / "sparse",
        "best_sparse": workspace / "sparse_best",
        "raw_ply": workspace / "cloud_colmap_raw.ply",
        "report": workspace / "validation_report.json",
        "input_report": workspace / "colmap_input_report.json",
        "limited_images": workspace / "images_subset",
    }
    paths["sparse"].mkdir(parents=True, exist_ok=True)
    return paths


def fail_if_workspace_has_outputs(paths: dict[str, Path]) -> None:
    existing = [
        paths["database"],
        paths["raw_ply"],
        paths["report"],
        paths["best_sparse"],
        paths["limited_images"],
    ]
    if any(path.exists() for path in existing):
        raise FileExistsError(
            f"{paths['workspace']} already contains COLMAP outputs. "
            "Pass --overwrite to rebuild it."
        )


def run_reconstruction(
    image_dir: Path,
    workspace: Path,
    camera: ColmapCamera,
    *,
    matching: str = "sequential",
    overlap: int = 15,
    max_num_features: int = 8192,
    image_limit: int | None = None,
    optimize_intrinsics: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    pycolmap_module = require_pycolmap()
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
        camera=camera,
        image_limit=image_limit,
    )
    reconstruction_image_dir = image_dir
    if image_limit is not None:
        reconstruction_image_dir = prepare_limited_image_dir(
            images,
            paths["limited_images"],
        )

    reader_options = pycolmap_module.ImageReaderOptions()
    reader_options.camera_model = camera.model
    reader_options.camera_params = camera.params_string

    extraction_options = pycolmap_module.FeatureExtractionOptions()
    extraction_options.use_gpu = False
    extraction_options.sift.max_num_features = max_num_features

    matching_options = pycolmap_module.FeatureMatchingOptions()
    matching_options.use_gpu = False

    pairing_options = pycolmap_module.SequentialPairingOptions()
    pairing_options.overlap = overlap

    mapper_options = pycolmap_module.IncrementalPipelineOptions()
    mapper_options.ba_refine_focal_length = optimize_intrinsics
    mapper_options.ba_refine_principal_point = optimize_intrinsics
    mapper_options.ba_refine_extra_params = False
    mapper_options.ba_use_gpu = False
    mapper_options.extract_colors = True
    mapper_options.mapper.abs_pose_refine_focal_length = optimize_intrinsics
    mapper_options.mapper.abs_pose_refine_extra_params = False

    pycolmap_module.extract_features(
        paths["database"],
        reconstruction_image_dir,
        camera_mode=pycolmap_module.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap_module.Device.cpu,
    )
    if matching == "sequential":
        pycolmap_module.match_sequential(
            paths["database"],
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=pycolmap_module.Device.cpu,
        )
    elif matching == "exhaustive":
        pycolmap_module.match_exhaustive(
            paths["database"],
            matching_options=matching_options,
            device=pycolmap_module.Device.cpu,
        )
    else:
        raise ValueError("matching must be 'sequential' or 'exhaustive'")
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
    if paths["best_sparse"].exists():
        shutil.rmtree(paths["best_sparse"])
    paths["best_sparse"].mkdir(parents=True, exist_ok=True)
    best_reconstruction.write(paths["best_sparse"])
    best_reconstruction.export_PLY(paths["raw_ply"])

    report = build_report(
        image_dir=image_dir,
        reconstruction_image_dir=reconstruction_image_dir,
        paths=paths,
        camera=camera,
        image_count=len(images),
        total_available_image_count=len(all_images),
        matching=matching,
        overlap=overlap,
        max_num_features=max_num_features,
        image_limit=image_limit,
        optimize_intrinsics=optimize_intrinsics,
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
    )
    paths["report"].write_text(json.dumps(
        report, indent=2) + "\n", encoding="utf-8")
    return report


def prepare_limited_image_dir(images: list[Path], limited_dir: Path) -> Path:
    if limited_dir.exists():
        shutil.rmtree(limited_dir)
    limited_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        target = limited_dir / image.name
        try:
            target.symlink_to(image)
        except OSError:
            shutil.copy2(image, target)
    return limited_dir


def build_report(
    *,
    image_dir: Path,
    reconstruction_image_dir: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    image_count: int,
    total_available_image_count: int,
    matching: str,
    overlap: int,
    max_num_features: int,
    image_limit: int | None,
    optimize_intrinsics: bool,
    reconstructions: dict[int, Any],
    best_model_id: int,
    best_reconstruction: Any,
) -> dict[str, Any]:
    pycolmap_module = require_pycolmap()
    database_stats = read_database_stats(paths["database"])
    model_stats = {
        str(model_id): reconstruction_stats(reconstruction)
        for model_id, reconstruction in sorted(reconstructions.items())
    }
    camera_report = camera.report()
    camera_report["intrinsics_refined"] = optimize_intrinsics
    return {
        "status": "raw_colmap_unscaled",
        "warning": (
            "This is a monocular COLMAP reconstruction in arbitrary COLMAP "
            "scale/frame. Do not use it as metric output until it is aligned "
            "with an external metric reference."
        ),
        "pycolmap_version": pycolmap_module.__version__,
        "image_dir": str(image_dir),
        "reconstruction_image_dir": str(reconstruction_image_dir),
        "workspace": str(paths["workspace"]),
        "database_path": str(paths["database"]),
        "sparse_path": str(paths["sparse"]),
        "best_sparse_path": str(paths["best_sparse"]),
        "raw_ply_path": str(paths["raw_ply"]),
        "input_report_path": str(paths["input_report"]),
        "camera": camera_report,
        "options": {
            "matching": matching,
            "sequential_overlap": overlap,
            "max_num_features": max_num_features,
            "use_gpu": False,
            "image_limit": image_limit,
            "optimize_intrinsics": optimize_intrinsics,
            "shared_single_camera": True,
        },
        "input_image_count": image_count,
        "total_available_image_count": total_available_image_count,
        "database": database_stats,
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": model_stats,
        "best_model": reconstruction_stats(best_reconstruction),
    }


def reconstruction_stats(reconstruction: Any) -> dict[str, Any]:
    return {
        "num_images": reconstruction.num_images(),
        "num_registered_images": reconstruction.num_reg_images(),
        "num_points3D": reconstruction.num_points3D(),
        "num_observations": reconstruction.compute_num_observations(),
        "mean_observations_per_registered_image": (
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_track_length": reconstruction.compute_mean_track_length(),
        "mean_reprojection_error_px": reconstruction.compute_mean_reprojection_error(),
    }


def read_database_stats(database_path: Path) -> dict[str, int]:
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

        def count(table: str) -> int:
            if table not in tables:
                return 0
            return int(cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        return {
            "cameras": count("cameras"),
            "images": count("images"),
            "keypoints": count("keypoints"),
            "descriptors": count("descriptors"),
            "matches": count("matches"),
            "two_view_geometries": count("two_view_geometries"),
        }


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required to run COLMAP reconstruction. Install the "
            "`colmap` extra or add `pycolmap` to the environment."
        )
    return pycolmap


def validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_fov(value: float, name: str) -> None:
    if not 0.0 < value < pi:
        raise ValueError(
            f"{name} must be greater than 0 and less than pi radians")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect or reconstruct generated pinhole images with fixed COLMAP "
            "PINHOLE intrinsics. Output is raw and unscaled."
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
        help=f"COLMAP workspace directory (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Virtual pinhole image width in pixels. Defaults to the renderer "
            f"default ({DEFAULT_WIDTH}) unless --optimize-intrinsics is used."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Virtual pinhole image height in pixels. Defaults to the renderer "
            f"default ({DEFAULT_HEIGHT}) unless --optimize-intrinsics is used."
        ),
    )
    parser.add_argument(
        "--fov-x",
        type=float,
        default=None,
        help=(
            "Virtual pinhole horizontal field of view in radians. Defaults to "
            f"the renderer default ({DEFAULT_FOV_X}) for fixed intrinsics."
        ),
    )
    parser.add_argument(
        "--fov-y",
        type=float,
        default=None,
        help=(
            "Virtual pinhole vertical field of view in radians. Defaults to "
            f"the renderer default ({DEFAULT_FOV_Y}) for fixed intrinsics."
        ),
    )
    parser.add_argument(
        "--matching",
        choices=("sequential", "exhaustive"),
        default="sequential",
        help=(
            "Feature matching strategy. Exhaustive is useful for small smoke "
            "subsets when sequential matching is too sparse."
        ),
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=15,
        help="Sequential matcher overlap in neighboring frames (default: 15)",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="Maximum SIFT features per image (default: 8192)",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=None,
        help="Use only the first N sorted images for smoke testing.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect inputs and write colmap_input_report.json without COLMAP.",
    )
    parser.add_argument(
        "--optimize-intrinsics",
        action="store_true",
        help=(
            "Let COLMAP refine focal length and principal point during mapping. "
            "All images still share one PINHOLE camera."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the COLMAP workspace before running.",
    )
    args = parser.parse_args(argv)
    if not args.image_dir.is_dir():
        parser.error(
            f"--image-dir must be an existing directory: {args.image_dir}")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive")
    if args.fov_x is not None and not 0.0 < args.fov_x < pi:
        parser.error("--fov-x must be greater than 0 and less than pi radians")
    if args.fov_y is not None and not 0.0 < args.fov_y < pi:
        parser.error("--fov-y must be greater than 0 and less than pi radians")
    if args.overlap <= 0:
        parser.error("--overlap must be positive")
    if args.max_num_features <= 0:
        parser.error("--max-num-features must be positive")
    if args.image_limit is not None and args.image_limit <= 0:
        parser.error("--image-limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    camera = camera_from_args(args)

    if args.inspect_only:
        report = inspect_input(
            image_dir=args.image_dir,
            workspace=args.workspace,
            camera=camera,
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
        print(f"first image: {report['first_image']}")
        print(f"last image: {report['last_image']}")
        print(f"camera params: {camera.params_string}")
        return 0

    report = run_reconstruction(
        image_dir=args.image_dir,
        workspace=args.workspace,
        camera=camera,
        matching=args.matching,
        overlap=args.overlap,
        max_num_features=args.max_num_features,
        image_limit=args.image_limit,
        optimize_intrinsics=args.optimize_intrinsics,
        overwrite=args.overwrite,
    )

    best = report["best_model"]
    print(f"workspace: {report['workspace']}")
    print(f"raw sparse cloud: {report['raw_ply_path']}")
    print(f"validation report: {report['workspace']}/validation_report.json")
    print(
        "best model: "
        f"{best['num_registered_images']}/{report['input_image_count']} images, "
        f"{best['num_points3D']} points, "
        f"{best['mean_reprojection_error_px']:.3f}px mean reprojection error"
    )
    print("status: raw COLMAP reconstruction is unscaled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
