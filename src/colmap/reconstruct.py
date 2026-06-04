"""Run raw COLMAP sparse reconstruction for generated pinhole images.

The output model is COLMAP's arbitrary monocular world. It is not metric until
it is aligned with an external scale source such as poses, LiDAR, or markers.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import pi
from pathlib import Path
from typing import Any

from colmap.camera import (
    ColmapCamera,
    camera_from_fov,
    camera_from_resolution_guess,
)
from colmap.input_inspection import (
    first_image_resolution,
    image_paths,
    inspect_input,
    limited_image_paths,
)
from colmap.pipeline import run_colmap_pipeline
from colmap.reporting import build_report
from colmap.workspace import (
    fail_if_workspace_has_outputs,
    prepare_limited_image_dir,
    prepare_workspace,
)
from ep_pinhole.defaults import (
    DEFAULT_PINHOLE_FOV_X,
    DEFAULT_PINHOLE_FOV_Y,
    DEFAULT_PINHOLE_HEIGHT,
    DEFAULT_PINHOLE_WIDTH,
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
DEFAULT_WIDTH = DEFAULT_PINHOLE_WIDTH
DEFAULT_HEIGHT = DEFAULT_PINHOLE_HEIGHT
DEFAULT_FOV_X = DEFAULT_PINHOLE_FOV_X
DEFAULT_FOV_Y = DEFAULT_PINHOLE_FOV_Y


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

    pipeline_result = run_colmap_pipeline(
        pycolmap_module=pycolmap_module,
        reconstruction_image_dir=reconstruction_image_dir,
        paths=paths,
        camera=camera,
        matching=matching,
        overlap=overlap,
        max_num_features=max_num_features,
        optimize_intrinsics=optimize_intrinsics,
    )

    report = build_report(
        pycolmap_version=pycolmap_module.__version__,
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
        reconstructions=pipeline_result.reconstructions,
        best_model_id=pipeline_result.best_model_id,
        best_reconstruction=pipeline_result.best_reconstruction,
    )
    paths["report"].write_text(json.dumps(
        report, indent=2) + "\n", encoding="utf-8")
    return report


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required to run COLMAP reconstruction. Install the "
            "`colmap` extra or add `pycolmap` to the environment."
        )
    return pycolmap


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
