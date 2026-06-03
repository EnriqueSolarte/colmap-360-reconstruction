"""Run a raw COLMAP sparse reconstruction with the challenge camera intrinsics.

This module intentionally stops before metric scale recovery. The output model
is COLMAP's arbitrary monocular world and must be aligned to the ArUco marker
trajectory before being treated as metric challenge output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pycolmap

from starter.camera_calibration import CameraCalibration


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "rgb"
DEFAULT_CALIBRATION = PROJECT_ROOT / "data" / "cfg.yaml"
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap"


@dataclass(frozen=True)
class ColmapCamera:
    model: str
    width: int
    height: int
    params: tuple[float, float, float, float]

    @property
    def params_string(self) -> str:
        return ",".join(f"{value:.12g}" for value in self.params)


def camera_from_calibration(calibration: CameraCalibration) -> ColmapCamera:
    return ColmapCamera(
        model="PINHOLE",
        width=calibration.image_width,
        height=calibration.image_height,
        params=(calibration.fx, calibration.fy,
                calibration.cx, calibration.cy),
    )


def image_paths(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(image_dir.glob(suffix))
    return sorted(paths)


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
    }
    paths["sparse"].mkdir(parents=True, exist_ok=True)
    return paths


def fail_if_workspace_has_outputs(paths: dict[str, Path]) -> None:
    existing = [
        paths["database"],
        paths["raw_ply"],
        paths["report"],
        paths["best_sparse"],
    ]
    if any(path.exists() for path in existing):
        raise FileExistsError(
            f"{paths['workspace']} already contains COLMAP outputs. "
            "Pass --overwrite to rebuild it."
        )


def run_reconstruction(
    image_dir: Path,
    calibration_path: Path,
    workspace: Path,
    *,
    overlap: int = 15,
    max_num_features: int = 8192,
    overwrite: bool = False,
) -> dict[str, Any]:
    images = image_paths(image_dir)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    calibration = CameraCalibration.from_yaml(calibration_path)
    camera = camera_from_calibration(calibration)
    paths = prepare_workspace(workspace, overwrite)
    if not overwrite:
        fail_if_workspace_has_outputs(paths)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera.model
    reader_options.camera_params = camera.params_string

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.use_gpu = False
    extraction_options.sift.max_num_features = max_num_features

    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.use_gpu = False

    pairing_options = pycolmap.SequentialPairingOptions()
    pairing_options.overlap = overlap

    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.ba_refine_focal_length = False
    mapper_options.ba_refine_principal_point = False
    mapper_options.ba_refine_extra_params = False
    mapper_options.ba_use_gpu = False
    mapper_options.extract_colors = True
    mapper_options.mapper.abs_pose_refine_focal_length = False
    mapper_options.mapper.abs_pose_refine_extra_params = False

    pycolmap.extract_features(
        paths["database"],
        image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=extraction_options,
        device=pycolmap.Device.cpu,
    )
    pycolmap.match_sequential(
        paths["database"],
        matching_options=matching_options,
        pairing_options=pairing_options,
        device=pycolmap.Device.cpu,
    )
    reconstructions = pycolmap.incremental_mapping(
        paths["database"],
        image_dir,
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
        calibration_path=calibration_path,
        paths=paths,
        camera=camera,
        image_count=len(images),
        overlap=overlap,
        max_num_features=max_num_features,
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
    )
    paths["report"].write_text(json.dumps(
        report, indent=2) + "\n", encoding="utf-8")
    return report


def build_report(
    *,
    image_dir: Path,
    calibration_path: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    image_count: int,
    overlap: int,
    max_num_features: int,
    reconstructions: dict[int, pycolmap.Reconstruction],
    best_model_id: int,
    best_reconstruction: pycolmap.Reconstruction,
) -> dict[str, Any]:
    database_stats = read_database_stats(paths["database"])
    model_stats = {
        str(model_id): reconstruction_stats(reconstruction)
        for model_id, reconstruction in sorted(reconstructions.items())
    }
    return {
        "status": "raw_colmap_unscaled",
        "warning": (
            "This is a monocular COLMAP reconstruction in arbitrary COLMAP "
            "scale/frame. Do not use it as metric challenge output until it "
            "is aligned to the ArUco marker trajectory."
        ),
        "pycolmap_version": pycolmap.__version__,
        "image_dir": str(image_dir),
        "calibration_path": str(calibration_path),
        "workspace": str(paths["workspace"]),
        "database_path": str(paths["database"]),
        "sparse_path": str(paths["sparse"]),
        "best_sparse_path": str(paths["best_sparse"]),
        "raw_ply_path": str(paths["raw_ply"]),
        "camera": {
            "model": camera.model,
            "width": camera.width,
            "height": camera.height,
            "params": list(camera.params),
            "params_order": ["fx", "fy", "cx", "cy"],
            "shared_single_camera": True,
            "intrinsics_refined": False,
        },
        "options": {
            "sequential_overlap": overlap,
            "max_num_features": max_num_features,
            "use_gpu": False,
        },
        "input_image_count": image_count,
        "database": database_stats,
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": model_stats,
        "best_model": reconstruction_stats(best_reconstruction),
    }


def reconstruction_stats(reconstruction: pycolmap.Reconstruction) -> dict[str, Any]:
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw COLMAP registration on data/rgb with fixed challenge "
            "camera calibration. This does not recover metric scale."
        )
    )
    parser.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"RGB image directory (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "-c",
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help=f"Camera calibration YAML (default: {DEFAULT_CALIBRATION})",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"COLMAP workspace directory (default: {DEFAULT_WORKSPACE})",
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
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the COLMAP workspace before running.",
    )
    args = parser.parse_args(argv)
    if args.overlap <= 0:
        parser.error("--overlap must be positive")
    if args.max_num_features <= 0:
        parser.error("--max-num-features must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run_reconstruction(
        image_dir=args.image_dir,
        calibration_path=args.calibration,
        workspace=args.workspace,
        overlap=args.overlap,
        max_num_features=args.max_num_features,
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
    print("status: raw COLMAP reconstruction is unscaled; marker alignment comes next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
