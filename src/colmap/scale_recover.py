"""Recover metric marker-frame alignment for a raw monocular COLMAP model.

COLMAP's monocular world is only defined up to an arbitrary similarity. This
module uses ArUco marker poses from registered images to estimate:

- metric scale from pairwise camera-baseline ratios;
- rotation and translation that map COLMAP world coordinates into the ArUco
  marker frame.

The exported COLMAP model and PLY are therefore in metres and in marker-frame
coordinates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pycolmap
from tqdm import tqdm

from starter.aruco.aruco_pose import estimate_camera_target_pose, invert_transform
from starter.camera_calibration import CameraCalibration
from starter.config import DEFAULT_CONFIG, ChallengeConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "rgb"
DEFAULT_CALIBRATION = PROJECT_ROOT / "data" / "cfg.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "data" / "colmap" / "sparse_best"
DEFAULT_OUTPUT_MODEL = PROJECT_ROOT / "data" / "colmap" / "sparse_metric"
DEFAULT_OUTPUT_PLY = PROJECT_ROOT / "data" / "colmap" / "cloud_colmap_metric.ply"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "colmap" / "scale_recovery_report.json"


@dataclass(frozen=True)
class ImageScalePose:
    """Image pose in both marker-metric and COLMAP-unscaled coordinates."""

    image_name: str
    T_marker_camera: np.ndarray
    colmap_center: np.ndarray
    reprojection_error_px: float


@dataclass(frozen=True)
class PairScaleEstimate:
    """Metric scale estimate from one image pair."""

    image_name1: str
    image_name2: str
    marker_baseline_m: float
    colmap_baseline: float
    scale: float


@dataclass(frozen=True)
class ScaleRecoveryResult:
    """Average scale estimate and the pair estimates used to compute it."""

    scale: float
    pair_estimates: tuple[PairScaleEstimate, ...]

    @property
    def pair_count(self) -> int:
        return len(self.pair_estimates)

    @property
    def median_scale(self) -> float:
        return float(np.median([pair.scale for pair in self.pair_estimates]))

    @property
    def std_scale(self) -> float:
        if len(self.pair_estimates) < 2:
            return 0.0
        return float(np.std([pair.scale for pair in self.pair_estimates], ddof=1))


@dataclass(frozen=True)
class MarkerFrameAlignment:
    """Similarity transform that maps COLMAP world points into marker frame."""

    scale: float
    rotation_marker_from_colmap: np.ndarray
    translation_marker_from_colmap: np.ndarray
    rmse_m: float
    max_error_m: float


def _pair_scale_estimate(
    pose1: ImageScalePose,
    pose2: ImageScalePose,
    *,
    min_marker_baseline_m: float = 0.03,
    min_colmap_baseline: float = 1e-8,
) -> PairScaleEstimate:
    """Estimate COLMAP scale from one pair of images that see the marker."""
    marker_baseline_m = aruco_relative_translation_norm(
        pose1.T_marker_camera,
        pose2.T_marker_camera,
    )
    colmap_baseline = float(np.linalg.norm(pose2.colmap_center - pose1.colmap_center))

    if marker_baseline_m < min_marker_baseline_m:
        raise ValueError(
            f"Marker baseline is too small: {marker_baseline_m:.6g} m "
            f"< {min_marker_baseline_m:.6g} m"
        )
    if colmap_baseline < min_colmap_baseline:
        raise ValueError(
            f"COLMAP baseline is too small: {colmap_baseline:.6g} "
            f"< {min_colmap_baseline:.6g}"
        )

    return PairScaleEstimate(
        image_name1=pose1.image_name,
        image_name2=pose2.image_name,
        marker_baseline_m=marker_baseline_m,
        colmap_baseline=colmap_baseline,
        scale=marker_baseline_m / colmap_baseline,
    )


def aruco_relative_translation_norm(
    T_marker_camera1: np.ndarray,
    T_marker_camera2: np.ndarray,
) -> float:
    """Return the relative camera translation norm between two marker poses."""
    T_camera1_camera2 = invert_transform(T_marker_camera1) @ T_marker_camera2
    return float(np.linalg.norm(T_camera1_camera2[:3, 3]))


def average_scale_recover(
    poses: list[ImageScalePose],
    *,
    min_marker_baseline_m: float = 0.03,
    min_colmap_baseline: float = 1e-8,
) -> ScaleRecoveryResult:
    """Average all valid pairwise scale estimates."""
    if len(poses) < 2:
        raise ValueError("Need at least two images with both ArUco and COLMAP poses")

    pair_estimates: list[PairScaleEstimate] = []
    for index1, pose1 in enumerate(poses[:-1]):
        for pose2 in poses[index1 + 1 :]:
            try:
                pair_estimates.append(
                    _pair_scale_estimate(
                        pose1,
                        pose2,
                        min_marker_baseline_m=min_marker_baseline_m,
                        min_colmap_baseline=min_colmap_baseline,
                    )
                )
            except ValueError:
                continue

    if not pair_estimates:
        raise ValueError(
            "No valid image pairs found for scale recovery. Try lowering "
            "--min-marker-baseline-m, or inspect ArUco/COLMAP pose overlap."
        )

    scale = float(np.mean([pair.scale for pair in pair_estimates]))
    return ScaleRecoveryResult(scale=scale, pair_estimates=tuple(pair_estimates))


def collect_image_scale_poses(
    reconstruction: pycolmap.Reconstruction,
    image_dir: Path,
    calibration: CameraCalibration,
    *,
    config: ChallengeConfig = DEFAULT_CONFIG,
    max_reprojection_error_px: float = 5.0,
) -> list[ImageScalePose]:
    """Estimate ArUco poses for registered COLMAP images."""
    poses: list[ImageScalePose] = []
    cv2 = _require_cv2()

    for image in tqdm(registered_images(reconstruction), "recovering scale"):
        image_path = image_dir / image.name
        if not image_path.exists():
            continue

        frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            continue

        aruco_pose = estimate_camera_target_pose(frame, calibration, config)
        if aruco_pose is None:
            continue
        if aruco_pose.reprojection_error_px > max_reprojection_error_px:
            continue

        poses.append(
            ImageScalePose(
                image_name=image.name,
                T_marker_camera=aruco_pose.T_marker_camera,
                colmap_center=np.asarray(image.projection_center(), dtype=np.float64),
                reprojection_error_px=aruco_pose.reprojection_error_px,
            )
        )

    return poses


def registered_images(reconstruction: pycolmap.Reconstruction) -> list[pycolmap.Image]:
    """Return registered COLMAP images sorted by image name."""
    images = [
        reconstruction.image(image_id)
        for image_id in reconstruction.reg_image_ids()
    ]
    return sorted(images, key=lambda image: image.name)


def estimate_marker_frame_alignment(
    poses: list[ImageScalePose],
    scale: float,
) -> MarkerFrameAlignment:
    """Estimate rotation/translation from COLMAP world to ArUco marker frame."""
    if len(poses) < 3:
        raise ValueError("Need at least three shared poses to align a 3D frame")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be positive and finite, got {scale}")

    colmap_points = np.vstack([pose.colmap_center for pose in poses])
    marker_points = np.vstack(
        [pose.T_marker_camera[:3, 3] for pose in poses]
    )
    colmap_mean = colmap_points.mean(axis=0)
    marker_mean = marker_points.mean(axis=0)

    colmap_centered = colmap_points - colmap_mean
    marker_centered = marker_points - marker_mean
    covariance = marker_centered.T @ colmap_centered
    U, _singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0.0:
        correction[2, 2] = -1.0
    rotation = U @ correction @ Vt
    translation = marker_mean - scale * rotation @ colmap_mean

    transformed = (scale * (rotation @ colmap_points.T)).T + translation
    errors = np.linalg.norm(transformed - marker_points, axis=1)
    return MarkerFrameAlignment(
        scale=scale,
        rotation_marker_from_colmap=rotation,
        translation_marker_from_colmap=translation,
        rmse_m=float(np.sqrt(np.mean(errors**2))),
        max_error_m=float(errors.max()),
    )


def apply_marker_frame_alignment(
    reconstruction: pycolmap.Reconstruction,
    alignment: MarkerFrameAlignment,
) -> None:
    """Apply COLMAP-world to marker-frame Sim3 transform in-place."""
    pycolmap_module = _require_pycolmap()

    transform = pycolmap_module.Sim3d(
        float(alignment.scale),
        pycolmap_module.Rotation3d(alignment.rotation_marker_from_colmap),
        alignment.translation_marker_from_colmap,
    )
    reconstruction.transform(transform)


def recover_colmap_scale(
    model_path: Path,
    image_dir: Path,
    calibration_path: Path,
    *,
    max_reprojection_error_px: float = 5.0,
    min_marker_baseline_m: float = 0.03,
    min_colmap_baseline: float = 1e-8,
) -> tuple[
    ScaleRecoveryResult,
    MarkerFrameAlignment,
    pycolmap.Reconstruction,
    list[ImageScalePose],
]:
    """Load COLMAP and ArUco poses, then recover marker-frame alignment."""
    _require_pycolmap()
    reconstruction = pycolmap.Reconstruction(model_path)
    calibration = CameraCalibration.from_yaml(calibration_path)
    poses = collect_image_scale_poses(
        reconstruction,
        image_dir,
        calibration,
        max_reprojection_error_px=max_reprojection_error_px,
    )
    result = average_scale_recover(
        poses,
        min_marker_baseline_m=min_marker_baseline_m,
        min_colmap_baseline=min_colmap_baseline,
    )
    alignment = estimate_marker_frame_alignment(poses, result.scale)
    return result, alignment, reconstruction, poses


def write_outputs(
    reconstruction: pycolmap.Reconstruction,
    result: ScaleRecoveryResult,
    alignment: MarkerFrameAlignment,
    poses: list[ImageScalePose],
    *,
    output_model: Path | None,
    output_ply: Path | None,
    report_path: Path | None,
) -> None:
    """Write the marker-frame COLMAP model, optional PLY, and JSON report."""
    apply_marker_frame_alignment(reconstruction, alignment)

    if output_model is not None:
        output_model.mkdir(parents=True, exist_ok=True)
        reconstruction.write(output_model)

    if output_ply is not None:
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        reconstruction.export_PLY(output_ply)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(build_report(result, alignment, poses), indent=2) + "\n",
            encoding="utf-8",
        )


def build_report(
    result: ScaleRecoveryResult,
    alignment: MarkerFrameAlignment,
    poses: list[ImageScalePose],
) -> dict[str, Any]:
    """Build a compact marker-frame alignment report."""
    scales = np.asarray([pair.scale for pair in result.pair_estimates], dtype=np.float64)
    marker_baselines = np.asarray(
        [pair.marker_baseline_m for pair in result.pair_estimates],
        dtype=np.float64,
    )
    colmap_baselines = np.asarray(
        [pair.colmap_baseline for pair in result.pair_estimates],
        dtype=np.float64,
    )
    reprojection_errors = np.asarray(
        [pose.reprojection_error_px for pose in poses],
        dtype=np.float64,
    )
    return {
        "status": "metric_marker_frame_recovered",
        "warning": (
            "The COLMAP reconstruction was transformed into the ArUco marker "
            "frame using camera-center similarity alignment."
        ),
        "output_frame": DEFAULT_CONFIG.marker_frame,
        "scale_m_per_colmap_unit": result.scale,
        "scale_median": result.median_scale,
        "scale_std": result.std_scale,
        "rotation_marker_from_colmap_world": (
            alignment.rotation_marker_from_colmap.tolist()
        ),
        "translation_marker_from_colmap_world_m": (
            alignment.translation_marker_from_colmap.tolist()
        ),
        "alignment_rmse_m": alignment.rmse_m,
        "alignment_max_error_m": alignment.max_error_m,
        "aruco_colmap_pose_count": len(poses),
        "pair_count": result.pair_count,
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "marker_baseline_m_min": float(marker_baselines.min()),
        "marker_baseline_m_max": float(marker_baselines.max()),
        "colmap_baseline_min": float(colmap_baselines.min()),
        "colmap_baseline_max": float(colmap_baselines.max()),
        "aruco_reprojection_error_px_mean": float(reprojection_errors.mean()),
        "aruco_reprojection_error_px_max": float(reprojection_errors.max()),
        "sample_pairs": [
            {
                "image1": pair.image_name1,
                "image2": pair.image_name2,
                "marker_baseline_m": pair.marker_baseline_m,
                "colmap_baseline": pair.colmap_baseline,
                "scale": pair.scale,
            }
            for pair in result.pair_estimates[:20]
        ],
    }


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "OpenCV is required for ArUco scale recovery. Install "
            "`opencv-contrib-python`."
        ) from exc
    return cv2


def _require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required for COLMAP scale recovery. Install `pycolmap`."
        )
    return pycolmap


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover metric marker-frame alignment for a COLMAP model using "
            "ArUco marker camera poses."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Input COLMAP model directory (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"RGB image directory (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
        help=f"Camera calibration YAML (default: {DEFAULT_CALIBRATION})",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_OUTPUT_MODEL,
        help=(
            "Marker-frame COLMAP model output directory "
            f"(default: {DEFAULT_OUTPUT_MODEL})"
        ),
    )
    parser.add_argument(
        "--output-ply",
        type=Path,
        default=DEFAULT_OUTPUT_PLY,
        help=f"Marker-frame sparse PLY output path (default: {DEFAULT_OUTPUT_PLY})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"JSON report path (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=5.0,
        help="Maximum ArUco reprojection error to accept (default: 5.0)",
    )
    parser.add_argument(
        "--min-marker-baseline-m",
        type=float,
        default=0.03,
        help="Minimum ArUco pair baseline in metres (default: 0.03)",
    )
    parser.add_argument(
        "--min-colmap-baseline",
        type=float,
        default=1e-8,
        help="Minimum COLMAP pair baseline in unscaled units (default: 1e-8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute scale but do not write scaled model, PLY, or report.",
    )
    args = parser.parse_args(argv)

    if args.max_reprojection_error_px <= 0.0:
        parser.error("--max-reprojection-error-px must be positive")
    if args.min_marker_baseline_m <= 0.0:
        parser.error("--min-marker-baseline-m must be positive")
    if args.min_colmap_baseline <= 0.0:
        parser.error("--min-colmap-baseline must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    result, alignment, reconstruction, poses = recover_colmap_scale(
        model_path=args.model,
        image_dir=args.image_dir,
        calibration_path=args.calibration,
        max_reprojection_error_px=args.max_reprojection_error_px,
        min_marker_baseline_m=args.min_marker_baseline_m,
        min_colmap_baseline=args.min_colmap_baseline,
    )

    if not args.dry_run:
        write_outputs(
            reconstruction,
            result,
            alignment,
            poses,
            output_model=args.output_model,
            output_ply=args.output_ply,
            report_path=args.report,
        )

    print(f"images with ArUco + COLMAP pose: {len(poses)}")
    print(f"valid image pairs: {result.pair_count}")
    print(f"scale: {result.scale:.12g} m / COLMAP unit")
    print(f"median scale: {result.median_scale:.12g}")
    print(f"scale std: {result.std_scale:.12g}")
    print(f"marker alignment rmse: {alignment.rmse_m:.6g} m")
    print(f"marker alignment max error: {alignment.max_error_m:.6g} m")
    if args.dry_run:
        print("dry run: no marker-frame model, PLY, or report written")
    else:
        print(f"marker-frame model: {args.output_model}")
        print(f"marker-frame PLY: {args.output_ply}")
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
