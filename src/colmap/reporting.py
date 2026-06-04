"""Report and statistics helpers for COLMAP reconstruction runs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from colmap.camera import ColmapCamera


def build_report(
    *,
    pycolmap_version: str,
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
