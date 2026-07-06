"""Report and metadata helpers for pinhole reconstruction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from colmap.panorama.database import read_database_stats
from colmap.panorama.reports import reconstruction_stats
from colmap.panorama.types import FeaturePipelineOptions, GeneratedImageBatch
from colmap.pinhole.image_import import PinholeImageImportResult

PINHOLE_CAMERA_SETUP = "pinhole_render_report"


def write_pinhole_metadata(
    path: Path,
    *,
    import_result: PinholeImageImportResult,
) -> None:
    data = {
        "schema_version": 1,
        "camera_setup": PINHOLE_CAMERA_SETUP,
        "render_report_path": str(import_result.render_report_path),
        "image_dir_override": (
            None
            if import_result.image_dir_override is None
            else str(import_result.image_dir_override)
        ),
        "camera": import_result.camera.report(),
        "import_options": {
            "copy_backend": "hardlink_or_symlink",
            "image_resolution_source": "actual_imported_files",
            "fov_source": "render_report",
        },
        "pinhole_images": [
            image.report() for image in import_result.imported_images
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_pinhole_import_report(
    *,
    paths: dict[str, Path],
    import_result: PinholeImageImportResult,
    import_workers: int,
) -> dict[str, Any]:
    return {
        "status": f"{PINHOLE_CAMERA_SETUP}_import_only",
        "camera_setup": PINHOLE_CAMERA_SETUP,
        "render_report_path": str(import_result.render_report_path),
        "image_dir_override": (
            None
            if import_result.image_dir_override is None
            else str(import_result.image_dir_override)
        ),
        "workspace": str(paths["workspace"]),
        "image_dir": str(paths["image_dir"]),
        "metadata_path": str(paths["metadata"]),
        "report_path": str(paths["import_report"]),
        "camera": import_result.camera.report(),
        "options": {
            "import_workers": import_workers,
            "copy_backend": "hardlink_or_symlink",
            "image_resolution_source": "actual_imported_files",
            "fov_source": "render_report",
        },
        "rendered_image_count": len(import_result.imported_images),
        "generated_image_count": len(import_result.image_names),
        "generated_image_names": list(import_result.image_names),
        "resolution_mismatch_count": import_result.resolution_mismatch_count,
        "pinhole_images": [
            image.report() for image in import_result.imported_images
        ],
        "skipped_stages": [
            "feature_extraction",
            "matching",
            "incremental_mapping",
            "best_model_write",
        ],
    }


def pinhole_feature_health_summary(
    feature_report: dict[str, dict[str, float | int]],
    *,
    expected_image_count: int,
) -> dict[str, Any]:
    image_count = sum(int(stats["image_count"]) for stats in feature_report.values())
    zero_feature_cameras = [
        camera_name
        for camera_name, stats in sorted(feature_report.items())
        if int(stats["total_features"]) == 0
    ]
    return {
        "camera_setup": PINHOLE_CAMERA_SETUP,
        "expected_pinhole_images": expected_image_count,
        "pinhole_images_with_features": image_count,
        "camera_count": len(feature_report),
        "cameras_with_features": len(feature_report) - len(zero_feature_cameras),
        "zero_feature_cameras": zero_feature_cameras,
    }


def build_pinhole_reconstruction_report(
    *,
    pycolmap_version: str,
    paths: dict[str, Path],
    import_result: PinholeImageImportResult,
    matcher: str,
    overlap: int,
    loop_detection: bool,
    ssl_cert_file: Path | None,
    feature_options: FeaturePipelineOptions,
    execution: dict[str, Any],
    feature_report: dict[str, dict[str, float | int]],
    feature_health: dict[str, Any],
    generated_image_batch: GeneratedImageBatch,
    reconstructions: dict[int, Any],
    best_model_id: int,
    best_reconstruction: Any,
    import_workers: int,
    pose_prior_summary: Any | None = None,
    use_position_priors: bool = False,
    written_position_prior_count: int | None = None,
    initial_pose_summary: Any | None = None,
) -> dict[str, Any]:
    database_report = read_database_stats(paths["database"])
    model_stats = {
        str(model_id): reconstruction_stats(reconstruction)
        for model_id, reconstruction in sorted(reconstructions.items())
    }
    return {
        "status": f"raw_colmap_{PINHOLE_CAMERA_SETUP}_unscaled",
        "warning": (
            "This is a monocular COLMAP reconstruction in arbitrary COLMAP "
            "scale/frame. The resulting world is not metric without an "
            "external scale."
        ),
        "camera_setup": PINHOLE_CAMERA_SETUP,
        "pycolmap_version": pycolmap_version,
        "render_report_path": str(import_result.render_report_path),
        "image_dir_override": (
            None
            if import_result.image_dir_override is None
            else str(import_result.image_dir_override)
        ),
        "workspace": str(paths["workspace"]),
        "image_dir": str(paths["image_dir"]),
        "database_path": str(paths["database"]),
        "sparse_path": str(paths["sparse"]),
        "best_sparse_path": str(paths["best_sparse"]),
        "raw_ply_path": str(paths["raw_ply"]),
        "metadata_path": str(paths["metadata"]),
        "report_path": str(paths["report"]),
        "camera": import_result.camera.report(),
        "options": {
            "matcher": matcher,
            "effective_pairing": matcher,
            "sequential_overlap": overlap,
            "loop_detection": loop_detection,
            "ssl_cert_file": (
                os.environ.get("SSL_CERT_FILE")
                if ssl_cert_file is None
                else str(ssl_cert_file)
            ),
            "feature_extractor": feature_options.extractor,
            "feature_matcher": feature_options.matcher,
            "max_num_features": feature_options.max_num_features,
            "max_image_size": feature_options.max_image_size,
            "num_threads": feature_options.num_threads,
            "guided_matching": feature_options.guided_matching,
            "import_workers": import_workers,
            "feature_extraction_mode": "single_database",
            "requested_device": execution["requested_device"],
            "resolved_device": execution["resolved_device"],
            "cuda_available": execution["cuda_available"],
            "gpu_index": execution["gpu_index"],
            "gpu_bundle_adjustment": execution["gpu_bundle_adjustment"],
            "bundle_adjustment_gpu_index": execution[
                "bundle_adjustment_gpu_index"
            ],
            "camera_mode": "SINGLE",
            "ba_refine_sensor_from_rig": False,
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
            "rig_verification": False,
            "skip_image_pairs_in_same_frame": False,
            "image_resolution_source": "actual_imported_files",
            "fov_source": "render_report",
            "use_position_priors": use_position_priors,
            "position_prior_count": written_position_prior_count,
            "initial_poses_from_pose_report": initial_pose_summary is not None,
        },
        "rendered_image_count": len(import_result.imported_images),
        "generated_image_count": len(import_result.image_names),
        "database_image_count": database_report["images"],
        "new_database_image_count": len(generated_image_batch.names),
        "new_database_image_ids": generated_image_batch.image_ids,
        "new_database_camera_ids": generated_image_batch.camera_ids,
        "new_database_image_names": list(generated_image_batch.names),
        "resolution_mismatch_count": import_result.resolution_mismatch_count,
        "database": database_report,
        "features_by_camera": feature_report,
        "feature_health": feature_health,
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": model_stats,
        "best_model": reconstruction_stats(best_reconstruction),
        "pinhole_images": [
            image.report() for image in import_result.imported_images
        ],
        "pose_priors": (
            None
            if pose_prior_summary is None
            else pose_prior_summary.report(
                use_position_priors=use_position_priors,
                written_prior_count=written_position_prior_count,
            )
        ),
        "initial_poses": (
            None if initial_pose_summary is None else initial_pose_summary.report()
        ),
    }
