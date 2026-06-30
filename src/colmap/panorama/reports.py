"""Report and metadata helpers for panorama reconstruction."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from colmap.camera import ColmapCamera
from colmap.panorama.constants import NATIVE_CAMERA_SETUP
from colmap.panorama.database import read_database_stats
from colmap.panorama.keyframe_matching import KeyframePairPlan, KeyframeSelectionOptions
from colmap.panorama.types import (
    FeaturePipelineOptions,
    GeneratedImageBatch,
    KeyframeAnchorScore,
)


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


def write_image_pair_list(
    path: Path,
    pairs: Sequence[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{image_name_a} {image_name_b}\n"
            for image_name_a, image_name_b in pairs
        ),
        encoding="utf-8",
    )


def format_elapsed_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    rounded_seconds = int(round(seconds))
    minutes, seconds_part = divmod(rounded_seconds, 60)
    hours, minutes_part = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes_part:02d}m {seconds_part:02d}s"
    return f"{minutes_part}m {seconds_part:02d}s"


def format_elapsed_seconds_by_stage(
    elapsed_seconds_by_stage: dict[str, float],
) -> dict[str, str]:
    return {
        stage: format_elapsed_seconds(seconds)
        for stage, seconds in elapsed_seconds_by_stage.items()
    }


def write_equirectangular_metadata(
    path: Path,
    input_image_path: Path,
    camera: ColmapCamera,
    panorama_paths: Sequence[Path],
    image_names: Sequence[str],
    *,
    image_limit: int | None,
    relative_name_prefix: Path | None,
) -> None:
    data = {
        "schema_version": 2,
        "camera_setup": NATIVE_CAMERA_SETUP,
        "input_image_path": str(input_image_path),
        "image_limit": image_limit,
        "camera": camera.report(),
        "import_options": {
            "copy_backend": "hardlink_or_symlink",
            "new_image_prefix": (
                None
                if relative_name_prefix is None
                else relative_name_prefix.as_posix()
            ),
        },
        "panorama_images": [
            {
                "source_path": str(panorama_path),
                "colmap_image_name": image_name,
            }
            for panorama_path, image_name in zip(panorama_paths, image_names)
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_panorama_import_report(
    *,
    input_image_path: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    panorama_paths: Sequence[Path],
    image_names: Sequence[str],
    image_limit: int | None,
    import_workers: int,
) -> dict[str, Any]:
    return {
        "status": f"{NATIVE_CAMERA_SETUP}_import_only",
        "camera_setup": NATIVE_CAMERA_SETUP,
        "input_image_path": str(input_image_path),
        "workspace": str(paths["workspace"]),
        "image_dir": str(paths["image_dir"]),
        "metadata_path": str(paths["metadata"]),
        "report_path": str(paths["import_report"]),
        "camera": camera.report(),
        "options": {
            "image_limit": image_limit,
            "import_workers": import_workers,
            "copy_backend": "hardlink_or_symlink",
        },
        "panorama_count": len(panorama_paths),
        "generated_image_count": len(image_names),
        "generated_image_names": list(image_names),
        "panorama_images": [
            {
                "source_path": str(panorama_path),
                "colmap_image_name": image_name,
            }
            for panorama_path, image_name in zip(panorama_paths, image_names)
        ],
        "skipped_stages": [
            "feature_extraction",
            "matching",
            "incremental_mapping",
            "best_model_write",
            "keyframe_scoring",
        ],
    }


def build_panorama_feature_report(
    *,
    pycolmap_version: str,
    input_image_path: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    panorama_count: int,
    feature_options: FeaturePipelineOptions,
    execution: dict[str, Any],
    feature_report: dict[str, dict[str, float | int]],
    feature_health: dict[str, Any],
    image_limit: int | None,
    import_workers: int,
    feature_workers: int,
    feature_batch_size: int | None,
    feature_shard_result: Any | None,
    generated_image_batch: GeneratedImageBatch,
) -> dict[str, Any]:
    database_report = read_database_stats(paths["database"])
    feature_extraction_mode = (
        "sharded_database" if feature_shard_result is not None else "single_database"
    )
    return {
        "status": f"{NATIVE_CAMERA_SETUP}_features_only",
        "camera_setup": NATIVE_CAMERA_SETUP,
        "pycolmap_version": pycolmap_version,
        "input_image_path": str(input_image_path),
        "workspace": str(paths["workspace"]),
        "image_dir": str(paths["image_dir"]),
        "database_path": str(paths["database"]),
        "metadata_path": str(paths["metadata"]),
        "report_path": str(paths["feature_report"]),
        "camera": camera.report(),
        "options": {
            "feature_extractor": feature_options.extractor,
            "feature_matcher": feature_options.matcher,
            "max_num_features": feature_options.max_num_features,
            "max_image_size": feature_options.max_image_size,
            "num_threads": feature_options.num_threads,
            "guided_matching": feature_options.guided_matching,
            "image_limit": image_limit,
            "import_workers": import_workers,
            "feature_workers": feature_workers,
            "feature_batch_size": feature_batch_size,
            "feature_extraction_mode": feature_extraction_mode,
            "requested_device": execution["requested_device"],
            "resolved_device": execution["resolved_device"],
            "cuda_available": execution["cuda_available"],
            "gpu_index": execution["gpu_index"],
            "camera_mode": "SINGLE",
        },
        "panorama_count": panorama_count,
        "generated_image_count": panorama_count,
        "database_image_count": database_report["images"],
        "new_database_image_count": len(generated_image_batch.names),
        "new_database_image_ids": generated_image_batch.image_ids,
        "new_database_camera_ids": generated_image_batch.camera_ids,
        "new_database_image_names": list(generated_image_batch.names),
        "database": database_report,
        "features_by_camera": feature_report,
        "feature_health": feature_health,
        "feature_shards": feature_shard_report(feature_shard_result),
        "skipped_stages": [
            "matching",
            "incremental_mapping",
            "best_model_write",
            "keyframe_scoring",
        ],
    }


def feature_shard_report(feature_shard_result: Any | None) -> dict[str, Any] | None:
    if feature_shard_result is None:
        return None
    return {
        "batch_count": len(feature_shard_result.batches),
        "batches": [list(batch) for batch in feature_shard_result.batches],
        "shard_database_paths": [
            str(path) for path in feature_shard_result.shard_database_paths
        ],
        "merged_database_path": str(feature_shard_result.merged_database_path),
        "shard_elapsed_seconds": list(feature_shard_result.shard_elapsed_seconds),
        "merge_elapsed_seconds": feature_shard_result.merge_elapsed_seconds,
        "camera_policy": "multiple_identical_equirectangular_cameras_are_expected",
    }


def build_panorama_report(
    *,
    pycolmap_version: str,
    input_image_path: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    panorama_count: int,
    matcher: str,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    ssl_cert_file: Path | None,
    feature_options: FeaturePipelineOptions,
    execution: dict[str, Any],
    feature_report: dict[str, dict[str, float | int]],
    feature_health: dict[str, Any],
    image_limit: int | None,
    import_workers: int,
    feature_workers: int,
    feature_batch_size: int | None,
    feature_shard_result: Any | None,
    generated_image_batch: GeneratedImageBatch,
    reconstructions: dict[int, Any],
    best_model_id: int,
    best_reconstruction: Any,
    keyframe_scores: Sequence[KeyframeAnchorScore],
    keyframe_options: KeyframeSelectionOptions | None = None,
    keyframe_pair_plan: KeyframePairPlan | None = None,
) -> dict[str, Any]:
    database_report = read_database_stats(paths["database"])
    model_stats = {
        str(model_id): reconstruction_stats(reconstruction)
        for model_id, reconstruction in sorted(reconstructions.items())
    }
    feature_extraction_mode = (
        "sharded_database" if feature_shard_result is not None else "single_database"
    )
    keyframe_pair_counts = (
        keyframe_pair_plan.pair_counts_by_source
        if keyframe_pair_plan is not None
        else {}
    )
    effective_pairing = (
        keyframe_pair_plan.effective_selection
        if keyframe_pair_plan is not None
        else matcher
    )
    keyframe_options_report = (
        {
            "keyframe_selection": keyframe_options.selection,
            "keyframe_selection_effective": (
                None
                if keyframe_pair_plan is None
                else keyframe_pair_plan.effective_selection
            ),
            "keyframe_interval": keyframe_options.interval,
            "keyframe_overlap": keyframe_options.overlap,
            "keyframe_backbone_policy": keyframe_options.backbone_policy,
            "keyframe_backbone_overlap": keyframe_options.backbone_overlap,
            "keyframe_cluster_max_size": keyframe_options.cluster_max_size,
            "keyframe_cluster_recent_window": keyframe_options.cluster_recent_window,
            "keyframe_vocabtree_neighbors": keyframe_options.vocabtree_neighbors,
            "keyframe_vocabtree_checks": keyframe_options.vocabtree_checks,
            "matching_pair_count": keyframe_pair_counts.get("all", 0),
            "keyframe_pair_count": keyframe_pair_counts.get("keyframe", 0),
            "vocabtree_pair_count": keyframe_pair_counts.get("vocabtree", 0),
            "sequential_backbone_pair_count": keyframe_pair_counts.get(
                "sequential_backbone",
                0,
            ),
            "match_pairs_path": (
                str(paths["match_pairs"])
                if keyframe_pair_plan is not None
                else None
            ),
        }
        if keyframe_options is not None
        else {}
    )
    return {
        "status": f"raw_colmap_{NATIVE_CAMERA_SETUP}_unscaled",
        "warning": (
            "This is a monocular COLMAP reconstruction in arbitrary COLMAP "
            "scale/frame. The resulting world is not metric without an "
            "external scale."
        ),
        "camera_setup": NATIVE_CAMERA_SETUP,
        "pycolmap_version": pycolmap_version,
        "input_image_path": str(input_image_path),
        "workspace": str(paths["workspace"]),
        "image_dir": str(paths["image_dir"]),
        "database_path": str(paths["database"]),
        "sparse_path": str(paths["sparse"]),
        "best_sparse_path": str(paths["best_sparse"]),
        "raw_ply_path": str(paths["raw_ply"]),
        "metadata_path": str(paths["metadata"]),
        "camera": camera.report(),
        "options": {
            "matcher": matcher,
            "effective_pairing": effective_pairing,
            "sequential_overlap": overlap,
            "loop_detection": loop_detection,
            "vocab_tree_path": (
                None if vocab_tree_path is None else str(vocab_tree_path)
            ),
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
            "render_backend": "source_equirectangular_images",
            "image_limit": image_limit,
            "import_workers": import_workers,
            "feature_workers": feature_workers,
            "feature_batch_size": feature_batch_size,
            "feature_extraction_mode": feature_extraction_mode,
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
            **keyframe_options_report,
        },
        "panorama_count": panorama_count,
        "generated_image_count": panorama_count,
        "database_image_count": database_report["images"],
        "new_database_image_count": len(generated_image_batch.names),
        "new_database_image_ids": generated_image_batch.image_ids,
        "new_database_camera_ids": generated_image_batch.camera_ids,
        "new_database_image_names": list(generated_image_batch.names),
        "database": database_report,
        "features_by_camera": feature_report,
        "feature_health": feature_health,
        "feature_shards": feature_shard_report(feature_shard_result),
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": model_stats,
        "best_model": reconstruction_stats(best_reconstruction),
        "keyframe_score_count": len(keyframe_scores),
        "keyframe_scores": [score.report() for score in keyframe_scores],
        "keyframes": (
            [] if keyframe_pair_plan is None else keyframe_pair_plan.report_keyframes()
        ),
        "keyframe_clusters": (
            [] if keyframe_pair_plan is None else keyframe_pair_plan.report_clusters()
        ),
    }


def equirectangular_feature_health_summary(
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
        "camera_setup": NATIVE_CAMERA_SETUP,
        "expected_panorama_images": expected_image_count,
        "panorama_images_with_features": image_count,
        "camera_count": len(feature_report),
        "cameras_with_features": len(feature_report) - len(zero_feature_cameras),
        "zero_feature_cameras": zero_feature_cameras,
    }
