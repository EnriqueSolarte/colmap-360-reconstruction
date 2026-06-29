"""High-level panorama reconstruction pipeline."""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from colmap.panorama.database import (
    read_feature_stats_by_database_camera,
    read_generated_image_batch,
)
from colmap.panorama.feature_pipeline import (
    FeatureShardExtractionResult,
    extract_panorama_features_to_database,
    extract_panorama_features_to_sharded_database,
)
from colmap.panorama.import_pipeline import import_panorama_images_to_workspace
from colmap.panorama.paths import panorama_workspace_paths
from colmap.panorama.pycolmap_adapter import (
    configure_ssl_cert_file,
    create_feature_matching_options,
    create_mapper_options,
    match_features,
    require_pycolmap,
    resolve_pipeline_execution,
    run_incremental_mapping,
    write_best_model,
)
from colmap.panorama.reports import (
    build_panorama_report,
    equirectangular_feature_health_summary,
    format_elapsed_seconds,
    format_elapsed_seconds_by_stage,
)
from colmap.panorama.scoring import score_keyframe_anchors
from colmap.panorama.types import FeaturePipelineOptions, PipelineExecutionOptions


def record_elapsed_stage(
    elapsed_seconds_by_stage: dict[str, float],
    stage: str,
    stage_started_at: float,
) -> None:
    elapsed_seconds_by_stage[stage] = elapsed_seconds_by_stage.get(
        stage,
        0.0,
    ) + (time.perf_counter() - stage_started_at)


def run_panorama_reconstruction(
    input_image_path: Path,
    workspace: Path,
    *,
    matcher: str = "sequential",
    overlap: int = 15,
    loop_detection: bool = True,
    vocab_tree_path: Path | None = None,
    ssl_cert_file: Path | None = None,
    feature_options: FeaturePipelineOptions = FeaturePipelineOptions(),
    execution_options: PipelineExecutionOptions = PipelineExecutionOptions(),
    image_limit: int | None = None,
    import_workers: int = 1,
    feature_workers: int = 1,
    feature_batch_size: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if feature_workers <= 0:
        raise ValueError("feature_workers must be positive")
    if feature_batch_size is not None and feature_batch_size <= 0:
        raise ValueError("feature_batch_size must be positive")
    started_at = time.perf_counter()
    elapsed_seconds_by_stage: dict[str, float] = {}
    feature_shard_result: FeatureShardExtractionResult | None = None
    stage_started_at = started_at
    pycolmap_module = require_pycolmap()
    pycolmap_module.set_random_seed(0)
    configure_ssl_cert_file(ssl_cert_file)
    execution = resolve_pipeline_execution(pycolmap_module, execution_options)
    paths = panorama_workspace_paths(workspace)
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    paths["image_dir"].mkdir(parents=True, exist_ok=True)
    paths["sparse"].mkdir(parents=True, exist_ok=True)
    if paths["database"].exists():
        raise FileExistsError(f"{paths['database']} already exists. Pass --overwrite.")
    record_elapsed_stage(elapsed_seconds_by_stage, "setup", stage_started_at)

    stage_started_at = time.perf_counter()
    panorama_paths, camera, generated_image_names = import_panorama_images_to_workspace(
        input_image_path=input_image_path,
        paths=paths,
        image_limit=image_limit,
        import_workers=import_workers,
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "import_equirectangular_images",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    if feature_workers == 1:
        extract_panorama_features_to_database(
            pycolmap_module,
            database_path=paths["database"],
            image_dir=paths["image_dir"],
            camera=camera,
            feature_options=feature_options,
            execution=execution,
        )
    else:
        effective_feature_batch_size = feature_batch_size or math.ceil(
            len(generated_image_names) / feature_workers,
        )
        feature_shard_result = extract_panorama_features_to_sharded_database(
            pycolmap_module,
            database_path=paths["database"],
            image_dir=paths["image_dir"],
            shard_dir=paths["feature_shards"],
            camera=camera,
            feature_options=feature_options,
            execution=execution,
            image_names=generated_image_names,
            batch_size=effective_feature_batch_size,
            feature_workers=feature_workers,
        )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "feature_extraction",
        stage_started_at,
    )
    if feature_shard_result is not None:
        elapsed_seconds_by_stage["feature_shard_merge"] = (
            feature_shard_result.merge_elapsed_seconds
        )

    stage_started_at = time.perf_counter()
    generated_image_batch = read_generated_image_batch(
        paths["database"],
        generated_image_names,
    )
    feature_report = read_feature_stats_by_database_camera(paths["database"])
    feature_health = equirectangular_feature_health_summary(
        feature_report,
        expected_image_count=len(generated_image_names),
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "feature_stats",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
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
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "matching",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    mapper_options = create_mapper_options(pycolmap_module, execution)
    reconstructions = run_incremental_mapping(
        pycolmap_module,
        paths,
        image_dir=paths["image_dir"],
        mapper_options=mapper_options,
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "incremental_mapping",
        stage_started_at,
    )
    if not reconstructions:
        raise RuntimeError("COLMAP did not produce any sparse reconstruction")

    stage_started_at = time.perf_counter()
    best_model_id, best_reconstruction = max(
        reconstructions.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
    )
    write_best_model(best_reconstruction, paths)
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "best_model_write",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    keyframe_scores = score_keyframe_anchors(paths["database"], best_reconstruction)
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "keyframe_scoring",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    report = build_panorama_report(
        pycolmap_version=pycolmap_module.__version__,
        input_image_path=input_image_path,
        paths=paths,
        camera=camera,
        panorama_count=len(panorama_paths),
        matcher=matcher,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        ssl_cert_file=ssl_cert_file,
        feature_options=feature_options,
        execution=execution,
        feature_report=feature_report,
        feature_health=feature_health,
        image_limit=image_limit,
        import_workers=import_workers,
        feature_workers=feature_workers,
        feature_batch_size=feature_batch_size,
        feature_shard_result=feature_shard_result,
        generated_image_batch=generated_image_batch,
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
        keyframe_scores=keyframe_scores,
    )
    elapsed_seconds_by_stage["report_writing"] = 0.0
    elapsed_seconds = time.perf_counter() - started_at
    report["elapsed_seconds"] = elapsed_seconds
    report["elapsed_time"] = format_elapsed_seconds(elapsed_seconds)
    report["elapsed_seconds_by_stage"] = elapsed_seconds_by_stage
    report["elapsed_time_by_stage"] = format_elapsed_seconds_by_stage(
        elapsed_seconds_by_stage,
    )
    json.dumps(report, indent=2)
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "report_writing",
        stage_started_at,
    )
    elapsed_seconds = time.perf_counter() - started_at
    report["elapsed_seconds"] = elapsed_seconds
    report["elapsed_time"] = format_elapsed_seconds(elapsed_seconds)
    report["elapsed_seconds_by_stage"] = elapsed_seconds_by_stage
    report["elapsed_time_by_stage"] = format_elapsed_seconds_by_stage(
        elapsed_seconds_by_stage,
    )
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
