"""High-level pinhole reconstruction pipeline."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from colmap.panorama.database import (
    read_feature_stats_by_database_camera,
    read_generated_image_batch,
)
from colmap.panorama.feature_pipeline import extract_panorama_features_to_database
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
    format_elapsed_seconds,
    format_elapsed_seconds_by_stage,
)
from colmap.panorama.types import FeaturePipelineOptions, PipelineExecutionOptions
from colmap.pinhole.colorize import colorize_reconstruction_points
from colmap.pinhole.image_import import import_pinhole_images_from_render_report
from colmap.pinhole.initial_poses import write_initial_pose_model_from_pose_report
from colmap.pinhole.paths import pinhole_workspace_paths
from colmap.pinhole.pose_priors import (
    build_position_prior_summary,
    write_position_priors_to_database,
)
from colmap.pinhole.reports import (
    build_pinhole_reconstruction_report,
    pinhole_feature_health_summary,
    write_pinhole_metadata,
)


def record_elapsed_stage(
    elapsed_seconds_by_stage: dict[str, float],
    stage: str,
    stage_started_at: float,
) -> None:
    elapsed_seconds_by_stage[stage] = elapsed_seconds_by_stage.get(
        stage,
        0.0,
    ) + (time.perf_counter() - stage_started_at)


def run_pinhole_reconstruction(
    render_report_path: Path,
    workspace: Path,
    *,
    image_dir: Path | None = None,
    pose_report_path: Path | None = None,
    use_position_priors: bool | None = None,
    initial_poses_from_pose_report: bool = False,
    overlap: int = 10,
    loop_detection: bool = True,
    ssl_cert_file: Path | None = None,
    feature_options: FeaturePipelineOptions = FeaturePipelineOptions(),
    execution_options: PipelineExecutionOptions = PipelineExecutionOptions(),
    import_workers: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    elapsed_seconds_by_stage: dict[str, float] = {}
    written_position_prior_count: int | None = None
    initial_pose_summary = None
    point_colorization_summary = None
    effective_use_position_priors = (
        pose_report_path is not None
        if use_position_priors is None
        else use_position_priors
    )
    if effective_use_position_priors and pose_report_path is None:
        raise ValueError("position priors require pose_report_path")
    if initial_poses_from_pose_report and pose_report_path is None:
        raise ValueError("--initial-poses-from-pose-report requires --pose-report")
    stage_started_at = started_at
    pycolmap_module = require_pycolmap()
    pycolmap_module.set_random_seed(0)
    configure_ssl_cert_file(ssl_cert_file)
    execution = resolve_pipeline_execution(pycolmap_module, execution_options)
    paths = pinhole_workspace_paths(workspace)
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    paths["image_dir"].mkdir(parents=True, exist_ok=True)
    paths["sparse"].mkdir(parents=True, exist_ok=True)
    if paths["database"].exists():
        raise FileExistsError(f"{paths['database']} already exists. Pass --overwrite.")
    record_elapsed_stage(elapsed_seconds_by_stage, "setup", stage_started_at)

    stage_started_at = time.perf_counter()
    import_result = import_pinhole_images_from_render_report(
        render_report_path=render_report_path,
        output_image_dir=paths["image_dir"],
        image_dir=image_dir,
        fail_on_existing=True,
        import_workers=import_workers,
    )
    write_pinhole_metadata(paths["metadata"], import_result=import_result)
    record_elapsed_stage(elapsed_seconds_by_stage, "import_pinhole_images", stage_started_at)

    stage_started_at = time.perf_counter()
    pose_prior_summary = (
        None
        if pose_report_path is None
        else build_position_prior_summary(
            pose_report_path=pose_report_path,
            imported_images=import_result.imported_images,
        )
    )
    if pose_prior_summary is not None:
        record_elapsed_stage(
            elapsed_seconds_by_stage,
            "pose_prior_parsing",
            stage_started_at,
        )

    stage_started_at = time.perf_counter()
    extract_panorama_features_to_database(
        pycolmap_module,
        database_path=paths["database"],
        image_dir=paths["image_dir"],
        camera=import_result.camera,
        feature_options=feature_options,
        execution=execution,
    )
    record_elapsed_stage(elapsed_seconds_by_stage, "feature_extraction", stage_started_at)

    stage_started_at = time.perf_counter()
    generated_image_batch = read_generated_image_batch(
        paths["database"],
        import_result.image_names,
    )
    feature_report = read_feature_stats_by_database_camera(paths["database"])
    feature_health = pinhole_feature_health_summary(
        feature_report,
        expected_image_count=len(import_result.image_names),
    )
    record_elapsed_stage(elapsed_seconds_by_stage, "feature_stats", stage_started_at)

    if effective_use_position_priors:
        if pose_prior_summary is None:
            raise ValueError("position priors require pose_report_path")
        stage_started_at = time.perf_counter()
        written_position_prior_count = write_position_priors_to_database(
            pycolmap_module,
            database_path=paths["database"],
            prior_summary=pose_prior_summary,
        )
        record_elapsed_stage(
            elapsed_seconds_by_stage,
            "pose_prior_database_write",
            stage_started_at,
        )

    if initial_poses_from_pose_report:
        stage_started_at = time.perf_counter()
        initial_pose_summary = write_initial_pose_model_from_pose_report(
            pose_report_path=pose_report_path,  # type: ignore[arg-type]
            model_path=paths["initial_model"],
            imported_images=import_result.imported_images,
            generated_image_batch=generated_image_batch,
            camera=import_result.camera,
        )
        record_elapsed_stage(
            elapsed_seconds_by_stage,
            "initial_pose_model_write",
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
        matcher="sequential",
        matching_options=matching_options,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=None,
        device=execution["device"],
    )
    record_elapsed_stage(elapsed_seconds_by_stage, "matching", stage_started_at)

    stage_started_at = time.perf_counter()
    mapper_options = create_mapper_options(pycolmap_module, execution)
    if effective_use_position_priors:
        mapper_options.use_prior_position = True
    reconstructions = run_incremental_mapping(
        pycolmap_module,
        paths,
        image_dir=paths["image_dir"],
        mapper_options=mapper_options,
        input_path=(
            initial_pose_summary.model_path
            if initial_pose_summary is not None
            else None
        ),
    )
    record_elapsed_stage(elapsed_seconds_by_stage, "incremental_mapping", stage_started_at)
    if not reconstructions:
        raise RuntimeError("COLMAP did not produce any sparse reconstruction")

    stage_started_at = time.perf_counter()
    best_model_id, best_reconstruction = max(
        reconstructions.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "best_model_selection",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    point_colorization_summary = colorize_reconstruction_points(
        best_reconstruction,
        image_dir=paths["image_dir"],
        workers=4,
    )
    if (
        point_colorization_summary.point_count > 0
        and point_colorization_summary.colored_point_count == 0
    ):
        raise RuntimeError(
            "COLMAP produced sparse points, but none could be colorized from "
            "the rendered pinhole images.",
        )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "point_colorization",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    write_best_model(best_reconstruction, paths)
    record_elapsed_stage(elapsed_seconds_by_stage, "best_model_write", stage_started_at)

    stage_started_at = time.perf_counter()
    report = build_pinhole_reconstruction_report(
        pycolmap_version=pycolmap_module.__version__,
        paths=paths,
        import_result=import_result,
        matcher="sequential",
        overlap=overlap,
        loop_detection=loop_detection,
        ssl_cert_file=ssl_cert_file,
        feature_options=feature_options,
        execution=execution,
        feature_report=feature_report,
        feature_health=feature_health,
        generated_image_batch=generated_image_batch,
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
        import_workers=import_workers,
        pose_prior_summary=pose_prior_summary,
        use_position_priors=effective_use_position_priors,
        written_position_prior_count=written_position_prior_count,
        initial_pose_summary=initial_pose_summary,
        point_colorization_summary=point_colorization_summary,
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
    record_elapsed_stage(elapsed_seconds_by_stage, "report_writing", stage_started_at)
    elapsed_seconds = time.perf_counter() - started_at
    report["elapsed_seconds"] = elapsed_seconds
    report["elapsed_time"] = format_elapsed_seconds(elapsed_seconds)
    report["elapsed_seconds_by_stage"] = elapsed_seconds_by_stage
    report["elapsed_time_by_stage"] = format_elapsed_seconds_by_stage(
        elapsed_seconds_by_stage,
    )
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
