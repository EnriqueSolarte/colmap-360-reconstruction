"""Feature-only panorama workspace pipeline."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from colmap.camera import ColmapCamera
from colmap.panorama.database import (
    read_feature_stats_by_database_camera,
    read_generated_image_batch,
)
from colmap.panorama.import_pipeline import import_panorama_images_to_workspace
from colmap.panorama.paths import panorama_workspace_paths
from colmap.panorama.pycolmap_adapter import (
    configure_ssl_cert_file,
    create_feature_extraction_options,
    require_pycolmap,
    resolve_pipeline_execution,
)
from colmap.panorama.reports import (
    build_panorama_feature_report,
    equirectangular_feature_health_summary,
    format_elapsed_seconds,
    format_elapsed_seconds_by_stage,
)
from colmap.panorama.types import FeaturePipelineOptions, PipelineExecutionOptions


@dataclass(frozen=True)
class FeatureShardExtractionResult:
    batches: tuple[tuple[str, ...], ...]
    shard_database_paths: tuple[Path, ...]
    merged_database_path: Path
    shard_elapsed_seconds: tuple[float, ...] = ()
    merge_elapsed_seconds: float = 0.0


def split_image_names_into_batches(
    image_names: Sequence[str],
    *,
    batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return tuple(
        tuple(image_names[index : index + batch_size])
        for index in range(0, len(image_names), batch_size)
    )


def compute_feature_batch_size(
    *,
    image_count: int,
    feature_workers: int,
) -> int:
    if image_count <= 0:
        raise ValueError("image_count must be positive")
    if feature_workers <= 0:
        raise ValueError("feature_workers must be positive")
    return math.ceil(image_count / feature_workers)


def extract_panorama_features_to_database(
    pycolmap_module: Any,
    *,
    database_path: Path,
    image_dir: Path,
    camera: ColmapCamera,
    feature_options: FeaturePipelineOptions,
    execution: dict[str, Any],
    image_names: Sequence[str] | None = None,
) -> None:
    extraction_options = create_feature_extraction_options(
        pycolmap_module,
        feature_options,
        use_gpu=execution["resolved_device"] == "cuda",
        gpu_index=execution["gpu_index"],
    )
    reader_options = pycolmap_module.ImageReaderOptions()
    reader_options.camera_model = camera.model
    reader_options.camera_params = camera.params_string
    kwargs: dict[str, Any] = {}
    if image_names is not None:
        kwargs["image_names"] = image_names
    pycolmap_module.extract_features(
        database_path,
        image_dir,
        reader_options=reader_options,
        camera_mode=pycolmap_module.CameraMode.SINGLE,
        extraction_options=extraction_options,
        device=execution["device"],
        **kwargs,
    )


def merge_feature_shard_databases(
    pycolmap_module: Any,
    *,
    shard_database_paths: Sequence[Path],
    merged_database_path: Path,
) -> None:
    if not shard_database_paths:
        raise ValueError("at least one shard database is required")
    if merged_database_path.exists():
        raise FileExistsError(f"{merged_database_path} already exists")
    merged_database_path.parent.mkdir(parents=True, exist_ok=True)
    if len(shard_database_paths) == 1:
        shutil.copy2(shard_database_paths[0], merged_database_path)
        return

    with tempfile.TemporaryDirectory(
        prefix="feature-shard-merge-",
        dir=merged_database_path.parent,
    ) as merge_dir:
        merge_dir_path = Path(merge_dir)
        current_database_path = merge_dir_path / "merged_0000.db"
        _merge_two_feature_databases(
            pycolmap_module,
            shard_database_paths[0],
            shard_database_paths[1],
            current_database_path,
        )
        for merge_index, shard_database_path in enumerate(
            shard_database_paths[2:],
            start=1,
        ):
            next_database_path = merge_dir_path / f"merged_{merge_index:04d}.db"
            _merge_two_feature_databases(
                pycolmap_module,
                current_database_path,
                shard_database_path,
                next_database_path,
            )
            current_database_path = next_database_path
        shutil.move(str(current_database_path), merged_database_path)


def _merge_two_feature_databases(
    pycolmap_module: Any,
    first_database_path: Path,
    second_database_path: Path,
    merged_database_path: Path,
) -> None:
    with (
        pycolmap_module.Database.open(first_database_path) as first_database,
        pycolmap_module.Database.open(second_database_path) as second_database,
        pycolmap_module.Database.open(merged_database_path) as merged_database,
    ):
        pycolmap_module.Database.merge(
            first_database,
            second_database,
            merged_database,
        )


def extract_panorama_features_to_sharded_database(
    pycolmap_module: Any,
    *,
    database_path: Path,
    image_dir: Path,
    shard_dir: Path,
    camera: ColmapCamera,
    feature_options: FeaturePipelineOptions,
    execution: dict[str, Any],
    image_names: Sequence[str],
    batch_size: int,
    feature_workers: int = 1,
) -> FeatureShardExtractionResult:
    if feature_workers <= 0:
        raise ValueError("feature_workers must be positive")
    batches = split_image_names_into_batches(image_names, batch_size=batch_size)
    if not batches:
        raise ValueError("at least one image name is required")
    if database_path.exists():
        raise FileExistsError(f"{database_path} already exists")
    shard_dir.mkdir(parents=True, exist_ok=True)

    def extract_batch(
        batch_index: int,
        batch_names: tuple[str, ...],
    ) -> tuple[Path, float]:
        shard_database_path = shard_dir / f"features_{batch_index:04d}.db"
        if shard_database_path.exists():
            raise FileExistsError(f"{shard_database_path} already exists")
        started_at = time.perf_counter()
        extract_panorama_features_to_database(
            pycolmap_module,
            database_path=shard_database_path,
            image_dir=image_dir,
            camera=camera,
            feature_options=feature_options,
            execution=execution,
            image_names=batch_names,
        )
        return shard_database_path, time.perf_counter() - started_at

    if feature_workers == 1:
        extraction_results = [
            extract_batch(batch_index, batch_names)
            for batch_index, batch_names in enumerate(batches)
        ]
    else:
        with ThreadPoolExecutor(max_workers=feature_workers) as executor:
            futures = [
                executor.submit(extract_batch, batch_index, batch_names)
                for batch_index, batch_names in enumerate(batches)
            ]
            extraction_results = [future.result() for future in futures]

    shard_database_paths = [path for path, _elapsed in extraction_results]
    shard_elapsed_seconds = tuple(elapsed for _path, elapsed in extraction_results)
    print(
        "Feature extraction batches finished; merging "
        f"{len(shard_database_paths)} shard database(s) into {database_path}",
        flush=True,
    )
    merge_started_at = time.perf_counter()
    merge_feature_shard_databases(
        pycolmap_module,
        shard_database_paths=shard_database_paths,
        merged_database_path=database_path,
    )
    merge_elapsed_seconds = time.perf_counter() - merge_started_at
    return FeatureShardExtractionResult(
        batches=batches,
        shard_database_paths=tuple(shard_database_paths),
        merged_database_path=database_path,
        shard_elapsed_seconds=shard_elapsed_seconds,
        merge_elapsed_seconds=merge_elapsed_seconds,
    )


def run_panorama_feature_extraction(
    input_image_path: Path,
    workspace: Path,
    *,
    ssl_cert_file: Path | None = None,
    feature_options: FeaturePipelineOptions = FeaturePipelineOptions(),
    execution_options: PipelineExecutionOptions = PipelineExecutionOptions(),
    image_limit: int | None = None,
    import_workers: int = 1,
    feature_workers: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    if feature_workers <= 0:
        raise ValueError("feature_workers must be positive")
    started_at = time.perf_counter()
    elapsed_seconds_by_stage: dict[str, float] = {}
    feature_shard_result: FeatureShardExtractionResult | None = None
    pycolmap_module = require_pycolmap()
    pycolmap_module.set_random_seed(0)
    configure_ssl_cert_file(ssl_cert_file)
    execution = resolve_pipeline_execution(pycolmap_module, execution_options)
    paths = panorama_workspace_paths(workspace)
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    existing_outputs = [
        path
        for path in (
            paths["database"],
            paths["feature_report"],
            paths["report"],
            paths["sparse"],
            paths["best_sparse"],
            paths["raw_ply"],
        )
        if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Feature workspace already contains COLMAP panorama outputs. "
            "Pass --overwrite to rebuild: "
            f"{existing_outputs[0]}"
        )
    workspace.mkdir(parents=True, exist_ok=True)

    stage_started_at = time.perf_counter()
    panorama_paths, camera, generated_image_names = import_panorama_images_to_workspace(
        input_image_path=input_image_path,
        paths=paths,
        image_limit=image_limit,
        import_workers=import_workers,
    )
    elapsed_seconds_by_stage["import_equirectangular_images"] = (
        time.perf_counter() - stage_started_at
    )

    stage_started_at = time.perf_counter()
    feature_batch_size = None
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
        feature_batch_size = compute_feature_batch_size(
            image_count=len(generated_image_names),
            feature_workers=feature_workers,
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
            batch_size=feature_batch_size,
            feature_workers=feature_workers,
        )
    elapsed_seconds_by_stage["feature_extraction"] = (
        time.perf_counter() - stage_started_at
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
    elapsed_seconds_by_stage["feature_stats"] = time.perf_counter() - stage_started_at

    report = build_panorama_feature_report(
        pycolmap_version=pycolmap_module.__version__,
        input_image_path=input_image_path,
        paths=paths,
        camera=camera,
        panorama_count=len(panorama_paths),
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
    )
    elapsed_seconds = time.perf_counter() - started_at
    report["elapsed_seconds"] = elapsed_seconds
    report["elapsed_time"] = format_elapsed_seconds(elapsed_seconds)
    report["elapsed_seconds_by_stage"] = elapsed_seconds_by_stage
    report["elapsed_time_by_stage"] = format_elapsed_seconds_by_stage(
        elapsed_seconds_by_stage,
    )
    paths["feature_report"].write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
