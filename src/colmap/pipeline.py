"""Raw pycolmap reconstruction pipeline."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from colmap.camera import ColmapCamera


@dataclass(frozen=True)
class ColmapPipelineResult:
    reconstructions: dict[int, Any]
    best_model_id: int
    best_reconstruction: Any


def run_colmap_pipeline(
    pycolmap_module: Any,
    reconstruction_image_dir: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    *,
    matching: str,
    overlap: int,
    max_num_features: int,
    optimize_intrinsics: bool,
) -> ColmapPipelineResult:
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
    write_best_model(best_reconstruction, paths)
    return ColmapPipelineResult(
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
    )


def write_best_model(best_reconstruction: Any, paths: dict[str, Path]) -> None:
    if paths["best_sparse"].exists():
        shutil.rmtree(paths["best_sparse"])
    paths["best_sparse"].mkdir(parents=True, exist_ok=True)
    best_reconstruction.write(paths["best_sparse"])
    best_reconstruction.export_PLY(paths["raw_ply"])
