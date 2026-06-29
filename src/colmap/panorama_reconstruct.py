"""Compatibility facade for the panorama reconstruction CLI and helpers."""

from __future__ import annotations

from colmap.panorama.cli import (
    DEFAULT_WORKSPACE,
    PROJECT_ROOT,
    PanoramaArgumentDefaultsHelpFormatter,
    main,
    parse_args,
)
from colmap.panorama.constants import (
    COLMAP_IMAGE_PAIR_ID_PRIME,
    FILENAME_TIMESTAMP_PATTERN,
    IMAGE_EXTENSIONS,
    NATIVE_CAMERA_SETUP,
)
from colmap.panorama.database import (
    decode_colmap_image_pair_id,
    read_database_stats,
    read_feature_stats_by_camera,
    read_feature_stats_by_database_camera,
    read_frame_connection_stats,
    read_generated_image_batch,
    read_image_records_by_name,
)
from colmap.panorama.image_import import (
    hardlink_or_symlink_file,
    image_resolution,
    import_equirectangular_panorama_images,
    panorama_image_paths,
    panorama_timestamp_sort_key,
    prefixed_panorama_relative_name,
    timestamp_from_image_name,
)
from colmap.panorama.paths import panorama_workspace_paths
from colmap.panorama.pipeline import record_elapsed_stage, run_panorama_reconstruction
from colmap.panorama.pycolmap_adapter import (
    configure_ssl_cert_file,
    create_feature_extraction_options,
    create_feature_matching_options,
    create_mapper_options,
    first_gpu_index,
    match_features,
    parse_gpu_index,
    require_pycolmap,
    resolve_pipeline_execution,
    run_incremental_mapping,
    validate_gpu_index,
    validate_vocab_tree_path,
    write_best_model,
)
from colmap.panorama.reports import (
    build_panorama_report,
    equirectangular_feature_health_summary,
    format_elapsed_seconds,
    format_elapsed_seconds_by_stage,
    reconstruction_stats,
    write_equirectangular_metadata,
    write_image_pair_list,
)
from colmap.panorama.scoring import (
    colmap_frame_id_for_image,
    linear_capped_score,
    log_capped_score,
    normalized_entropy,
    score_keyframe_anchors,
)
from colmap.panorama.types import (
    FeaturePipelineOptions,
    GeneratedImageBatch,
    KeyframeAnchorScore,
    PipelineExecutionOptions,
    optional_report_float,
    round_report_float,
)

__all__ = [
    "COLMAP_IMAGE_PAIR_ID_PRIME",
    "DEFAULT_WORKSPACE",
    "FILENAME_TIMESTAMP_PATTERN",
    "FeaturePipelineOptions",
    "GeneratedImageBatch",
    "IMAGE_EXTENSIONS",
    "KeyframeAnchorScore",
    "NATIVE_CAMERA_SETUP",
    "PROJECT_ROOT",
    "PanoramaArgumentDefaultsHelpFormatter",
    "PipelineExecutionOptions",
    "build_panorama_report",
    "colmap_frame_id_for_image",
    "configure_ssl_cert_file",
    "create_feature_extraction_options",
    "create_feature_matching_options",
    "create_mapper_options",
    "decode_colmap_image_pair_id",
    "equirectangular_feature_health_summary",
    "first_gpu_index",
    "format_elapsed_seconds",
    "format_elapsed_seconds_by_stage",
    "hardlink_or_symlink_file",
    "image_resolution",
    "import_equirectangular_panorama_images",
    "linear_capped_score",
    "log_capped_score",
    "main",
    "match_features",
    "normalized_entropy",
    "optional_report_float",
    "panorama_image_paths",
    "panorama_timestamp_sort_key",
    "panorama_workspace_paths",
    "parse_args",
    "parse_gpu_index",
    "prefixed_panorama_relative_name",
    "read_database_stats",
    "read_feature_stats_by_camera",
    "read_feature_stats_by_database_camera",
    "read_frame_connection_stats",
    "read_generated_image_batch",
    "read_image_records_by_name",
    "reconstruction_stats",
    "record_elapsed_stage",
    "require_pycolmap",
    "resolve_pipeline_execution",
    "round_report_float",
    "run_incremental_mapping",
    "run_panorama_reconstruction",
    "score_keyframe_anchors",
    "timestamp_from_image_name",
    "validate_gpu_index",
    "validate_vocab_tree_path",
    "write_best_model",
    "write_equirectangular_metadata",
    "write_image_pair_list",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
