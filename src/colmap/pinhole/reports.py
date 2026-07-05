"""Report and metadata helpers for pinhole reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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

