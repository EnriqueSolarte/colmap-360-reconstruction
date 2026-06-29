"""Import-only panorama workspace pipeline."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from colmap.panorama.image_import import import_equirectangular_panorama_images
from colmap.panorama.paths import panorama_workspace_paths
from colmap.panorama.reports import (
    build_panorama_import_report,
    format_elapsed_seconds,
    format_elapsed_seconds_by_stage,
    write_equirectangular_metadata,
)


def import_panorama_images_to_workspace(
    *,
    input_image_path: Path,
    paths: dict[str, Path],
    image_limit: int | None = None,
    import_workers: int = 1,
    fail_on_existing: bool = False,
) -> tuple[list[Path], Any, list[str]]:
    paths["image_dir"].mkdir(parents=True, exist_ok=True)
    panorama_paths, camera, image_names = import_equirectangular_panorama_images(
        input_image_path=input_image_path,
        output_image_dir=paths["image_dir"],
        image_limit=image_limit,
        fail_on_existing=fail_on_existing,
        import_workers=import_workers,
    )
    write_equirectangular_metadata(
        paths["metadata"],
        input_image_path,
        camera,
        panorama_paths,
        image_names,
        image_limit=image_limit,
        relative_name_prefix=None,
    )
    return panorama_paths, camera, image_names


def run_panorama_image_import(
    input_image_path: Path,
    workspace: Path,
    *,
    image_limit: int | None = None,
    import_workers: int = 1,
    overwrite: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    paths = panorama_workspace_paths(workspace)
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    existing_outputs = [
        path
        for path in (
            paths["metadata"],
            paths["import_report"],
            paths["database"],
            paths["sparse"],
            paths["best_sparse"],
            paths["raw_ply"],
        )
        if path.exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Import workspace already contains COLMAP panorama outputs. "
            "Pass --overwrite to rebuild: "
            f"{existing_outputs[0]}"
        )
    workspace.mkdir(parents=True, exist_ok=True)

    stage_started_at = time.perf_counter()
    panorama_paths, camera, image_names = import_panorama_images_to_workspace(
        input_image_path=input_image_path,
        paths=paths,
        image_limit=image_limit,
        fail_on_existing=True,
        import_workers=import_workers,
    )
    import_elapsed_seconds = time.perf_counter() - stage_started_at
    elapsed_seconds_by_stage = {
        "import_equirectangular_images": import_elapsed_seconds,
    }

    report = build_panorama_import_report(
        input_image_path=input_image_path,
        paths=paths,
        camera=camera,
        panorama_paths=panorama_paths,
        image_names=image_names,
        image_limit=image_limit,
        import_workers=import_workers,
    )
    elapsed_seconds = time.perf_counter() - started_at
    report["elapsed_seconds"] = elapsed_seconds
    report["elapsed_time"] = format_elapsed_seconds(elapsed_seconds)
    report["elapsed_seconds_by_stage"] = elapsed_seconds_by_stage
    report["elapsed_time_by_stage"] = format_elapsed_seconds_by_stage(
        elapsed_seconds_by_stage,
    )
    paths["import_report"].write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
