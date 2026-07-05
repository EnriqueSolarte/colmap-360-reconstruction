"""Import-only pipeline for rendered pinhole COLMAP workspaces."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from colmap.panorama.reports import (
    format_elapsed_seconds,
    format_elapsed_seconds_by_stage,
)
from colmap.pinhole.image_import import import_pinhole_images_from_render_report
from colmap.pinhole.paths import pinhole_workspace_paths
from colmap.pinhole.reports import (
    build_pinhole_import_report,
    write_pinhole_metadata,
)


def run_pinhole_image_import(
    render_report_path: Path,
    workspace: Path,
    *,
    image_dir: Path | None = None,
    import_workers: int = 1,
    overwrite: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    paths = pinhole_workspace_paths(workspace)
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
            "Import workspace already contains COLMAP pinhole outputs. "
            "Pass --overwrite to rebuild: "
            f"{existing_outputs[0]}",
        )
    workspace.mkdir(parents=True, exist_ok=True)

    stage_started_at = time.perf_counter()
    import_result = import_pinhole_images_from_render_report(
        render_report_path=render_report_path,
        output_image_dir=paths["image_dir"],
        image_dir=image_dir,
        fail_on_existing=True,
        import_workers=import_workers,
        show_progress=show_progress,
    )
    import_elapsed_seconds = time.perf_counter() - stage_started_at
    elapsed_seconds_by_stage = {
        "import_pinhole_images": import_elapsed_seconds,
    }

    write_pinhole_metadata(paths["metadata"], import_result=import_result)
    report = build_pinhole_import_report(
        paths=paths,
        import_result=import_result,
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

