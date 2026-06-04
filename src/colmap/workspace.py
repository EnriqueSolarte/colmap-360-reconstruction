"""Workspace preparation helpers for COLMAP reconstruction outputs."""

from __future__ import annotations

import shutil
from pathlib import Path


def prepare_workspace(workspace: Path, overwrite: bool) -> dict[str, Path]:
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    paths = {
        "workspace": workspace,
        "database": workspace / "database.db",
        "sparse": workspace / "sparse",
        "best_sparse": workspace / "sparse_best",
        "raw_ply": workspace / "cloud_colmap_raw.ply",
        "report": workspace / "validation_report.json",
        "input_report": workspace / "colmap_input_report.json",
        "limited_images": workspace / "images_subset",
    }
    paths["sparse"].mkdir(parents=True, exist_ok=True)
    return paths


def fail_if_workspace_has_outputs(paths: dict[str, Path]) -> None:
    existing = [
        paths["database"],
        paths["raw_ply"],
        paths["report"],
        paths["best_sparse"],
        paths["limited_images"],
    ]
    if any(path.exists() for path in existing):
        raise FileExistsError(
            f"{paths['workspace']} already contains COLMAP outputs. "
            "Pass --overwrite to rebuild it."
        )


def prepare_limited_image_dir(images: list[Path], limited_dir: Path) -> Path:
    if limited_dir.exists():
        shutil.rmtree(limited_dir)
    limited_dir.mkdir(parents=True, exist_ok=True)
    for image in images:
        target = limited_dir / image.name
        try:
            target.symlink_to(image)
        except OSError:
            shutil.copy2(image, target)
    return limited_dir
