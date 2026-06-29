"""Workspace path helpers for panorama reconstruction."""

from __future__ import annotations

from pathlib import Path


def panorama_workspace_paths(workspace: Path) -> dict[str, Path]:
    return {
        "workspace": workspace,
        "image_dir": workspace / "images",
        "database": workspace / "database.db",
        "sparse": workspace / "sparse",
        "best_sparse": workspace / "sparse_best",
        "raw_ply": workspace / "cloud_colmap_raw.ply",
        "report": workspace / "panorama_equirectangular_report.json",
        "metadata": workspace / "panorama_equirectangular_config.json",
        "match_pairs": workspace / "match_pairs.txt",
    }
