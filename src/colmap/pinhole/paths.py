"""Workspace path helpers for pinhole reconstruction."""

from __future__ import annotations

from pathlib import Path


def pinhole_workspace_paths(workspace: Path) -> dict[str, Path]:
    return {
        "workspace": workspace,
        "image_dir": workspace / "images",
        "database": workspace / "database.db",
        "feature_shards": workspace / "feature_shards",
        "sparse": workspace / "sparse",
        "best_sparse": workspace / "sparse_best",
        "initial_model": workspace / "initial_model",
        "raw_ply": workspace / "cloud_colmap_raw.ply",
        "report": workspace / "pinhole_reconstruction_report.json",
        "import_report": workspace / "pinhole_import_report.json",
        "metadata": workspace / "pinhole_config.json",
        "match_pairs": workspace / "match_pairs.txt",
    }
