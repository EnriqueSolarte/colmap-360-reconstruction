"""COLMAP model filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def is_colmap_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_names = ("cameras", "images", "points3D")
    return all(
        any((path / f"{name}.{extension}").is_file() for extension in ("bin", "txt"))
        for name in required_names
    )
