"""Input discovery and validation for COLMAP pinhole reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from colmap.camera import ColmapCamera

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def first_image_resolution(image_dir: Path, image_limit: int | None = None) -> tuple[int, int]:
    images = limited_image_paths(image_paths(image_dir), image_limit)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    with Image.open(images[0]) as image:
        return image.size


def image_paths(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in IMAGE_EXTENSIONS:
        paths.extend(image_dir.glob(f"*{suffix}"))
        paths.extend(image_dir.glob(f"*{suffix.upper()}"))
    return sorted(set(paths))


def inspect_input(
    image_dir: Path,
    workspace: Path,
    camera: ColmapCamera,
    *,
    image_limit: int | None = None,
    verify_all_resolutions: bool = True,
) -> dict[str, Any]:
    images = limited_image_paths(image_paths(image_dir), image_limit)
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    resolutions = image_resolutions(images if verify_all_resolutions else images[:20])
    expected_resolution = (camera.width, camera.height)
    mismatches = {
        str(path): list(resolution)
        for path, resolution in resolutions.items()
        if resolution != expected_resolution
    }
    if mismatches:
        examples = list(mismatches.items())[:5]
        raise ValueError(
            "Image resolution does not match camera intrinsics. "
            f"Expected {expected_resolution[0]}x{expected_resolution[1]}, "
            f"examples: {examples}"
        )

    report = {
        "status": "colmap_input_inspected",
        "warning": (
            "This report assumes all pinhole images share one virtual PINHOLE "
            "camera. Confirm the original render FOV before trusting COLMAP "
            "intrinsics."
        ),
        "image_dir": str(image_dir),
        "workspace": str(workspace),
        "input_image_count": len(images),
        "image_limit": image_limit,
        "first_image": images[0].name,
        "last_image": images[-1].name,
        "sample_images": [path.name for path in images[:5]],
        "image_resolution": {
            "width": expected_resolution[0],
            "height": expected_resolution[1],
        },
        "camera": camera.report(),
    }
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "colmap_input_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def image_resolutions(paths: list[Path]) -> dict[Path, tuple[int, int]]:
    resolutions: dict[Path, tuple[int, int]] = {}
    for path in paths:
        with Image.open(path) as image:
            resolutions[path] = image.size
    return resolutions


def limited_image_paths(paths: list[Path], image_limit: int | None) -> list[Path]:
    if image_limit is None:
        return paths
    if image_limit <= 0:
        raise ValueError("image_limit must be positive")
    return paths[:image_limit]
