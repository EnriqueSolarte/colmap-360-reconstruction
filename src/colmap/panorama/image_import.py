"""Panorama image discovery and import helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from colmap.camera import ColmapCamera, camera_from_equirectangular
from colmap.panorama.constants import (
    FILENAME_TIMESTAMP_PATTERN,
    IMAGE_EXTENSIONS,
)


def hardlink_or_symlink_file(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except FileExistsError:
        if os.path.samefile(source, destination):
            return destination
        raise
    except OSError:
        try:
            os.symlink(os.path.abspath(source), destination)
        except FileExistsError:
            if os.path.samefile(source, destination):
                return destination
            raise
    return destination


def panorama_image_paths(input_image_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_image_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=panorama_timestamp_sort_key,
    )


def panorama_timestamp_sort_key(path: Path) -> tuple[int, int | str, str]:
    timestamp = timestamp_from_image_name(path.name)
    if timestamp is None:
        return (1, path.name, path.name)
    return (0, timestamp, path.name)


def import_equirectangular_panorama_images(
    input_image_path: Path,
    output_image_dir: Path,
    *,
    image_limit: int | None = None,
    fail_on_existing: bool = False,
    relative_name_prefix: Path | None = None,
    import_workers: int = 1,
) -> tuple[list[Path], ColmapCamera, list[str]]:
    if import_workers <= 0:
        raise ValueError("import_workers must be positive")
    panorama_paths = panorama_image_paths(input_image_path)
    if image_limit is not None:
        panorama_paths = panorama_paths[:image_limit]
    if not panorama_paths:
        raise RuntimeError(f"No panorama images found in {input_image_path}")

    first_width, first_height = image_resolution(panorama_paths[0])
    camera = camera_from_equirectangular(first_width, first_height)

    def import_panorama(panorama_path: Path) -> str:
        width, height = image_resolution(panorama_path)
        if (width, height) != (first_width, first_height):
            raise ValueError(
                f"Panorama {panorama_path} has size {width}x{height}; "
                f"expected {first_width}x{first_height}"
            )
        relative_name = prefixed_panorama_relative_name(
            input_image_path,
            panorama_path,
            relative_name_prefix,
        )
        output_path = output_image_dir / relative_name
        if fail_on_existing and output_path.exists():
            raise FileExistsError(
                "New panorama image would overwrite a previous session file: "
                f"{output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hardlink_or_symlink_file(str(panorama_path), str(output_path))
        return relative_name.as_posix()

    progress_options = {
        "desc": "Importing equirectangular panoramas",
        "unit": "image",
        "total": len(panorama_paths),
    }
    if import_workers == 1:
        image_names = [
            import_panorama(panorama_path)
            for panorama_path in tqdm(panorama_paths, **progress_options)
        ]
    else:
        with ThreadPoolExecutor(max_workers=import_workers) as executor:
            image_names = list(
                tqdm(
                    executor.map(import_panorama, panorama_paths),
                    **progress_options,
                ),
            )

    return panorama_paths, camera, image_names


def image_resolution(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def prefixed_panorama_relative_name(
    input_image_path: Path,
    panorama_path: Path,
    relative_name_prefix: Path | None = None,
) -> Path:
    relative_name = panorama_path.relative_to(input_image_path)
    if relative_name_prefix is None:
        return relative_name
    return Path(f"{relative_name_prefix.as_posix()}_{relative_name.as_posix()}")


def timestamp_from_image_name(name: str) -> int | None:
    match = FILENAME_TIMESTAMP_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1))
