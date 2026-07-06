"""Colorize COLMAP sparse points from rendered pinhole image observations."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


COLOR_SOURCE = "mean_rgb_from_observed_pinhole_pixels"


@dataclass(frozen=True)
class PointColorizationSummary:
    image_dir: Path
    point_count: int
    track_observation_count: int
    sampled_observation_count: int
    colored_point_count: int
    black_point_count_before: int
    black_point_count_after: int
    requested_image_count: int
    loaded_image_count: int
    missing_image_count: int
    missing_image_observation_count: int
    missing_point2d_count: int
    out_of_bounds_observation_count: int
    workers: int

    def report(self) -> dict[str, Any]:
        return {
            "source": COLOR_SOURCE,
            "image_dir": str(self.image_dir),
            "point_count": self.point_count,
            "track_observation_count": self.track_observation_count,
            "sampled_observation_count": self.sampled_observation_count,
            "colored_point_count": self.colored_point_count,
            "black_point_count_before": self.black_point_count_before,
            "black_point_count_after": self.black_point_count_after,
            "requested_image_count": self.requested_image_count,
            "loaded_image_count": self.loaded_image_count,
            "missing_image_count": self.missing_image_count,
            "missing_image_observation_count": self.missing_image_observation_count,
            "missing_point2d_count": self.missing_point2d_count,
            "out_of_bounds_observation_count": self.out_of_bounds_observation_count,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class PointColorRequest:
    point3d_id: int
    x: int
    y: int


@dataclass(frozen=True)
class ImageColorizationResult:
    color_sums: dict[int, np.ndarray]
    sample_counts: dict[int, int]
    sampled_observation_count: int
    loaded_image_count: int
    missing_image_count: int
    missing_image_observation_count: int
    out_of_bounds_observation_count: int


def colorize_reconstruction_points(
    reconstruction: Any,
    *,
    image_dir: Path,
    show_progress: bool = True,
    workers: int = 4,
) -> PointColorizationSummary:
    """Assign each 3D point the mean RGB of its observed rendered-image pixels."""
    if workers <= 0:
        raise ValueError("workers must be positive")

    image_by_id = dict(reconstruction.images.items())
    requests_by_image_id: dict[int, list[PointColorRequest]] = defaultdict(list)
    point_count = 0
    track_observation_count = 0
    black_point_count_before = 0
    missing_point2d_count = 0

    point_items = reconstruction.points3D.items()
    for point3d_id, point3d in tqdm(
        point_items,
        desc="Indexing sparse point tracks",
        unit="point",
        total=len(reconstruction.points3D),
        disable=not show_progress,
    ):
        point_count += 1
        if is_black_color(point3d.color):
            black_point_count_before += 1
        for track_element in point3d.track.elements:
            track_observation_count += 1
            image = image_by_id.get(int(track_element.image_id))
            if image is None:
                missing_point2d_count += 1
                continue
            try:
                point2d = image.point2D(int(track_element.point2D_idx))
                xy = np.asarray(point2d.xy, dtype=np.float64)
            except (IndexError, TypeError, ValueError, AttributeError):
                missing_point2d_count += 1
                continue
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                missing_point2d_count += 1
                continue
            requests_by_image_id[int(track_element.image_id)].append(
                PointColorRequest(
                    point3d_id=int(point3d_id),
                    x=int(round(float(xy[0]))),
                    y=int(round(float(xy[1]))),
                ),
            )

    color_sums: dict[int, np.ndarray] = {}
    sample_counts: dict[int, int] = {}
    sampled_observation_count = 0
    loaded_image_count = 0
    missing_image_count = 0
    missing_image_observation_count = 0
    out_of_bounds_observation_count = 0

    image_jobs = tuple(
        (image_dir / image_by_id[image_id].name, tuple(requests))
        for image_id, requests in requests_by_image_id.items()
    )
    if workers == 1:
        results = (
            sample_image_color_requests(image_path, requests)
            for image_path, requests in tqdm(
                image_jobs,
                desc="Sampling pinhole image colors",
                unit="image",
                total=len(image_jobs),
                disable=not show_progress,
            )
        )
        for result in results:
            merge_image_colorization_result(
                result,
                color_sums=color_sums,
                sample_counts=sample_counts,
            )
            sampled_observation_count += result.sampled_observation_count
            loaded_image_count += result.loaded_image_count
            missing_image_count += result.missing_image_count
            missing_image_observation_count += result.missing_image_observation_count
            out_of_bounds_observation_count += result.out_of_bounds_observation_count
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(sample_image_color_requests, image_path, requests)
                for image_path, requests in image_jobs
            ]
            for future in tqdm(
                as_completed(futures),
                desc="Sampling pinhole image colors",
                unit="image",
                total=len(futures),
                disable=not show_progress,
            ):
                result = future.result()
                merge_image_colorization_result(
                    result,
                    color_sums=color_sums,
                    sample_counts=sample_counts,
                )
                sampled_observation_count += result.sampled_observation_count
                loaded_image_count += result.loaded_image_count
                missing_image_count += result.missing_image_count
                missing_image_observation_count += (
                    result.missing_image_observation_count
                )
                out_of_bounds_observation_count += (
                    result.out_of_bounds_observation_count
                )

    colored_point_count = 0
    for point3d_id, color_sum in tqdm(
        color_sums.items(),
        desc="Assigning sparse point colors",
        unit="point",
        total=len(color_sums),
        disable=not show_progress,
    ):
        count = sample_counts[point3d_id]
        color = np.rint(color_sum / count).clip(0, 255).astype(np.uint8)
        reconstruction.points3D[point3d_id].color = color
        colored_point_count += 1

    black_point_count_after = sum(
        1
        for point3d in reconstruction.points3D.values()
        if is_black_color(point3d.color)
    )
    return PointColorizationSummary(
        image_dir=image_dir,
        point_count=point_count,
        track_observation_count=track_observation_count,
        sampled_observation_count=sampled_observation_count,
        colored_point_count=colored_point_count,
        black_point_count_before=black_point_count_before,
        black_point_count_after=black_point_count_after,
        requested_image_count=len(requests_by_image_id),
        loaded_image_count=loaded_image_count,
        missing_image_count=missing_image_count,
        missing_image_observation_count=missing_image_observation_count,
        missing_point2d_count=missing_point2d_count,
        out_of_bounds_observation_count=out_of_bounds_observation_count,
        workers=workers,
    )


def colorize_reconstruction_model(
    pycolmap_module: Any,
    *,
    model_path: Path,
    image_dir: Path,
    output_model_path: Path,
    output_ply_path: Path | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
    workers: int = 4,
) -> PointColorizationSummary:
    reconstruction = pycolmap_module.Reconstruction(model_path)
    summary = colorize_reconstruction_points(
        reconstruction,
        image_dir=image_dir,
        show_progress=show_progress,
        workers=workers,
    )
    if summary.point_count > 0 and summary.colored_point_count == 0:
        raise RuntimeError(
            "No sparse points could be colorized. Check that --image-dir contains "
            "the registered COLMAP image names.",
        )
    write_reconstruction_outputs(
        reconstruction,
        output_model_path=output_model_path,
        output_ply_path=output_ply_path,
        overwrite=overwrite,
    )
    return summary


def write_reconstruction_outputs(
    reconstruction: Any,
    *,
    output_model_path: Path,
    output_ply_path: Path | None = None,
    overwrite: bool = False,
) -> None:
    if output_model_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output COLMAP model already exists: {output_model_path}. "
            "Pass --overwrite or choose a different --output-model.",
        )
    temporary_path = output_model_path.with_name(
        f".{output_model_path.name}.colorizing-tmp",
    )
    if temporary_path.exists():
        shutil.rmtree(temporary_path)
    temporary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(temporary_path)
    if output_model_path.exists():
        shutil.rmtree(output_model_path)
    shutil.move(str(temporary_path), str(output_model_path))
    if output_ply_path is not None:
        output_ply_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruction.export_PLY(output_ply_path)


def infer_image_dir(model_path: Path) -> Path:
    candidates = (
        model_path.parent / "images",
        model_path.parent.parent / "images",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return model_path.parent / "images"


def default_output_model_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.name}_colorized")


def sample_image_color_requests(
    image_path: Path,
    requests: tuple[PointColorRequest, ...],
) -> ImageColorizationResult:
    color_sums: dict[int, np.ndarray] = {}
    sample_counts: dict[int, int] = {}
    sampled_observation_count = 0
    out_of_bounds_observation_count = 0
    if not image_path.is_file():
        return ImageColorizationResult(
            color_sums=color_sums,
            sample_counts=sample_counts,
            sampled_observation_count=0,
            loaded_image_count=0,
            missing_image_count=1,
            missing_image_observation_count=len(requests),
            out_of_bounds_observation_count=0,
        )

    pixels = load_rgb_image(image_path)
    height, width = pixels.shape[:2]
    for request in requests:
        if (
            request.x < 0
            or request.x >= width
            or request.y < 0
            or request.y >= height
        ):
            out_of_bounds_observation_count += 1
            continue
        color = pixels[request.y, request.x].astype(np.float64)
        color_sums.setdefault(request.point3d_id, np.zeros(3, dtype=np.float64))
        color_sums[request.point3d_id] += color
        sample_counts[request.point3d_id] = (
            sample_counts.get(request.point3d_id, 0) + 1
        )
        sampled_observation_count += 1

    return ImageColorizationResult(
        color_sums=color_sums,
        sample_counts=sample_counts,
        sampled_observation_count=sampled_observation_count,
        loaded_image_count=1,
        missing_image_count=0,
        missing_image_observation_count=0,
        out_of_bounds_observation_count=out_of_bounds_observation_count,
    )


def merge_image_colorization_result(
    result: ImageColorizationResult,
    *,
    color_sums: dict[int, np.ndarray],
    sample_counts: dict[int, int],
) -> None:
    for point3d_id, color_sum in result.color_sums.items():
        color_sums.setdefault(point3d_id, np.zeros(3, dtype=np.float64))
        color_sums[point3d_id] += color_sum
        sample_counts[point3d_id] = (
            sample_counts.get(point3d_id, 0)
            + result.sample_counts[point3d_id]
        )


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def is_black_color(color: Any) -> bool:
    values = np.asarray(color, dtype=np.uint8)
    return bool(values.shape == (3,) and np.all(values == 0))
