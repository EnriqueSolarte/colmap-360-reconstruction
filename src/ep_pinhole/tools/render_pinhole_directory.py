"""Render multiple pinhole images from every EP image in a directory."""

from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path

import numpy as np
from PIL import Image

from ep_pinhole import EquirectangularToPinhole
from ep_pinhole.defaults import (
    DEFAULT_EP_LATITUDE,
    DEFAULT_EP_LATITUDE_MAX,
    DEFAULT_EP_LATITUDE_MIN,
    DEFAULT_EP_LONGITUDE_MAX,
    DEFAULT_EP_LONGITUDE_MIN,
    DEFAULT_PINHOLE_FOV_X,
    DEFAULT_PINHOLE_FOV_Y,
    DEFAULT_PINHOLE_HEIGHT,
    DEFAULT_PINHOLE_INTERPOLATION,
    DEFAULT_PINHOLE_WIDTH,
)

DEFAULT_WIDTH = DEFAULT_PINHOLE_WIDTH
DEFAULT_HEIGHT = DEFAULT_PINHOLE_HEIGHT
DEFAULT_FOV_X = DEFAULT_PINHOLE_FOV_X
DEFAULT_FOV_Y = DEFAULT_PINHOLE_FOV_Y
DEFAULT_INTERPOLATION = DEFAULT_PINHOLE_INTERPOLATION
DEFAULT_NUM_VIEWS = 5
DEFAULT_LATITUDE = DEFAULT_EP_LATITUDE
DEFAULT_LATITUDE_MIN = DEFAULT_EP_LATITUDE_MIN
DEFAULT_LATITUDE_MAX = DEFAULT_EP_LATITUDE_MAX
DEFAULT_LONGITUDE_JITTER = 0.0
DEFAULT_LATITUDE_JITTER = 0.0
DEFAULT_LONGITUDE_MIN = DEFAULT_EP_LONGITUDE_MIN
DEFAULT_LONGITUDE_MAX = DEFAULT_EP_LONGITUDE_MAX
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    image_paths = list(_iter_image_paths(args.input_dir))
    if not image_paths:
        print(f"No EP images found in {args.input_dir}")
        return 1

    rng = np.random.default_rng(args.seed)
    total_outputs = 0
    metadata = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "num_views": args.num_views,
        "total_views_per_image": args.num_views,
        "height": args.height,
        "width": args.width,
        "latitude_min": args.latitude_min,
        "latitude_max": args.latitude_max,
        "longitude_min": args.longitude_min,
        "longitude_max": args.longitude_max,
        "longitude_jitter": args.longitude_jitter,
        "latitude_jitter": args.latitude_jitter,
        "fov_x": args.fov_x,
        "fov_y": args.fov_y,
        "interpolation": args.interpolation,
        "seed": args.seed,
        "images": [],
    }

    for image_path in image_paths:
        renderer = EquirectangularToPinhole.from_file(image_path)
        input_width, input_height = _resolution(renderer.image)
        print(f"Input EP: {image_path.name} ({input_width}x{input_height})")
        views = _view_plan(
            num_views=args.num_views,
            rng=rng,
            longitude_min=args.longitude_min,
            longitude_max=args.longitude_max,
            latitude_min=args.latitude_min,
            latitude_max=args.latitude_max,
            longitude_jitter=args.longitude_jitter,
            latitude_jitter=args.latitude_jitter,
        )

        for view_index, view in enumerate(views):
            pinhole = renderer.render(
                height=args.height,
                width=args.width,
                longitude=view["longitude"],
                latitude=view["latitude"],
                fov_x=args.fov_x,
                fov_y=args.fov_y,
                interpolation=args.interpolation,
            )
            output_path = _output_path(
                args.output_dir,
                image_path,
                view_index,
                args.output_extension,
            )
            _write_image(output_path, pinhole)
            output_width, output_height = _resolution(pinhole)
            metadata["images"].append(
                {
                    "source_file": str(image_path),
                    "source_name": image_path.name,
                    "source_width": input_width,
                    "source_height": input_height,
                    "file": str(output_path),
                    "file_name": output_path.name,
                    "view_index": view_index,
                    "width": output_width,
                    "height": output_height,
                    "longitude": view["longitude"],
                    "latitude": view["latitude"],
                    "base_longitude": view["base_longitude"],
                    "base_latitude": view["base_latitude"],
                    "fov_x": args.fov_x,
                    "fov_y": args.fov_y,
                    "interpolation": args.interpolation,
                }
            )
            total_outputs += 1
            print(
                f"  view {view_index:02d}: "
                f"longitude={view['longitude']:.6f}, "
                f"latitude={view['latitude']:.6f} -> {output_path}"
            )

    metadata_path = args.output_dir / "metadata.json"
    _write_metadata(metadata_path, metadata)
    print(f"Metadata: {metadata_path}")
    print(f"Rendered {total_outputs} pinhole images from {len(image_paths)} EP images.")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render multiple pinhole images from every EP image in a directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing input equirectangular images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where rendered pinhole images will be written.",
    )
    parser.add_argument(
        "--num-views",
        default=DEFAULT_NUM_VIEWS,
        type=int,
        help="Number of pinhole views to render per EP image.",
    )
    parser.add_argument(
        "--height",
        default=DEFAULT_HEIGHT,
        type=int,
        help="Output image height in pixels.",
    )
    parser.add_argument(
        "--width",
        default=DEFAULT_WIDTH,
        type=int,
        help="Output image width in pixels.",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        help="Fixed view latitude/pitch in radians for every rendered view.",
    )
    parser.add_argument(
        "--latitude-min",
        default=DEFAULT_LATITUDE_MIN,
        type=float,
        help="Minimum random latitude/pitch in radians.",
    )
    parser.add_argument(
        "--latitude-max",
        default=DEFAULT_LATITUDE_MAX,
        type=float,
        help="Maximum random latitude/pitch in radians.",
    )
    parser.add_argument(
        "--longitude-min",
        default=DEFAULT_LONGITUDE_MIN,
        type=float,
        help="Minimum random longitude/yaw in radians.",
    )
    parser.add_argument(
        "--longitude-max",
        default=DEFAULT_LONGITUDE_MAX,
        type=float,
        help="Maximum random longitude/yaw in radians.",
    )
    parser.add_argument(
        "--longitude-jitter",
        default=DEFAULT_LONGITUDE_JITTER,
        type=float,
        help="Uniform random longitude jitter radius in radians.",
    )
    parser.add_argument(
        "--latitude-jitter",
        default=DEFAULT_LATITUDE_JITTER,
        type=float,
        help="Uniform random latitude jitter radius in radians.",
    )
    parser.add_argument(
        "--fov-x",
        default=DEFAULT_FOV_X,
        type=float,
        help="Horizontal field of view in radians.",
    )
    parser.add_argument(
        "--fov-y",
        default=DEFAULT_FOV_Y,
        type=float,
        help="Vertical field of view in radians.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("linear", "bilinear", "nearest"),
        default=DEFAULT_INTERPOLATION,
        help="Interpolation mode.",
    )
    parser.add_argument(
        "--output-extension",
        default=None,
        help=(
            "Optional output file extension such as .jpg or .png. "
            "Defaults to each input image extension."
        ),
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional random seed for reproducible latitude sampling and jitter.",
    )
    args = parser.parse_args(argv)

    if args.latitude is not None:
        args.latitude_min = args.latitude
        args.latitude_max = args.latitude
    if not args.input_dir.is_dir():
        parser.error(f"--input-dir must be an existing directory: {args.input_dir}")
    if args.num_views <= 0:
        parser.error("--num-views must be positive")
    if args.latitude_min > args.latitude_max:
        parser.error("--latitude-min must be less than or equal to --latitude-max")
    if args.longitude_min > args.longitude_max:
        parser.error("--longitude-min must be less than or equal to --longitude-max")
    if args.longitude_jitter < 0:
        parser.error("--longitude-jitter must be non-negative")
    if args.latitude_jitter < 0:
        parser.error("--latitude-jitter must be non-negative")
    if args.output_extension is not None:
        args.output_extension = _normalize_extension(args.output_extension)

    return args


def _iter_image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _view_plan(
    num_views: int,
    rng: np.random.Generator,
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
    longitude_jitter: float,
    latitude_jitter: float,
) -> list[dict[str, float]]:
    views = []
    for _ in range(num_views):
        base_longitude = _random_longitude(rng, longitude_min, longitude_max)
        base_latitude = _random_latitude(rng, latitude_min, latitude_max)
        longitude = _jitter_value(rng, base_longitude, longitude_jitter)
        latitude = _jitter_value(rng, base_latitude, latitude_jitter)
        views.append(_view_record(base_longitude, base_latitude, longitude, latitude))
    return views


def _random_longitude(
    rng: np.random.Generator,
    longitude_min: float,
    longitude_max: float,
) -> float:
    if longitude_min == longitude_max:
        return longitude_min
    return float(rng.uniform(longitude_min, longitude_max))


def _view_record(
    base_longitude: float,
    base_latitude: float,
    longitude: float,
    latitude: float,
) -> dict[str, float]:
    return {
        "base_longitude": float(base_longitude),
        "base_latitude": float(base_latitude),
        "longitude": _wrap_longitude(longitude),
        "latitude": _clamp(latitude, -pi / 2, pi / 2),
    }


def _random_latitude(
    rng: np.random.Generator,
    latitude_min: float,
    latitude_max: float,
) -> float:
    if latitude_min == latitude_max:
        return latitude_min
    return float(rng.uniform(latitude_min, latitude_max))


def _jitter_value(
    rng: np.random.Generator,
    value: float,
    jitter: float,
) -> float:
    if jitter == 0:
        return value
    return float(value + rng.uniform(-jitter, jitter))


def _wrap_longitude(longitude: float) -> float:
    return float((longitude + pi) % (2 * pi) - pi)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return float(min(max(value, minimum), maximum))


def _output_path(
    output_dir: Path,
    input_path: Path,
    view_index: int,
    output_extension: str | None,
) -> Path:
    extension = output_extension if output_extension is not None else input_path.suffix
    return output_dir / f"{input_path.stem}_pinhole_{view_index:02d}{extension}"


def _normalize_extension(extension: str) -> str:
    return extension if extension.startswith(".") else f".{extension}"


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _resolution(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return width, height


if __name__ == "__main__":
    raise SystemExit(main())
