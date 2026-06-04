"""Render one pinhole image from a random equirectangular orientation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ep_pinhole import EquirectangularToPinhole
from ep_pinhole.defaults import (
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
DEFAULT_LONGITUDE_MIN = DEFAULT_EP_LONGITUDE_MIN
DEFAULT_LONGITUDE_MAX = DEFAULT_EP_LONGITUDE_MAX
DEFAULT_LATITUDE_MIN = DEFAULT_EP_LATITUDE_MIN
DEFAULT_LATITUDE_MAX = DEFAULT_EP_LATITUDE_MAX


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)
    longitude = float(rng.uniform(args.longitude_min, args.longitude_max))
    latitude = float(rng.uniform(args.latitude_min, args.latitude_max))

    renderer = EquirectangularToPinhole.from_file(args.input)
    print(f"Input EP resolution: {_format_resolution(renderer.image)}")
    print(f"Random longitude: {longitude}")
    print(f"Random latitude: {latitude}")

    pinhole = renderer.render(
        height=args.height,
        width=args.width,
        longitude=longitude,
        latitude=latitude,
        fov_x=args.fov_x,
        fov_y=args.fov_y,
        interpolation=args.interpolation,
    )

    _write_image(args.output, pinhole)
    print(f"Output pinhole resolution: {_format_resolution(pinhole)}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one pinhole image from a random equirectangular orientation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input equirectangular image path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output pinhole image path.",
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
        "--seed",
        default=None,
        type=int,
        help="Optional random seed for reproducible smoke renders.",
    )
    return parser.parse_args(argv)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _format_resolution(image: np.ndarray) -> str:
    height, width = image.shape[:2]
    return f"{width}x{height}"


if __name__ == "__main__":
    raise SystemExit(main())
