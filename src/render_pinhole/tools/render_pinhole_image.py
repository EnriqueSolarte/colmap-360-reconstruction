"""Render one pinhole image from an equirectangular image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from render_pinhole import EquirectangularToPinhole
from render_pinhole.defaults import (
    DEFAULT_EP_LATITUDE,
    DEFAULT_EP_LONGITUDE,
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
DEFAULT_LONGITUDE = DEFAULT_EP_LONGITUDE
DEFAULT_LATITUDE = DEFAULT_EP_LATITUDE
DEFAULT_INTERPOLATION = DEFAULT_PINHOLE_INTERPOLATION


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    renderer = EquirectangularToPinhole.from_file(args.input)
    print(f"Input EP resolution: {_format_resolution(renderer.image)}")

    pinhole = renderer.render(
        height=args.height,
        width=args.width,
        longitude=args.longitude,
        latitude=args.latitude,
        fov_x=args.fov_x,
        fov_y=args.fov_y,
        interpolation=args.interpolation,
    )

    _write_image(args.output, pinhole)
    print(f"Output pinhole resolution: {_format_resolution(pinhole)}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one pinhole image from an equirectangular image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input", required=True, type=Path,
        help="Input equirectangular image path.")
    parser.add_argument(
        "-o",
        "--output", required=True, type=Path,
        help="Output pinhole image path.")
    parser.add_argument(
        "--height", default=DEFAULT_HEIGHT, type=int,
        help="Output image height in pixels.")
    parser.add_argument(
        "--width", default=DEFAULT_WIDTH, type=int,
        help="Output image width in pixels.")
    parser.add_argument(
        "--longitude", default=DEFAULT_LONGITUDE,
        type=float, help="View longitude/yaw in radians.")
    parser.add_argument(
        "--latitude", default=DEFAULT_LATITUDE, type=float,
        help="View latitude/pitch in radians.")
    parser.add_argument(
        "--fov-x", default=DEFAULT_FOV_X, type=float,
        help="Horizontal field of view in radians.")
    parser.add_argument(
        "--fov-y", default=DEFAULT_FOV_Y, type=float,
        help="Vertical field of view in radians.")
    parser.add_argument(
        "--interpolation",
        choices=("linear", "bilinear", "nearest"),
        default=DEFAULT_INTERPOLATION,
        help="Interpolation mode.",
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
