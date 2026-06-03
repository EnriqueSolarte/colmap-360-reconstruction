"""Renderer wrapper around py360convert's equirectangular-to-perspective API."""

from __future__ import annotations

from math import degrees, pi
from pathlib import Path
from typing import Literal

import numpy as np
import py360convert
from PIL import Image

Interpolation = Literal["linear", "bilinear", "nearest"]

_INTERPOLATION_MODES: dict[str, str] = {
    "linear": "bilinear",
    "bilinear": "bilinear",
    "nearest": "nearest",
}


class EquirectangularToPinhole:
    """Render pinhole views from one equirectangular image."""

    def __init__(self, image: np.ndarray):
        self.image = _validate_image(image)

    @classmethod
    def from_file(cls, path: str | Path) -> "EquirectangularToPinhole":
        """Load an equirectangular image from disk using Pillow."""
        image = np.asarray(Image.open(path))
        return cls(image)

    def render(
        self,
        height: int,
        width: int,
        longitude: float,
        latitude: float,
        fov_x: float,
        fov_y: float,
        interpolation: Interpolation = "linear",
    ) -> np.ndarray:
        """Render one pinhole view.

        Orientation and field-of-view values are in radians. Longitude is yaw,
        latitude is pitch, and roll is fixed at zero for v0.
        """
        return render_pinhole(
            self.image,
            height=height,
            width=width,
            longitude=longitude,
            latitude=latitude,
            fov_x=fov_x,
            fov_y=fov_y,
            interpolation=interpolation,
        )


def render_pinhole(
    image: np.ndarray,
    height: int,
    width: int,
    longitude: float,
    latitude: float,
    fov_x: float,
    fov_y: float,
    interpolation: Interpolation = "linear",
) -> np.ndarray:
    """Render one pinhole view from a loaded equirectangular image."""
    source = _validate_image(image)
    _validate_output_size(height, width)
    _validate_fov(fov_x, "fov_x")
    _validate_fov(fov_y, "fov_y")
    mode = _resolve_interpolation(interpolation)

    return py360convert.e2p(
        source,
        fov_deg=(degrees(fov_x), degrees(fov_y)),
        u_deg=degrees(longitude),
        v_deg=degrees(latitude),
        out_hw=(height, width),
        mode=mode,
    )


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    if image.ndim not in (2, 3):
        raise ValueError(
            "image must have shape (height, width) or (height, width, channels)")

    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image height and width must be positive")

    return image


def _validate_output_size(height: int, width: int) -> None:
    if not isinstance(height, int) or not isinstance(width, int):
        raise TypeError("height and width must be integers")

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")


def _validate_fov(value: float, name: str) -> None:
    if not np.isscalar(value):
        raise TypeError(f"{name} must be a scalar")

    if not 0 < float(value) < pi:
        raise ValueError(
            f"{name} must be greater than 0 and less than pi radians")


def _resolve_interpolation(interpolation: str) -> str:
    try:
        return _INTERPOLATION_MODES[interpolation]
    except KeyError as error:
        valid = ", ".join(sorted(_INTERPOLATION_MODES))
        raise ValueError(f"interpolation must be one of: {valid}") from error
