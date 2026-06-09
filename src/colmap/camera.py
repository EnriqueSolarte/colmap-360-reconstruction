"""COLMAP camera model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, tan
from typing import Any


@dataclass(frozen=True)
class ColmapCamera:
    model: str
    width: int
    height: int
    params: tuple[float, float, float, float]
    fov_x: float | None = None
    fov_y: float | None = None

    @property
    def params_string(self) -> str:
        return ",".join(f"{value:.12g}" for value in self.params)

    def report(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "params": list(self.params),
            "params_order": ["fx", "fy", "cx", "cy"],
            "fov_x": self.fov_x,
            "fov_y": self.fov_y,
            "shared_single_camera": True,
            "intrinsics_refined": False,
        }


def camera_from_fov(width: int, height: int, fov_x: float, fov_y: float) -> ColmapCamera:
    validate_positive_int(width, "width")
    validate_positive_int(height, "height")
    validate_fov(fov_x, "fov_x")
    validate_fov(fov_y, "fov_y")

    fx = (width / 2.0) / tan(fov_x / 2.0)
    fy = (height / 2.0) / tan(fov_y / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return ColmapCamera(
        model="PINHOLE",
        width=width,
        height=height,
        params=(fx, fy, cx, cy),
        fov_x=fov_x,
        fov_y=fov_y,
    )


def camera_from_resolution_guess(width: int, height: int) -> ColmapCamera:
    """Build a generic initial PINHOLE guess for BA to refine."""
    validate_positive_int(width, "width")
    validate_positive_int(height, "height")

    focal = 1.2 * max(width, height)
    return ColmapCamera(
        model="PINHOLE",
        width=width,
        height=height,
        params=(focal, focal, width / 2.0, height / 2.0),
    )


def camera_from_intrinsics(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> ColmapCamera:
    validate_positive_int(width, "width")
    validate_positive_int(height, "height")
    validate_positive_float(fx, "fx")
    validate_positive_float(fy, "fy")
    return ColmapCamera(
        model="PINHOLE",
        width=width,
        height=height,
        params=(fx, fy, cx, cy),
    )


def validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_fov(value: float, name: str) -> None:
    if not 0.0 < value < pi:
        raise ValueError(f"{name} must be greater than 0 and less than pi radians")


def validate_positive_float(value: float, name: str) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
