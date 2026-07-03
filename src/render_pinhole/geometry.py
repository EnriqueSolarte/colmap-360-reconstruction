"""Geometry helpers for panorama-to-pinhole view directions."""

from __future__ import annotations

from math import degrees, pi
from typing import Any

import numpy as np


def wrap_longitude(longitude: float) -> float:
    return float((longitude + pi) % (2 * pi) - pi)


def validate_latitude(latitude: float) -> None:
    if not -pi / 2 <= latitude <= pi / 2:
        raise ValueError("latitude must be in [-pi / 2, pi / 2] radians")


def pinhole_camera_pose_report(
    pano_camera_pose: dict[str, Any] | None,
    longitude: float,
    latitude: float,
) -> dict[str, Any] | None:
    if pano_camera_pose is None:
        return None
    rotation_world_from_pano = np.asarray(
        pano_camera_pose["rotation_world_from_pano"],
        dtype=np.float64,
    )
    center_world = np.asarray(pano_camera_pose["center_world"], dtype=np.float64)
    cam_from_pano = cam_from_pano_rotation(longitude, latitude)
    rotation_world_from_pinhole = rotation_world_from_pano @ cam_from_pano.T
    transform_world_from_pinhole = np.eye(4, dtype=np.float64)
    transform_world_from_pinhole[:3, :3] = rotation_world_from_pinhole
    transform_world_from_pinhole[:3, 3] = center_world
    return {
        "coordinate_frame": pano_camera_pose.get(
            "coordinate_frame",
            "colmap_world_unscaled",
        ),
        "longitude": longitude,
        "latitude": latitude,
        "longitude_degrees": degrees(longitude),
        "latitude_degrees": degrees(latitude),
        "center_world": center_world.tolist(),
        "rotation_world_from_pinhole": rotation_world_from_pinhole.tolist(),
        "transform_world_from_pinhole": transform_world_from_pinhole.tolist(),
        "cam_from_pano_rotation": cam_from_pano.tolist(),
    }


def cam_from_pano_rotation(longitude: float, latitude: float) -> np.ndarray:
    return rotation_x(-latitude) @ rotation_y(-longitude)


def rotation_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def rotation_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )
