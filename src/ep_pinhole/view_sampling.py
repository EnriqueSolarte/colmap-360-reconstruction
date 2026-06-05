"""Sampling plans for equirectangular longitude/latitude view points."""

from __future__ import annotations

from math import pi
from typing import Optional

import numpy as np

VIEW_SAMPLING_METHODS = ("random", "uniform_longitude")
VIEW_SAMPLING_DESCRIPTIONS = {
    "random": (
        "Samples each view independently from uniform longitude and latitude "
        "ranges, then applies optional random jitter."
    ),
    "uniform_longitude": (
        "Places views at evenly spaced longitudes over the requested range "
        "using one fixed latitude, then applies optional random jitter."
    ),
}


def random_views(
    *,
    num_views: int,
    rng: np.random.Generator,
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
    longitude_jitter: float = 0.0,
    latitude_jitter: float = 0.0,
) -> list[dict[str, float]]:
    """Sample independent random longitude/latitude views within configured ranges."""
    views = []
    for _ in range(num_views):
        base_longitude = _random_value(rng, longitude_min, longitude_max)
        base_latitude = _random_value(rng, latitude_min, latitude_max)
        longitude = _jitter_value(rng, base_longitude, longitude_jitter)
        latitude = _jitter_value(rng, base_latitude, latitude_jitter)
        views.append(_view_record(base_longitude, base_latitude, longitude, latitude))
    return views


def uniform_longitude_views(
    *,
    num_views: int,
    longitude_min: float,
    longitude_max: float,
    latitude: float,
    rng: Optional[np.random.Generator] = None,
    longitude_jitter: float = 0.0,
    latitude_jitter: float = 0.0,
) -> list[dict[str, float]]:
    """Sample deterministic, evenly spaced longitude views at one latitude."""
    views = []
    for base_longitude in _uniform_longitudes(
        num_views,
        longitude_min,
        longitude_max,
    ):
        longitude = _jitter_value(rng, base_longitude, longitude_jitter)
        view_latitude = _jitter_value(rng, latitude, latitude_jitter)
        views.append(
            _view_record(base_longitude, latitude, longitude, view_latitude)
        )
    return views


def _uniform_longitudes(
    num_views: int,
    longitude_min: float,
    longitude_max: float,
) -> list[float]:
    if num_views == 1:
        return [float((longitude_min + longitude_max) / 2)]
    if longitude_min == longitude_max:
        return [float(longitude_min)] * num_views
    return [
        float(longitude)
        for longitude in np.linspace(
            longitude_min,
            longitude_max,
            num_views,
            endpoint=False,
        )
    ]


def _random_value(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
) -> float:
    if minimum == maximum:
        return minimum
    return float(rng.uniform(minimum, maximum))


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


def _jitter_value(
    rng: Optional[np.random.Generator],
    value: float,
    jitter: float,
) -> float:
    if jitter == 0:
        return value
    if rng is None:
        raise ValueError("rng is required when jitter is non-zero")
    return float(value + rng.uniform(-jitter, jitter))


def _wrap_longitude(longitude: float) -> float:
    return float((longitude + pi) % (2 * pi) - pi)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return float(min(max(value, minimum), maximum))
