"""Sampling plans for equirectangular longitude/latitude view points."""

from __future__ import annotations

from math import asin, cos, pi, radians, sin
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


def spherical_roi_points_by_delta(
    *,
    longitude_min: float,
    longitude_max: float,
    latitude_min: float,
    latitude_max: float,
    delta_degrees: float,
) -> list[tuple[float, float]]:
    """Sample an ROI with approximately uniform angular spacing on the sphere."""
    if delta_degrees <= 0:
        raise ValueError("ROI delta angle must be positive")
    longitude_min = float(longitude_min)
    longitude_max = float(longitude_max)
    latitude_min = float(latitude_min)
    latitude_max = float(latitude_max)
    _validate_latitude(latitude_min)
    _validate_latitude(latitude_max)
    if longitude_max <= longitude_min:
        raise ValueError("ROI longitude_max must be greater than longitude_min")
    if latitude_max <= latitude_min:
        raise ValueError("ROI latitude_max must be greater than latitude_min")

    delta = radians(float(delta_degrees))
    latitude_values = list(
        reversed(_axis_values_by_delta(latitude_min, latitude_max, delta))
    )
    points = []
    for latitude in latitude_values:
        longitude_delta = longitude_delta_for_spherical_spacing(
            latitude=latitude,
            angular_delta=delta,
        )
        points.extend(
            (longitude, latitude)
            for longitude in _axis_values_by_delta(
                longitude_min,
                longitude_max,
                longitude_delta,
            )
        )
    return points


def longitude_delta_for_spherical_spacing(
    *,
    latitude: float,
    angular_delta: float,
) -> float:
    """Return longitude spacing for a target great-circle spacing at latitude."""
    if angular_delta <= 0:
        raise ValueError("angular_delta must be positive")
    _validate_latitude(latitude)
    cos_latitude = abs(cos(latitude))
    if cos_latitude <= 0.0:
        return 2.0 * pi
    scaled = sin(angular_delta / 2.0) / cos_latitude
    if scaled >= 1.0:
        return 2.0 * pi
    return 2.0 * asin(scaled)


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


def _axis_values_by_delta(minimum: float, maximum: float, delta: float) -> list[float]:
    span = maximum - minimum
    if span <= delta:
        return [(minimum + maximum) / 2.0]
    value = minimum + delta / 2.0
    points = []
    while value < maximum:
        points.append(value)
        value += delta
    return points


def _validate_latitude(latitude: float) -> None:
    if not -pi / 2 <= latitude <= pi / 2:
        raise ValueError("latitude must be in [-pi / 2, pi / 2] radians")


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
