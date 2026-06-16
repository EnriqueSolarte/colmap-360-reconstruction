from math import pi

import numpy as np
import pytest

from render_pinhole import render_pinhole

EP_HEIGHT = 181
EP_WIDTH = 360
ANGLE_TOLERANCE = 0.02


def make_longitude_latitude_image() -> np.ndarray:
    x = np.arange(EP_WIDTH, dtype=np.float32)[None, :]
    y = np.arange(EP_HEIGHT, dtype=np.float32)[:, None]

    longitude = ((x + 0.5) / EP_WIDTH) * (2 * pi) - pi
    latitude = (pi / 2) - ((y + 0.5) / EP_HEIGHT) * pi

    image = np.zeros((EP_HEIGHT, EP_WIDTH, 2), dtype=np.float32)
    image[..., 0] = longitude
    image[..., 1] = latitude
    return image


def center_sample(image: np.ndarray) -> np.ndarray:
    return image[image.shape[0] // 2, image.shape[1] // 2]


def angular_error(actual: float, expected: float) -> float:
    return abs((actual - expected + pi) % (2 * pi) - pi)


@pytest.mark.parametrize(
    "longitude,latitude",
    [
        (0.0, 0.0),
        (pi / 2, 0.0),
        (-pi / 2, 0.0),
        (0.0, pi / 4),
        (0.0, -pi / 4),
    ],
)
def test_center_pixel_samples_requested_orientation(longitude: float, latitude: float) -> None:
    result = render_pinhole(
        make_longitude_latitude_image(),
        height=3,
        width=3,
        longitude=longitude,
        latitude=latitude,
        fov_x=pi / 3,
        fov_y=pi / 3,
        interpolation="nearest",
    )

    actual_longitude, actual_latitude = center_sample(result)

    assert angular_error(float(actual_longitude), longitude) < ANGLE_TOLERANCE
    assert abs(float(actual_latitude) - latitude) < ANGLE_TOLERANCE


@pytest.mark.parametrize("longitude", [pi - 1e-3, -pi + 1e-3])
def test_center_pixel_handles_horizontal_wrapping(longitude: float) -> None:
    result = render_pinhole(
        make_longitude_latitude_image(),
        height=3,
        width=3,
        longitude=longitude,
        latitude=0.0,
        fov_x=pi / 3,
        fov_y=pi / 3,
        interpolation="nearest",
    )

    actual_longitude, actual_latitude = center_sample(result)

    assert angular_error(float(actual_longitude), longitude) < ANGLE_TOLERANCE
    assert abs(float(actual_latitude)) < ANGLE_TOLERANCE


def test_positive_longitude_moves_view_toward_larger_ep_longitudes() -> None:
    center_view = render_pinhole(
        make_longitude_latitude_image(),
        height=3,
        width=3,
        longitude=0.0,
        latitude=0.0,
        fov_x=pi / 3,
        fov_y=pi / 3,
        interpolation="nearest",
    )
    right_view = render_pinhole(
        make_longitude_latitude_image(),
        height=3,
        width=3,
        longitude=pi / 2,
        latitude=0.0,
        fov_x=pi / 3,
        fov_y=pi / 3,
        interpolation="nearest",
    )

    assert center_sample(right_view)[0] > center_sample(center_view)[0]


def test_positive_latitude_moves_view_upward() -> None:
    center_view = render_pinhole(
        make_longitude_latitude_image(),
        height=3,
        width=3,
        longitude=0.0,
        latitude=0.0,
        fov_x=pi / 3,
        fov_y=pi / 3,
        interpolation="nearest",
    )
    upward_view = render_pinhole(
        make_longitude_latitude_image(),
        height=3,
        width=3,
        longitude=0.0,
        latitude=pi / 4,
        fov_x=pi / 3,
        fov_y=pi / 3,
        interpolation="nearest",
    )

    assert center_sample(upward_view)[1] > center_sample(center_view)[1]


def test_fov_controls_horizontal_and_vertical_extent() -> None:
    result = render_pinhole(
        make_longitude_latitude_image(),
        height=3,
        width=3,
        longitude=0.0,
        latitude=0.0,
        fov_x=pi / 2,
        fov_y=pi / 2,
        interpolation="nearest",
    )

    left_longitude = float(result[1, 0, 0])
    center_longitude = float(result[1, 1, 0])
    right_longitude = float(result[1, 2, 0])
    top_latitude = float(result[0, 1, 1])
    center_latitude = float(result[1, 1, 1])
    bottom_latitude = float(result[2, 1, 1])

    assert left_longitude < center_longitude < right_longitude
    assert angular_error(left_longitude, -pi / 4) < ANGLE_TOLERANCE
    assert angular_error(right_longitude, pi / 4) < ANGLE_TOLERANCE
    assert top_latitude > center_latitude > bottom_latitude
    assert abs(top_latitude - pi / 4) < ANGLE_TOLERANCE
    assert abs(bottom_latitude + pi / 4) < ANGLE_TOLERANCE
