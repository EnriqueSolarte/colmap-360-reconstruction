from math import pi

import numpy as np
import pytest

from ep_pinhole import EquirectangularToPinhole, render_pinhole
from ep_pinhole.tools.render_image import main as render_image_main


def make_equirectangular_image() -> np.ndarray:
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[..., 0] = np.arange(32, dtype=np.uint8)
    image[..., 1] = np.arange(16, dtype=np.uint8)[:, None]
    image[..., 2] = 127
    return image


def test_constructor_accepts_valid_image() -> None:
    image = make_equirectangular_image()

    renderer = EquirectangularToPinhole(image)

    assert renderer.image is image


def test_constructor_rejects_invalid_image_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        EquirectangularToPinhole(np.zeros((3,), dtype=np.uint8))


def test_render_pinhole_returns_requested_shape() -> None:
    result = render_pinhole(
        make_equirectangular_image(),
        height=8,
        width=10,
        longitude=0.0,
        latitude=0.0,
        fov_x=pi / 2,
        fov_y=pi / 3,
    )

    assert result.shape == (8, 10, 3)


@pytest.mark.parametrize("height,width", [(0, 10), (8, 0)])
def test_render_pinhole_rejects_invalid_output_size(height: int, width: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        render_pinhole(
            make_equirectangular_image(),
            height=height,
            width=width,
            longitude=0.0,
            latitude=0.0,
            fov_x=pi / 2,
            fov_y=pi / 3,
        )


@pytest.mark.parametrize("fov_x,fov_y", [(0.0, pi / 3), (pi / 2, pi)])
def test_render_pinhole_rejects_invalid_fov(fov_x: float, fov_y: float) -> None:
    with pytest.raises(ValueError, match="fov_"):
        render_pinhole(
            make_equirectangular_image(),
            height=8,
            width=10,
            longitude=0.0,
            latitude=0.0,
            fov_x=fov_x,
            fov_y=fov_y,
        )


@pytest.mark.parametrize("interpolation", ["linear", "bilinear", "nearest"])
def test_render_pinhole_accepts_supported_interpolation(interpolation: str) -> None:
    result = render_pinhole(
        make_equirectangular_image(),
        height=8,
        width=10,
        longitude=0.0,
        latitude=0.0,
        fov_x=pi / 2,
        fov_y=pi / 3,
        interpolation=interpolation,
    )

    assert result.shape == (8, 10, 3)


def test_render_pinhole_rejects_unknown_interpolation() -> None:
    with pytest.raises(ValueError, match="interpolation"):
        render_pinhole(
            make_equirectangular_image(),
            height=8,
            width=10,
            longitude=0.0,
            latitude=0.0,
            fov_x=pi / 2,
            fov_y=pi / 3,
            interpolation="cubic",
        )


def test_render_image_cli_writes_output(tmp_path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(input_path)

    exit_code = render_image_main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--height",
            "8",
            "--width",
            "10",
            "--longitude",
            "0.0",
            "--latitude",
            "0.0",
            "--fov-x",
            str(pi / 2),
            "--fov-y",
            str(pi / 3),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert np.asarray(Image.open(output_path)).shape == (8, 10, 3)
