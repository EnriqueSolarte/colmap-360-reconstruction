import numpy as np
import pytest
from PIL import Image

from tools.shift_panorama import main as shift_panorama_main
from tools.shift_panorama import shift_image_array


def test_shift_image_array_rolls_columns_by_degrees() -> None:
    image = np.array([[[0], [1], [2], [3]]], dtype=np.uint8)

    shifted, pixel_shift = shift_image_array(image, 180.0)

    assert pixel_shift == 2
    np.testing.assert_array_equal(
        shifted,
        np.array([[[2], [3], [0], [1]]], dtype=np.uint8),
    )


def test_shift_panorama_cli_processes_directory_in_place_after_confirmation(
    tmp_path, capsys, monkeypatch
) -> None:
    source_dir = tmp_path / "panos"
    source_dir.mkdir()
    image = np.array([[[0], [64], [128], [255]]], dtype=np.uint8)
    source_path = source_dir / "pano.png"
    Image.fromarray(image[:, :, 0], mode="L").save(source_path)
    monkeypatch.setattr("builtins.input", lambda _: "yes")

    exit_code = shift_panorama_main(["--input", str(source_dir), "--theta", "90"])

    assert exit_code == 0
    with Image.open(source_path) as shifted_image:
        shifted = np.asarray(shifted_image)
    np.testing.assert_array_equal(
        shifted,
        np.array([[255, 0, 64, 128]], dtype=np.uint8),
    )

    output = capsys.readouterr().out
    assert "WARNING: no --output was provided" in output
    assert "theta=90" in output
    assert "1 px" in output


def test_shift_panorama_cli_processes_directory_with_workers(tmp_path) -> None:
    source_dir = tmp_path / "panos"
    output_dir = tmp_path / "shifted"
    source_dir.mkdir()
    first = source_dir / "first.png"
    second = source_dir / "second.png"
    Image.fromarray(np.array([[0, 1, 2, 3]], dtype=np.uint8), mode="L").save(first)
    Image.fromarray(np.array([[4, 5, 6, 7]], dtype=np.uint8), mode="L").save(second)

    exit_code = shift_panorama_main(
        [
            "--input",
            str(source_dir),
            "--output",
            str(output_dir),
            "--theta",
            "90",
            "--workers",
            "2",
        ]
    )

    assert exit_code == 0
    with Image.open(output_dir / "first.png") as image:
        shifted_first = np.asarray(image)
    with Image.open(output_dir / "second.png") as image:
        shifted_second = np.asarray(image)
    np.testing.assert_array_equal(shifted_first, np.array([[3, 0, 1, 2]], dtype=np.uint8))
    np.testing.assert_array_equal(shifted_second, np.array([[7, 4, 5, 6]], dtype=np.uint8))


def test_shift_panorama_cli_aborts_in_place_without_confirmation(
    tmp_path, capsys, monkeypatch
) -> None:
    source = tmp_path / "pano.png"
    Image.fromarray(np.array([[0, 1]], dtype=np.uint8), mode="L").save(source)
    monkeypatch.setattr("builtins.input", lambda _: "no")

    exit_code = shift_panorama_main(["--input", str(source)])

    assert exit_code == 1
    with Image.open(source) as image:
        unchanged = np.asarray(image)
    np.testing.assert_array_equal(unchanged, np.array([[0, 1]], dtype=np.uint8))

    output = capsys.readouterr().out
    assert "WARNING: no --output was provided" in output
    assert "Aborted; no images were changed." in output


def test_shift_panorama_cli_refuses_existing_output_without_overwrite(tmp_path) -> None:
    source = tmp_path / "pano.png"
    output = tmp_path / "shifted.png"
    Image.fromarray(np.array([[0, 1]], dtype=np.uint8), mode="L").save(source)
    output.write_text("existing")

    with pytest.raises(FileExistsError, match="Output image already exists"):
        shift_panorama_main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
            ]
        )
