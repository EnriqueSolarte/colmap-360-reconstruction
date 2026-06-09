import json

import numpy as np
import pytest

import colmap.pinhole_reconstruction as pinhole_reconstruction
from colmap.pinhole_reconstruction import (
    camera_initialization_from_args,
    main as colmap_pinhole_reconstruct_main,
    parse_args,
    read_calibration_yaml,
)


def write_image(path, *, width: int = 10, height: int = 8) -> None:
    Image = pytest.importorskip("PIL.Image")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_read_calibration_yaml_accepts_opencv_camera_matrix(tmp_path) -> None:
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(
        """
image_width: 1280
image_height: 960
camera_matrix:
  rows: 3
  cols: 3
  data: [914.0, 0.0, 640.0, 0.0, 913.5, 480.0, 0.0, 0.0, 1.0]
distortion_coefficients:
  data: [0, 0, 0, 0]
""",
        encoding="utf-8",
    )

    calibration = read_calibration_yaml(calibration_path)

    assert calibration["width"] == 1280
    assert calibration["height"] == 960
    assert calibration["fx"] == 914.0
    assert calibration["fy"] == 913.5
    assert calibration["cx"] == 640.0
    assert calibration["cy"] == 480.0


def test_camera_initialization_defaults_to_optimized_shared_guess(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    write_image(image_dir / "frame.png", width=12, height=7)
    args = parse_args(["--image-dir", str(image_dir)])

    camera_init = camera_initialization_from_args(args)

    assert camera_init.source == "image_resolution_initial_guess"
    assert camera_init.optimize_intrinsics is True
    assert camera_init.camera.model == "PINHOLE"
    assert camera_init.camera.width == 12
    assert camera_init.camera.height == 7
    assert camera_init.camera.params == (
        14.399999999999999,
        14.399999999999999,
        6.0,
        3.5,
    )


def test_camera_initialization_uses_calibration_without_refinement_by_default(
    tmp_path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(
        """
width: 10
height: 8
fx: 5.0
fy: 6.0
cx: 5.0
cy: 4.0
""",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--image-dir",
            str(image_dir),
            "--calibration",
            str(calibration_path),
        ]
    )

    camera_init = camera_initialization_from_args(args)

    assert camera_init.source == "calibration_yaml"
    assert camera_init.optimize_intrinsics is False
    assert camera_init.camera.params == (5.0, 6.0, 5.0, 4.0)


def test_pinhole_reconstruct_cli_inspect_only_with_calibration(tmp_path, capsys) -> None:
    image_dir = tmp_path / "images"
    workspace = tmp_path / "workspace"
    image_dir.mkdir()
    write_image(image_dir / "frame.png", width=10, height=8)
    calibration_path = tmp_path / "calibration.yaml"
    calibration_path.write_text(
        """
image_width: 10
image_height: 8
K: [5.0, 0.0, 5.0, 0.0, 6.0, 4.0, 0.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )

    exit_code = colmap_pinhole_reconstruct_main(
        [
            "--image-dir",
            str(image_dir),
            "--workspace",
            str(workspace),
            "--calibration",
            str(calibration_path),
            "--inspect-only",
        ]
    )

    assert exit_code == 0
    report = json.loads(
        (workspace / "colmap_input_report.json").read_text(encoding="utf-8")
    )
    assert report["camera"]["params"] == [5.0, 6.0, 5.0, 4.0]
    output = capsys.readouterr().out
    assert "camera source: calibration_yaml" in output
    assert "optimize intrinsics: False" in output


def test_require_pycolmap_reports_missing_dependency(monkeypatch) -> None:
    monkeypatch.setattr(pinhole_reconstruction, "pycolmap", None)

    with pytest.raises(RuntimeError, match="pycolmap is required"):
        pinhole_reconstruction.require_pycolmap()
