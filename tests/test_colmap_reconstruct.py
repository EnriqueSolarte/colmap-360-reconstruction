import json
from math import pi, tan

import numpy as np
import pytest

import colmap.reconstruct as reconstruct
from colmap.reconstruct import (
    camera_from_fov,
    camera_from_args,
    inspect_input,
    main as colmap_reconstruct_main,
    parse_args,
)
from ep_pinhole.tools import render_pinhole_directory


def write_image(path, *, width: int = 10, height: int = 8) -> None:
    Image = pytest.importorskip("PIL.Image")
    image = np.zeros((height, width, 3), dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_colmap_defaults_follow_directory_renderer_defaults() -> None:
    assert reconstruct.DEFAULT_WIDTH == render_pinhole_directory.DEFAULT_WIDTH
    assert reconstruct.DEFAULT_HEIGHT == render_pinhole_directory.DEFAULT_HEIGHT
    assert reconstruct.DEFAULT_FOV_X == render_pinhole_directory.DEFAULT_FOV_X
    assert reconstruct.DEFAULT_FOV_Y == render_pinhole_directory.DEFAULT_FOV_Y


def test_camera_from_fov_computes_pinhole_intrinsics() -> None:
    camera = camera_from_fov(width=1280, height=960, fov_x=pi / 2, fov_y=pi / 3)

    assert camera.model == "PINHOLE"
    assert camera.width == 1280
    assert camera.height == 960
    assert camera.params == pytest.approx(
        (
            (1280 / 2) / tan((pi / 2) / 2),
            (960 / 2) / tan((pi / 3) / 2),
            640.0,
            480.0,
        )
    )
    assert camera.params_string == "640,831.384387633,640,480"


def test_inspect_input_writes_report(tmp_path) -> None:
    image_dir = tmp_path / "images"
    workspace = tmp_path / "workspace"
    image_dir.mkdir()
    write_image(image_dir / "frame_0002.png")
    write_image(image_dir / "frame_0001.png")
    camera = camera_from_fov(width=10, height=8, fov_x=pi / 2, fov_y=pi / 3)

    report = inspect_input(image_dir=image_dir, workspace=workspace, camera=camera)

    assert report["status"] == "colmap_input_inspected"
    assert report["input_image_count"] == 2
    assert report["first_image"] == "frame_0001.png"
    assert report["last_image"] == "frame_0002.png"
    assert report["image_resolution"] == {"width": 10, "height": 8}
    assert report["camera"]["params"] == pytest.approx(list(camera.params))

    report_path = workspace / "colmap_input_report.json"
    assert report_path.exists()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["camera"]["shared_single_camera"] is True


def test_inspect_input_rejects_resolution_mismatch(tmp_path) -> None:
    image_dir = tmp_path / "images"
    workspace = tmp_path / "workspace"
    image_dir.mkdir()
    write_image(image_dir / "frame.png", width=12, height=8)
    camera = camera_from_fov(width=10, height=8, fov_x=pi / 2, fov_y=pi / 3)

    with pytest.raises(ValueError, match="resolution"):
        inspect_input(image_dir=image_dir, workspace=workspace, camera=camera)


def test_colmap_reconstruct_cli_inspect_only(tmp_path, capsys) -> None:
    image_dir = tmp_path / "images"
    workspace = tmp_path / "workspace"
    image_dir.mkdir()
    write_image(image_dir / "frame.png", width=10, height=8)

    exit_code = colmap_reconstruct_main(
        [
            "--image-dir",
            str(image_dir),
            "--workspace",
            str(workspace),
            "--width",
            "10",
            "--height",
            "8",
            "--fov-x",
            str(pi / 2),
            "--fov-y",
            str(pi / 3),
            "--inspect-only",
        ]
    )

    assert exit_code == 0
    assert (workspace / "colmap_input_report.json").exists()
    output = capsys.readouterr().out
    assert "images: 1 at 10x8" in output
    assert "camera params: 5,6.92820323028,5,4" in output


def test_optimize_intrinsics_infers_image_resolution(tmp_path, capsys) -> None:
    image_dir = tmp_path / "images"
    workspace = tmp_path / "workspace"
    image_dir.mkdir()
    write_image(image_dir / "frame.png", width=12, height=7)

    exit_code = colmap_reconstruct_main(
        [
            "--image-dir",
            str(image_dir),
            "--workspace",
            str(workspace),
            "--optimize-intrinsics",
            "--inspect-only",
        ]
    )

    assert exit_code == 0
    report = json.loads(
        (workspace / "colmap_input_report.json").read_text(encoding="utf-8")
    )
    assert report["image_resolution"] == {"width": 12, "height": 7}
    assert report["camera"]["width"] == 12
    assert report["camera"]["height"] == 7
    assert report["camera"]["params"] == [14.399999999999999, 14.399999999999999, 6.0, 3.5]
    output = capsys.readouterr().out
    assert "images: 1 at 12x7" in output


def test_optimize_intrinsics_requires_width_and_height_together(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    write_image(image_dir / "frame.png", width=12, height=7)
    args = parse_args(
        [
            "--image-dir",
            str(image_dir),
            "--optimize-intrinsics",
            "--width",
            "12",
        ]
    )

    with pytest.raises(ValueError, match="both --width and --height"):
        camera_from_args(args)


def test_parse_args_supports_shared_intrinsics_optimization(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    args = parse_args(
        [
            "--image-dir",
            str(image_dir),
            "--optimize-intrinsics",
        ]
    )

    assert args.optimize_intrinsics is True


def test_require_pycolmap_reports_missing_dependency(monkeypatch) -> None:
    monkeypatch.setattr(reconstruct, "pycolmap", None)

    with pytest.raises(RuntimeError, match="pycolmap is required"):
        reconstruct.require_pycolmap()
