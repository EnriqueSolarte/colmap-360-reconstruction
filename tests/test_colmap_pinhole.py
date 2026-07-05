import json
from math import tan
from pathlib import Path

import pytest
from PIL import Image

from colmap.pinhole.cli_import import parse_import_args
from colmap.pinhole.import_pipeline import run_pinhole_image_import
from colmap.pinhole.render_report import (
    load_render_report,
    render_records_from_report,
)


def write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


def write_render_report(
    path: Path,
    *,
    image_dir: Path,
    output_names: tuple[str, ...] = ("a__sel-000.png", "b__sel-000.png"),
    report_size: tuple[int, int] = (10, 8),
    fovs: tuple[tuple[float, float], ...] | None = None,
) -> None:
    fovs = fovs or tuple((1.2, 1.0) for _name in output_names)
    renders = []
    for index, output_name in enumerate(output_names):
        width, height = report_size
        fov_x, fov_y = fovs[index]
        renders.append(
            {
                "selection_id": f"pano_{index:03d}__sel-000",
                "pano_image_name": f"pano_{index:03d}.png",
                "selection_index": 0,
                "source_file": str(path.parent / "panos" / f"pano_{index:03d}.png"),
                "output_file": str(image_dir / output_name),
                "output_name": output_name,
                "width": width,
                "height": height,
                "longitude": 0.1 * index,
                "latitude": -0.1,
                "longitude_degrees": 5.0 * index,
                "latitude_degrees": -5.0,
                "fov_x": fov_x,
                "fov_y": fov_y,
                "interpolation": "bilinear",
            },
        )
    report = {
        "schema_version": 1,
        "source": {"image_dir": str(path.parent / "panos")},
        "output": {
            "output_dir": str(image_dir),
            "render_report": str(path),
        },
        "render_params": {
            "width": report_size[0],
            "height": report_size[1],
            "fov_x": fovs[0][0],
            "fov_y": fovs[0][1],
        },
        "rendered_count": len(renders),
        "renders": renders,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def test_render_records_use_report_output_files_by_default(tmp_path: Path) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=image_dir)

    records = render_records_from_report(
        load_render_report(report_path),
        render_report_path=report_path,
    )

    assert [record.output_name for record in records] == [
        "a__sel-000.png",
        "b__sel-000.png",
    ]
    assert records[0].image_path == image_dir / "a__sel-000.png"


def test_pinhole_import_uses_actual_image_resolution_from_image_dir_override(
    tmp_path: Path,
) -> None:
    old_image_dir = tmp_path / "old-renders"
    highres_image_dir = tmp_path / "highres-renders"
    write_image(highres_image_dir / "a__sel-000.png", (20, 16))
    write_image(highres_image_dir / "b__sel-000.png", (20, 16))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=old_image_dir, report_size=(10, 8))

    workspace = tmp_path / "workspace"
    report = run_pinhole_image_import(
        render_report_path=report_path,
        workspace=workspace,
        image_dir=highres_image_dir,
        show_progress=False,
    )

    assert report["camera"]["model"] == "PINHOLE"
    assert report["camera"]["width"] == 20
    assert report["camera"]["height"] == 16
    assert report["resolution_mismatch_count"] == 2
    assert report["generated_image_names"] == ["a__sel-000.png", "b__sel-000.png"]
    assert (workspace / "images" / "a__sel-000.png").is_file()
    assert (workspace / "pinhole_config.json").is_file()
    assert (workspace / "pinhole_import_report.json").is_file()
    expected_fx = (20 / 2.0) / tan(1.2 / 2.0)
    assert report["camera"]["params"][0] == pytest.approx(expected_fx)
    assert report["pinhole_images"][0]["input_image_path"] == str(
        highres_image_dir / "a__sel-000.png",
    )


def test_pinhole_import_rejects_mixed_fov(tmp_path: Path) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(
        report_path,
        image_dir=image_dir,
        fovs=((1.2, 1.0), (1.3, 1.0)),
    )

    with pytest.raises(ValueError, match="fov_x"):
        run_pinhole_image_import(
            render_report_path=report_path,
            workspace=tmp_path / "workspace",
            show_progress=False,
        )


def test_pinhole_import_rejects_missing_image_dir_override_file(
    tmp_path: Path,
) -> None:
    old_image_dir = tmp_path / "old-renders"
    override_image_dir = tmp_path / "override-renders"
    write_image(override_image_dir / "a__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=old_image_dir)

    with pytest.raises(FileNotFoundError, match="b__sel-000.png"):
        run_pinhole_image_import(
            render_report_path=report_path,
            workspace=tmp_path / "workspace",
            image_dir=override_image_dir,
            show_progress=False,
        )


def test_pinhole_import_rejects_mixed_actual_resolution(tmp_path: Path) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (12, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=image_dir)

    with pytest.raises(ValueError, match="expected 10x8"):
        run_pinhole_image_import(
            render_report_path=report_path,
            workspace=tmp_path / "workspace",
            show_progress=False,
        )


def test_pinhole_import_cli_accepts_image_dir_override(tmp_path: Path) -> None:
    image_dir = tmp_path / "renders"
    image_dir.mkdir()
    report_path = tmp_path / "render_report.json"
    report_path.write_text('{"renders": []}\n', encoding="utf-8")

    args = parse_import_args(
        [
            "--render-report",
            str(report_path),
            "--image-dir",
            str(image_dir),
            "--workspace",
            str(tmp_path / "workspace"),
        ],
    )

    assert args.render_report == report_path
    assert args.image_dir == image_dir
