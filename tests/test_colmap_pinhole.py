import json
from math import tan
from pathlib import Path
import sqlite3

import pytest
from PIL import Image

from colmap.panorama.types import GeneratedImageBatch
from colmap.pinhole import pipeline as pinhole_pipeline
from colmap.pinhole.cli_colorize import parse_args as parse_colorize_args
from colmap.pinhole.cli import parse_args as parse_reconstruct_args
from colmap.pinhole.colorize import (
    PointColorizationSummary,
    colorize_reconstruction_points,
)
from colmap.pinhole.initial_poses import (
    initial_pose_from_view,
    write_initial_pose_model_from_pose_report,
)
from colmap.pinhole.cli_import import parse_import_args
from colmap.pinhole.import_pipeline import run_pinhole_image_import
from colmap.pinhole.pose_priors import (
    build_position_prior_summary,
    write_position_priors_to_database,
)
from colmap.pinhole.reports import build_pinhole_reconstruction_report
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


def write_pose_report(
    path: Path,
    views: list[dict],
) -> None:
    report = {
        "schema_version": 1,
        "source": {
            "render_report_path": str(path.parent / "render_report.json"),
        },
        "views": views,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def pose_view(
    *,
    selection_id: str | None,
    output_name: str,
    center_world: list[float] | None = None,
    pose_status: str = "matched",
) -> dict:
    view = {
        "selection_id": selection_id,
        "output_name": output_name,
        "pano_image_name": "pano.png",
        "pose_status": pose_status,
        "source_colmap_image_name": "pano.png",
    }
    if center_world is not None:
        view["pinhole_camera_pose"] = {
            "coordinate_frame": "colmap_world_unscaled",
            "center_world": center_world,
            "rotation_world_from_pinhole": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        }
    return view


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


def test_pinhole_colorize_cli_defaults_to_sibling_output_model(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "workspace" / "sparse_best"
    model_path.mkdir(parents=True)
    image_dir = tmp_path / "workspace" / "images"
    image_dir.mkdir()

    args = parse_colorize_args(["--model", str(model_path)])

    assert args.model == model_path
    assert args.image_dir == image_dir
    assert args.output_model == tmp_path / "workspace" / "sparse_best_colorized"
    assert args.in_place is False
    assert args.workers == 4
    assert args.no_progress is False


def test_pinhole_colorize_cli_in_place_overwrites_model(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "workspace" / "sparse_best"
    model_path.mkdir(parents=True)
    image_dir = tmp_path / "workspace" / "images"
    image_dir.mkdir()

    args = parse_colorize_args(["--model", str(model_path), "--in-place"])

    assert args.output_model == model_path
    assert args.overwrite is True


def test_pinhole_colorize_cli_rejects_nonpositive_workers(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "workspace" / "sparse_best"
    model_path.mkdir(parents=True)
    image_dir = tmp_path / "workspace" / "images"
    image_dir.mkdir()

    with pytest.raises(SystemExit):
        parse_colorize_args(["--model", str(model_path), "--workers", "0"])


def test_pinhole_reconstruct_cli_accepts_baseline_options(tmp_path: Path) -> None:
    image_dir = tmp_path / "renders"
    image_dir.mkdir()
    report_path = tmp_path / "render_report.json"
    report_path.write_text('{"renders": []}\n', encoding="utf-8")
    pose_report_path = tmp_path / "pose_report.json"
    pose_report_path.write_text('{"views": []}\n', encoding="utf-8")

    args = parse_reconstruct_args(
        [
            "--render-report",
            str(report_path),
            "--image-dir",
            str(image_dir),
            "--pose-report",
            str(pose_report_path),
            "--initial-poses-from-pose-report",
            "--workspace",
            str(tmp_path / "workspace"),
            "--overlap",
            "7",
            "--device",
            "cpu",
        ],
    )

    assert args.render_report == report_path
    assert args.image_dir == image_dir
    assert args.pose_report == pose_report_path
    assert args.use_position_priors is True
    assert args.initial_poses_from_pose_report is True
    assert args.overlap == 7
    assert args.device == "cpu"
    assert args.loop_detection is True


def test_pinhole_reconstruct_cli_can_disable_default_position_priors(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "render_report.json"
    report_path.write_text('{"renders": []}\n', encoding="utf-8")
    pose_report_path = tmp_path / "pose_report.json"
    pose_report_path.write_text('{"views": []}\n', encoding="utf-8")

    args = parse_reconstruct_args(
        [
            "--render-report",
            str(report_path),
            "--pose-report",
            str(pose_report_path),
            "--no-use-position-priors",
        ],
    )

    assert args.use_position_priors is False


def test_pinhole_reconstruct_cli_defaults_position_priors_off_without_pose_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "render_report.json"
    report_path.write_text('{"renders": []}\n', encoding="utf-8")

    args = parse_reconstruct_args(
        [
            "--render-report",
            str(report_path),
        ],
    )

    assert args.use_position_priors is False


def test_pinhole_reconstruct_cli_rejects_positive_position_priors_flag(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "render_report.json"
    report_path.write_text('{"renders": []}\n', encoding="utf-8")
    pose_report_path = tmp_path / "pose_report.json"
    pose_report_path.write_text('{"views": []}\n', encoding="utf-8")

    with pytest.raises(SystemExit):
        parse_reconstruct_args(
            [
                "--render-report",
                str(report_path),
                "--pose-report",
                str(pose_report_path),
                "--use-position-priors",
            ],
        )


def test_pinhole_reconstruct_cli_requires_pose_report_for_initial_poses(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "render_report.json"
    report_path.write_text('{"renders": []}\n', encoding="utf-8")

    with pytest.raises(SystemExit):
        parse_reconstruct_args(
            [
                "--render-report",
                str(report_path),
                "--initial-poses-from-pose-report",
            ],
        )


def test_position_prior_summary_matches_selection_id_and_output_name_fallback(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=image_dir)
    import_result = pinhole_pipeline.import_pinhole_images_from_render_report(
        render_report_path=report_path,
        output_image_dir=tmp_path / "workspace" / "images",
        show_progress=False,
    )
    pose_report_path = tmp_path / "pose_report.json"
    write_pose_report(
        pose_report_path,
        [
            pose_view(
                selection_id="pano_000__sel-000",
                output_name="a__sel-000.png",
                center_world=[1.0, 2.0, 3.0],
            ),
            pose_view(
                selection_id=None,
                output_name="b__sel-000.png",
                center_world=[4.0, 5.0, 6.0],
            ),
        ],
    )

    summary = build_position_prior_summary(
        pose_report_path=pose_report_path,
        imported_images=import_result.imported_images,
    )

    assert summary.report()["matched_count"] == 2
    assert summary.report()["missing_count"] == 0
    assert summary.report()["unusable_count"] == 0
    assert summary.priors[0].center_world == (1.0, 2.0, 3.0)
    assert summary.priors[1].center_world == (4.0, 5.0, 6.0)


def test_position_prior_summary_reports_missing_and_unusable_views(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=image_dir)
    import_result = pinhole_pipeline.import_pinhole_images_from_render_report(
        render_report_path=report_path,
        output_image_dir=tmp_path / "workspace" / "images",
        show_progress=False,
    )
    pose_report_path = tmp_path / "pose_report.json"
    write_pose_report(
        pose_report_path,
        [
            pose_view(
                selection_id="pano_000__sel-000",
                output_name="a__sel-000.png",
                pose_status="missing",
            ),
        ],
    )

    report = build_position_prior_summary(
        pose_report_path=pose_report_path,
        imported_images=import_result.imported_images,
    ).report()

    assert report["matched_count"] == 0
    assert report["missing_count"] == 1
    assert report["unusable_count"] == 1
    assert report["use_position_priors"] is False
    assert report["unusable"][0]["reason"] == "pose_status is missing"


def test_initial_pose_from_view_converts_world_pose_to_colmap_cam_from_world() -> None:
    pose, reason = initial_pose_from_view(
        pose_view(
            selection_id="sel",
            output_name="image.png",
            center_world=[1.0, 2.0, 3.0],
        ),
        fallback_selection_id="sel",
        fallback_output_name="image.png",
        image_id=7,
        camera_id=3,
    )

    assert reason is None
    assert pose is not None
    assert pose.qvec == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert pose.tvec == pytest.approx((-1.0, -2.0, -3.0))
    assert pose.image_id == 7
    assert pose.camera_id == 3


def test_write_initial_pose_model_from_pose_report_writes_colmap_text_model(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=image_dir)
    import_result = pinhole_pipeline.import_pinhole_images_from_render_report(
        render_report_path=report_path,
        output_image_dir=tmp_path / "workspace" / "images",
        show_progress=False,
    )
    pose_report_path = tmp_path / "pose_report.json"
    write_pose_report(
        pose_report_path,
        [
            pose_view(
                selection_id="pano_000__sel-000",
                output_name="a__sel-000.png",
                center_world=[1.0, 2.0, 3.0],
            ),
            pose_view(
                selection_id="pano_001__sel-000",
                output_name="b__sel-000.png",
                center_world=[4.0, 5.0, 6.0],
            ),
        ],
    )
    generated_batch = GeneratedImageBatch(
        names=("a__sel-000.png", "b__sel-000.png"),
        records_by_name={
            "a__sel-000.png": (10, 2),
            "b__sel-000.png": (11, 2),
        },
    )
    model_path = tmp_path / "initial_model"

    summary = write_initial_pose_model_from_pose_report(
        pose_report_path=pose_report_path,
        model_path=model_path,
        imported_images=import_result.imported_images,
        generated_image_batch=generated_batch,
        camera=import_result.camera,
    )

    assert summary.report()["initialized_count"] == 2
    assert summary.report()["fixed_poses"] is False
    assert (model_path / "cameras.txt").is_file()
    images_text = (model_path / "images.txt").read_text(encoding="utf-8")
    assert "10 1 0 0 0 -1 -2 -3 2 a__sel-000.png" in images_text
    assert "11 1 0 0 0 -4 -5 -6 2 b__sel-000.png" in images_text
    assert (model_path / "points3D.txt").read_text(encoding="utf-8").count("\n") == 3


class FakeColorTrackElement:
    def __init__(self, image_id: int, point2d_idx: int) -> None:
        self.image_id = image_id
        self.point2D_idx = point2d_idx


class FakeColorTrack:
    def __init__(self, elements: list[FakeColorTrackElement]) -> None:
        self.elements = elements


class FakeColorPoint3D:
    def __init__(self, elements: list[FakeColorTrackElement]) -> None:
        self.track = FakeColorTrack(elements)
        self.color = [0, 0, 0]


class FakeColorPoint2D:
    def __init__(self, xy: tuple[float, float]) -> None:
        self.xy = xy


class FakeColorImage:
    def __init__(self, name: str, points2d: dict[int, tuple[float, float]]) -> None:
        self.name = name
        self.points2d = points2d

    def point2D(self, point2d_idx: int) -> FakeColorPoint2D:
        return FakeColorPoint2D(self.points2d[point2d_idx])


class FakeColorReconstruction:
    def __init__(self) -> None:
        self.images = {
            1: FakeColorImage("a.png", {0: (0.0, 0.0)}),
            2: FakeColorImage("b.png", {0: (1.0, 0.0)}),
        }
        self.points3D = {
            10: FakeColorPoint3D(
                [
                    FakeColorTrackElement(1, 0),
                    FakeColorTrackElement(2, 0),
                ],
            ),
        }


def test_colorize_reconstruction_points_uses_mean_observation_rgb(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (2, 1), color=(0, 0, 0)).save(image_dir / "unused.png")
    image_a = Image.new("RGB", (2, 1), color=(0, 0, 0))
    image_a.putpixel((0, 0), (10, 20, 30))
    image_a.save(image_dir / "a.png")
    image_b = Image.new("RGB", (2, 1), color=(0, 0, 0))
    image_b.putpixel((1, 0), (30, 40, 50))
    image_b.save(image_dir / "b.png")
    reconstruction = FakeColorReconstruction()

    summary = colorize_reconstruction_points(
        reconstruction,
        image_dir=image_dir,
        show_progress=False,
        workers=2,
    )

    assert reconstruction.points3D[10].color.tolist() == [20, 30, 40]
    assert summary.report()["colored_point_count"] == 1
    assert summary.report()["sampled_observation_count"] == 2
    assert summary.report()["black_point_count_before"] == 1
    assert summary.report()["black_point_count_after"] == 0
    assert summary.report()["workers"] == 2


class FakeDevice:
    cpu = "cpu"
    cuda = "cuda"


class FakePycolmap:
    __version__ = "fake-pycolmap"
    has_cuda = False
    Device = FakeDevice

    def set_random_seed(self, value: int) -> None:
        self.random_seed = value


class FakeReconstruction:
    def __init__(self, *, registered_images: int = 2, points: int = 5):
        self.registered_images = registered_images
        self.points = points

    def num_images(self) -> int:
        return self.registered_images

    def num_reg_images(self) -> int:
        return self.registered_images

    def num_points3D(self) -> int:
        return self.points

    def compute_num_observations(self) -> int:
        return 12

    def compute_mean_observations_per_reg_image(self) -> float:
        return 6.0

    def compute_mean_track_length(self) -> float:
        return 2.4

    def compute_mean_reprojection_error(self) -> float:
        return 0.7

    def write(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "cameras.txt").write_text("# fake\n", encoding="utf-8")
        (path / "images.txt").write_text("# fake\n", encoding="utf-8")
        (path / "points3D.txt").write_text("# fake\n", encoding="utf-8")

    def export_PLY(self, path: Path) -> None:
        path.write_text("ply\n", encoding="utf-8")


class FakeMapperOptions:
    def __init__(self) -> None:
        self.use_prior_position = False


def fake_extract_features(
    _pycolmap_module,
    *,
    database_path: Path,
    image_dir: Path,
    camera,
    feature_options,
    execution,
) -> None:
    image_names = sorted(path.name for path in image_dir.iterdir() if path.is_file())
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY)")
        connection.execute(
            "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)",
        )
        connection.execute(
            "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER)",
        )
        connection.execute("INSERT INTO cameras(camera_id) VALUES (1)")
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, 1)",
            [
                (image_index + 1, image_name)
                for image_index, image_name in enumerate(image_names)
            ],
        )
        connection.executemany(
            "INSERT INTO keypoints(image_id, rows) VALUES (?, ?)",
            [
                (image_index + 1, 50)
                for image_index, _image_name in enumerate(image_names)
            ],
        )


def test_pinhole_reconstruction_pipeline_runs_baseline_with_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=image_dir)
    pose_report_path = tmp_path / "pose_report.json"
    write_pose_report(
        pose_report_path,
        [
            pose_view(
                selection_id="pano_000__sel-000",
                output_name="a__sel-000.png",
                center_world=[1.0, 2.0, 3.0],
            ),
            pose_view(
                selection_id="pano_001__sel-000",
                output_name="b__sel-000.png",
                center_world=[4.0, 5.0, 6.0],
            ),
        ],
    )

    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        pinhole_pipeline,
        "require_pycolmap",
        lambda: FakePycolmap(),
    )
    monkeypatch.setattr(
        pinhole_pipeline,
        "extract_panorama_features_to_database",
        fake_extract_features,
    )
    monkeypatch.setattr(
        pinhole_pipeline,
        "create_feature_matching_options",
        lambda *args, **kwargs: {"matching": True},
    )
    mapper_options = FakeMapperOptions()
    monkeypatch.setattr(
        pinhole_pipeline,
        "create_mapper_options",
        lambda *args, **kwargs: mapper_options,
    )
    monkeypatch.setattr(
        pinhole_pipeline,
        "write_position_priors_to_database",
        lambda *args, **kwargs: 2,
    )
    monkeypatch.setattr(
        pinhole_pipeline,
        "colorize_reconstruction_points",
        lambda _reconstruction, *, image_dir, workers: PointColorizationSummary(
            image_dir=image_dir,
            point_count=7,
            track_observation_count=14,
            sampled_observation_count=14,
            colored_point_count=7,
            black_point_count_before=7,
            black_point_count_after=0,
            requested_image_count=2,
            loaded_image_count=2,
            missing_image_count=0,
            missing_image_observation_count=0,
            missing_point2d_count=0,
            out_of_bounds_observation_count=0,
            workers=4,
        ),
    )

    def fake_match_features(*args, **kwargs) -> None:
        calls.append(("matcher", kwargs["matcher"]))
        calls.append(("overlap", kwargs["overlap"]))

    def fake_mapping(*args, **kwargs):
        calls.append(("image_dir", kwargs["image_dir"]))
        calls.append(("input_path", kwargs.get("input_path")))
        return {0: FakeReconstruction(registered_images=2, points=7)}

    monkeypatch.setattr(pinhole_pipeline, "match_features", fake_match_features)
    monkeypatch.setattr(pinhole_pipeline, "run_incremental_mapping", fake_mapping)

    report = pinhole_pipeline.run_pinhole_reconstruction(
        render_report_path=report_path,
        workspace=tmp_path / "workspace",
        pose_report_path=pose_report_path,
        initial_poses_from_pose_report=True,
        overlap=4,
        overwrite=True,
    )

    assert report["status"] == "raw_colmap_pinhole_render_report_unscaled"
    assert report["camera"]["model"] == "PINHOLE"
    assert report["rendered_image_count"] == 2
    assert report["feature_health"]["pinhole_images_with_features"] == 2
    assert report["best_model"]["num_registered_images"] == 2
    assert report["best_model"]["num_points3D"] == 7
    assert report["pose_priors"]["matched_count"] == 2
    assert report["pose_priors"]["use_position_priors"] is True
    assert report["pose_priors"]["written_prior_count"] == 2
    assert report["options"]["use_position_priors"] is True
    assert report["options"]["position_prior_count"] == 2
    assert report["initial_poses"]["initialized_count"] == 2
    assert report["initial_poses"]["fixed_poses"] is False
    assert report["options"]["initial_poses_from_pose_report"] is True
    assert report["options"]["point_colorization"] is True
    assert report["point_colorization"]["colored_point_count"] == 7
    assert report["point_colorization"]["black_point_count_after"] == 0
    assert report["point_colorization"]["workers"] == 4
    assert mapper_options.use_prior_position is True
    assert ("matcher", "sequential") in calls
    assert ("overlap", 4) in calls
    assert ("input_path", tmp_path / "workspace" / "initial_model") in calls
    assert (tmp_path / "workspace" / "sparse_best" / "cameras.txt").is_file()
    assert (tmp_path / "workspace" / "cloud_colmap_raw.ply").is_file()
    assert (tmp_path / "workspace" / "pinhole_reconstruction_report.json").is_file()


def test_build_pinhole_reconstruction_report_records_pose_priors_as_future_stage(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=image_dir)
    import_report = run_pinhole_image_import(
        render_report_path=report_path,
        workspace=tmp_path / "import-workspace",
        show_progress=False,
    )
    metadata = json.loads(
        (tmp_path / "import-workspace" / "pinhole_config.json").read_text(
            encoding="utf-8",
        ),
    )
    imported_images = metadata["pinhole_images"]

    fake_import_result = type("ImportResult", (), {})()
    fake_import_result.render_report_path = report_path
    fake_import_result.image_dir_override = None
    fake_import_result.camera = type(
        "Camera",
        (),
        {"report": lambda self: import_report["camera"]},
    )()
    fake_import_result.imported_images = tuple(
        type("Image", (), {"report": lambda self, image=image: image})()
        for image in imported_images
    )
    fake_import_result.image_names = tuple(
        image["colmap_image_name"] for image in imported_images
    )
    fake_import_result.resolution_mismatch_count = 0

    paths = {
        "workspace": tmp_path / "workspace",
        "image_dir": tmp_path / "workspace" / "images",
        "database": tmp_path / "database.db",
        "sparse": tmp_path / "workspace" / "sparse",
        "best_sparse": tmp_path / "workspace" / "sparse_best",
        "raw_ply": tmp_path / "workspace" / "cloud_colmap_raw.ply",
        "metadata": tmp_path / "workspace" / "pinhole_config.json",
        "report": tmp_path / "workspace" / "pinhole_reconstruction_report.json",
    }
    with sqlite3.connect(paths["database"]) as connection:
        connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY)")

    report = build_pinhole_reconstruction_report(
        pycolmap_version="fake",
        paths=paths,
        import_result=fake_import_result,
        matcher="sequential",
        overlap=3,
        loop_detection=True,
        ssl_cert_file=None,
        feature_options=pinhole_pipeline.FeaturePipelineOptions(),
        execution={
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "cuda_available": False,
            "gpu_index": "-1",
            "gpu_bundle_adjustment": False,
            "bundle_adjustment_gpu_index": "-1",
        },
        feature_report={},
        feature_health={},
        generated_image_batch=GeneratedImageBatch(names=(), records_by_name={}),
        reconstructions={0: FakeReconstruction()},
        best_model_id=0,
        best_reconstruction=FakeReconstruction(),
        import_workers=1,
    )

    assert report["pose_priors"] is None
    assert report["options"]["effective_pairing"] == "sequential"


class FakeDatabaseImage:
    def __init__(self, data_id: object) -> None:
        self.data_id = data_id


class FakePosePrior:
    def __init__(self) -> None:
        self.position = None
        self.coordinate_system = None
        self.corr_data_id = None


class FakePosePriorCoordinateSystem:
    CARTESIAN = "cartesian"


class FakeDatabase:
    instances: list["FakeDatabase"] = []

    def __init__(self) -> None:
        self.images = {
            "a__sel-000.png": FakeDatabaseImage("data-a"),
            "b__sel-000.png": FakeDatabaseImage("data-b"),
        }
        self.pose_priors: list[FakePosePrior] = []
        self.closed = False

    @classmethod
    def open(cls, _database_path: Path) -> "FakeDatabase":
        instance = cls()
        cls.instances.append(instance)
        return instance

    def read_image_with_name(self, name: str) -> FakeDatabaseImage | None:
        return self.images.get(name)

    def write_pose_prior(self, pose_prior: FakePosePrior) -> int:
        self.pose_priors.append(pose_prior)
        return len(self.pose_priors)

    def close(self) -> None:
        self.closed = True


class FakePriorPycolmap:
    Database = FakeDatabase
    PosePrior = FakePosePrior
    PosePriorCoordinateSystem = FakePosePriorCoordinateSystem


def test_write_position_priors_to_database_sets_corr_data_id_and_cartesian(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "renders"
    write_image(image_dir / "a__sel-000.png", (10, 8))
    write_image(image_dir / "b__sel-000.png", (10, 8))
    report_path = tmp_path / "render_report.json"
    write_render_report(report_path, image_dir=image_dir)
    import_result = pinhole_pipeline.import_pinhole_images_from_render_report(
        render_report_path=report_path,
        output_image_dir=tmp_path / "workspace" / "images",
        show_progress=False,
    )
    pose_report_path = tmp_path / "pose_report.json"
    write_pose_report(
        pose_report_path,
        [
            pose_view(
                selection_id="pano_000__sel-000",
                output_name="a__sel-000.png",
                center_world=[1.0, 2.0, 3.0],
            ),
            pose_view(
                selection_id="pano_001__sel-000",
                output_name="b__sel-000.png",
                center_world=[4.0, 5.0, 6.0],
            ),
        ],
    )
    summary = build_position_prior_summary(
        pose_report_path=pose_report_path,
        imported_images=import_result.imported_images,
    )
    FakeDatabase.instances.clear()

    written_count = write_position_priors_to_database(
        FakePriorPycolmap,
        database_path=tmp_path / "database.db",
        prior_summary=summary,
    )

    database = FakeDatabase.instances[-1]
    assert written_count == 2
    assert database.closed is True
    assert database.pose_priors[0].corr_data_id == "data-a"
    assert database.pose_priors[0].coordinate_system == "cartesian"
    assert database.pose_priors[0].position.tolist() == [1.0, 2.0, 3.0]
