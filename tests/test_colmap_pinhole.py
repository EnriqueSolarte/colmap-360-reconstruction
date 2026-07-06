import json
from math import tan
from pathlib import Path
import sqlite3

import pytest
from PIL import Image

from colmap.panorama.types import GeneratedImageBatch
from colmap.pinhole import pipeline as pinhole_pipeline
from colmap.pinhole.cli import parse_args as parse_reconstruct_args
from colmap.pinhole.cli_import import parse_import_args
from colmap.pinhole.import_pipeline import run_pinhole_image_import
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


def test_pinhole_reconstruct_cli_accepts_baseline_options(tmp_path: Path) -> None:
    image_dir = tmp_path / "renders"
    image_dir.mkdir()
    report_path = tmp_path / "render_report.json"
    report_path.write_text('{"renders": []}\n', encoding="utf-8")

    args = parse_reconstruct_args(
        [
            "--render-report",
            str(report_path),
            "--image-dir",
            str(image_dir),
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
    assert args.overlap == 7
    assert args.device == "cpu"
    assert args.loop_detection is True


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
    monkeypatch.setattr(
        pinhole_pipeline,
        "create_mapper_options",
        lambda *args, **kwargs: {"mapper": True},
    )

    def fake_match_features(*args, **kwargs) -> None:
        calls.append(("matcher", kwargs["matcher"]))
        calls.append(("overlap", kwargs["overlap"]))

    def fake_mapping(*args, **kwargs):
        calls.append(("image_dir", kwargs["image_dir"]))
        return {0: FakeReconstruction(registered_images=2, points=7)}

    monkeypatch.setattr(pinhole_pipeline, "match_features", fake_match_features)
    monkeypatch.setattr(pinhole_pipeline, "run_incremental_mapping", fake_mapping)

    report = pinhole_pipeline.run_pinhole_reconstruction(
        render_report_path=report_path,
        workspace=tmp_path / "workspace",
        overlap=4,
        overwrite=True,
    )

    assert report["status"] == "raw_colmap_pinhole_render_report_unscaled"
    assert report["camera"]["model"] == "PINHOLE"
    assert report["rendered_image_count"] == 2
    assert report["feature_health"]["pinhole_images_with_features"] == 2
    assert report["best_model"]["num_registered_images"] == 2
    assert report["best_model"]["num_points3D"] == 7
    assert ("matcher", "sequential") in calls
    assert ("overlap", 4) in calls
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
