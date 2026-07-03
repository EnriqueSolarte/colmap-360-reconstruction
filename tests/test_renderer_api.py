import io
import json
from math import degrees, pi

import numpy as np
import pytest

from render_pinhole import EquirectangularToPinhole, render_pinhole
from render_pinhole.tools.render_pinhole_image import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    main as render_image_main,
)
from render_pinhole.defaults import DEFAULT_PINHOLE_FOV_X, DEFAULT_PINHOLE_FOV_Y
from render_pinhole.selection_schema import (
    auto_grid_selection_store,
    build_selection_store,
    selection_record,
)
from render_pinhole.tools.manual_pinhole_select_server import (
    INDEX_HTML,
    append_flat_selection,
    flat_api_state,
    image_dir_panorama_records,
    load_flat_selection_store,
    pixel_to_longitude_latitude,
    replace_flat_auto_grid_selections,
    render_flat_panorama_display_jpeg,
    render_flat_preview_png,
    reset_flat_collection_selections,
    roi_points_by_delta,
)
from render_pinhole.tools.manual_pinhole_select import _parse_args as parse_select_args
from render_pinhole.tools.random_pinhole_image import main as random_render_image_main
from render_pinhole.tools.render_pinhole_directory import main as render_directory_main
from render_pinhole.tools.render_selected_pinhole import (
    DEFAULT_FOV_X_DEGREES,
    DEFAULT_FOV_Y_DEGREES,
    RENDER_REPORT_FILENAME,
    _parse_args as parse_render_selected_args,
    main as render_selected_main,
)
from render_pinhole.tools.attach_pinhole_poses import (
    attach_pinhole_poses,
    write_colmap_text_export,
    lookup_panorama_pose,
    native_panorama_pose_record,
    panorama_match_keys,
)
from render_pinhole.view_sampling import (
    VIEW_SAMPLING_DESCRIPTIONS,
    random_views,
    uniform_longitude_views,
)


def make_equirectangular_image() -> np.ndarray:
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    image[..., 0] = np.arange(32, dtype=np.uint8)
    image[..., 1] = np.arange(16, dtype=np.uint8)[:, None]
    image[..., 2] = 127
    return image


def angular_distance(a: float, b: float) -> float:
    return abs((a - b + pi) % (2 * pi) - pi)


def test_random_views_preserves_existing_sampling_shape() -> None:
    rng = np.random.default_rng(123)

    views = random_views(
        num_views=2,
        rng=rng,
        longitude_min=-0.3,
        longitude_max=0.3,
        latitude_min=-0.2,
        latitude_max=0.2,
        longitude_jitter=0.05,
        latitude_jitter=0.03,
    )

    assert len(views) == 2
    assert all(-0.3 <= view["base_longitude"] <= 0.3 for view in views)
    assert all(-0.2 <= view["base_latitude"] <= 0.2 for view in views)
    assert all(
        angular_distance(view["longitude"], view["base_longitude"]) <= 0.05
        for view in views
    )
    assert all(
        abs(view["latitude"] - view["base_latitude"]) <= 0.03
        for view in views
    )


def test_uniform_longitude_views_evenly_spaces_longitudes() -> None:
    views = uniform_longitude_views(
        num_views=4,
        longitude_min=-pi,
        longitude_max=pi,
        latitude=0.25,
    )

    assert [view["base_longitude"] for view in views] == [
        -pi,
        -pi / 2,
        0.0,
        pi / 2,
    ]
    assert [view["longitude"] for view in views] == [
        -pi,
        -pi / 2,
        0.0,
        pi / 2,
    ]
    assert [view["latitude"] for view in views] == [0.25, 0.25, 0.25, 0.25]


def test_uniform_longitude_single_view_uses_midpoint() -> None:
    views = uniform_longitude_views(
        num_views=1,
        longitude_min=-pi,
        longitude_max=pi,
        latitude=0.0,
    )

    assert views[0]["longitude"] == 0.0


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


def test_render_image_cli_writes_output(tmp_path, capsys) -> None:
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
    output = capsys.readouterr().out
    assert "Input EP resolution: 32x16" in output
    assert "Output pinhole resolution: 10x8" in output


def test_render_image_cli_uses_reconstruction_defaults(tmp_path, capsys) -> None:
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
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert np.asarray(Image.open(output_path)).shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)
    output = capsys.readouterr().out
    assert "Input EP resolution: 32x16" in output
    assert f"Output pinhole resolution: {DEFAULT_WIDTH}x{DEFAULT_HEIGHT}" in output


def test_random_render_image_cli_writes_output(tmp_path, capsys) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "random-output.png"
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(input_path)

    exit_code = random_render_image_main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--height",
            "8",
            "--width",
            "10",
            "--fov-x",
            str(pi / 2),
            "--fov-y",
            str(pi / 3),
            "--seed",
            "123",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert np.asarray(Image.open(output_path)).shape == (8, 10, 3)
    output = capsys.readouterr().out
    assert "Input EP resolution: 32x16" in output
    assert "Random longitude:" in output
    assert "Random latitude:" in output
    assert "Output pinhole resolution: 10x8" in output


def test_manual_select_image_dir_defaults_output_dir_to_image_parent(tmp_path) -> None:
    image_dir = tmp_path / "workspace" / "images"
    explicit_output_dir = tmp_path / "manual-selections"
    image_dir.mkdir(parents=True)

    default_args = parse_select_args(["--image-dir", str(image_dir)])
    explicit_args = parse_select_args(
        ["--image-dir", str(image_dir), "--output-dir", str(explicit_output_dir)]
    )

    assert default_args.output_dir == image_dir.parent
    assert explicit_args.output_dir == explicit_output_dir


def test_manual_select_ui_cmd_click_and_arrow_navigation_defaults() -> None:
    assert "cmdSaveOnly" not in INDEX_HTML
    assert "Cmd-click save only" not in INDEX_HTML
    assert "saveSelectionAndAdvance(event.shiftKey)" in INDEX_HTML
    assert "function navigatePanorama(delta)" in INDEX_HTML
    assert 'event.key === "ArrowLeft"' in INDEX_HTML
    assert 'event.key === "ArrowRight"' in INDEX_HTML


def test_manual_select_ui_highlighted_marker_can_be_deleted() -> None:
    assert ".selection-marker.highlighted" in INDEX_HTML
    assert "const HIGHLIGHT_DISTANCE_PX" in INDEX_HTML
    assert "function nearestSelectionIndex(point)" in INDEX_HTML
    assert "function updateHighlightedSelection(point)" in INDEX_HTML
    assert 'event.key.toLowerCase() === "d"' in INDEX_HTML
    assert "deleteSelection(highlightedSelectionIndex)" in INDEX_HTML


def test_render_selected_pinhole_cli_writes_images_and_report(tmp_path, capsys) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(image_dir / "front.png")
    Image.fromarray(make_equirectangular_image()).save(image_dir / "back.png")
    selections_path = tmp_path / "selections.json"
    selections_store = build_selection_store(
        image_dir=image_dir,
        selections=[
            selection_record(
                pano_image_name="front.png",
                selection_index=0,
                longitude=pi / 4,
                latitude=0.0,
                selected_by="manual-pinhole-select",
            ),
            selection_record(
                pano_image_name="back.png",
                selection_index=0,
                longitude=-pi / 4,
                latitude=0.1,
                selected_by="manual-pinhole-select",
            )
        ],
        selection_mode="manual",
    )
    selections_path.write_text(
        json.dumps(selections_store),
        encoding="utf-8",
    )

    exit_code = render_selected_main(
        [
            "-s",
            str(selections_path),
            "-h",
            "8",
            "-w",
            "10",
            "--fov-x-deg",
            "90",
            "--fov-y-deg",
            "60",
            "--interpolation",
            "nearest",
            "--workers",
            "2",
        ]
    )

    output_dir = tmp_path / "renders"
    output_path = output_dir / "front__sel-000.png"
    second_output_path = output_dir / "back__sel-000.png"
    report = json.loads(
        (output_dir / RENDER_REPORT_FILENAME).read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert output_path.exists()
    assert second_output_path.exists()
    assert np.asarray(Image.open(output_path)).shape == (8, 10, 3)
    assert report["source"]["image_dir"] == str(image_dir)
    assert report["options"]["workers"] == 2
    assert report["render_params"]["width"] == 10
    assert report["render_params"]["height"] == 8
    assert report["render_params"]["fov_x_degrees"] == pytest.approx(90.0)
    assert report["render_params"]["fov_y_degrees"] == pytest.approx(60.0)
    assert report["rendered_count"] == 2
    assert report["skipped_count"] == 0
    assert report["failed_count"] == 0
    assert [record["output_file"] for record in report["renders"]] == [
        str(output_path),
        str(second_output_path),
    ]
    captured = capsys.readouterr()
    assert "Rendered: 2" in captured.out
    assert "Rendering selected pinholes" in captured.err


def test_render_selected_pinhole_cli_reports_missing_sources(tmp_path) -> None:
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "selected-renders"
    image_dir.mkdir()
    selections_path = tmp_path / "selections.json"
    selections_store = build_selection_store(
        image_dir=image_dir,
        selections=[
            selection_record(
                pano_image_name="missing.png",
                selection_index=0,
                longitude=0.0,
                latitude=0.0,
                selected_by="manual-pinhole-select",
            )
        ],
        selection_mode="manual",
    )
    selections_path.write_text(
        json.dumps(selections_store),
        encoding="utf-8",
    )

    exit_code = render_selected_main(
        [
            "--selections",
            str(selections_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    report = json.loads(
        (output_dir / RENDER_REPORT_FILENAME).read_text(encoding="utf-8")
    )
    assert exit_code == 1
    assert report["rendered_count"] == 0
    assert report["skipped_count"] == 1
    assert report["skipped"][0]["pano_image_name"] == "missing.png"


def test_render_selected_pinhole_cli_uses_degree_fov_only(tmp_path) -> None:
    selections_path = tmp_path / "selections.json"
    selections_path.write_text("{}", encoding="utf-8")

    args = parse_render_selected_args(["-s", str(selections_path)])

    assert args.fov_x_deg == DEFAULT_FOV_X_DEGREES
    assert args.fov_y_deg == DEFAULT_FOV_Y_DEGREES
    assert args.fov_x == pytest.approx(DEFAULT_PINHOLE_FOV_X)
    assert args.fov_y == pytest.approx(DEFAULT_PINHOLE_FOV_Y)
    with pytest.raises(SystemExit):
        parse_render_selected_args(
            ["-s", str(selections_path), "--fov-x", str(pi / 2)]
        )


def test_attach_pinhole_poses_reports_matched_and_missing_native_poses(tmp_path) -> None:
    selections_path = tmp_path / "selections.json"
    render_report_path = tmp_path / "renders" / "render_report.json"
    output_dir = tmp_path / "posed-renders"
    selections_store = build_selection_store(
        image_dir=tmp_path / "images",
        selections=[
            selection_record(
                pano_image_name="front.png",
                selection_index=0,
                longitude=pi / 4,
                latitude=0.0,
                selected_by="manual-pinhole-select",
            ),
            selection_record(
                pano_image_name="missing.png",
                selection_index=0,
                longitude=0.0,
                latitude=0.0,
                selected_by="manual-pinhole-select",
            ),
        ],
        selection_mode="manual",
    )
    render_report = {
        "renders": [
            {
                "selection_id": "front__sel-000",
                "pano_image_name": "front.png",
                "output_file": str(tmp_path / "renders" / "front__sel-000.png"),
                "output_name": "front__sel-000.png",
                "width": 10,
                "height": 8,
                "fov_x": pi / 2,
                "fov_y": pi / 3,
            },
            {
                "selection_id": "missing__sel-000",
                "pano_image_name": "missing.png",
                "output_file": str(tmp_path / "renders" / "missing__sel-000.png"),
                "output_name": "missing__sel-000.png",
                "width": 10,
                "height": 8,
                "fov_x": pi / 2,
                "fov_y": pi / 3,
            },
        ],
    }
    reconstruction = FakeColmapReconstruction(
        {
            7: FakePoint3D(
                error=0.25,
                image_ids=[],
                xyz=(1.0, 2.0, 3.0),
                color=(10, 20, 30),
            ),
        },
        num_reg_images=1,
        images=(
            FakeColmapImage(
                image_id=7,
                name="nested/front.png",
                point3d_ids=[],
            ),
        ),
    )

    pose_report = attach_pinhole_poses(
        selections_store=selections_store,
        render_report=render_report,
        selections_path=selections_path,
        render_report_path=render_report_path,
        sparse_best_path=tmp_path / "sparse_best",
        output_dir=output_dir,
        reconstruction=reconstruction,
    )

    assert pose_report["selected_view_count"] == 2
    assert pose_report["rendered_view_count"] == 2
    assert pose_report["registered_panorama_count"] == 1
    assert pose_report["pose_attached_view_count"] == 1
    assert pose_report["missing_pose_view_count"] == 1
    matched = pose_report["views"][0]
    missing = pose_report["views"][1]
    assert matched["pose_status"] == "matched"
    assert matched["source_colmap_image_name"] == "nested/front.png"
    assert matched["pinhole_camera_pose"]["coordinate_frame"] == "colmap_world_unscaled"
    assert matched["pinhole_camera_pose"]["center_world"] == [1.0, 2.0, 3.0]
    assert missing == {
        "selection_id": "missing__sel-000",
        "pano_image_name": "missing.png",
        "output_file": str(tmp_path / "renders" / "missing__sel-000.png"),
        "pose_status": "missing",
        "missing_reason": "panorama pose not registered in COLMAP model",
    }
    manifest = write_colmap_text_export(
        pose_report=pose_report,
        render_report=render_report,
        reconstruction=reconstruction,
        output_dir=output_dir,
    )
    sparse_dir = output_dir / "sparse" / "0"
    assert manifest == {
        "status": "exported",
        "format": "colmap_text",
        "coordinate_frame": "colmap_world_unscaled",
        "sparse_dir": str(sparse_dir),
        "camera_count": 1,
        "image_count": 1,
        "point3D_count": 1,
        "points3D_tracks": "omitted_for_derived_pinhole_views",
    }
    cameras_text = (sparse_dir / "cameras.txt").read_text(encoding="utf-8")
    images_text = (sparse_dir / "images.txt").read_text(encoding="utf-8")
    points_text = (sparse_dir / "points3D.txt").read_text(encoding="utf-8")
    assert "1 PINHOLE 10 8 5 6.92820323028 5 4" in cameras_text
    assert "1 0.923879532511 0 -0.382683432365 0 1.41421356237 -2 -2.82842712475 1 front__sel-000.png" in images_text
    assert "7 1 2 3 10 20 30 0.25" in points_text


def test_native_panorama_pose_matching_uses_basename() -> None:
    image = FakeColmapImage(
        image_id=1,
        name="images/front.png",
        point3d_ids=[],
    )
    pose = native_panorama_pose_record(image)

    assert panorama_match_keys("images/front.png") == (
        "images/front.png",
        "front.png",
    )
    assert lookup_panorama_pose({"front.png": pose}, "nested/front.png") is pose


def test_render_directory_cli_writes_multiple_outputs_per_image(tmp_path, capsys) -> None:
    input_dir = tmp_path / "eps"
    output_dir = tmp_path / "pinholes"
    input_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(input_dir / "front.png")
    Image.fromarray(make_equirectangular_image()).save(input_dir / "back.jpg")
    (input_dir / "notes.txt").write_text("not an image")

    exit_code = render_directory_main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--num-views",
            "3",
            "--height",
            "8",
            "--width",
            "10",
            "--fov-x",
            str(pi / 2),
            "--fov-y",
            str(pi / 3),
            "--output-extension",
            ".png",
            "--latitude-min",
            "-0.2",
            "--latitude-max",
            "0.2",
            "--longitude-min",
            "-0.3",
            "--longitude-max",
            "0.3",
            "--seed",
            "123",
        ]
    )

    assert exit_code == 0
    output_paths = sorted(output_dir.glob("*.png"))
    assert [path.name for path in output_paths] == [
        "back_pinhole_00.png",
        "back_pinhole_01.png",
        "back_pinhole_02.png",
        "front_pinhole_00.png",
        "front_pinhole_01.png",
        "front_pinhole_02.png",
    ]
    for output_path in output_paths:
        assert np.asarray(Image.open(output_path)).shape == (8, 10, 3)
    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["view_sampling"] == "random"
    assert metadata["view_sampling_description"] == VIEW_SAMPLING_DESCRIPTIONS["random"]
    assert metadata["num_views"] == 3
    assert metadata["height"] == 8
    assert metadata["width"] == 10
    assert metadata["latitude_min"] == -0.2
    assert metadata["latitude_max"] == 0.2
    assert metadata["longitude_min"] == -0.3
    assert metadata["longitude_max"] == 0.3
    assert metadata["longitude_jitter"] == 0.0
    assert metadata["latitude_jitter"] == 0.0
    assert metadata["fov_x"] == pi / 2
    assert metadata["fov_y"] == pi / 3
    assert metadata["interpolation"] == "bilinear"
    assert metadata["seed"] == 123
    assert len(metadata["images"]) == 6
    first_image = metadata["images"][0].copy()
    first_longitude = first_image.pop("longitude")
    first_base_longitude = first_image.pop("base_longitude")
    first_latitude = first_image.pop("latitude")
    first_base_latitude = first_image.pop("base_latitude")
    assert -0.3 <= first_longitude <= 0.3
    assert angular_distance(first_longitude, first_base_longitude) < 1e-12
    assert -0.2 <= first_latitude <= 0.2
    assert first_latitude == first_base_latitude
    assert first_image == {
        "source_file": str(input_dir / "back.jpg"),
        "source_name": "back.jpg",
        "source_width": 32,
        "source_height": 16,
        "file": str(output_dir / "back_pinhole_00.png"),
        "file_name": "back_pinhole_00.png",
        "view_index": 0,
        "width": 10,
        "height": 8,
        "fov_x": pi / 2,
        "fov_y": pi / 3,
        "interpolation": "bilinear",
    }
    assert all(-0.3 <= image["longitude"] <= 0.3 for image in metadata["images"])
    assert all(-0.2 <= image["latitude"] <= 0.2 for image in metadata["images"])
    back_views = [
        (image["longitude"], image["latitude"])
        for image in metadata["images"]
        if image["source_name"] == "back.jpg"
    ]
    front_views = [
        (image["longitude"], image["latitude"])
        for image in metadata["images"]
        if image["source_name"] == "front.png"
    ]
    assert back_views != front_views
    output = capsys.readouterr().out
    assert f"Metadata: {metadata_path}" in output
    assert "Rendered 6 pinhole images from 2 EP images." in output


def test_render_directory_cli_supports_fixed_latitude_shortcut(tmp_path) -> None:
    input_dir = tmp_path / "eps"
    output_dir = tmp_path / "pinholes"
    input_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(input_dir / "front.png")

    exit_code = render_directory_main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--num-views",
            "2",
            "--height",
            "8",
            "--width",
            "10",
            "--latitude",
            "0.25",
            "--fov-x",
            str(pi / 2),
            "--fov-y",
            str(pi / 3),
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["latitude_min"] == 0.25
    assert metadata["latitude_max"] == 0.25
    assert [image["latitude"] for image in metadata["images"]] == [0.25, 0.25]


def test_render_directory_cli_supports_random_views_and_jitter(tmp_path) -> None:
    input_dir = tmp_path / "eps"
    output_dir = tmp_path / "pinholes"
    input_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(input_dir / "front.png")

    exit_code = render_directory_main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--num-views",
            "2",
            "--height",
            "8",
            "--width",
            "10",
            "--latitude-min",
            "-0.35",
            "--latitude-max",
            "0.35",
            "--longitude-jitter",
            "0.05",
            "--latitude-jitter",
            "0.03",
            "--seed",
            "123",
            "--fov-x",
            str(pi / 2),
            "--fov-y",
            str(pi / 3),
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    records = metadata["images"]

    assert exit_code == 0
    assert metadata["latitude_min"] == -0.35
    assert metadata["latitude_max"] == 0.35
    assert metadata["longitude_jitter"] == 0.05
    assert metadata["latitude_jitter"] == 0.03
    assert metadata["total_views_per_image"] == 2
    assert len(records) == 2
    assert all(-0.35 <= record["base_latitude"] <= 0.35 for record in records)
    assert all(
        angular_distance(record["longitude"], record["base_longitude"]) <= 0.05
        for record in records
    )
    assert all(
        abs(record["latitude"] - record["base_latitude"]) <= 0.03
        for record in records
    )


def test_render_directory_cli_supports_uniform_longitude_views(tmp_path) -> None:
    input_dir = tmp_path / "eps"
    output_dir = tmp_path / "pinholes"
    input_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(input_dir / "front.png")

    exit_code = render_directory_main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--num-views",
            "4",
            "--view-sampling",
            "uniform_longitude",
            "--height",
            "8",
            "--width",
            "10",
            "--latitude",
            "0.25",
            "--fov-x",
            str(pi / 2),
            "--fov-y",
            str(pi / 3),
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    records = metadata["images"]

    assert exit_code == 0
    assert metadata["view_sampling"] == "uniform_longitude"
    assert (
        metadata["view_sampling_description"]
        == VIEW_SAMPLING_DESCRIPTIONS["uniform_longitude"]
    )
    assert [record["base_longitude"] for record in records] == [
        -pi,
        -pi / 2,
        0.0,
        pi / 2,
    ]
    assert [record["longitude"] for record in records] == [
        -pi,
        -pi / 2,
        0.0,
        pi / 2,
    ]
    assert [record["base_latitude"] for record in records] == [
        0.25,
        0.25,
        0.25,
        0.25,
    ]
    assert [record["latitude"] for record in records] == [
        0.25,
        0.25,
        0.25,
        0.25,
    ]


def test_flat_selection_store_supports_multiple_selections_per_panorama(tmp_path) -> None:
    selections = [
        selection_record(
            pano_image_name="1782890858308.png",
            selection_index=0,
            longitude=0.52,
            latitude=-0.12,
            selected_by="manual-pinhole-select",
        ),
        selection_record(
            pano_image_name="1782890858308.png",
            selection_index=1,
            longitude=1.2,
            latitude=0.05,
            selected_by="manual-pinhole-select",
        ),
        selection_record(
            pano_image_name="nested/1782890859000.png",
            selection_index=0,
            longitude=-0.25,
            latitude=0.1,
            selected_by="manual-pinhole-select",
        ),
    ]

    store = build_selection_store(
        image_dir=tmp_path / "images",
        selections=selections,
        selection_mode="manual",
    )

    assert store["schema_version"] == 1
    assert store["source"] == {"image_dir": str(tmp_path / "images")}
    assert store["selection_mode"] == "manual"
    assert store["selection_summary"] == {
        "selected_panorama_count": 2,
        "selection_count": 3,
    }
    assert [selection["selection_id"] for selection in store["selections"]] == [
        "1782890858308__sel-000",
        "1782890858308__sel-001",
        "nested__1782890859000__sel-000",
    ]
    assert store["selections"][0]["longitude_degrees"] == pytest.approx(
        degrees(0.52)
    )
    assert store["selections"][0]["latitude_degrees"] == pytest.approx(
        degrees(-0.12)
    )


def test_flat_auto_grid_selection_store_uses_same_selection_shape(tmp_path) -> None:
    store = auto_grid_selection_store(
        image_dir=tmp_path / "images",
        pano_image_names=["a.png", "b.png"],
    )

    assert store["selection_mode"] == "auto_grid"
    assert store["selection_summary"] == {
        "selected_panorama_count": 2,
        "selection_count": 64,
    }
    assert store["auto_selection"] == {
        "strategy": "longitude_latitude_grid",
        "delta_degrees": 45.0,
        "longitude_sample_count": 8,
        "latitude_sample_count": 4,
        "views_per_panorama": 32,
        "selection_count": 64,
    }
    first_selection = store["selections"][0]
    assert first_selection == {
        "selection_id": "a__sel-000",
        "pano_image_name": "a.png",
        "selection_index": 0,
        "longitude": pytest.approx(-7 * pi / 8),
        "latitude": pytest.approx(3 * pi / 8),
        "longitude_degrees": pytest.approx(-157.5),
        "latitude_degrees": pytest.approx(67.5),
        "selected_by": "auto_grid",
    }
    assert store["selections"][32]["selection_id"] == "b__sel-000"
    assert set(store["selections"][0]) == set(store["selections"][32])


def test_manual_select_pixel_coordinates_map_to_lon_lat() -> None:
    longitude, latitude = pixel_to_longitude_latitude(50.0, 25.0, 100, 50)

    assert longitude == pytest.approx(0.0)
    assert latitude == pytest.approx(0.0)

    longitude, latitude = pixel_to_longitude_latitude(0.0, 0.0, 100, 50)

    assert longitude == pytest.approx(-pi)
    assert latitude == pytest.approx(pi / 2)


def test_manual_select_roi_points_are_spaced_by_delta() -> None:
    points = roi_points_by_delta(
        longitude_min=-1.0,
        longitude_max=1.0,
        latitude_min=-1.0,
        latitude_max=1.0,
        delta_degrees=degrees(1.0),
    )

    assert points == [
        (-0.5, 0.5),
        (0.5, 0.5),
        (-0.5, -0.5),
        (0.5, -0.5),
    ]


def test_image_dir_selector_state_uses_flat_selections(tmp_path) -> None:
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "selection"
    nested_dir = image_dir / "nested"
    nested_dir.mkdir(parents=True)
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(image_dir / "b.png")
    Image.fromarray(make_equirectangular_image()).save(nested_dir / "a.png")

    image_records = image_dir_panorama_records(image_dir)
    store = load_flat_selection_store(output_dir, image_dir, image_records)
    first_selection = append_flat_selection(store, image_records[0], pi / 4, 0.0)
    second_selection = append_flat_selection(store, image_records[0], -pi / 4, 0.1)
    state = flat_api_state(
        image_dir=image_dir,
        output_dir=output_dir,
        image_records=image_records,
        store=store,
    )

    assert [record["pano_image_name"] for record in image_records] == [
        "b.png",
        "nested/a.png",
    ]
    assert first_selection["selection_id"] == "b__sel-000"
    assert second_selection["selection_id"] == "b__sel-001"
    assert state["selector_mode"] == "image_dir"
    assert state["selection_summary"] == {
        "selected_panorama_count": 1,
        "selection_count": 2,
    }
    assert state["panoramas"][0]["selections"] == [first_selection, second_selection]
    assert state["panoramas"][1]["selections"] == []


def test_image_dir_selector_preview_and_reset_are_colmap_free(tmp_path) -> None:
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "selection"
    image_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(image_dir / "pano.png")
    image_records = image_dir_panorama_records(image_dir)
    store = load_flat_selection_store(output_dir, image_dir, image_records)

    append_flat_selection(store, image_records[0], pi / 4, 0.0)
    preview = render_flat_preview_png(
        image_records[0],
        pi / 4,
        0.0,
        {
            "width": 10,
            "height": 8,
            "fov_x": pi / 2,
            "fov_y": pi / 3,
            "interpolation": "nearest",
        },
    )
    removed_count = reset_flat_collection_selections(store)

    assert preview.startswith(b"\x89PNG")
    assert np.asarray(Image.open(io.BytesIO(preview))).shape == (8, 10, 3)
    assert removed_count == 1
    assert store["selection_summary"] == {
        "selected_panorama_count": 0,
        "selection_count": 0,
    }


def test_image_dir_selector_auto_grid_replaces_with_flat_store(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(image_dir / "a.png")
    Image.fromarray(make_equirectangular_image()).save(image_dir / "b.png")
    image_records = image_dir_panorama_records(image_dir)

    store = replace_flat_auto_grid_selections(
        image_dir=image_dir,
        image_records=image_records,
        delta_degrees=90.0,
    )

    assert store["selection_mode"] == "auto_grid"
    assert store["selection_summary"] == {
        "selected_panorama_count": 2,
        "selection_count": 16,
    }
    assert store["auto_selection"] == {
        "strategy": "longitude_latitude_grid",
        "delta_degrees": 90.0,
        "longitude_sample_count": 4,
        "latitude_sample_count": 2,
        "views_per_panorama": 8,
        "selection_count": 16,
    }
    assert store["selections"][0]["selected_by"] == "auto_grid"
    assert store["selections"][0]["selection_id"] == "a__sel-000"
    assert store["selections"][8]["selection_id"] == "b__sel-000"


def test_manual_select_panorama_display_uses_resized_jpeg(tmp_path) -> None:
    source_path = tmp_path / "pano.png"
    Image = pytest.importorskip("PIL.Image")
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[..., 0] = np.arange(64, dtype=np.uint8)
    image[..., 1] = np.arange(32, dtype=np.uint8)[:, None]
    Image.fromarray(image).save(source_path)
    display = render_flat_panorama_display_jpeg(
        {"source_path": str(source_path)},
        max_width=16,
        max_height=8,
    )

    assert display.startswith(b"\xff\xd8")
    result = Image.open(io.BytesIO(display))
    assert result.format == "JPEG"
    assert result.size == (16, 8)




class FakeColmapImage:
    def __init__(self, image_id: int, name: str, point3d_ids: list[int]):
        self.image_id = image_id
        self.name = name
        self.camera_id = image_id + 100
        self.points2D = [FakePoint2D(point3d_id) for point3d_id in point3d_ids]
        self.num_points3D = len(point3d_ids)

    def projection_center(self):
        return np.asarray([1.0, 2.0, 3.0])

    def cam_from_world(self):
        return FakePose()


class FakePoint2D:
    def __init__(self, point3d_id: int | None):
        self.point3D_id = point3d_id

    def has_point3D(self) -> bool:
        return self.point3D_id is not None


class FakePoint3D:
    def __init__(
        self,
        error: float,
        image_ids: list[int],
        xyz: tuple[float, float, float] = (1.0, 2.0, 3.0),
        color: tuple[int, int, int] = (255, 255, 255),
    ):
        self.error = error
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.color = np.asarray(color, dtype=np.uint8)
        self.track = FakeTrack(image_ids)


class FakeTrack:
    def __init__(self, image_ids: list[int]):
        self.elements = [FakeTrackElement(image_id) for image_id in image_ids]

    def length(self) -> int:
        return len(self.elements)


class FakeTrackElement:
    def __init__(self, image_id: int):
        self.image_id = image_id


class FakeColmapReconstruction:
    def __init__(
        self,
        points: dict[int, FakePoint3D],
        num_reg_images: int,
        images: tuple[FakeColmapImage, ...] = (),
    ):
        self._points = points
        self.points3D = points
        self._num_reg_images = num_reg_images
        self._images = {int(image.image_id): image for image in images}

    def point3D(self, point_id: int) -> FakePoint3D:
        return self._points[point_id]

    def num_reg_images(self) -> int:
        return self._num_reg_images

    def reg_image_ids(self) -> list[int]:
        return sorted(self._images)

    def image(self, image_id: int) -> FakeColmapImage:
        return self._images[int(image_id)]


class FakePose:
    translation = np.asarray([0.0, 0.0, 0.0])
    rotation = None

    def __init__(self):
        self.rotation = FakeRotation()


class FakeRotation:
    def matrix(self):
        return np.eye(3)
