import json
from math import pi

import numpy as np
import pytest

from render_pinhole import EquirectangularToPinhole, render_pinhole
from render_pinhole.tools.render_pinhole_image import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    main as render_image_main,
)
from render_pinhole.tools.manual_pinhole_render import (
    PanoramaRigViewMetadata,
    SceneContext,
    ScenePanoramaGroup,
    build_metadata,
    rendered_filename,
    main as manual_render_main,
)
from render_pinhole.tools.random_pinhole_image import main as random_render_image_main
from render_pinhole.tools.render_pinhole_directory import main as render_directory_main
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


def test_manual_render_cli_initializes_metadata(tmp_path, capsys) -> None:
    pano_dir = tmp_path / "panos"
    output_dir = tmp_path / "pinhole-renders"
    pano_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(pano_dir / "pano_002.png")
    Image.fromarray(make_equirectangular_image()).save(pano_dir / "pano_001.jpg")
    (pano_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    exit_code = manual_render_main(
        [
            "-i",
            str(pano_dir),
            "-o",
            str(output_dir),
            "--width",
            "10",
            "--height",
            "8",
            "--fov-x",
            str(pi / 2),
            "--fov-y",
            str(pi / 3),
            "--interpolation",
            "nearest",
            "--output-extension",
            "jpg",
        ]
    )

    metadata_path = output_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert metadata["schema_version"] == 1
    assert metadata["tool"] == "manual-pinhole-render"
    assert metadata["pano_dir"] == str(pano_dir)
    assert metadata["scene_workspace"] is None
    assert metadata["output_dir"] == str(output_dir)
    assert metadata["metadata_path"] == str(metadata_path)
    assert metadata["pose_status"] == "missing"
    assert metadata["source_pano_count"] == 2
    assert metadata["scene"] is None
    assert metadata["render_defaults"]["width"] == 10
    assert metadata["render_defaults"]["height"] == 8
    assert metadata["render_defaults"]["fov_x"] == pi / 2
    assert metadata["render_defaults"]["fov_y"] == pi / 3
    assert metadata["render_defaults"]["interpolation"] == "nearest"
    assert metadata["render_defaults"]["output_extension"] == ".jpg"
    assert metadata["render_defaults"]["camera"]["model"] == "PINHOLE"
    assert metadata["render_defaults"]["camera"]["width"] == 10
    assert metadata["render_defaults"]["camera"]["height"] == 8
    assert metadata["panorama_count"] == 2
    assert [record["pano_filename"] for record in metadata["panoramas"]] == [
        "pano_001.jpg",
        "pano_002.png",
    ]
    assert metadata["panoramas"][0]["pano_index"] == 0
    assert metadata["panoramas"][0]["pano_camera_pose"] is None
    assert metadata["panoramas"][0]["pose_status"] == "missing"
    assert metadata["panoramas"][0]["source_status"] == "matched"
    assert metadata["panoramas"][0]["colmap_quality"] is None
    assert metadata["panoramas"][0]["selections"] == []
    output = capsys.readouterr().out
    assert f"Metadata: {metadata_path}" in output
    assert "Initialized 2 panoramas." in output


def test_manual_render_cli_refuses_to_overwrite_metadata(tmp_path) -> None:
    pano_dir = tmp_path / "panos"
    output_dir = tmp_path / "pinhole-renders"
    pano_dir.mkdir()
    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(make_equirectangular_image()).save(pano_dir / "pano.png")

    assert manual_render_main(["-i", str(pano_dir), "-o", str(output_dir)]) == 0
    metadata_path = output_dir / "metadata.json"
    original_metadata = metadata_path.read_text(encoding="utf-8")

    assert manual_render_main(["-i", str(pano_dir), "-o", str(output_dir)]) == 1
    assert metadata_path.read_text(encoding="utf-8") == original_metadata

    assert (
        manual_render_main(
            ["-i", str(pano_dir), "-o", str(output_dir), "--overwrite"]
        )
        == 0
    )


def test_manual_render_cli_rejects_invalid_panorama_image(tmp_path, capsys) -> None:
    pano_dir = tmp_path / "panos"
    output_dir = tmp_path / "pinhole-renders"
    pano_dir.mkdir()
    (pano_dir / "broken.png").write_text("not an image", encoding="utf-8")

    exit_code = manual_render_main(["-i", str(pano_dir), "-o", str(output_dir)])

    assert exit_code == 1
    assert not (output_dir / "metadata.json").exists()
    assert "Invalid panorama image" in capsys.readouterr().out


def test_manual_render_filename_uses_pm_angle_convention() -> None:
    assert (
        rendered_filename(
            "shelf-1781171031546_000123",
            selection_index=2,
            longitude_degrees=45.0,
            latitude_degrees=-10.0,
            width=1280,
            height=960,
            extension="png",
        )
        == (
            "shelf-1781171031546_000123__sel-002"
            "__lon-p045.0__lat-m010.0__w1280_h960.png"
        )
    )


def test_manual_render_scene_metadata_uses_registered_colmap_panoramas(tmp_path) -> None:
    source_path = tmp_path / "shelf-0001.png"
    source_path.write_bytes(b"placeholder")
    scene_name = "img-0002_shelf-0001.png"
    views = (
        PanoramaRigViewMetadata(
            index=0,
            image_prefix="pano_camera00/",
            yaw=0.0,
            pitch=0.0,
            cam_from_pano_rotation=np.eye(3),
        ),
        PanoramaRigViewMetadata(
            index=1,
            image_prefix="pano_camera01/",
            yaw=0.0,
            pitch=0.0,
            cam_from_pano_rotation=np.eye(3),
        ),
    )
    images = (
        FakeColmapImage(
            image_id=1,
            name=f"pano_camera00/{scene_name}",
            point3d_ids=[10],
        ),
        FakeColmapImage(
            image_id=2,
            name=f"pano_camera01/{scene_name}",
            point3d_ids=[11],
        ),
    )
    reconstruction = FakeColmapReconstruction(
        {
            10: FakePoint3D(1.0, [1, 3]),
            11: FakePoint3D(3.0, [2]),
        },
        num_reg_images=2,
    )
    scene_context = SceneContext(
        workspace=tmp_path / "scene",
        model_path=tmp_path / "scene" / "sparse_best",
        rig_metadata_path=tmp_path / "scene" / "panorama_rig_config.json",
        reconstruction=reconstruction,
        views=views,
        groups=(ScenePanoramaGroup(name=scene_name, images=images),),
        image_frame_names={
            1: scene_name,
            2: scene_name,
            3: "connected.png",
        },
    )

    metadata = build_metadata(
        pano_paths=[source_path],
        pano_dir=tmp_path,
        output_dir=tmp_path / "renders",
        scene_workspace=tmp_path / "scene",
        width=10,
        height=8,
        fov_x=pi / 2,
        fov_y=pi / 3,
        interpolation="bilinear",
        output_extension=".png",
        scene_context=scene_context,
    )

    record = metadata["panoramas"][0]
    quality = record["colmap_quality"]

    assert metadata["panorama_count"] == 1
    assert metadata["source_pano_count"] == 1
    assert metadata["pose_status"] == "matched"
    assert metadata["scene"]["registered_virtual_image_count"] == 2
    assert metadata["scene"]["registered_panorama_count"] == 1
    assert record["scene_pano_filename"] == scene_name
    assert record["source_pano_filename"] == "shelf-0001.png"
    assert record["source_pano_path"] == str(source_path)
    assert record["source_status"] == "matched"
    assert record["source_resolution_strategy"] == "strip_first_prefix"
    assert record["pose_status"] == "matched"
    assert record["pano_camera_pose"]["coordinate_frame"] == "colmap_world_unscaled"
    assert record["pano_camera_pose"]["center_world"] == [1.0, 2.0, 3.0]
    assert quality["registered_virtual_image_count"] == 2
    assert quality["expected_virtual_image_count"] == 2
    assert quality["active_virtual_camera_count"] == 2
    assert quality["triangulated_observation_count"] == 2
    assert quality["unique_point3D_count"] == 2
    assert quality["mean_reprojection_error_px"] == 2.0
    assert quality["median_reprojection_error_px"] == 2.0
    assert quality["p90_reprojection_error_px"] == pytest.approx(2.8)
    assert quality["mean_track_length"] == 1.5
    assert quality["connected_panorama_count"] == 1
    assert quality["top_connected_panoramas"] == [
        {"scene_pano_filename": "connected.png", "shared_track_count": 1}
    ]
    assert [camera["virtual_camera_index"] for camera in quality["per_virtual_camera"]] == [
        0,
        1,
    ]


def test_manual_render_scene_metadata_marks_missing_source_panoramas(tmp_path) -> None:
    scene_name = "img-0002_missing.png"
    views = (
        PanoramaRigViewMetadata(
            index=0,
            image_prefix="pano_camera00/",
            yaw=0.0,
            pitch=0.0,
            cam_from_pano_rotation=np.eye(3),
        ),
    )
    scene_context = SceneContext(
        workspace=tmp_path / "scene",
        model_path=tmp_path / "scene" / "sparse_best",
        rig_metadata_path=tmp_path / "scene" / "panorama_rig_config.json",
        reconstruction=FakeColmapReconstruction({}, num_reg_images=1),
        views=views,
        groups=(
            ScenePanoramaGroup(
                name=scene_name,
                images=(
                    FakeColmapImage(
                        image_id=1,
                        name=f"pano_camera00/{scene_name}",
                        point3d_ids=[],
                    ),
                ),
            ),
        ),
        image_frame_names={1: scene_name},
    )

    metadata = build_metadata(
        pano_paths=[],
        pano_dir=tmp_path,
        output_dir=tmp_path / "renders",
        scene_workspace=tmp_path / "scene",
        width=10,
        height=8,
        fov_x=pi / 2,
        fov_y=pi / 3,
        interpolation="bilinear",
        output_extension=".png",
        scene_context=scene_context,
    )

    record = metadata["panoramas"][0]

    assert record["scene_pano_filename"] == scene_name
    assert record["source_pano_filename"] is None
    assert record["source_pano_path"] is None
    assert record["source_status"] == "missing"
    assert record["source_resolution_strategy"] == "unresolved"


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
    def __init__(self, error: float, image_ids: list[int]):
        self.error = error
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
    def __init__(self, points: dict[int, FakePoint3D], num_reg_images: int):
        self._points = points
        self._num_reg_images = num_reg_images

    def point3D(self, point_id: int) -> FakePoint3D:
        return self._points[point_id]

    def num_reg_images(self) -> int:
        return self._num_reg_images


class FakePose:
    translation = np.asarray([0.0, 0.0, 0.0])
    rotation = None

    def __init__(self):
        self.rotation = FakeRotation()


class FakeRotation:
    def matrix(self):
        return np.eye(3)
