import json
import os
import sqlite3
from math import pi
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from colmap.camera import camera_from_fov
from colmap.panorama_reconstruct import (
    FeaturePipelineOptions,
    PanoramaRenderOptions,
    PipelineExecutionOptions,
    VirtualImageOutputOptions,
    assign_camera_ids_by_prefix,
    build_rig_remap_maps,
    build_initial_session_report,
    build_rig_views,
    configure_ssl_cert_file,
    copy_initial_panorama_session,
    copytree_hardlink_or_copy,
    create_feature_extraction_options,
    create_feature_matching_options,
    create_mapper_options,
    create_rig_config,
    default_render_options,
    feature_health_summary,
    format_elapsed_seconds,
    generated_virtual_image_names,
    mask_statistics,
    match_features,
    read_camera_ids_by_prefix,
    read_image_records_by_name,
    resolve_initial_panorama_session,
    run_incremental_mapping,
    parse_args,
    pad_equirectangular_image,
    read_rendered_rig_masks,
    render_remapped_view,
    render_panorama_rig_images,
    replace_generated_frames_with_seed_rig_frames,
    resolve_pipeline_execution,
    rig_view_masks,
    validate_initial_session_compatible,
    validate_rig_masks,
)
from ep_pinhole import EquirectangularToPinhole


def create_previous_panorama_session(
    workspace,
    *,
    options=None,
    output_options=None,
) -> None:
    options = default_render_options("non-overlapping") if options is None else options
    output_options = (
        VirtualImageOutputOptions(save_workers=1)
        if output_options is None
        else output_options
    )
    workspace.mkdir()
    (workspace / "images" / "pano_camera00").mkdir(parents=True)
    (workspace / "masks" / "pano_camera00").mkdir(parents=True)
    (workspace / "sparse_best").mkdir()
    (workspace / "database.db").write_bytes(b"database")
    for stem in ("cameras", "images", "points3D"):
        (workspace / "sparse_best" / f"{stem}.bin").write_bytes(b"model")
    metadata = {
        "render_options": {
            "num_yaw_steps": options.num_yaw_steps,
            "pitches": list(options.pitches),
            "width": options.width,
            "height": options.height,
            "fov_x": options.fov_x,
            "fov_y": options.fov_y,
            "interpolation": options.interpolation,
            "virtual_image_format": output_options.image_format,
        },
        "views": [
            {"index": view.index, "image_prefix": view.image_prefix}
            for view in build_rig_views(options)
        ],
    }
    (workspace / "panorama_rig_config.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_configure_ssl_cert_file_uses_explicit_bundle(tmp_path, monkeypatch) -> None:
    cert_file = tmp_path / "ca-certificates.crt"
    cert_file.write_text("test certificate bundle", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    configure_ssl_cert_file(cert_file)

    assert os.environ["SSL_CERT_FILE"] == str(cert_file)


def test_resolve_initial_panorama_session_accepts_workspace_or_model_path(tmp_path) -> None:
    workspace = tmp_path / "previous"
    create_previous_panorama_session(workspace)

    from_workspace = resolve_initial_panorama_session(workspace)
    from_model = resolve_initial_panorama_session(workspace / "sparse_best")

    assert from_workspace.workspace == workspace
    assert from_workspace.model_path == workspace / "sparse_best"
    assert from_model.workspace == workspace
    assert from_model.model_path == workspace / "sparse_best"


def test_resolve_initial_panorama_session_rejects_non_panorama_workspace(tmp_path) -> None:
    model = tmp_path / "sparse_best"
    model.mkdir()
    for stem in ("cameras", "images", "points3D"):
        (model / f"{stem}.bin").write_bytes(b"model")

    with pytest.raises(ValueError, match="previous colmap-panorama-reconstruct"):
        resolve_initial_panorama_session(model)


def test_initial_session_copy_preserves_rig_and_frame_model_files(tmp_path) -> None:
    previous = tmp_path / "previous"
    create_previous_panorama_session(previous)
    previous_image = previous / "images" / "pano_camera00" / "old.png"
    previous_mask = previous / "masks" / "pano_camera00" / "old.png.png"
    previous_image.write_bytes(b"image")
    previous_mask.write_bytes(b"mask")
    (previous / "sparse_best" / "rigs.bin").write_bytes(b"rigs")
    (previous / "sparse_best" / "frames.bin").write_bytes(b"frames")
    output = tmp_path / "output"
    paths = {
        "image_dir": output / "images",
        "mask_dir": output / "masks",
        "database": output / "database.db",
        "initial_sparse": output / "initial_sparse",
    }
    paths["image_dir"].mkdir(parents=True)
    paths["mask_dir"].mkdir(parents=True)
    session = resolve_initial_panorama_session(previous)

    copy_initial_panorama_session(session, paths)

    assert (paths["initial_sparse"] / "cameras.bin").is_file()
    assert (paths["initial_sparse"] / "images.bin").is_file()
    assert (paths["initial_sparse"] / "points3D.bin").is_file()
    assert (paths["initial_sparse"] / "rigs.bin").read_bytes() == b"rigs"
    assert (paths["initial_sparse"] / "frames.bin").read_bytes() == b"frames"
    assert os.path.samefile(
        previous_image,
        paths["image_dir"] / "pano_camera00" / "old.png",
    )
    assert os.path.samefile(
        previous_mask,
        paths["mask_dir"] / "pano_camera00" / "old.png.png",
    )


def test_initial_session_tree_copy_falls_back_when_hardlink_fails(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    source_file = source / "image.png"
    source_file.write_bytes(b"image")

    def raise_cross_device_link(_source, _destination):
        raise OSError("cross-device link")

    monkeypatch.setattr("colmap.panorama_reconstruct.os.link", raise_cross_device_link)

    copytree_hardlink_or_copy(source, destination)

    destination_file = destination / "image.png"
    assert destination_file.read_bytes() == b"image"
    assert not os.path.samefile(source_file, destination_file)


def test_initial_panorama_session_must_match_current_render_options(tmp_path) -> None:
    workspace = tmp_path / "previous"
    previous_options = default_render_options("non-overlapping")
    create_previous_panorama_session(workspace, options=previous_options)
    session = resolve_initial_panorama_session(workspace)
    current_options = PanoramaRenderOptions(
        num_yaw_steps=previous_options.num_yaw_steps,
        pitches=previous_options.pitches,
        width=previous_options.width + 1,
        height=previous_options.height,
        fov_x=previous_options.fov_x,
        fov_y=previous_options.fov_y,
    )

    with pytest.raises(ValueError, match="render settings do not match"):
        validate_initial_session_compatible(
            session,
            current_options,
            VirtualImageOutputOptions(save_workers=1),
        )


def test_initial_session_report_marks_existing_poses_fixed(tmp_path) -> None:
    workspace = tmp_path / "previous"
    create_previous_panorama_session(workspace)
    session = resolve_initial_panorama_session(workspace)
    paths = {"initial_sparse": tmp_path / "continued" / "initial_sparse"}
    metadata = {
        "views": [
            {"index": view.index, "image_prefix": view.image_prefix}
            for view in build_rig_views(default_render_options("non-overlapping"))
        ],
    }

    report = build_initial_session_report(session, paths, metadata)

    assert report["copied_model_preserves_rig_frames"] is True
    assert report["existing_camera_poses_refinable"] is False
    assert report["intrinsics_refinable"] is False


def test_overlapping_preset_builds_fixed_virtual_camera_rig() -> None:
    options = default_render_options("overlapping")

    views = build_rig_views(options)

    assert len(views) == 12
    assert views[0].image_prefix == "pano_camera00/"
    assert views[-1].image_prefix == "pano_camera11/"
    assert all(view.cam_from_pano_rotation.shape == (3, 3) for view in views)


def test_non_overlapping_preset_builds_four_equatorial_views() -> None:
    options = default_render_options("non-overlapping")

    views = build_rig_views(options)

    assert len(views) == 4
    assert [view.pitch for view in views] == [0.0, 0.0, 0.0, 0.0]
    assert [view.yaw for view in views] == pytest.approx([0.0, pi / 2, -pi, -pi / 2])


def test_rig_view_masks_are_binary_and_match_camera_shape() -> None:
    options = default_render_options("non-overlapping")
    views = build_rig_views(options)
    camera = camera_from_fov(width=32, height=24, fov_x=pi / 2, fov_y=pi / 2)

    masks = rig_view_masks(camera, views)

    assert set(masks) == {0, 1, 2, 3}
    for mask in masks.values():
        assert mask.shape == (24, 32)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})


def test_overlapping_rig_masks_assign_every_virtual_camera() -> None:
    options = default_render_options("overlapping")
    views = build_rig_views(options)
    camera = camera_from_fov(
        width=64,
        height=64,
        fov_x=options.fov_x,
        fov_y=options.fov_y,
    )

    masks = rig_view_masks(camera, views)
    validate_rig_masks(camera, views, masks)

    assert len(masks) == 12
    for mask in masks.values():
        assert np.count_nonzero(mask) > 0
        assert mask[camera.height // 2, camera.width // 2] == 255


def test_validate_rig_masks_rejects_empty_virtual_camera() -> None:
    options = default_render_options("non-overlapping")
    views = build_rig_views(options)
    camera = camera_from_fov(width=32, height=24, fov_x=pi / 2, fov_y=pi / 2)
    masks = rig_view_masks(camera, views)
    masks[2] = np.zeros_like(masks[2])

    with pytest.raises(ValueError, match="Mask 2 is empty"):
        validate_rig_masks(camera, views, masks)


def test_mask_statistics_reports_active_fraction() -> None:
    masks = {
        0: np.asarray([[0, 255], [0, 255]], dtype=np.uint8),
        1: np.full((2, 2), 255, dtype=np.uint8),
    }

    stats = mask_statistics(masks)

    assert stats["0"] == {"active_pixels": 2, "active_fraction": 0.5}
    assert stats["1"] == {"active_pixels": 4, "active_fraction": 1.0}


def test_feature_health_summary_identifies_zero_feature_camera() -> None:
    views = build_rig_views(default_render_options("non-overlapping"))
    report = {
        view.image_prefix.rstrip("/"): {"total_features": 10}
        for view in views
    }
    report["pano_camera02"]["total_features"] = 0

    health = feature_health_summary(views, report)

    assert health["virtual_camera_count"] == 4
    assert health["virtual_cameras_with_features"] == 3
    assert health["zero_feature_virtual_cameras"] == ["pano_camera02"]


def test_seeded_camera_ids_are_reused_by_virtual_camera_prefix(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [
                (1, "pano_camera00/old.png", 1),
                (2, "pano_camera01/old.png", 2),
            ],
        )
    views = build_rig_views(
        PanoramaRenderOptions(
            num_yaw_steps=2,
            pitches=(0.0,),
            width=8,
            height=8,
            fov_x=pi / 2,
            fov_y=pi / 2,
        )
    )

    camera_ids = read_camera_ids_by_prefix(database_path, views)
    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [
                (3, "pano_camera00/img-0002_new.png", 13),
                (4, "pano_camera01/img-0002_new.png", 14),
            ],
        )
    assign_camera_ids_by_prefix(database_path, camera_ids)

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
    assert rows == [
        ("pano_camera00/old.png", 1),
        ("pano_camera01/old.png", 2),
        ("pano_camera00/img-0002_new.png", 1),
        ("pano_camera01/img-0002_new.png", 2),
    ]


def test_seeded_generated_frames_are_moved_into_existing_rig(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY)")
        connection.execute(
            """
            CREATE TABLE images(
                image_id INTEGER PRIMARY KEY,
                name TEXT,
                camera_id INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE rigs(
                rig_id INTEGER PRIMARY KEY,
                ref_sensor_id INTEGER,
                ref_sensor_type INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE rig_sensors(
                rig_id INTEGER,
                sensor_id INTEGER,
                sensor_type INTEGER,
                sensor_from_rig BLOB
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE frames(
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                rig_id INTEGER
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE frame_data(
                frame_id INTEGER,
                data_id INTEGER,
                sensor_id INTEGER,
                sensor_type INTEGER
            )
            """
        )
        connection.executemany(
            "INSERT INTO cameras(camera_id) VALUES (?)",
            [(1,), (2,), (13,), (14,)],
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [
                (1, "pano_camera00/old.png", 1),
                (2, "pano_camera01/old.png", 2),
                (3, "pano_camera00/img-0001_shelf.png", 13),
                (4, "pano_camera01/img-0001_shelf.png", 14),
            ],
        )
        connection.executemany(
            "INSERT INTO rigs(rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, 0)",
            [(13, 1), (14, 13), (15, 14)],
        )
        connection.execute(
            "INSERT INTO rig_sensors(rig_id, sensor_id, sensor_type) VALUES (13, 2, 0)"
        )
        connection.executemany(
            "INSERT INTO frames(frame_id, rig_id) VALUES (?, ?)",
            [(577, 13), (625, 14), (626, 15)],
        )
        connection.executemany(
            """
            INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type)
            VALUES (?, ?, ?, 0)
            """,
            [(577, 1, 1), (577, 2, 2), (625, 3, 13), (626, 4, 14)],
        )
    input_dir = tmp_path / "img-0001"
    panorama_path = input_dir / "shelf.png"
    input_dir.mkdir()
    panorama_path.write_bytes(b"image")
    views = build_rig_views(
        PanoramaRenderOptions(
            num_yaw_steps=2,
            pitches=(0.0,),
            width=8,
            height=8,
            fov_x=pi / 2,
            fov_y=pi / 2,
        )
    )
    output_options = VirtualImageOutputOptions(save_workers=1)
    image_names = generated_virtual_image_names(
        input_dir,
        [panorama_path],
        views,
        output_options,
        Path("img-0001"),
    )
    generated_records = read_image_records_by_name(database_path, image_names)
    assign_camera_ids_by_prefix(
        database_path,
        {"pano_camera00/": 1, "pano_camera01/": 2},
    )

    replace_generated_frames_with_seed_rig_frames(
        database_path,
        input_image_path=input_dir,
        panorama_paths=[panorama_path],
        views=views,
        output_options=output_options,
        relative_name_prefix=Path("img-0001"),
        seed_camera_ids={"pano_camera00/": 1, "pano_camera01/": 2},
        generated_records=generated_records,
    )

    with sqlite3.connect(database_path) as connection:
        camera_rows = connection.execute(
            "SELECT camera_id FROM cameras ORDER BY camera_id"
        ).fetchall()
        rig_rows = connection.execute(
            "SELECT rig_id FROM rigs ORDER BY rig_id"
        ).fetchall()
        rows = connection.execute(
            """
            SELECT frames.rig_id, frame_data.data_id, frame_data.sensor_id
            FROM frames
            JOIN frame_data USING(frame_id)
            WHERE frame_data.data_id IN (3, 4)
            ORDER BY frame_data.data_id
            """
        ).fetchall()
    assert camera_rows == [(1,), (2,)]
    assert rig_rows == [(13,)]
    assert rows == [(13, 3, 1), (13, 4, 2)]


def test_create_rig_config_uses_pinhole_cameras() -> None:
    pycolmap = pytest.importorskip("pycolmap")
    options = default_render_options("non-overlapping")
    views = build_rig_views(options)
    camera = camera_from_fov(options.width, options.height, options.fov_x, options.fov_y)

    rig_config = create_rig_config(pycolmap, camera, views)

    assert len(rig_config.cameras) == 4
    assert rig_config.cameras[0].ref_sensor is True
    assert rig_config.cameras[0].image_prefix == "pano_camera00/"
    assert rig_config.cameras[0].camera.model_name == "PINHOLE"
    assert rig_config.cameras[1].cam_from_rig is not None


def test_sift_feature_options_preserve_rig_matching_constraints() -> None:
    pycolmap = pytest.importorskip("pycolmap")
    options = FeaturePipelineOptions(
        max_num_features=4096,
        guided_matching=True,
    )

    extraction = create_feature_extraction_options(pycolmap, options)
    matching = create_feature_matching_options(pycolmap, options)

    assert extraction.type == pycolmap.FeatureExtractorType.SIFT
    assert extraction.sift.max_num_features == 4096
    assert matching.type == pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    assert matching.guided_matching is True
    assert matching.rig_verification is True
    assert matching.skip_image_pairs_in_same_frame is True


def test_gpu_feature_options_propagate_multi_gpu_indices() -> None:
    pycolmap = pytest.importorskip("pycolmap")
    options = FeaturePipelineOptions()

    extraction = create_feature_extraction_options(
        pycolmap,
        options,
        use_gpu=True,
        gpu_index="0,1",
    )
    matching = create_feature_matching_options(
        pycolmap,
        options,
        use_gpu=True,
        gpu_index="0,1",
    )

    assert extraction.use_gpu is True
    assert extraction.gpu_index == "0,1"
    assert matching.use_gpu is True
    assert matching.gpu_index == "0,1"


def test_format_elapsed_seconds_uses_readable_units() -> None:
    assert format_elapsed_seconds(12.34) == "12.3s"
    assert format_elapsed_seconds(125.2) == "2m 05s"
    assert format_elapsed_seconds(3725.2) == "1h 02m 05s"


def test_resolve_pipeline_execution_uses_cpu_fallback() -> None:
    class FakePycolmap:
        has_cuda = False

        class Device:
            cpu = "cpu"
            cuda = "cuda"

    execution = resolve_pipeline_execution(
        FakePycolmap,
        PipelineExecutionOptions(device="auto", gpu_index="0,1"),
    )

    assert execution["resolved_device"] == "cpu"
    assert execution["device"] == "cpu"
    assert execution["cuda_available"] is False
    assert execution["gpu_bundle_adjustment"] is False
    assert execution["bundle_adjustment_gpu_index"] == "0"


def test_resolve_pipeline_execution_rejects_unavailable_cuda() -> None:
    class FakePycolmap:
        has_cuda = False

        class Device:
            cpu = "cpu"
            cuda = "cuda"

    with pytest.raises(RuntimeError, match="pycolmap build has no CUDA support"):
        resolve_pipeline_execution(
            FakePycolmap,
            PipelineExecutionOptions(device="cuda"),
        )


def test_resolve_pipeline_execution_uses_cuda_bundle_adjustment() -> None:
    class FakePycolmap:
        has_cuda = True

        class Device:
            cpu = "cpu"
            cuda = "cuda"

    execution = resolve_pipeline_execution(
        FakePycolmap,
        PipelineExecutionOptions(device="cuda", gpu_index="2,3"),
    )

    assert execution["resolved_device"] == "cuda"
    assert execution["device"] == "cuda"
    assert execution["gpu_index"] == "2,3"
    assert execution["gpu_bundle_adjustment"] is True
    assert execution["bundle_adjustment_gpu_index"] == "2"


def test_mapper_options_enable_gpu_bundle_adjustment() -> None:
    pycolmap = pytest.importorskip("pycolmap")
    execution = {
        "gpu_bundle_adjustment": True,
        "bundle_adjustment_gpu_index": "2",
    }

    options = create_mapper_options(pycolmap, execution)

    assert options.ba_use_gpu is True
    assert options.ba_gpu_index == "2"
    assert options.ba_refine_sensor_from_rig is False
    assert options.ba_refine_focal_length is False


def test_mapper_options_disable_multiple_models_when_continuing_seed() -> None:
    pycolmap = pytest.importorskip("pycolmap")
    execution = {
        "gpu_bundle_adjustment": False,
        "bundle_adjustment_gpu_index": "-1",
    }

    options = create_mapper_options(
        pycolmap,
        execution,
        continue_existing_model=True,
    )

    assert options.multiple_models is False
    assert options.max_num_models == 1
    assert options.fix_existing_frames is True
    assert options.mapper.fix_existing_frames is True


def test_precomputed_rig_remaps_match_existing_renderer() -> None:
    height, width = 24, 48
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    options = PanoramaRenderOptions(
        num_yaw_steps=2,
        pitches=(-pi / 6, pi / 6),
        width=12,
        height=10,
        fov_x=pi / 2,
        fov_y=pi / 2,
    )
    views = build_rig_views(options)
    maps = build_rig_remap_maps(image.shape[:2], options, views)
    padded = pad_equirectangular_image(image)
    existing_renderer = EquirectangularToPinhole(image)

    for view in views:
        expected = existing_renderer.render(
            height=options.height,
            width=options.width,
            longitude=view.yaw,
            latitude=view.pitch,
            fov_x=options.fov_x,
            fov_y=options.fov_y,
            interpolation=options.interpolation,
        )
        actual = render_remapped_view(padded, maps[view.index])

        np.testing.assert_array_equal(actual, expected)


def test_sequential_matching_expands_rig_and_enables_loop_detection(
    tmp_path,
) -> None:
    class SequentialPairingOptions:
        def __init__(self):
            self.overlap = 0
            self.expand_rig_images = False
            self.loop_detection = False
            self.vocab_tree_path = None

    class FakePycolmap:
        class Device:
            cpu = "cpu"

        calls = []

        @classmethod
        def match_sequential(
            cls,
            database_path,
            *,
            matching_options,
            pairing_options,
            device,
        ):
            cls.calls.append((database_path, matching_options, pairing_options, device))

    FakePycolmap.SequentialPairingOptions = SequentialPairingOptions
    vocab_tree = tmp_path / "tree.bin"
    vocab_tree.write_bytes(b"tree")
    matching_options = object()

    match_features(
        FakePycolmap,
        tmp_path / "database.db",
        matcher="sequential",
        matching_options=matching_options,
        overlap=25,
        loop_detection=True,
        vocab_tree_path=vocab_tree,
    )

    _database, _matching, pairing, device = FakePycolmap.calls[0]
    assert pairing.overlap == 25
    assert pairing.expand_rig_images is True
    assert pairing.loop_detection is True
    assert pairing.vocab_tree_path == vocab_tree
    assert device == "cpu"


def test_run_incremental_mapping_passes_initial_model_path(tmp_path) -> None:
    class FakePycolmap:
        calls = []

        @classmethod
        def incremental_mapping(cls, database_path, image_dir, sparse_dir, **kwargs):
            cls.calls.append((database_path, image_dir, sparse_dir, kwargs))
            return {"ok": True}

    paths = {
        "database": tmp_path / "database.db",
        "sparse": tmp_path / "sparse",
    }
    initial_model = tmp_path / "initial_sparse"
    mapper_options = object()

    result = run_incremental_mapping(
        FakePycolmap,
        paths,
        image_dir=tmp_path / "images",
        mapper_options=mapper_options,
        initial_model_path=initial_model,
    )

    assert result == {"ok": True}
    _database, _images, _sparse, kwargs = FakePycolmap.calls[0]
    assert kwargs["options"] is mapper_options
    assert kwargs["input_path"] == initial_model


def test_parse_args_accepts_image_limit(tmp_path) -> None:
    previous = tmp_path / "previous"
    previous.mkdir()

    args = parse_args(
        [
            "--input-image-path",
            str(tmp_path),
            "--image-limit",
            "3",
            "--initial-session",
            str(previous),
        ]
    )

    assert args.image_limit == 3
    assert args.initial_session == previous
    assert args.loop_detection is True
    assert args.feature_extractor == "sift"
    assert args.feature_matcher == "sift-bruteforce"
    assert args.virtual_image_format == "png"
    assert args.png_compression == 1
    assert args.jpeg_quality == 95
    assert args.image_save_workers == 12
    assert args.device == "auto"
    assert args.gpu_index == "-1"
    assert args.gpu_bundle_adjustment is True


def test_render_panorama_rig_images_respects_image_limit(tmp_path, monkeypatch) -> None:
    progress_state = {}

    class RecordingProgress:
        def __init__(self, *, total, desc, unit):
            progress_state.update(total=total, desc=desc, unit=unit, updates=0)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def update(self):
            progress_state["updates"] += 1

    monkeypatch.setattr("colmap.panorama_reconstruct.tqdm", RecordingProgress)
    input_dir = tmp_path / "panos"
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    input_dir.mkdir()
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    Image.fromarray(image).save(input_dir / "b.png")
    Image.fromarray(image).save(input_dir / "a.png")
    options = PanoramaRenderOptions(
        num_yaw_steps=1,
        pitches=(0.0,),
        width=8,
        height=8,
        fov_x=pi / 2,
        fov_y=pi / 2,
    )

    panorama_paths, _camera, views = render_panorama_rig_images(
        input_image_path=input_dir,
        output_image_dir=image_dir,
        mask_dir=mask_dir,
        options=options,
        image_limit=1,
    )

    assert [path.name for path in panorama_paths] == ["a.png"]
    assert len(views) == 1
    assert (image_dir / "pano_camera00" / "a.png").exists()
    assert not (image_dir / "pano_camera00" / "b.png").exists()
    assert (mask_dir / "pano_camera00" / "a.png.png").exists()

    masks = read_rendered_rig_masks(mask_dir, input_dir, panorama_paths[0], views)
    assert set(masks) == {0}
    assert masks[0].shape == (8, 8)
    assert progress_state == {
        "total": 1,
        "desc": "Rendering virtual pinhole views",
        "unit": "view",
        "updates": 1,
    }


def test_render_panorama_rig_images_rejects_existing_seeded_views(tmp_path) -> None:
    input_dir = tmp_path / "panos"
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    input_dir.mkdir()
    (image_dir / "pano_camera00").mkdir(parents=True)
    (mask_dir / "pano_camera00").mkdir(parents=True)
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    Image.fromarray(image).save(input_dir / "pano.png")
    (image_dir / "pano_camera00" / "pano.png").write_bytes(b"old")
    options = PanoramaRenderOptions(
        num_yaw_steps=1,
        pitches=(0.0,),
        width=8,
        height=8,
        fov_x=pi / 2,
        fov_y=pi / 2,
    )

    with pytest.raises(FileExistsError, match="overwrite previous session files"):
        render_panorama_rig_images(
            input_image_path=input_dir,
            output_image_dir=image_dir,
            mask_dir=mask_dir,
            options=options,
            fail_on_existing=True,
        )


def test_render_panorama_rig_images_can_namespace_seeded_views(tmp_path) -> None:
    input_dir = tmp_path / "img-0001"
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    input_dir.mkdir()
    (image_dir / "pano_camera00").mkdir(parents=True)
    (mask_dir / "pano_camera00").mkdir(parents=True)
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    Image.fromarray(image).save(input_dir / "shelf.png")
    (image_dir / "pano_camera00" / "shelf.png").write_bytes(b"old")
    options = PanoramaRenderOptions(
        num_yaw_steps=1,
        pitches=(0.0,),
        width=8,
        height=8,
        fov_x=pi / 2,
        fov_y=pi / 2,
    )

    panorama_paths, _camera, views = render_panorama_rig_images(
        input_image_path=input_dir,
        output_image_dir=image_dir,
        mask_dir=mask_dir,
        options=options,
        fail_on_existing=True,
        relative_name_prefix=Path(input_dir.name),
    )

    assert (image_dir / "pano_camera00" / "shelf.png").read_bytes() == b"old"
    assert (image_dir / "pano_camera00" / "img-0001_shelf.png").exists()
    masks = read_rendered_rig_masks(
        mask_dir,
        input_dir,
        panorama_paths[0],
        views,
        relative_name_prefix=Path(input_dir.name),
    )
    assert masks[0].shape == (8, 8)


def test_render_panorama_rig_images_supports_jpeg_output(tmp_path) -> None:
    input_dir = tmp_path / "panos"
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    input_dir.mkdir()
    image = np.zeros((16, 32, 3), dtype=np.uint8)
    Image.fromarray(image).save(input_dir / "pano.png")
    render_options = PanoramaRenderOptions(
        num_yaw_steps=1,
        pitches=(0.0,),
        width=8,
        height=8,
        fov_x=pi / 2,
        fov_y=pi / 2,
    )
    output_options = VirtualImageOutputOptions(
        image_format="jpeg",
        jpeg_quality=95,
        save_workers=1,
    )

    panorama_paths, _camera, views = render_panorama_rig_images(
        input_image_path=input_dir,
        output_image_dir=image_dir,
        mask_dir=mask_dir,
        options=render_options,
        output_options=output_options,
    )

    assert (image_dir / "pano_camera00" / "pano.jpg").exists()
    assert (mask_dir / "pano_camera00" / "pano.jpg.png").exists()
    masks = read_rendered_rig_masks(
        mask_dir,
        input_dir,
        panorama_paths[0],
        views,
        output_options,
    )
    assert masks[0].shape == (8, 8)
