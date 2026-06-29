import json
import os
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from colmap.camera import camera_from_equirectangular
from colmap.panorama_reconstruct import (
    COLMAP_IMAGE_PAIR_ID_PRIME,
    FeaturePipelineOptions,
    GeneratedImageBatch,
    NATIVE_CAMERA_SETUP,
    PipelineExecutionOptions,
    build_panorama_report,
    configure_ssl_cert_file,
    create_feature_extraction_options,
    create_feature_matching_options,
    create_mapper_options,
    equirectangular_feature_health_summary,
    format_elapsed_seconds,
    format_elapsed_seconds_by_stage,
    import_equirectangular_panorama_images,
    match_features,
    panorama_image_paths,
    parse_args,
    read_feature_stats_by_database_camera,
    read_generated_image_batch,
    read_image_records_by_name,
    run_incremental_mapping,
    resolve_pipeline_execution,
    score_keyframe_anchors,
    write_equirectangular_metadata,
    write_image_pair_list,
)


class FakeReconstructionStats:
    def num_images(self):
        return 2

    def num_reg_images(self):
        return 2

    def num_points3D(self):
        return 5

    def compute_num_observations(self):
        return 10

    def compute_mean_observations_per_reg_image(self):
        return 5.0

    def compute_mean_track_length(self):
        return 2.0

    def compute_mean_reprojection_error(self):
        return 0.5


def test_configure_ssl_cert_file_uses_explicit_bundle(tmp_path, monkeypatch) -> None:
    cert_file = tmp_path / "ca-certificates.crt"
    cert_file.write_text("test certificate bundle", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    configure_ssl_cert_file(cert_file)

    assert os.environ["SSL_CERT_FILE"] == str(cert_file)


def test_native_equirectangular_import_keeps_original_images(tmp_path) -> None:
    input_dir = tmp_path / "input"
    image_dir = tmp_path / "workspace" / "images"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / "a.png")
    Image.new("RGB", (16, 8), color=(0, 255, 0)).save(nested_dir / "b.jpg")

    panorama_paths, camera, image_names = import_equirectangular_panorama_images(
        input_dir,
        image_dir,
    )

    assert [path.name for path in panorama_paths] == ["a.png", "b.jpg"]
    assert camera.model == "EQUIRECTANGULAR"
    assert camera.params == (16.0, 8.0)
    assert image_names == ["a.png", "nested/b.jpg"]
    assert os.path.samefile(input_dir / "a.png", image_dir / "a.png")
    assert os.path.samefile(nested_dir / "b.jpg", image_dir / "nested" / "b.jpg")


def test_native_equirectangular_import_symlinks_when_hardlink_fails(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    image_dir = tmp_path / "workspace" / "images"
    input_dir.mkdir()
    source_image = input_dir / "a.png"
    Image.new("RGB", (16, 8), color=(255, 0, 0)).save(source_image)

    def raise_cross_device_link(_source, _destination):
        raise OSError("cross-device link")

    monkeypatch.setattr("colmap.panorama.image_import.os.link", raise_cross_device_link)

    import_equirectangular_panorama_images(input_dir, image_dir)

    imported_image = image_dir / "a.png"
    assert imported_image.is_symlink()
    assert os.path.samefile(source_image, imported_image)


def test_native_equirectangular_import_rejects_mixed_sizes(tmp_path) -> None:
    input_dir = tmp_path / "input"
    image_dir = tmp_path / "workspace" / "images"
    input_dir.mkdir()
    Image.new("RGB", (16, 8)).save(input_dir / "a.png")
    Image.new("RGB", (32, 16)).save(input_dir / "b.png")

    with pytest.raises(ValueError, match="expected 16x8"):
        import_equirectangular_panorama_images(input_dir, image_dir)


def test_panorama_image_paths_sort_by_filename_timestamp(tmp_path) -> None:
    input_dir = tmp_path / "panos"
    input_dir.mkdir()
    for name in (
        "plain-b.png",
        "shelf-2000000000000_00000001.png",
        "shelf-1000000000000_00000002.png",
        "plain-a.png",
    ):
        (input_dir / name).write_bytes(b"image")

    assert [path.name for path in panorama_image_paths(input_dir)] == [
        "shelf-1000000000000_00000002.png",
        "shelf-2000000000000_00000001.png",
        "plain-a.png",
        "plain-b.png",
    ]


def test_write_equirectangular_metadata_records_native_images(tmp_path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    panorama_paths = [input_dir / "a.png", input_dir / "b.png"]
    path = tmp_path / "panorama_equirectangular_config.json"

    write_equirectangular_metadata(
        path,
        input_dir,
        camera_from_equirectangular(16, 8),
        panorama_paths,
        ["a.png", "b.png"],
        image_limit=None,
        relative_name_prefix=None,
    )

    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["camera_setup"] == NATIVE_CAMERA_SETUP
    assert metadata["camera"]["model"] == "EQUIRECTANGULAR"
    assert metadata["import_options"]["copy_backend"] == "hardlink_or_symlink"
    assert metadata["panorama_images"][0]["colmap_image_name"] == "a.png"
    assert "views" not in metadata


def test_generated_image_batch_preserves_database_identity(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [(7, "a.png", 1), (9, "b.png", 1), (3, "old.png", 1)],
        )

    batch = read_generated_image_batch(database_path, ("a.png", "b.png"))

    assert batch.names == ("a.png", "b.png")
    assert batch.image_ids == [7, 9]
    assert batch.camera_ids == [1, 1]
    assert read_image_records_by_name(database_path, ("a.png",)) == {"a.png": (7, 1)}


def test_read_feature_stats_by_database_camera(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        connection.execute(
            "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER)"
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [(1, "a.png", 1), (2, "b.png", 1)],
        )
        connection.executemany(
            "INSERT INTO keypoints(image_id, rows) VALUES (?, ?)",
            [(1, 12), (2, 8)],
        )

    stats = read_feature_stats_by_database_camera(database_path)

    assert stats == {
        "camera_1": {
            "image_count": 2,
            "total_features": 20,
            "mean_features_per_image": 10.0,
        }
    }


def test_equirectangular_feature_health_summary_identifies_zero_feature_camera() -> None:
    health = equirectangular_feature_health_summary(
        {
            "camera_1": {"image_count": 2, "total_features": 20},
            "camera_2": {"image_count": 1, "total_features": 0},
        },
        expected_image_count=3,
    )

    assert health["camera_setup"] == NATIVE_CAMERA_SETUP
    assert health["panorama_images_with_features"] == 3
    assert health["zero_feature_cameras"] == ["camera_2"]


def test_panorama_report_marks_native_equirectangular_camera(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [(1, "a.png", 1), (2, "b.png", 1)],
        )
    paths = {
        "workspace": tmp_path,
        "image_dir": tmp_path / "images",
        "database": database_path,
        "sparse": tmp_path / "sparse",
        "best_sparse": tmp_path / "sparse_best",
        "raw_ply": tmp_path / "cloud_colmap_raw.ply",
        "metadata": tmp_path / "panorama_equirectangular_config.json",
    }
    batch = GeneratedImageBatch(
        names=("a.png", "b.png"),
        records_by_name={"a.png": (1, 1), "b.png": (2, 1)},
    )

    report = build_panorama_report(
        pycolmap_version="test",
        input_image_path=tmp_path / "input",
        paths=paths,
        camera=camera_from_equirectangular(16, 8),
        panorama_count=2,
        matcher="sequential",
        overlap=15,
        loop_detection=True,
        vocab_tree_path=None,
        ssl_cert_file=None,
        feature_options=FeaturePipelineOptions(),
        execution={
            "requested_device": "auto",
            "resolved_device": "cpu",
            "cuda_available": False,
            "gpu_index": "-1",
            "gpu_bundle_adjustment": False,
            "bundle_adjustment_gpu_index": "-1",
        },
        feature_report={"camera_1": {"image_count": 2, "total_features": 20}},
        feature_health={
            "camera_setup": NATIVE_CAMERA_SETUP,
            "panorama_images_with_features": 2,
            "expected_panorama_images": 2,
        },
        image_limit=None,
        generated_image_batch=batch,
        reconstructions={0: FakeReconstructionStats()},
        best_model_id=0,
        best_reconstruction=FakeReconstructionStats(),
        keyframe_scores=[],
    )

    assert report["camera_setup"] == NATIVE_CAMERA_SETUP
    assert report["camera"]["model"] == "EQUIRECTANGULAR"
    assert report["camera"]["params"] == [16.0, 8.0]
    assert report["metadata_path"].endswith("panorama_equirectangular_config.json")
    assert report["options"]["camera_mode"] == "SINGLE"
    assert report["options"]["rig_verification"] is False
    assert report["generated_image_count"] == 2
    assert "virtual_camera_count" not in report
    assert "features_by_virtual_camera" not in report


def test_score_keyframe_anchors_reports_quality_components(tmp_path) -> None:
    class FakePoint2D:
        def __init__(self, point3d_id):
            self.point3D_id = point3d_id

        def has_point3D(self):
            return self.point3D_id >= 0

    class FakeImage:
        def __init__(self, image_id, frame_id, name, point3d_ids):
            self.image_id = image_id
            self.frame_id = frame_id
            self.name = name
            self.points2D = [FakePoint2D(point3d_id) for point3d_id in point3d_ids]

    class FakePoint3D:
        def __init__(self, error):
            self.error = error

        def has_error(self):
            return True

    class FakeReconstruction:
        def __init__(self):
            self._images = {
                1: FakeImage(1, 10, "a.png", [101, 102, -1]),
                2: FakeImage(2, 10, "b.png", [103]),
                3: FakeImage(3, 11, "c.png", [201]),
                4: FakeImage(4, 11, "d.png", [202]),
            }
            self.points3D = {
                101: FakePoint3D(0.5),
                102: FakePoint3D(1.0),
                103: FakePoint3D(1.5),
                201: FakePoint3D(3.0),
                202: FakePoint3D(3.5),
            }

        def reg_image_ids(self):
            return list(self._images)

        def image(self, image_id):
            return self._images[image_id]

    def pair_id(image_id1, image_id2):
        image_id1, image_id2 = sorted((image_id1, image_id2))
        return COLMAP_IMAGE_PAIR_ID_PRIME * image_id1 + image_id2

    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY, rows INTEGER)"
        )
        connection.executemany(
            "INSERT INTO two_view_geometries(pair_id, rows) VALUES (?, ?)",
            [
                (pair_id(1, 3), 80),
                (pair_id(2, 4), 40),
                (pair_id(1, 2), 999),
            ],
        )

    scores = score_keyframe_anchors(
        database_path,
        FakeReconstruction(),
        min_active_image_points=1,
        connectivity_frame_cap=2,
        verified_inlier_cap=100,
        triangulated_point_cap=10,
    )
    score_by_frame = {score.frame_id: score for score in scores}

    assert set(score_by_frame) == {10, 11}
    assert score_by_frame[10].connected_frame_count == 1
    assert score_by_frame[10].verified_inlier_count == 120
    assert score_by_frame[10].triangulated_point_count == 3
    assert score_by_frame[10].active_image_count == 2
    assert score_by_frame[10].image_distribution_score > 0.9
    assert score_by_frame[10].median_reprojection_error_px == 1.0
    assert score_by_frame[10].anchor_score > 0.0


def test_sift_feature_options_disable_rig_matching_constraints() -> None:
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
    assert matching.rig_verification is False
    assert matching.skip_image_pairs_in_same_frame is False


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


def test_format_elapsed_seconds_by_stage_preserves_stage_names() -> None:
    assert format_elapsed_seconds_by_stage(
        {
            "feature_extraction": 12.34,
            "incremental_mapping": 125.2,
        },
    ) == {
        "feature_extraction": "12.3s",
        "incremental_mapping": "2m 05s",
    }


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


def test_sequential_matching_uses_native_image_pairs(tmp_path) -> None:
    class SequentialPairingOptions:
        def __init__(self):
            self.overlap = 0
            self.expand_rig_images = True
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
    assert pairing.expand_rig_images is False
    assert pairing.loop_detection is True
    assert pairing.vocab_tree_path == vocab_tree
    assert device == "cpu"


def test_imported_pair_matching_uses_image_pair_file(tmp_path) -> None:
    class ImportedPairingOptions:
        def __init__(self):
            self.match_list_path = None

    class FakePycolmap:
        class Device:
            cpu = "cpu"

        calls = []

        @classmethod
        def match_image_pairs(
            cls,
            database_path,
            *,
            matching_options,
            pairing_options,
            device,
        ):
            cls.calls.append((database_path, matching_options, pairing_options, device))

    FakePycolmap.ImportedPairingOptions = ImportedPairingOptions
    pair_list = tmp_path / "pairs.txt"
    write_image_pair_list(pair_list, [("a.png", "b.png")])
    matching_options = object()

    match_features(
        FakePycolmap,
        tmp_path / "database.db",
        matcher="sequential",
        matching_options=matching_options,
        overlap=25,
        loop_detection=True,
        vocab_tree_path=None,
        image_pairs_path=pair_list,
    )

    assert pair_list.read_text(encoding="utf-8") == "a.png b.png\n"
    _database, matching, pairing, device = FakePycolmap.calls[0]
    assert matching is matching_options
    assert pairing.match_list_path == pair_list
    assert device == "cpu"


def test_run_incremental_mapping_passes_mapper_options(tmp_path) -> None:
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
    mapper_options = object()

    result = run_incremental_mapping(
        FakePycolmap,
        paths,
        image_dir=tmp_path / "images",
        mapper_options=mapper_options,
    )

    assert result == {"ok": True}
    _database, _images, _sparse, kwargs = FakePycolmap.calls[0]
    assert kwargs["options"] is mapper_options
    assert "input_path" not in kwargs


def test_parse_args_accepts_image_limit(tmp_path) -> None:
    args = parse_args(
        [
            "--input-image-path",
            str(tmp_path),
            "--image-limit",
            "3",
        ]
    )

    assert args.image_limit == 3
    assert args.loop_detection is True
    assert args.feature_extractor == "sift"
    assert args.feature_matcher == "sift-bruteforce"
    assert args.device == "auto"
    assert args.gpu_index == "-1"
    assert args.gpu_bundle_adjustment is True
    assert not hasattr(args, "initial_session")
    assert not hasattr(args, "match_scope")
    assert not hasattr(args, "virtual_image_format")


@pytest.mark.parametrize(
    "legacy_arg",
    [
        "--pano-render-type",
        "--num-yaw-steps",
        "--pitches-deg",
        "--width",
        "--height",
        "--fov-x",
        "--fov-y",
        "--match-scope",
        "--last-k-frames",
        "--top-k-frames",
        "--virtual-image-format",
        "--png-compression",
        "--jpeg-quality",
        "--image-save-workers",
        "--initial-session",
    ],
)
def test_parse_args_rejects_legacy_rig_arguments(tmp_path, legacy_arg) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--input-image-path", str(tmp_path), legacy_arg, "1"])
