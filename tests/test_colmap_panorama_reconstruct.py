import json
import os
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from colmap.camera import camera_from_equirectangular
from colmap.panorama import cluster_pipeline as panorama_cluster_pipeline
from colmap.panorama import cli_clusters as panorama_cli_clusters
from colmap.panorama import cli_features as panorama_cli_features
from colmap.panorama import feature_pipeline as panorama_feature_pipeline
from colmap.panorama import pipeline as panorama_pipeline
from colmap.panorama_reconstruct import (
    build_panorama_feature_report,
    build_panorama_import_report,
    COLMAP_IMAGE_PAIR_ID_PRIME,
    FeaturePipelineOptions,
    FeatureShardExtractionResult,
    GeneratedImageBatch,
    KeyframeCluster,
    KeyframePairPlan,
    KeyframeSelectionOptions,
    NATIVE_CAMERA_SETUP,
    PipelineExecutionOptions,
    build_backbone_pairs_for_policy,
    build_interval_keyframe_pair_plan,
    build_interval_keyframes,
    build_keyframe_pair_plan,
    build_panorama_report,
    build_sequential_backbone_pairs,
    build_sequential_vocabtree_clusters,
    build_sequential_vocabtree_pair_plan,
    build_vocabtree_candidate_pairs,
    compute_feature_batch_size,
    configure_ssl_cert_file,
    create_feature_extraction_options,
    create_feature_matching_options,
    create_mapper_options,
    deduplicate_image_pairs,
    equirectangular_feature_health_summary,
    extract_panorama_features_to_database,
    extract_panorama_features_to_sharded_database,
    format_elapsed_seconds,
    format_elapsed_seconds_by_stage,
    import_equirectangular_panorama_images,
    match_features,
    merge_feature_shard_databases,
    nearest_keyframe_name,
    image_id_pairs_to_name_pairs,
    normalize_image_pair,
    panorama_image_paths,
    parse_args,
    read_feature_stats_by_database_camera,
    read_feature_counts_by_image_name,
    read_generated_image_batch,
    read_image_records_by_name,
    run_incremental_mapping,
    run_panorama_feature_extraction,
    run_panorama_reconstruction,
    run_panorama_image_import,
    resolve_pipeline_execution,
    score_keyframe_anchors,
    select_cluster_keyframe,
    split_image_names_into_batches,
    write_equirectangular_metadata,
    write_image_pair_list,
)
from colmap.panorama.cli_import import main as import_main
from colmap.panorama.cli_import import parse_import_args
from colmap.panorama.cli_clusters import main as cluster_main
from colmap.panorama.cli_clusters import parse_cluster_args
from colmap.panorama.cli_clusters import read_metadata_image_names
from colmap.panorama.cli_features import main as feature_main
from colmap.panorama.cli_features import parse_feature_args


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


def write_cluster_cli_workspace(
    workspace: Path,
    image_names: tuple[str, ...],
) -> None:
    workspace.mkdir()
    metadata = {
        "schema_version": 2,
        "camera_setup": NATIVE_CAMERA_SETUP,
        "panorama_images": [
            {
                "source_path": str(workspace / "source" / image_name),
                "colmap_image_name": image_name,
            }
            for image_name in image_names
        ],
    }
    (workspace / "panorama_equirectangular_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    with sqlite3.connect(workspace / "database.db") as connection:
        connection.execute(
            "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        connection.execute(
            "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER)"
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [
                (image_index + 1, image_name, 1)
                for image_index, image_name in enumerate(image_names)
            ],
        )
        connection.executemany(
            "INSERT INTO keypoints(image_id, rows) VALUES (?, ?)",
            [
                (image_index + 1, (image_index + 1) * 10)
                for image_index, _image_name in enumerate(image_names)
            ],
        )


def test_normalize_image_pair_orders_pair_and_skips_self_pair() -> None:
    assert normalize_image_pair(("b.png", "a.png")) == ("a.png", "b.png")
    assert normalize_image_pair(("a.png", "b.png")) == ("a.png", "b.png")
    assert normalize_image_pair(("a.png", "a.png")) is None


def test_deduplicate_image_pairs_is_deterministic_and_unordered() -> None:
    pairs = [
        ("b.png", "a.png"),
        ("a.png", "b.png"),
        ("a.png", "a.png"),
        ("c.png", "b.png"),
    ]

    assert deduplicate_image_pairs(pairs) == (
        ("a.png", "b.png"),
        ("b.png", "c.png"),
    )


def test_keyframe_pair_plan_reports_source_counts_and_all_pairs() -> None:
    plan = KeyframePairPlan(
        selection="interval",
        effective_selection="interval",
        keyframe_pairs=(("a.png", "c.png"), ("c.png", "a.png")),
        vocabtree_pairs=(("b.png", "c.png"),),
        sequential_backbone_pairs=(("a.png", "b.png"), ("b.png", "c.png")),
        keyframes=("a.png", "c.png"),
    )

    assert plan.all_pairs == (
        ("a.png", "c.png"),
        ("b.png", "c.png"),
        ("a.png", "b.png"),
    )
    assert plan.pair_counts_by_source == {
        "keyframe": 1,
        "vocabtree": 1,
        "sequential_backbone": 2,
        "all": 3,
    }
    assert plan.report_keyframes() == [
        {"image_name": "a.png", "selection": "interval"},
        {"image_name": "c.png", "selection": "interval"},
    ]


def test_image_id_pairs_to_name_pairs_uses_generated_batch_identity() -> None:
    batch = GeneratedImageBatch(
        names=("a.png", "b.png", "c.png"),
        records_by_name={
            "a.png": (7, 1),
            "b.png": (3, 1),
            "c.png": (5, 1),
        },
    )

    assert image_id_pairs_to_name_pairs(
        [(3, 7), (7, 999), (5, 5), (3, 5)],
        batch,
    ) == (
        ("a.png", "b.png"),
        ("b.png", "c.png"),
    )


def test_cluster_pipeline_builds_interval_keyframe_pair_plan(tmp_path) -> None:
    batch = GeneratedImageBatch(
        names=("a.png", "b.png", "c.png"),
        records_by_name={
            "a.png": (1, 1),
            "b.png": (2, 1),
            "c.png": (3, 1),
        },
    )

    plan = build_keyframe_pair_plan(
        object(),
        database_path=tmp_path / "database.db",
        generated_image_batch=batch,
        keyframe_options=KeyframeSelectionOptions(
            selection="interval",
            interval=2,
            overlap=1,
            backbone_policy="gaps",
            backbone_overlap=1,
        ),
        vocab_tree_path=None,
    )

    assert plan.effective_selection == "interval"
    assert plan.keyframes == ("a.png", "c.png")
    assert plan.all_pairs == (
        ("a.png", "b.png"),
        ("a.png", "c.png"),
        ("b.png", "c.png"),
    )


def test_keyframe_cluster_report_records_keyframe_and_source() -> None:
    cluster = KeyframeCluster(
        cluster_id=2,
        image_names=("a.png", "b.png"),
        image_ids=(10, 11),
        keyframe_image_name="b.png",
        keyframe_image_id=11,
        source="sequential-vocabtree",
    )

    assert cluster.report() == {
        "cluster_id": 2,
        "image_count": 2,
        "image_names": ["a.png", "b.png"],
        "keyframe_image_name": "b.png",
        "keyframe_image_id": 11,
        "source": "sequential-vocabtree",
    }


def test_keyframe_selection_options_defaults_match_rollout_plan() -> None:
    options = KeyframeSelectionOptions()

    assert options.selection == "interval"
    assert options.interval == 10
    assert options.overlap == 3
    assert options.backbone_policy == "gaps"
    assert options.backbone_overlap == 1


def test_build_sequential_backbone_pairs_uses_ordered_overlap() -> None:
    assert build_sequential_backbone_pairs(
        ("a.png", "b.png", "c.png", "d.png"),
        2,
    ) == (
        ("a.png", "b.png"),
        ("a.png", "c.png"),
        ("b.png", "c.png"),
        ("b.png", "d.png"),
        ("c.png", "d.png"),
    )


def test_build_interval_keyframes_includes_first_last_and_interval() -> None:
    assert build_interval_keyframes(
        ("a.png", "b.png", "c.png", "d.png", "e.png"),
        2,
    ) == ("a.png", "c.png", "e.png")
    assert build_interval_keyframes(("a.png", "b.png", "c.png"), 10) == (
        "a.png",
        "c.png",
    )


def test_nearest_keyframe_name_prefers_earlier_keyframe_on_tie() -> None:
    image_order = {"a.png": 0, "b.png": 1, "c.png": 2}

    assert nearest_keyframe_name(
        "b.png",
        ("a.png", "c.png"),
        image_order,
    ) == "a.png"


def test_interval_keyframe_pair_plan_builds_clusters_and_pairs() -> None:
    plan = build_interval_keyframe_pair_plan(
        ("a.png", "b.png", "c.png", "d.png", "e.png"),
        KeyframeSelectionOptions(
            selection="interval",
            interval=2,
            overlap=1,
            backbone_policy="gaps",
            backbone_overlap=1,
        ),
    )

    assert plan.keyframes == ("a.png", "c.png", "e.png")
    assert [cluster.image_names for cluster in plan.clusters] == [
        ("a.png", "b.png"),
        ("c.png", "d.png"),
        ("e.png",),
    ]
    assert plan.keyframe_pairs == (
        ("a.png", "b.png"),
        ("c.png", "d.png"),
        ("a.png", "c.png"),
        ("c.png", "e.png"),
    )
    assert plan.sequential_backbone_pairs == (("b.png", "c.png"), ("d.png", "e.png"))


def test_build_backbone_pairs_for_gaps_only_adds_missing_local_edges() -> None:
    pairs = build_backbone_pairs_for_policy(
        ("a.png", "b.png", "c.png"),
        existing_pairs=(("a.png", "b.png"),),
        policy="gaps",
        overlap=1,
    )

    assert pairs == (("b.png", "c.png"),)


def test_select_cluster_keyframe_uses_degree_features_middle_and_order() -> None:
    keyframe = select_cluster_keyframe(
        ("a.png", "b.png", "c.png"),
        candidate_pairs=(
            ("a.png", "b.png"),
            ("b.png", "c.png"),
        ),
        feature_counts_by_name={"a.png": 30, "b.png": 10, "c.png": 50},
    )

    assert keyframe == "b.png"


def test_select_cluster_keyframe_uses_feature_count_as_degree_tie_break() -> None:
    keyframe = select_cluster_keyframe(
        ("a.png", "b.png"),
        candidate_pairs=(("a.png", "b.png"),),
        feature_counts_by_name={"a.png": 30, "b.png": 50},
    )

    assert keyframe == "b.png"


def test_sequential_vocabtree_clusters_split_when_recent_edges_disappear() -> None:
    clusters = build_sequential_vocabtree_clusters(
        ("a.png", "b.png", "c.png", "d.png", "e.png"),
        candidate_pairs=(
            ("a.png", "b.png"),
            ("b.png", "c.png"),
            ("d.png", "e.png"),
        ),
        options=KeyframeSelectionOptions(
            selection="sequential-vocabtree",
            cluster_recent_window=1,
        ),
        feature_counts_by_name={
            "a.png": 10,
            "b.png": 30,
            "c.png": 20,
            "d.png": 10,
            "e.png": 20,
        },
    )

    assert [cluster.image_names for cluster in clusters] == [
        ("a.png", "b.png", "c.png"),
        ("d.png", "e.png"),
    ]
    assert [cluster.keyframe_image_name for cluster in clusters] == [
        "b.png",
        "e.png",
    ]


def test_sequential_vocabtree_clusters_use_recent_window_for_visual_drift() -> None:
    clusters = build_sequential_vocabtree_clusters(
        ("a.png", "b.png", "c.png"),
        candidate_pairs=(
            ("a.png", "b.png"),
            ("b.png", "c.png"),
        ),
        options=KeyframeSelectionOptions(
            selection="sequential-vocabtree",
            cluster_recent_window=1,
        ),
    )

    assert [cluster.image_names for cluster in clusters] == [
        ("a.png", "b.png", "c.png"),
    ]


def test_sequential_vocabtree_pair_plan_tracks_sources_and_backbone_gaps() -> None:
    plan = build_sequential_vocabtree_pair_plan(
        ("a.png", "b.png", "c.png", "d.png"),
        candidate_pairs=(
            ("a.png", "b.png"),
            ("b.png", "c.png"),
        ),
        options=KeyframeSelectionOptions(
            selection="sequential-vocabtree",
            overlap=1,
            backbone_policy="gaps",
            backbone_overlap=1,
            cluster_recent_window=1,
        ),
        feature_counts_by_name={
            "a.png": 10,
            "b.png": 30,
            "c.png": 20,
            "d.png": 5,
        },
    )

    assert plan.selection == "sequential-vocabtree"
    assert plan.keyframes == ("b.png", "d.png")
    assert plan.keyframe_pairs == (
        ("a.png", "b.png"),
        ("b.png", "c.png"),
        ("b.png", "d.png"),
    )
    assert plan.vocabtree_pairs == (
        ("a.png", "b.png"),
        ("b.png", "c.png"),
    )
    assert plan.sequential_backbone_pairs == (("c.png", "d.png"),)
    assert plan.pair_counts_by_source == {
        "keyframe": 3,
        "vocabtree": 2,
        "sequential_backbone": 1,
        "all": 4,
    }


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


def test_native_equirectangular_import_workers_preserve_sorted_order(tmp_path) -> None:
    input_dir = tmp_path / "input"
    image_dir = tmp_path / "workspace" / "images"
    input_dir.mkdir()
    for name, color in (
        ("plain-b.png", (255, 0, 0)),
        ("shelf-2000000000000_00000001.png", (0, 255, 0)),
        ("shelf-1000000000000_00000002.png", (0, 0, 255)),
        ("plain-a.png", (255, 255, 0)),
    ):
        Image.new("RGB", (16, 8), color=color).save(input_dir / name)

    panorama_paths, camera, image_names = import_equirectangular_panorama_images(
        input_dir,
        image_dir,
        import_workers=2,
    )

    assert [path.name for path in panorama_paths] == [
        "shelf-1000000000000_00000002.png",
        "shelf-2000000000000_00000001.png",
        "plain-a.png",
        "plain-b.png",
    ]
    assert camera.model == "EQUIRECTANGULAR"
    assert image_names == [path.name for path in panorama_paths]
    for image_name in image_names:
        assert os.path.samefile(input_dir / image_name, image_dir / image_name)


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


def test_native_equirectangular_import_workers_reject_mixed_sizes(tmp_path) -> None:
    input_dir = tmp_path / "input"
    image_dir = tmp_path / "workspace" / "images"
    input_dir.mkdir()
    Image.new("RGB", (16, 8)).save(input_dir / "a.png")
    Image.new("RGB", (32, 16)).save(input_dir / "b.png")

    with pytest.raises(ValueError, match="expected 16x8"):
        import_equirectangular_panorama_images(
            input_dir,
            image_dir,
            import_workers=2,
        )


def test_native_equirectangular_import_rejects_invalid_worker_count(tmp_path) -> None:
    input_dir = tmp_path / "input"
    image_dir = tmp_path / "workspace" / "images"
    input_dir.mkdir()
    Image.new("RGB", (16, 8)).save(input_dir / "a.png")

    with pytest.raises(ValueError, match="import_workers must be positive"):
        import_equirectangular_panorama_images(
            input_dir,
            image_dir,
            import_workers=0,
        )


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


def test_panorama_import_report_marks_skipped_colmap_stages(tmp_path) -> None:
    paths = {
        "workspace": tmp_path,
        "image_dir": tmp_path / "images",
        "metadata": tmp_path / "panorama_equirectangular_config.json",
        "import_report": tmp_path / "panorama_equirectangular_import_report.json",
    }

    report = build_panorama_import_report(
        input_image_path=tmp_path / "input",
        paths=paths,
        camera=camera_from_equirectangular(16, 8),
        panorama_paths=[tmp_path / "input" / "a.png"],
        image_names=["a.png"],
        image_limit=None,
        import_workers=2,
    )

    assert report["status"] == f"{NATIVE_CAMERA_SETUP}_import_only"
    assert report["camera"]["model"] == "EQUIRECTANGULAR"
    assert report["options"]["import_workers"] == 2
    assert report["panorama_count"] == 1
    assert "feature_extraction" in report["skipped_stages"]
    assert report["generated_image_names"] == ["a.png"]


def test_run_panorama_image_import_writes_metadata_report_without_colmap_outputs(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / "a.png")
    Image.new("RGB", (16, 8), color=(0, 255, 0)).save(nested_dir / "b.jpg")

    def fail_if_pycolmap_is_required():
        raise AssertionError("import-only mode should not require pycolmap")

    monkeypatch.setattr(
        "colmap.panorama.pycolmap_adapter.require_pycolmap",
        fail_if_pycolmap_is_required,
    )
    monkeypatch.setattr(
        "colmap.panorama.pipeline.require_pycolmap",
        fail_if_pycolmap_is_required,
    )

    report = run_panorama_image_import(
        input_dir,
        workspace,
        import_workers=2,
    )

    metadata_path = workspace / "panorama_equirectangular_config.json"
    report_path = workspace / "panorama_equirectangular_import_report.json"
    stored_report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert report["status"] == f"{NATIVE_CAMERA_SETUP}_import_only"
    assert report["panorama_count"] == 2
    assert report["options"]["import_workers"] == 2
    assert stored_report["generated_image_names"] == ["a.png", "nested/b.jpg"]
    assert metadata["panorama_images"][1]["colmap_image_name"] == "nested/b.jpg"
    assert os.path.samefile(input_dir / "a.png", workspace / "images" / "a.png")
    assert os.path.samefile(
        nested_dir / "b.jpg",
        workspace / "images" / "nested" / "b.jpg",
    )
    assert not (workspace / "database.db").exists()
    assert not (workspace / "sparse").exists()
    assert "feature_extraction" in stored_report["skipped_stages"]


def test_run_panorama_image_import_refuses_existing_outputs_without_overwrite(
    tmp_path,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / "a.png")
    run_panorama_image_import(input_dir, workspace)

    with pytest.raises(FileExistsError, match="Pass --overwrite"):
        run_panorama_image_import(input_dir, workspace)


def test_run_panorama_reconstruction_uses_shared_import_pipeline(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    import_calls = []
    extract_calls = []

    class FakePycolmap:
        __version__ = "fake"

        class CameraMode:
            SINGLE = "single"

        class ImageReaderOptions:
            def __init__(self):
                self.camera_model = None
                self.camera_params = None

        @staticmethod
        def set_random_seed(_seed):
            return None

        @staticmethod
        def extract_features(*args, **kwargs):
            extract_calls.append((args, kwargs))

    def fake_import_panorama_images_to_workspace(**kwargs):
        import_calls.append(kwargs)
        return (
            [input_dir / "a.png"],
            camera_from_equirectangular(16, 8),
            ["a.png"],
        )

    monkeypatch.setattr(panorama_pipeline, "require_pycolmap", lambda: FakePycolmap)
    monkeypatch.setattr(panorama_pipeline, "configure_ssl_cert_file", lambda _path: None)
    monkeypatch.setattr(
        panorama_pipeline,
        "resolve_pipeline_execution",
        lambda _pycolmap, _options: {
            "requested_device": "auto",
            "resolved_device": "cpu",
            "device": "cpu",
            "cuda_available": False,
            "gpu_index": "-1",
            "gpu_bundle_adjustment": False,
            "bundle_adjustment_gpu_index": "-1",
        },
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "import_panorama_images_to_workspace",
        fake_import_panorama_images_to_workspace,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "create_feature_extraction_options",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "read_generated_image_batch",
        lambda _database, image_names: GeneratedImageBatch(
            names=tuple(image_names),
            records_by_name={"a.png": (1, 1)},
        ),
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "read_feature_stats_by_database_camera",
        lambda _database: {"camera_1": {"image_count": 1, "total_features": 10}},
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "equirectangular_feature_health_summary",
        lambda _report, expected_image_count: {
            "expected_panorama_images": expected_image_count,
            "panorama_images_with_features": expected_image_count,
        },
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "create_feature_matching_options",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(panorama_pipeline, "match_features", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        panorama_pipeline,
        "create_mapper_options",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "run_incremental_mapping",
        lambda *_args, **_kwargs: {0: FakeReconstructionStats()},
    )
    monkeypatch.setattr(panorama_pipeline, "write_best_model", lambda *_args: None)
    monkeypatch.setattr(panorama_pipeline, "score_keyframe_anchors", lambda *_args: [])
    monkeypatch.setattr(
        panorama_pipeline,
        "build_panorama_report",
        lambda **_kwargs: {
            "workspace": str(workspace),
            "camera_setup": NATIVE_CAMERA_SETUP,
            "camera": {"model": "EQUIRECTANGULAR"},
            "panorama_count": 1,
            "options": {"resolved_device": "cpu"},
            "feature_health": {
                "panorama_images_with_features": 1,
                "expected_panorama_images": 1,
            },
            "best_sparse_path": str(workspace / "sparse_best"),
            "best_model": {"num_registered_images": 1, "num_points3D": 0},
        },
    )

    run_panorama_reconstruction(
        input_dir,
        workspace,
        import_workers=3,
    )

    assert len(import_calls) == 1
    assert import_calls[0]["input_image_path"] == input_dir
    assert import_calls[0]["paths"]["workspace"] == workspace
    assert import_calls[0]["image_limit"] is None
    assert import_calls[0]["import_workers"] == 3
    assert len(extract_calls) == 1


def test_run_panorama_reconstruction_interval_keyframes_write_pair_list(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / name)
    match_calls = []

    class FakePycolmap:
        __version__ = "fake"

        @staticmethod
        def set_random_seed(_seed):
            return None

    def fake_extract_panorama_features_to_database(*_args, **kwargs):
        with sqlite3.connect(kwargs["database_path"]) as connection:
            connection.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
            )
            connection.execute(
                "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER)"
            )
            connection.executemany("INSERT INTO cameras(camera_id) VALUES (?)", [(1,)])
            connection.executemany(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                [(1, "a.png", 1), (2, "b.png", 1), (3, "c.png", 1)],
            )
            connection.executemany(
                "INSERT INTO keypoints(image_id, rows) VALUES (?, ?)",
                [(1, 10), (2, 20), (3, 30)],
            )

    def fake_match_features(*_args, **kwargs):
        match_calls.append(kwargs)

    monkeypatch.setattr(panorama_pipeline, "require_pycolmap", lambda: FakePycolmap)
    monkeypatch.setattr(panorama_pipeline, "configure_ssl_cert_file", lambda _path: None)
    monkeypatch.setattr(
        panorama_pipeline,
        "resolve_pipeline_execution",
        lambda _pycolmap, _options: {
            "requested_device": "auto",
            "resolved_device": "cpu",
            "device": "cpu",
            "cuda_available": False,
            "gpu_index": "-1",
            "gpu_bundle_adjustment": False,
            "bundle_adjustment_gpu_index": "-1",
        },
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "extract_panorama_features_to_database",
        fake_extract_panorama_features_to_database,
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "create_feature_matching_options",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(panorama_pipeline, "match_features", fake_match_features)
    monkeypatch.setattr(panorama_pipeline, "create_mapper_options", lambda *_args: object())
    monkeypatch.setattr(
        panorama_pipeline,
        "run_incremental_mapping",
        lambda *_args, **_kwargs: {0: FakeReconstructionStats()},
    )
    monkeypatch.setattr(panorama_pipeline, "write_best_model", lambda *_args: None)
    monkeypatch.setattr(panorama_pipeline, "score_keyframe_anchors", lambda *_args: [])

    report = run_panorama_reconstruction(
        input_dir,
        workspace,
        matcher="keyframes",
        keyframe_options=KeyframeSelectionOptions(
            selection="interval",
            interval=2,
            overlap=1,
            backbone_policy="gaps",
            backbone_overlap=1,
        ),
    )

    pair_list = workspace / "match_pairs.txt"
    assert match_calls[0]["image_pairs_path"] == pair_list
    assert pair_list.read_text(encoding="utf-8") == (
        "a.png b.png\n"
        "a.png c.png\n"
        "b.png c.png\n"
    )
    assert report["options"]["matcher"] == "keyframes"
    assert report["options"]["effective_pairing"] == "interval"
    assert report["options"]["matching_pair_count"] == 3
    assert report["options"]["sequential_backbone_pair_count"] == 1
    assert [keyframe["image_name"] for keyframe in report["keyframes"]] == [
        "a.png",
        "c.png",
    ]


def test_run_panorama_reconstruction_sequential_vocabtree_keyframes(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    for name in ("a.png", "b.png", "c.png", "d.png"):
        Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / name)
    vocab_tree = tmp_path / "tree.bin"
    vocab_tree.write_bytes(b"tree")
    match_calls = []

    class FakePycolmap:
        __version__ = "fake"

        @staticmethod
        def set_random_seed(_seed):
            return None

    def fake_extract_panorama_features_to_database(*_args, **kwargs):
        with sqlite3.connect(kwargs["database_path"]) as connection:
            connection.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
            )
            connection.execute(
                "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER)"
            )
            connection.executemany("INSERT INTO cameras(camera_id) VALUES (?)", [(1,)])
            connection.executemany(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                [
                    (1, "a.png", 1),
                    (2, "b.png", 1),
                    (3, "c.png", 1),
                    (4, "d.png", 1),
                ],
            )
            connection.executemany(
                "INSERT INTO keypoints(image_id, rows) VALUES (?, ?)",
                [(1, 10), (2, 30), (3, 20), (4, 5)],
            )

    monkeypatch.setattr(panorama_pipeline, "require_pycolmap", lambda: FakePycolmap)
    monkeypatch.setattr(panorama_pipeline, "configure_ssl_cert_file", lambda _path: None)
    monkeypatch.setattr(
        panorama_pipeline,
        "resolve_pipeline_execution",
        lambda _pycolmap, _options: {
            "requested_device": "auto",
            "resolved_device": "cpu",
            "device": "cpu",
            "cuda_available": False,
            "gpu_index": "-1",
            "gpu_bundle_adjustment": False,
            "bundle_adjustment_gpu_index": "-1",
        },
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "extract_panorama_features_to_database",
        fake_extract_panorama_features_to_database,
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "create_feature_matching_options",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        panorama_cluster_pipeline,
        "build_vocabtree_candidate_pairs",
        lambda *_args, **_kwargs: ((1, 2), (2, 3)),
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "match_features",
        lambda *_args, **kwargs: match_calls.append(kwargs),
    )
    monkeypatch.setattr(panorama_pipeline, "create_mapper_options", lambda *_args: object())
    monkeypatch.setattr(
        panorama_pipeline,
        "run_incremental_mapping",
        lambda *_args, **_kwargs: {0: FakeReconstructionStats()},
    )
    monkeypatch.setattr(panorama_pipeline, "write_best_model", lambda *_args: None)
    monkeypatch.setattr(panorama_pipeline, "score_keyframe_anchors", lambda *_args: [])

    report = run_panorama_reconstruction(
        input_dir,
        workspace,
        matcher="keyframes",
        vocab_tree_path=vocab_tree,
        keyframe_options=KeyframeSelectionOptions(
            selection="sequential-vocabtree",
            overlap=1,
            backbone_policy="gaps",
            backbone_overlap=1,
            cluster_recent_window=1,
        ),
    )

    pair_list = workspace / "match_pairs.txt"
    assert match_calls[0]["image_pairs_path"] == pair_list
    assert pair_list.read_text(encoding="utf-8") == (
        "a.png b.png\n"
        "b.png c.png\n"
        "b.png d.png\n"
        "c.png d.png\n"
    )
    assert report["options"]["effective_pairing"] == "sequential-vocabtree"
    assert report["options"]["vocabtree_pair_count"] == 2
    assert report["options"]["sequential_backbone_pair_count"] == 1
    assert [keyframe["image_name"] for keyframe in report["keyframes"]] == [
        "b.png",
        "d.png",
    ]


def test_run_panorama_reconstruction_uses_sharded_feature_extraction_when_requested(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    sharded_calls = []
    report_calls = []

    class FakePycolmap:
        __version__ = "fake"

        @staticmethod
        def set_random_seed(_seed):
            return None

    def fake_import_panorama_images_to_workspace(**_kwargs):
        return (
            [input_dir / "a.png", input_dir / "b.png", input_dir / "c.png"],
            camera_from_equirectangular(16, 8),
            ["a.png", "b.png", "c.png"],
        )

    def fake_extract_panorama_features_to_sharded_database(*_args, **kwargs):
        sharded_calls.append(kwargs)
        return FeatureShardExtractionResult(
            batches=(("a.png", "b.png"), ("c.png",)),
            shard_database_paths=(
                workspace / "feature_shards" / "features_0000.db",
                workspace / "feature_shards" / "features_0001.db",
            ),
            merged_database_path=workspace / "database.db",
            shard_elapsed_seconds=(1.0, 1.5),
            merge_elapsed_seconds=0.25,
        )

    def fake_build_panorama_report(**kwargs):
        report_calls.append(kwargs)
        shard_result = kwargs["feature_shard_result"]
        return {
            "workspace": str(workspace),
            "camera_setup": NATIVE_CAMERA_SETUP,
            "camera": {"model": "EQUIRECTANGULAR"},
            "panorama_count": 3,
            "options": {
                "feature_workers": kwargs["feature_workers"],
                "feature_batch_size": kwargs["feature_batch_size"],
                "feature_extraction_mode": "sharded_database",
            },
            "feature_shards": {
                "batch_count": len(shard_result.batches),
                "camera_policy": (
                    "multiple_identical_equirectangular_cameras_are_expected"
                ),
            },
            "feature_health": {
                "panorama_images_with_features": 3,
                "expected_panorama_images": 3,
            },
            "best_sparse_path": str(workspace / "sparse_best"),
            "best_model": {"num_registered_images": 3, "num_points3D": 0},
        }

    monkeypatch.setattr(panorama_pipeline, "require_pycolmap", lambda: FakePycolmap)
    monkeypatch.setattr(panorama_pipeline, "configure_ssl_cert_file", lambda _path: None)
    monkeypatch.setattr(
        panorama_pipeline,
        "resolve_pipeline_execution",
        lambda _pycolmap, _options: {
            "requested_device": "auto",
            "resolved_device": "cpu",
            "device": "cpu",
            "cuda_available": False,
            "gpu_index": "-1",
            "gpu_bundle_adjustment": False,
            "bundle_adjustment_gpu_index": "-1",
        },
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "import_panorama_images_to_workspace",
        fake_import_panorama_images_to_workspace,
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "extract_panorama_features_to_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "extract_panorama_features_to_sharded_database",
        fake_extract_panorama_features_to_sharded_database,
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "read_generated_image_batch",
        lambda _database, image_names: GeneratedImageBatch(
            names=tuple(image_names),
            records_by_name={
                "a.png": (1, 1),
                "b.png": (2, 1),
                "c.png": (3, 2),
            },
        ),
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "read_feature_stats_by_database_camera",
        lambda _database: {
            "camera_1": {"image_count": 2, "total_features": 20},
            "camera_2": {"image_count": 1, "total_features": 10},
        },
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "equirectangular_feature_health_summary",
        lambda _report, expected_image_count: {
            "expected_panorama_images": expected_image_count,
            "panorama_images_with_features": expected_image_count,
            "camera_count": 2,
        },
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "create_feature_matching_options",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(panorama_pipeline, "match_features", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        panorama_pipeline,
        "create_mapper_options",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        panorama_pipeline,
        "run_incremental_mapping",
        lambda *_args, **_kwargs: {0: FakeReconstructionStats()},
    )
    monkeypatch.setattr(panorama_pipeline, "write_best_model", lambda *_args: None)
    monkeypatch.setattr(panorama_pipeline, "score_keyframe_anchors", lambda *_args: [])
    monkeypatch.setattr(panorama_pipeline, "build_panorama_report", fake_build_panorama_report)

    report = run_panorama_reconstruction(
        input_dir,
        workspace,
        feature_workers=2,
    )

    assert len(sharded_calls) == 1
    assert sharded_calls[0]["database_path"] == workspace / "database.db"
    assert sharded_calls[0]["shard_dir"] == workspace / "feature_shards"
    assert sharded_calls[0]["image_names"] == ["a.png", "b.png", "c.png"]
    assert sharded_calls[0]["batch_size"] == 2
    assert sharded_calls[0]["feature_workers"] == 2
    assert report_calls[0]["feature_workers"] == 2
    assert report_calls[0]["feature_batch_size"] == 2
    assert report_calls[0]["feature_shard_result"].merge_elapsed_seconds == 0.25
    assert report["options"]["feature_extraction_mode"] == "sharded_database"
    assert report["feature_shards"]["camera_policy"].startswith("multiple_identical")


def test_extract_panorama_features_to_database_passes_optional_image_names(
    tmp_path,
    monkeypatch,
) -> None:
    calls = []

    class FakePycolmap:
        class CameraMode:
            SINGLE = "single"

        class ImageReaderOptions:
            def __init__(self):
                self.camera_model = None
                self.camera_params = None

        @staticmethod
        def extract_features(*args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(
        panorama_feature_pipeline,
        "create_feature_extraction_options",
        lambda *_args, **_kwargs: "options",
    )

    extract_panorama_features_to_database(
        FakePycolmap,
        database_path=tmp_path / "database.db",
        image_dir=tmp_path / "images",
        camera=camera_from_equirectangular(16, 8),
        feature_options=FeaturePipelineOptions(),
        execution={"resolved_device": "cpu", "gpu_index": "-1", "device": "cpu"},
        image_names=["a.png"],
    )

    args, kwargs = calls[0]
    assert args[0] == tmp_path / "database.db"
    assert args[1] == tmp_path / "images"
    assert kwargs["reader_options"].camera_model == "EQUIRECTANGULAR"
    assert kwargs["reader_options"].camera_params == "16,8"
    assert kwargs["camera_mode"] == "single"
    assert kwargs["extraction_options"] == "options"
    assert kwargs["device"] == "cpu"
    assert kwargs["image_names"] == ["a.png"]


def test_split_image_names_into_batches_preserves_order() -> None:
    assert split_image_names_into_batches(
        ["a.png", "b.png", "c.png", "d.png", "e.png"],
        batch_size=2,
    ) == (("a.png", "b.png"), ("c.png", "d.png"), ("e.png",))

    with pytest.raises(ValueError, match="batch_size"):
        split_image_names_into_batches(["a.png"], batch_size=0)


def test_compute_feature_batch_size_balances_images_across_workers() -> None:
    assert compute_feature_batch_size(image_count=3, feature_workers=2) == 2
    assert compute_feature_batch_size(image_count=12, feature_workers=4) == 3
    assert compute_feature_batch_size(image_count=3, feature_workers=8) == 1

    with pytest.raises(ValueError, match="image_count"):
        compute_feature_batch_size(image_count=0, feature_workers=2)
    with pytest.raises(ValueError, match="feature_workers"):
        compute_feature_batch_size(image_count=3, feature_workers=0)


def test_merge_feature_shard_databases_merges_iteratively(tmp_path) -> None:
    shard_paths = [
        tmp_path / "shard_0.db",
        tmp_path / "shard_1.db",
        tmp_path / "shard_2.db",
    ]
    for index, shard_path in enumerate(shard_paths):
        shard_path.write_text(f"shard {index}", encoding="utf-8")
    merged_database_path = tmp_path / "database.db"
    merge_calls = []

    class FakeDatabaseHandle:
        def __init__(self, path):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    class FakeDatabase:
        @staticmethod
        def open(path):
            return FakeDatabaseHandle(path)

        @staticmethod
        def merge(first_database, second_database, merged_database):
            merge_calls.append(
                (
                    first_database.path,
                    second_database.path,
                    merged_database.path,
                )
            )
            merged_database.path.write_text(
                f"{first_database.path.name}+{second_database.path.name}",
                encoding="utf-8",
            )

    class FakePycolmap:
        Database = FakeDatabase

    merge_feature_shard_databases(
        FakePycolmap,
        shard_database_paths=shard_paths,
        merged_database_path=merged_database_path,
    )

    assert merged_database_path.is_file()
    assert len(merge_calls) == 2
    assert merge_calls[0][0] == shard_paths[0]
    assert merge_calls[0][1] == shard_paths[1]
    assert merge_calls[1][0].name == "merged_0000.db"
    assert merge_calls[1][1] == shard_paths[2]


def test_extract_panorama_features_to_sharded_database_extracts_batches_and_merges(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    extract_calls = []
    merge_calls = []

    def fake_extract_panorama_features_to_database(*_args, **kwargs):
        extract_calls.append(kwargs)
        kwargs["database_path"].write_text("shard", encoding="utf-8")

    def fake_merge_feature_shard_databases(*_args, **kwargs):
        merge_calls.append(kwargs)
        kwargs["merged_database_path"].write_text("merged", encoding="utf-8")

    monkeypatch.setattr(
        panorama_feature_pipeline,
        "extract_panorama_features_to_database",
        fake_extract_panorama_features_to_database,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "merge_feature_shard_databases",
        fake_merge_feature_shard_databases,
    )

    result = extract_panorama_features_to_sharded_database(
        object(),
        database_path=tmp_path / "database.db",
        image_dir=tmp_path / "images",
        shard_dir=tmp_path / "feature_shards",
        camera=camera_from_equirectangular(16, 8),
        feature_options=FeaturePipelineOptions(),
        execution={"resolved_device": "cpu", "gpu_index": "-1", "device": "cpu"},
        image_names=["a.png", "b.png", "c.png"],
        batch_size=2,
    )

    assert isinstance(result, FeatureShardExtractionResult)
    assert result.batches == (("a.png", "b.png"), ("c.png",))
    assert result.shard_database_paths == (
        tmp_path / "feature_shards" / "features_0000.db",
        tmp_path / "feature_shards" / "features_0001.db",
    )
    assert result.merged_database_path == tmp_path / "database.db"
    assert [call["image_names"] for call in extract_calls] == [
        ("a.png", "b.png"),
        ("c.png",),
    ]
    assert merge_calls[0]["shard_database_paths"] == list(result.shard_database_paths)
    assert merge_calls[0]["merged_database_path"] == result.merged_database_path
    output = capsys.readouterr().out
    assert "Feature extraction batches finished; merging 2 shard database(s)" in output
    assert str(tmp_path / "database.db") in output


def test_sharded_feature_database_result_can_be_passed_to_matching(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_extract_panorama_features_to_database(*_args, **kwargs):
        kwargs["database_path"].write_text("shard", encoding="utf-8")

    def fake_merge_feature_shard_databases(*_args, **kwargs):
        kwargs["merged_database_path"].write_text("merged", encoding="utf-8")

    monkeypatch.setattr(
        panorama_feature_pipeline,
        "extract_panorama_features_to_database",
        fake_extract_panorama_features_to_database,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "merge_feature_shard_databases",
        fake_merge_feature_shard_databases,
    )
    match_calls = []

    class FakePycolmap:
        @staticmethod
        def match_exhaustive(database_path, **kwargs):
            match_calls.append((database_path, kwargs))

    result = extract_panorama_features_to_sharded_database(
        FakePycolmap,
        database_path=tmp_path / "database.db",
        image_dir=tmp_path / "images",
        shard_dir=tmp_path / "feature_shards",
        camera=camera_from_equirectangular(16, 8),
        feature_options=FeaturePipelineOptions(),
        execution={"resolved_device": "cpu", "gpu_index": "-1", "device": "cpu"},
        image_names=["a.png", "b.png"],
        batch_size=1,
    )
    match_features(
        FakePycolmap,
        result.merged_database_path,
        matcher="exhaustive",
        matching_options=object(),
        overlap=15,
        loop_detection=True,
        vocab_tree_path=None,
        device="cpu",
    )

    assert match_calls[0][0] == tmp_path / "database.db"
    assert (tmp_path / "database.db").read_text(encoding="utf-8") == "merged"


def test_run_panorama_feature_extraction_writes_feature_report_without_mapping(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / "a.png")

    class FakePycolmap:
        __version__ = "fake"
        has_cuda = False

        class Device:
            cpu = "cpu"
            cuda = "cuda"

        class CameraMode:
            SINGLE = "single"

        class ImageReaderOptions:
            def __init__(self):
                self.camera_model = None
                self.camera_params = None

        @staticmethod
        def set_random_seed(_seed):
            return None

        @staticmethod
        def extract_features(database_path, _image_dir, **_kwargs):
            with sqlite3.connect(database_path) as connection:
                connection.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY)")
                connection.execute(
                    "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
                )
                connection.execute(
                    "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER)"
                )
                connection.execute(
                    "CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, rows INTEGER)"
                )
                connection.execute("INSERT INTO cameras(camera_id) VALUES (1)")
                connection.execute(
                    "INSERT INTO images(image_id, name, camera_id) VALUES (1, 'a.png', 1)"
                )
                connection.execute(
                    "INSERT INTO keypoints(image_id, rows) VALUES (1, 12)"
                )
                connection.execute(
                    "INSERT INTO descriptors(image_id, rows) VALUES (1, 12)"
                )

    monkeypatch.setattr(
        panorama_feature_pipeline,
        "require_pycolmap",
        lambda: FakePycolmap,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "configure_ssl_cert_file",
        lambda _path: None,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "create_feature_extraction_options",
        lambda *_args, **_kwargs: object(),
    )

    report = run_panorama_feature_extraction(
        input_dir,
        workspace,
        import_workers=2,
    )

    stored_report = json.loads(
        (workspace / "panorama_equirectangular_feature_report.json").read_text(
            encoding="utf-8",
        )
    )
    assert report["status"] == f"{NATIVE_CAMERA_SETUP}_features_only"
    assert report["database"]["images"] == 1
    assert report["feature_health"]["panorama_images_with_features"] == 1
    assert stored_report["new_database_image_names"] == ["a.png"]
    assert "matching" in stored_report["skipped_stages"]
    assert (workspace / "database.db").is_file()
    assert (workspace / "panorama_equirectangular_config.json").is_file()
    assert not (workspace / "sparse").exists()
    assert not (workspace / "sparse_best").exists()


def test_run_panorama_feature_extraction_can_reuse_import_workspace(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / "a.png")
    run_panorama_image_import(input_dir, workspace)

    class FakePycolmap:
        __version__ = "fake"
        has_cuda = False

        class Device:
            cpu = "cpu"
            cuda = "cuda"

        class CameraMode:
            SINGLE = "single"

        class ImageReaderOptions:
            pass

        @staticmethod
        def set_random_seed(_seed):
            return None

        @staticmethod
        def extract_features(database_path, _image_dir, **_kwargs):
            with sqlite3.connect(database_path) as connection:
                connection.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY)")
                connection.execute(
                    "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
                )
                connection.execute(
                    "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER)"
                )
                connection.execute(
                    "CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, rows INTEGER)"
                )
                connection.execute("INSERT INTO cameras(camera_id) VALUES (1)")
                connection.execute(
                    "INSERT INTO images(image_id, name, camera_id) VALUES (1, 'a.png', 1)"
                )
                connection.execute(
                    "INSERT INTO keypoints(image_id, rows) VALUES (1, 12)"
                )
                connection.execute(
                    "INSERT INTO descriptors(image_id, rows) VALUES (1, 12)"
                )

    monkeypatch.setattr(
        panorama_feature_pipeline,
        "require_pycolmap",
        lambda: FakePycolmap,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "configure_ssl_cert_file",
        lambda _path: None,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "create_feature_extraction_options",
        lambda *_args, **_kwargs: object(),
    )

    report = run_panorama_feature_extraction(input_dir, workspace)

    assert report["database_image_count"] == 1


def test_run_panorama_feature_extraction_uses_feature_workers(
    tmp_path,
    monkeypatch,
) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / name)
    sharded_calls = []

    class FakePycolmap:
        __version__ = "fake"
        has_cuda = False

        class Device:
            cpu = "cpu"
            cuda = "cuda"

        @staticmethod
        def set_random_seed(_seed):
            return None

    def fake_extract_panorama_features_to_sharded_database(*_args, **kwargs):
        sharded_calls.append(kwargs)
        with sqlite3.connect(kwargs["database_path"]) as connection:
            connection.execute("CREATE TABLE cameras(camera_id INTEGER PRIMARY KEY)")
            connection.execute(
                "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
            )
            connection.execute(
                "CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY, rows INTEGER)"
            )
            connection.execute(
                "CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY, rows INTEGER)"
            )
            connection.executemany(
                "INSERT INTO cameras(camera_id) VALUES (?)",
                [(1,), (2,)],
            )
            connection.executemany(
                "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
                [(1, "a.png", 1), (2, "b.png", 1), (3, "c.png", 2)],
            )
            connection.executemany(
                "INSERT INTO keypoints(image_id, rows) VALUES (?, ?)",
                [(1, 12), (2, 14), (3, 16)],
            )
            connection.executemany(
                "INSERT INTO descriptors(image_id, rows) VALUES (?, ?)",
                [(1, 12), (2, 14), (3, 16)],
            )
        return FeatureShardExtractionResult(
            batches=(("a.png", "b.png"), ("c.png",)),
            shard_database_paths=(
                workspace / "feature_shards" / "features_0000.db",
                workspace / "feature_shards" / "features_0001.db",
            ),
            merged_database_path=workspace / "database.db",
            shard_elapsed_seconds=(1.0, 1.5),
            merge_elapsed_seconds=0.25,
        )

    monkeypatch.setattr(
        panorama_feature_pipeline,
        "require_pycolmap",
        lambda: FakePycolmap,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "configure_ssl_cert_file",
        lambda _path: None,
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "extract_panorama_features_to_database",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError),
    )
    monkeypatch.setattr(
        panorama_feature_pipeline,
        "extract_panorama_features_to_sharded_database",
        fake_extract_panorama_features_to_sharded_database,
    )

    report = run_panorama_feature_extraction(
        input_dir,
        workspace,
        feature_workers=2,
    )

    assert len(sharded_calls) == 1
    assert sharded_calls[0]["image_names"] == ["a.png", "b.png", "c.png"]
    assert sharded_calls[0]["batch_size"] == 2
    assert sharded_calls[0]["feature_workers"] == 2
    assert report["options"]["feature_workers"] == 2
    assert report["options"]["feature_batch_size"] == 2
    assert report["options"]["feature_extraction_mode"] == "sharded_database"
    assert report["feature_shards"]["batch_count"] == 2
    assert (
        report["feature_shards"]["camera_policy"]
        == "multiple_identical_equirectangular_cameras_are_expected"
    )


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


def test_read_feature_counts_by_image_name_preserves_requested_names(tmp_path) -> None:
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
        connection.execute("INSERT INTO keypoints(image_id, rows) VALUES (?, ?)", (1, 12))

    counts = read_feature_counts_by_image_name(
        database_path,
        ("a.png", "b.png", "missing.png"),
    )

    assert counts == {"a.png": 12, "b.png": 0, "missing.png": 0}


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
        import_workers=1,
        feature_workers=1,
        feature_batch_size=None,
        feature_shard_result=None,
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
    assert report["options"]["max_image_size"] == -1
    assert report["options"]["num_threads"] == -1
    assert report["options"]["import_workers"] == 1
    assert report["options"]["feature_workers"] == 1
    assert report["options"]["feature_batch_size"] is None
    assert report["options"]["feature_extraction_mode"] == "single_database"
    assert report["feature_shards"] is None
    assert report["options"]["rig_verification"] is False
    assert report["generated_image_count"] == 2
    assert "virtual_camera_count" not in report
    assert "features_by_virtual_camera" not in report


def test_panorama_report_records_sharded_feature_camera_policy(tmp_path) -> None:
    database_path = tmp_path / "database.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT, camera_id INTEGER)"
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [(1, "a.png", 1), (2, "b.png", 2)],
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
        records_by_name={"a.png": (1, 1), "b.png": (2, 2)},
    )
    shard_result = FeatureShardExtractionResult(
        batches=(("a.png",), ("b.png",)),
        shard_database_paths=(tmp_path / "shard_0.db", tmp_path / "shard_1.db"),
        merged_database_path=database_path,
        shard_elapsed_seconds=(1.0, 1.5),
        merge_elapsed_seconds=0.25,
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
        feature_report={
            "camera_1": {"image_count": 1, "total_features": 10},
            "camera_2": {"image_count": 1, "total_features": 12},
        },
        feature_health={
            "camera_setup": NATIVE_CAMERA_SETUP,
            "panorama_images_with_features": 2,
            "expected_panorama_images": 2,
            "camera_count": 2,
        },
        image_limit=None,
        import_workers=1,
        feature_workers=2,
        feature_batch_size=1,
        feature_shard_result=shard_result,
        generated_image_batch=batch,
        reconstructions={0: FakeReconstructionStats()},
        best_model_id=0,
        best_reconstruction=FakeReconstructionStats(),
        keyframe_scores=[],
    )

    assert report["options"]["feature_extraction_mode"] == "sharded_database"
    assert report["options"]["feature_workers"] == 2
    assert report["feature_shards"]["batch_count"] == 2
    assert report["feature_shards"]["batches"] == [["a.png"], ["b.png"]]
    assert report["feature_shards"]["merge_elapsed_seconds"] == 0.25
    assert (
        report["feature_shards"]["camera_policy"]
        == "multiple_identical_equirectangular_cameras_are_expected"
    )


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
        max_image_size=2048,
        num_threads=4,
        guided_matching=True,
    )

    extraction = create_feature_extraction_options(pycolmap, options)
    matching = create_feature_matching_options(pycolmap, options)

    assert extraction.type == pycolmap.FeatureExtractorType.SIFT
    assert extraction.sift.max_num_features == 4096
    assert extraction.max_image_size == 2048
    assert extraction.num_threads == 4
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


def test_build_vocabtree_candidate_pairs_uses_pair_generator(tmp_path) -> None:
    class VocabTreePairingOptions:
        def __init__(self):
            self.num_nearest_neighbors = 0
            self.num_checks = 0
            self.vocab_tree_path = None

    class FakeDatabase:
        opened_path = None
        closed = False

        @classmethod
        def open(cls, path):
            cls.opened_path = path
            return cls()

        def close(self):
            type(self).closed = True

    class VocabTreePairGenerator:
        calls = []

        def __init__(self, pairing_options, database, image_ids):
            self.calls.append((pairing_options, database, image_ids))

        def all_pairs(self):
            return [(1, 2), (2, 2), (3, 4)]

    class FakePycolmap:
        pass

    FakePycolmap.VocabTreePairingOptions = VocabTreePairingOptions
    FakePycolmap.VocabTreePairGenerator = VocabTreePairGenerator
    FakePycolmap.Database = FakeDatabase

    vocab_tree = tmp_path / "tree.bin"
    vocab_tree.write_bytes(b"tree")

    pairs = build_vocabtree_candidate_pairs(
        FakePycolmap,
        tmp_path / "database.db",
        [1, 2, 3, 4],
        KeyframeSelectionOptions(
            vocabtree_neighbors=7,
            vocabtree_checks=32,
        ),
        vocab_tree_path=vocab_tree,
    )

    assert pairs == ((1, 2), (3, 4))
    assert FakeDatabase.opened_path == tmp_path / "database.db"
    assert FakeDatabase.closed is True
    pairing_options, _database, image_ids = VocabTreePairGenerator.calls[0]
    assert pairing_options.num_nearest_neighbors == 7
    assert pairing_options.num_checks == 32
    assert pairing_options.vocab_tree_path == vocab_tree
    assert image_ids == [1, 2, 3, 4]


def test_build_vocabtree_candidate_pairs_explains_legacy_flann_tree(
    tmp_path,
) -> None:
    class VocabTreePairingOptions:
        def __init__(self):
            self.vocab_tree_path = None

    class FakeDatabase:
        closed = False

        @classmethod
        def open(cls, _path):
            return cls()

        def close(self):
            type(self).closed = True

    class VocabTreePairGenerator:
        def __init__(self, *_args):
            raise ValueError(
                "Failed to read faiss index. This may be caused by reading "
                "a legacy flann-based index."
            )

    class FakePycolmap:
        pass

    FakePycolmap.VocabTreePairingOptions = VocabTreePairingOptions
    FakePycolmap.VocabTreePairGenerator = VocabTreePairGenerator
    FakePycolmap.Database = FakeDatabase

    vocab_tree = tmp_path / "tree.bin"
    vocab_tree.write_bytes(b"tree")

    with pytest.raises(ValueError, match="legacy FLANN-format tree"):
        build_vocabtree_candidate_pairs(
            FakePycolmap,
            tmp_path / "database.db",
            [1, 2],
            KeyframeSelectionOptions(),
            vocab_tree_path=vocab_tree,
        )

    assert FakeDatabase.closed is True


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
    assert args.max_image_size == -1
    assert args.num_threads == -1
    assert args.import_workers == 1
    assert args.feature_workers == 1
    assert args.device == "auto"
    assert args.gpu_index == "-1"
    assert args.gpu_bundle_adjustment is True
    assert args.keyframe_selection == "sequential-vocabtree"
    assert args.keyframe_backbone_policy == "gaps"
    assert not hasattr(args, "initial_session")
    assert not hasattr(args, "match_scope")
    assert not hasattr(args, "virtual_image_format")


def test_parse_args_accepts_feature_extraction_tuning_options(tmp_path) -> None:
    args = parse_args(
        [
            "--input-image-path",
            str(tmp_path),
            "--max-image-size",
            "2048",
            "--num-threads",
            "4",
        ]
    )

    assert args.max_image_size == 2048
    assert args.num_threads == 4


def test_parse_args_accepts_keyframe_interval_mode_without_vocabtree(tmp_path) -> None:
    args = parse_args(
        [
            "--input-image-path",
            str(tmp_path),
            "--matcher",
            "keyframes",
            "--keyframe-selection",
            "interval",
            "--keyframe-interval",
            "5",
            "--keyframe-overlap",
            "2",
            "--keyframe-backbone-policy",
            "none",
        ]
    )

    assert args.matcher == "keyframes"
    assert args.keyframe_selection == "interval"
    assert args.keyframe_interval == 5
    assert args.keyframe_overlap == 2
    assert args.keyframe_backbone_policy == "none"


def test_parse_args_rejects_keyframe_vocabtree_without_vocab_tree_path(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--input-image-path",
                str(tmp_path),
                "--matcher",
                "keyframes",
                "--keyframe-selection",
                "sequential-vocabtree",
            ]
        )


def test_read_metadata_image_names_uses_imported_colmap_names(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    write_cluster_cli_workspace(workspace, ("a.png", "b.png"))

    assert read_metadata_image_names(
        workspace / "panorama_equirectangular_config.json",
    ) == ("a.png", "b.png")


def test_parse_cluster_args_accepts_interval_mode_without_vocabtree(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    write_cluster_cli_workspace(workspace, ("a.png", "b.png", "c.png"))

    args = parse_cluster_args(
        [
            "--workspace",
            str(workspace),
            "--keyframe-selection",
            "interval",
            "--keyframe-interval",
            "2",
            "--keyframe-overlap",
            "1",
        ]
    )

    assert args.workspace == workspace
    assert args.keyframe_selection == "interval"
    assert args.keyframe_interval == 2
    assert args.keyframe_overlap == 1


def test_parse_cluster_args_rejects_vocabtree_mode_without_tree(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    write_cluster_cli_workspace(workspace, ("a.png", "b.png"))

    with pytest.raises(SystemExit):
        parse_cluster_args(
            [
                "--workspace",
                str(workspace),
                "--keyframe-selection",
                "sequential-vocabtree",
            ]
        )


def test_parse_cluster_args_accepts_vocabtree_mode_with_tree(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    write_cluster_cli_workspace(workspace, ("a.png", "b.png"))
    vocab_tree = tmp_path / "tree.bin"
    vocab_tree.write_bytes(b"tree")

    args = parse_cluster_args(
        [
            "--workspace",
            str(workspace),
            "--vocab-tree-path",
            str(vocab_tree),
        ]
    )

    assert args.keyframe_selection == "sequential-vocabtree"
    assert args.vocab_tree_path == vocab_tree


def test_parse_args_accepts_import_workers(tmp_path) -> None:
    args = parse_args(
        [
            "--input-image-path",
            str(tmp_path),
            "--import-workers",
            "4",
        ]
    )

    assert args.import_workers == 4


def test_parse_args_accepts_feature_workers(tmp_path) -> None:
    args = parse_args(
        [
            "--input-image-path",
            str(tmp_path),
            "--feature-workers",
            "3",
        ]
    )

    assert args.feature_workers == 3


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--max-image-size", "-2"),
        ("--max-image-size", "0"),
        ("--num-threads", "-2"),
        ("--num-threads", "0"),
    ],
)
def test_parse_args_rejects_invalid_feature_extraction_tuning_options(
    tmp_path,
    option,
    value,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--input-image-path", str(tmp_path), option, value])


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parse_args_rejects_invalid_import_workers(tmp_path, value) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--input-image-path", str(tmp_path), "--import-workers", value])


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--feature-workers", "0"),
        ("--feature-workers", "-1"),
    ],
)
def test_parse_args_rejects_invalid_feature_workers(tmp_path, option, value) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--input-image-path", str(tmp_path), option, value])


def test_parse_import_args_accepts_import_options(tmp_path) -> None:
    args = parse_import_args(
        [
            "--input-image-path",
            str(tmp_path),
            "--image-limit",
            "3",
            "--import-workers",
            "2",
        ]
    )

    assert args.image_limit == 3
    assert args.import_workers == 2
    assert args.overwrite is False


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--image-limit", "0"),
        ("--import-workers", "0"),
        ("--import-workers", "-1"),
    ],
)
def test_parse_import_args_rejects_invalid_options(tmp_path, option, value) -> None:
    with pytest.raises(SystemExit):
        parse_import_args(["--input-image-path", str(tmp_path), option, value])


def test_parse_feature_args_accepts_feature_options(tmp_path) -> None:
    args = parse_feature_args(
        [
            "--input-image-path",
            str(tmp_path),
            "--max-num-features",
            "4096",
            "--max-image-size",
            "2048",
            "--num-threads",
            "4",
            "--device",
            "cpu",
            "--gpu-index",
            "0,1",
            "--image-limit",
            "3",
            "--import-workers",
            "2",
            "--feature-workers",
            "4",
        ]
    )

    assert args.max_num_features == 4096
    assert args.max_image_size == 2048
    assert args.num_threads == 4
    assert args.device == "cpu"
    assert args.gpu_index == "0,1"
    assert args.image_limit == 3
    assert args.import_workers == 2
    assert args.feature_workers == 4


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--image-limit", "0"),
        ("--import-workers", "0"),
        ("--feature-workers", "0"),
        ("--feature-workers", "-1"),
        ("--max-num-features", "0"),
        ("--max-image-size", "0"),
        ("--max-image-size", "-2"),
        ("--num-threads", "0"),
        ("--num-threads", "-2"),
    ],
)
def test_parse_feature_args_rejects_invalid_options(tmp_path, option, value) -> None:
    with pytest.raises(SystemExit):
        parse_feature_args(["--input-image-path", str(tmp_path), option, value])


def test_panorama_features_cli_prints_feature_report(tmp_path, monkeypatch, capsys) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()

    def fake_run_panorama_feature_extraction(**kwargs):
        assert kwargs["input_image_path"] == input_dir
        assert kwargs["workspace"] == workspace
        assert kwargs["feature_workers"] == 3
        return {
            "workspace": str(workspace),
            "camera_setup": NATIVE_CAMERA_SETUP,
            "camera": {"model": "EQUIRECTANGULAR"},
            "panorama_count": 1,
            "options": {"resolved_device": "cpu"},
            "feature_health": {
                "panorama_images_with_features": 1,
                "expected_panorama_images": 1,
            },
            "elapsed_time": "0.1s",
            "metadata_path": str(workspace / "panorama_equirectangular_config.json"),
            "database_path": str(workspace / "database.db"),
            "report_path": str(
                workspace / "panorama_equirectangular_feature_report.json"
            ),
        }

    monkeypatch.setattr(
        panorama_cli_features,
        "run_panorama_feature_extraction",
        fake_run_panorama_feature_extraction,
    )

    assert (
        feature_main(
            [
                "--input-image-path",
                str(input_dir),
                "--workspace",
                str(workspace),
                "--feature-workers",
                "3",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "panorama images: 1" in output
    assert "panorama images with features: 1/1" in output
    assert "database:" in output


def test_panorama_clusters_cli_writes_pair_list_and_report(tmp_path, capsys) -> None:
    workspace = tmp_path / "workspace"
    write_cluster_cli_workspace(workspace, ("a.png", "b.png", "c.png"))

    assert (
        cluster_main(
            [
                "--workspace",
                str(workspace),
                "--keyframe-selection",
                "interval",
                "--keyframe-interval",
                "2",
                "--keyframe-overlap",
                "1",
                "--keyframe-backbone-policy",
                "gaps",
                "--keyframe-backbone-overlap",
                "1",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "panorama images: 3" in output
    assert "keyframes: 2" in output
    assert "matching pairs: 3" in output
    assert "effective pairing: interval" in output
    assert (workspace / "match_pairs.txt").read_text(encoding="utf-8") == (
        "a.png b.png\n"
        "a.png c.png\n"
        "b.png c.png\n"
    )
    report = json.loads(
        (workspace / "panorama_keyframe_cluster_report.json").read_text(
            encoding="utf-8",
        )
    )
    assert report["options"]["effective_pairing"] == "interval"
    assert report["options"]["matching_pair_count"] == 3
    assert report["pair_counts_by_source"] == {
        "keyframe": 2,
        "vocabtree": 0,
        "sequential_backbone": 1,
        "all": 3,
    }


def test_panorama_clusters_cli_prints_vocabtree_plan(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    workspace = tmp_path / "workspace"
    write_cluster_cli_workspace(workspace, ("a.png", "b.png"))
    vocab_tree = tmp_path / "tree.bin"
    vocab_tree.write_bytes(b"tree")

    def fake_build_keyframe_pair_plan(*_args, **_kwargs):
        return KeyframePairPlan(
            selection="sequential-vocabtree",
            effective_selection="sequential-vocabtree",
            keyframe_pairs=(("a.png", "b.png"),),
            vocabtree_pairs=(("a.png", "b.png"),),
            keyframes=("a.png",),
            clusters=(
                KeyframeCluster(
                    cluster_id=0,
                    image_names=("a.png", "b.png"),
                    keyframe_image_name="a.png",
                    keyframe_image_id=1,
                    source="sequential-vocabtree",
                ),
            ),
        )

    monkeypatch.setattr(panorama_cli_clusters, "require_pycolmap", lambda: object())
    monkeypatch.setattr(
        panorama_cli_clusters,
        "build_keyframe_pair_plan",
        fake_build_keyframe_pair_plan,
    )

    assert (
        cluster_main(
            [
                "--workspace",
                str(workspace),
                "--vocab-tree-path",
                str(vocab_tree),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "effective pairing: sequential-vocabtree" in output
    assert (workspace / "match_pairs.txt").read_text(encoding="utf-8") == (
        "a.png b.png\n"
    )


def test_panorama_import_cli_writes_import_workspace(tmp_path, capsys) -> None:
    input_dir = tmp_path / "input"
    workspace = tmp_path / "workspace"
    input_dir.mkdir()
    Image.new("RGB", (16, 8), color=(255, 0, 0)).save(input_dir / "a.png")

    assert (
        import_main(
            [
                "--input-image-path",
                str(input_dir),
                "--workspace",
                str(workspace),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "panorama images: 1" in output
    assert "report:" in output
    assert (workspace / "panorama_equirectangular_config.json").is_file()
    assert (workspace / "panorama_equirectangular_import_report.json").is_file()
    assert not (workspace / "database.db").exists()


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
