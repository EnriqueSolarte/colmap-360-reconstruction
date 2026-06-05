import os
from math import pi

import numpy as np
import pytest
from PIL import Image

from colmap.camera import camera_from_fov
from colmap.panorama_reconstruct import (
    FeaturePipelineOptions,
    PanoramaRenderOptions,
    VirtualImageOutputOptions,
    build_rig_remap_maps,
    build_rig_views,
    configure_ssl_cert_file,
    create_feature_extraction_options,
    create_feature_matching_options,
    create_rig_config,
    default_render_options,
    feature_health_summary,
    mask_statistics,
    match_features,
    parse_args,
    pad_equirectangular_image,
    read_rendered_rig_masks,
    render_remapped_view,
    render_panorama_rig_images,
    rig_view_masks,
    validate_rig_masks,
)
from ep_pinhole import EquirectangularToPinhole


def test_configure_ssl_cert_file_uses_explicit_bundle(tmp_path, monkeypatch) -> None:
    cert_file = tmp_path / "ca-certificates.crt"
    cert_file.write_text("test certificate bundle", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    configure_ssl_cert_file(cert_file)

    assert os.environ["SSL_CERT_FILE"] == str(cert_file)


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


def test_parse_args_accepts_image_limit(tmp_path) -> None:
    args = parse_args(["--input-image-path", str(tmp_path), "--image-limit", "3"])

    assert args.image_limit == 3
    assert args.loop_detection is True
    assert args.feature_extractor == "sift"
    assert args.feature_matcher == "sift-bruteforce"
    assert args.virtual_image_format == "png"
    assert args.png_compression == 1
    assert args.jpeg_quality == 95
    assert args.image_save_workers == 12


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
