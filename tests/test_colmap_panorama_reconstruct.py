from math import pi

import numpy as np
import pytest
from PIL import Image

from colmap.camera import camera_from_fov
from colmap.panorama_reconstruct import (
    PanoramaRenderOptions,
    build_rig_views,
    create_rig_config,
    default_render_options,
    parse_args,
    render_panorama_rig_images,
    rig_view_masks,
)


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


def test_parse_args_accepts_image_limit(tmp_path) -> None:
    args = parse_args(["--input-image-path", str(tmp_path), "--image-limit", "3"])

    assert args.image_limit == 3


def test_render_panorama_rig_images_respects_image_limit(tmp_path) -> None:
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
