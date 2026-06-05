"""Run COLMAP rig SfM directly from equirectangular panoramas."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import cos, pi, radians, sin, tan
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
from py360convert.utils import uv2coor, xyz2uv, xyzpers
from tqdm import tqdm

from colmap.camera import ColmapCamera, camera_from_fov
from colmap.input_inspection import IMAGE_EXTENSIONS
from colmap.pipeline import write_best_model
from colmap.reporting import read_database_stats, reconstruction_stats

try:  # Keep argument parsing and tests usable before pycolmap is installed.
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-panorama-rig"


@dataclass(frozen=True)
class PanoramaRigView:
    index: int
    yaw: float
    pitch: float
    cam_from_pano_rotation: np.ndarray

    @property
    def image_prefix(self) -> str:
        return f"pano_camera{self.index:02d}/"


@dataclass(frozen=True)
class PanoramaRenderOptions:
    num_yaw_steps: int
    pitches: tuple[float, ...]
    width: int
    height: int
    fov_x: float
    fov_y: float
    interpolation: str = "bilinear"


@dataclass(frozen=True)
class FeaturePipelineOptions:
    extractor: str = "sift"
    matcher: str = "sift-bruteforce"
    max_num_features: int = 8192
    guided_matching: bool = False


@dataclass(frozen=True)
class VirtualImageOutputOptions:
    image_format: str = "png"
    png_compression: int = 1
    jpeg_quality: int = 95
    save_workers: int = 12


def default_render_options(render_type: str) -> PanoramaRenderOptions:
    if render_type == "overlapping":
        return PanoramaRenderOptions(
            num_yaw_steps=4,
            pitches=tuple(radians(value) for value in (-35.0, 0.0, 35.0)),
            width=1024,
            height=1024,
            fov_x=pi / 2,
            fov_y=pi / 2,
        )
    if render_type == "non-overlapping":
        return PanoramaRenderOptions(
            num_yaw_steps=4,
            pitches=(0.0,),
            width=1024,
            height=1024,
            fov_x=pi / 2,
            fov_y=pi / 2,
        )
    raise ValueError("render_type must be 'overlapping' or 'non-overlapping'")


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required for panorama rig reconstruction. Install the "
            "`colmap` extra or add `pycolmap` to the environment."
        )
    return pycolmap


def configure_ssl_cert_file(ssl_cert_file: Path | None = None) -> None:
    """Configure the CA bundle used by COLMAP/libcurl for optional downloads.

    Prefer an explicit bundle, preserve an existing environment configuration,
    and otherwise use the common system bundle when available. This lets
    vocabulary-tree downloads verify TLS certificates without overriding a
    caller's custom trust store.
    """
    if ssl_cert_file is not None:
        if not ssl_cert_file.is_file():
            raise FileNotFoundError(f"SSL certificate bundle not found: {ssl_cert_file}")
        os.environ["SSL_CERT_FILE"] = str(ssl_cert_file)
        return

    if os.environ.get("SSL_CERT_FILE"):
        return

    default_cert_file = Path("/etc/ssl/certs/ca-certificates.crt")
    if default_cert_file.is_file():
        os.environ["SSL_CERT_FILE"] = str(default_cert_file)


def panorama_image_paths(input_image_path: Path) -> list[Path]:
    return sorted(
        path
        for path in input_image_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_rig_views(options: PanoramaRenderOptions) -> list[PanoramaRigView]:
    if options.num_yaw_steps <= 0:
        raise ValueError("num_yaw_steps must be positive")
    if not options.pitches:
        raise ValueError("at least one pitch is required")

    views: list[PanoramaRigView] = []
    base_yaws = np.linspace(0.0, 2 * pi, options.num_yaw_steps, endpoint=False)
    for pitch in options.pitches:
        yaw_offset = (pi / options.num_yaw_steps) if pitch > 0 else 0.0
        for yaw in base_yaws + yaw_offset:
            views.append(
                PanoramaRigView(
                    index=len(views),
                    yaw=float(wrap_radians(yaw)),
                    pitch=float(pitch),
                    cam_from_pano_rotation=cam_from_pano_rotation(yaw, pitch),
                )
            )
    return views


def cam_from_pano_rotation(yaw: float, pitch: float) -> np.ndarray:
    """Rotation matching COLMAP's panorama rig example convention."""
    return rotation_x(-pitch) @ rotation_y(-yaw)


def rotation_x(angle: float) -> np.ndarray:
    c = cos(angle)
    s = sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def rotation_y(angle: float) -> np.ndarray:
    c = cos(angle)
    s = sin(angle)
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def wrap_radians(value: float) -> float:
    return float((value + pi) % (2 * pi) - pi)


def render_panorama_rig_images(
    input_image_path: Path,
    output_image_dir: Path,
    mask_dir: Path,
    options: PanoramaRenderOptions,
    *,
    output_options: VirtualImageOutputOptions = VirtualImageOutputOptions(),
    image_limit: int | None = None,
) -> tuple[list[Path], ColmapCamera, list[PanoramaRigView]]:
    validate_virtual_image_output_options(output_options)
    panorama_paths = panorama_image_paths(input_image_path)
    if image_limit is not None:
        panorama_paths = panorama_paths[:image_limit]
    if not panorama_paths:
        raise RuntimeError(f"No panorama images found in {input_image_path}")

    views = build_rig_views(options)
    camera = camera_from_fov(options.width, options.height, options.fov_x, options.fov_y)
    masks = rig_view_masks(camera, views)
    validate_rig_masks(camera, views, masks)

    first_panorama = read_panorama_image(panorama_paths[0])
    source_shape = first_panorama.shape[:2]
    remap_maps = build_rig_remap_maps(source_shape, options, views)
    total_views = len(panorama_paths) * len(views)

    with (
        ThreadPoolExecutor(max_workers=output_options.save_workers) as save_pool,
        tqdm(
            total=total_views,
            desc="Rendering virtual pinhole views",
            unit="view",
        ) as progress,
    ):
        for panorama_index, panorama_path in enumerate(panorama_paths):
            panorama = (
                first_panorama
                if panorama_index == 0
                else read_panorama_image(panorama_path)
            )
            if panorama.shape[:2] != source_shape:
                raise ValueError(
                    f"Panorama {panorama_path} has size {panorama.shape[1]}x"
                    f"{panorama.shape[0]}; expected {source_shape[1]}x{source_shape[0]}"
                )
            padded_panorama = pad_equirectangular_image(panorama)
            relative_name = virtual_image_relative_name(
                panorama_path.relative_to(input_image_path),
                output_options.image_format,
            )
            futures = []
            for view in views:
                image = render_remapped_view(padded_panorama, remap_maps[view.index])
                output_path = output_image_dir / view.image_prefix / relative_name
                output_path.parent.mkdir(parents=True, exist_ok=True)

                mask_path = mask_dir / f"{view.image_prefix}{relative_name}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                futures.append(
                    save_pool.submit(
                        save_virtual_view,
                        image,
                        output_path,
                        masks[view.index],
                        mask_path,
                        output_options,
                    )
                )

            for future in as_completed(futures):
                future.result()
                progress.update()

    return panorama_paths, camera, views


def validate_virtual_image_output_options(options: VirtualImageOutputOptions) -> None:
    if options.image_format not in {"png", "jpeg"}:
        raise ValueError("image_format must be 'png' or 'jpeg'")
    if not 0 <= options.png_compression <= 9:
        raise ValueError("png_compression must be between 0 and 9")
    if not 1 <= options.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if options.save_workers <= 0:
        raise ValueError("save_workers must be positive")


def read_panorama_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def build_rig_remap_maps(
    source_shape: tuple[int, int],
    options: PanoramaRenderOptions,
    views: Sequence[PanoramaRigView],
) -> dict[int, tuple[np.ndarray, np.ndarray | None, int]]:
    source_height, source_width = source_shape
    interpolation = cv2_interpolation(options.interpolation)
    nearest = interpolation == cv2.INTER_NEAREST
    maps = {}
    for view in views:
        xyz = xyzpers(
            options.fov_x,
            options.fov_y,
            -view.yaw,
            view.pitch,
            (options.height, options.width),
            0.0,
        )
        u, v = xyz2uv(xyz)
        x, y = uv2coor(u, v, source_height, source_width)
        map1, map2 = cv2.convertMaps(
            x + 1,
            y + 1,
            cv2.CV_16SC2,
            nninterpolation=nearest,
        )
        maps[view.index] = (map1, map2, interpolation)
    return maps


def cv2_interpolation(interpolation: str) -> int:
    if interpolation in {"linear", "bilinear"}:
        return cv2.INTER_LINEAR
    if interpolation == "nearest":
        return cv2.INTER_NEAREST
    raise ValueError("interpolation must be 'linear', 'bilinear', or 'nearest'")


def pad_equirectangular_image(image: np.ndarray) -> np.ndarray:
    # Match py360convert's seam and pole padding for pixel-identical remapping.
    padding = ((1, 1), (1, 1)) + ((0, 0),) * (image.ndim - 2)
    padded = np.pad(image, padding, mode="empty")
    width = image.shape[1]
    padded[0, 1:-1] = np.roll(image[0], width // 2, axis=0)
    padded[-1, 1:-1] = np.roll(image[-1], width // 2, axis=0)
    padded[:, 0] = padded[:, -2]
    padded[:, -1] = padded[:, 1]
    return padded


def render_remapped_view(
    padded_panorama: np.ndarray,
    remap: tuple[np.ndarray, np.ndarray | None, int],
) -> np.ndarray:
    map1, map2, interpolation = remap
    return cv2.remap(padded_panorama, map1, map2, interpolation=interpolation)


def virtual_image_relative_name(relative_path: Path, image_format: str) -> Path:
    suffix = ".png" if image_format == "png" else ".jpg"
    return relative_path.with_suffix(suffix)


def save_virtual_view(
    image: np.ndarray,
    output_path: Path,
    mask: np.ndarray,
    mask_path: Path,
    options: VirtualImageOutputOptions,
) -> None:
    if options.image_format == "png":
        Image.fromarray(image).save(
            output_path,
            format="PNG",
            compress_level=options.png_compression,
        )
    else:
        Image.fromarray(image).save(
            output_path,
            format="JPEG",
            quality=options.jpeg_quality,
            subsampling=0,
        )
    Image.fromarray(mask).save(mask_path, format="PNG", compress_level=1)


def read_rendered_rig_masks(
    mask_dir: Path,
    input_image_path: Path,
    panorama_path: Path,
    views: Sequence[PanoramaRigView],
    output_options: VirtualImageOutputOptions = VirtualImageOutputOptions(),
) -> dict[int, np.ndarray]:
    relative_name = virtual_image_relative_name(
        panorama_path.relative_to(input_image_path),
        output_options.image_format,
    ).as_posix()
    masks: dict[int, np.ndarray] = {}
    for view in views:
        path = mask_dir / f"{view.image_prefix}{relative_name}.png"
        with Image.open(path) as mask:
            masks[view.index] = np.asarray(mask).copy()
    return masks


def rig_view_masks(
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
) -> dict[int, np.ndarray]:
    rays = virtual_camera_rays(camera)
    forward = np.asarray([0.0, 0.0, 1.0])
    centers = np.vstack(
        [view.cam_from_pano_rotation.T @ forward for view in views]
    )
    masks: dict[int, np.ndarray] = {}
    for view in views:
        rays_in_pano = rays @ view.cam_from_pano_rotation
        closest_camera = np.argmax(rays_in_pano @ centers.T, axis=1)
        masks[view.index] = (
            (closest_camera == view.index)
            .astype(np.uint8)
            .reshape(camera.height, camera.width)
            * 255
        )
    return masks


def validate_rig_masks(
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
    masks: dict[int, np.ndarray],
) -> None:
    expected_indices = {view.index for view in views}
    if set(masks) != expected_indices:
        raise ValueError(
            f"Rig mask indices {sorted(masks)} do not match views "
            f"{sorted(expected_indices)}"
        )

    principal_x = min(max(int(camera.params[2]), 0), camera.width - 1)
    principal_y = min(max(int(camera.params[3]), 0), camera.height - 1)
    for view in views:
        mask = masks[view.index]
        if mask.shape != (camera.height, camera.width):
            raise ValueError(
                f"Mask {view.index} has shape {mask.shape}; expected "
                f"{(camera.height, camera.width)}"
            )
        if mask.dtype != np.uint8 or not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"Mask {view.index} must be binary uint8")
        if not np.any(mask):
            raise ValueError(f"Mask {view.index} is empty")
        if mask[principal_y, principal_x] == 0:
            raise ValueError(
                f"Mask {view.index} excludes its virtual camera principal ray"
            )


def mask_statistics(masks: dict[int, np.ndarray]) -> dict[str, dict[str, float | int]]:
    return {
        str(index): {
            "active_pixels": int(np.count_nonzero(mask)),
            "active_fraction": float(np.count_nonzero(mask) / mask.size),
        }
        for index, mask in sorted(masks.items())
    }


def virtual_camera_rays(camera: ColmapCamera) -> np.ndarray:
    fx, fy, cx, cy = camera.params
    x, y = np.meshgrid(
        np.arange(camera.width, dtype=np.float64) + 0.5,
        np.arange(camera.height, dtype=np.float64) + 0.5,
    )
    rays = np.stack(((x - cx) / fx, (y - cy) / fy, np.ones_like(x)), axis=-1)
    rays = rays.reshape(-1, 3)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def create_pycolmap_camera(pycolmap_module: Any, camera: ColmapCamera) -> Any:
    return pycolmap_module.Camera(
        camera_id=0,
        model=camera.model,
        width=camera.width,
        height=camera.height,
        params=list(camera.params),
    )


def create_rig_config(
    pycolmap_module: Any,
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
    *,
    ref_index: int = 0,
) -> Any:
    if not views:
        raise ValueError("at least one rig view is required")
    ref_rotation = views[ref_index].cam_from_pano_rotation
    pycolmap_camera = create_pycolmap_camera(pycolmap_module, camera)
    rig_cameras = []
    zero_translation = np.zeros((3, 1), dtype=np.float64)
    for view in views:
        if view.index == ref_index:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = view.cam_from_pano_rotation @ ref_rotation.T
            cam_from_rig = pycolmap_module.Rigid3d(
                pycolmap_module.Rotation3d(cam_from_ref_rotation),
                zero_translation,
            )
        rig_camera = pycolmap_module.RigConfigCamera(
            ref_sensor=view.index == ref_index,
            image_prefix=view.image_prefix,
            cam_from_rig=cam_from_rig,
        )
        rig_camera.camera = pycolmap_camera
        rig_cameras.append(rig_camera)
    return pycolmap_module.RigConfig(cameras=rig_cameras)


def create_feature_extraction_options(
    pycolmap_module: Any,
    options: FeaturePipelineOptions,
) -> Any:
    if options.extractor != "sift":
        raise ValueError(f"Unsupported feature extractor: {options.extractor}")
    extraction_options = pycolmap_module.FeatureExtractionOptions()
    extraction_options.type = pycolmap_module.FeatureExtractorType.SIFT
    extraction_options.use_gpu = False
    extraction_options.sift.max_num_features = options.max_num_features
    return extraction_options


def create_feature_matching_options(
    pycolmap_module: Any,
    options: FeaturePipelineOptions,
) -> Any:
    if options.matcher != "sift-bruteforce":
        raise ValueError(f"Unsupported feature matcher: {options.matcher}")
    matching_options = pycolmap_module.FeatureMatchingOptions()
    matching_options.type = pycolmap_module.FeatureMatcherType.SIFT_BRUTEFORCE
    matching_options.use_gpu = False
    matching_options.guided_matching = options.guided_matching
    matching_options.rig_verification = True
    matching_options.skip_image_pairs_in_same_frame = True
    return matching_options


def validate_vocab_tree_path(vocab_tree_path: Path | None) -> None:
    if vocab_tree_path is not None and not vocab_tree_path.is_file():
        raise FileNotFoundError(f"Vocabulary tree not found: {vocab_tree_path}")


def match_features(
    pycolmap_module: Any,
    database_path: Path,
    *,
    matcher: str,
    matching_options: Any,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
) -> None:
    validate_vocab_tree_path(vocab_tree_path)
    if matcher == "sequential":
        pairing_options = pycolmap_module.SequentialPairingOptions()
        pairing_options.overlap = overlap
        pairing_options.expand_rig_images = True
        pairing_options.loop_detection = loop_detection
        if vocab_tree_path is not None:
            pairing_options.vocab_tree_path = vocab_tree_path
        pycolmap_module.match_sequential(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=pycolmap_module.Device.cpu,
        )
    elif matcher == "exhaustive":
        pycolmap_module.match_exhaustive(
            database_path,
            matching_options=matching_options,
            device=pycolmap_module.Device.cpu,
        )
    elif matcher == "vocabtree":
        pairing_options = pycolmap_module.VocabTreePairingOptions()
        if vocab_tree_path is not None:
            pairing_options.vocab_tree_path = vocab_tree_path
        pycolmap_module.match_vocabtree(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=pycolmap_module.Device.cpu,
        )
    else:
        raise ValueError("matcher must be 'sequential', 'exhaustive', or 'vocabtree'")


def run_panorama_reconstruction(
    input_image_path: Path,
    workspace: Path,
    *,
    render_options: PanoramaRenderOptions,
    matcher: str = "sequential",
    overlap: int = 15,
    loop_detection: bool = True,
    vocab_tree_path: Path | None = None,
    ssl_cert_file: Path | None = None,
    feature_options: FeaturePipelineOptions = FeaturePipelineOptions(),
    output_options: VirtualImageOutputOptions = VirtualImageOutputOptions(),
    image_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    pycolmap_module = require_pycolmap()
    pycolmap_module.set_random_seed(0)
    configure_ssl_cert_file(ssl_cert_file)
    paths = panorama_workspace_paths(workspace)
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    paths["image_dir"].mkdir(parents=True, exist_ok=True)
    paths["mask_dir"].mkdir(parents=True, exist_ok=True)
    paths["sparse"].mkdir(parents=True, exist_ok=True)
    if paths["database"].exists():
        raise FileExistsError(f"{paths['database']} already exists. Pass --overwrite.")

    panorama_paths, camera, views = render_panorama_rig_images(
        input_image_path=input_image_path,
        output_image_dir=paths["image_dir"],
        mask_dir=paths["mask_dir"],
        options=render_options,
        output_options=output_options,
        image_limit=image_limit,
    )
    masks = read_rendered_rig_masks(
        paths["mask_dir"],
        input_image_path,
        panorama_paths[0],
        views,
        output_options,
    )
    validate_rig_masks(camera, views, masks)
    masks_report = mask_statistics(masks)
    write_rig_metadata(
        paths["rig_metadata"],
        input_image_path,
        render_options,
        camera,
        views,
        masks_report,
        output_options,
        image_limit=image_limit,
    )

    extraction_options = create_feature_extraction_options(
        pycolmap_module, feature_options
    )
    pycolmap_module.extract_features(
        paths["database"],
        paths["image_dir"],
        reader_options=pycolmap_module.ImageReaderOptions(mask_path=paths["mask_dir"]),
        camera_mode=pycolmap_module.CameraMode.PER_FOLDER,
        extraction_options=extraction_options,
        device=pycolmap_module.Device.cpu,
    )
    feature_report = read_feature_stats_by_camera(paths["database"])
    feature_health = feature_health_summary(views, feature_report)
    rig_config = create_rig_config(pycolmap_module, camera, views)
    with pycolmap_module.Database.open(paths["database"]) as database:
        pycolmap_module.apply_rig_config([rig_config], database)

    matching_options = create_feature_matching_options(
        pycolmap_module, feature_options
    )
    match_features(
        pycolmap_module,
        paths["database"],
        matcher=matcher,
        matching_options=matching_options,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
    )

    mapper_options = pycolmap_module.IncrementalPipelineOptions()
    mapper_options.ba_refine_sensor_from_rig = False
    mapper_options.ba_refine_focal_length = False
    mapper_options.ba_refine_principal_point = False
    mapper_options.ba_refine_extra_params = False
    mapper_options.ba_use_gpu = False
    reconstructions = pycolmap_module.incremental_mapping(
        paths["database"],
        paths["image_dir"],
        paths["sparse"],
        options=mapper_options,
    )
    if not reconstructions:
        raise RuntimeError("COLMAP did not produce any sparse reconstruction")

    best_model_id, best_reconstruction = max(
        reconstructions.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
    )
    write_best_model(best_reconstruction, paths)
    report = build_panorama_report(
        pycolmap_version=pycolmap_module.__version__,
        input_image_path=input_image_path,
        paths=paths,
        camera=camera,
        views=views,
        panorama_count=len(panorama_paths),
        matcher=matcher,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        ssl_cert_file=ssl_cert_file,
        feature_options=feature_options,
        output_options=output_options,
        masks_report=masks_report,
        feature_report=feature_report,
        feature_health=feature_health,
        image_limit=image_limit,
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
    )
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def panorama_workspace_paths(workspace: Path) -> dict[str, Path]:
    return {
        "workspace": workspace,
        "image_dir": workspace / "images",
        "mask_dir": workspace / "masks",
        "database": workspace / "database.db",
        "sparse": workspace / "sparse",
        "best_sparse": workspace / "sparse_best",
        "raw_ply": workspace / "cloud_colmap_raw.ply",
        "report": workspace / "panorama_rig_report.json",
        "rig_metadata": workspace / "panorama_rig_config.json",
    }


def write_rig_metadata(
    path: Path,
    input_image_path: Path,
    options: PanoramaRenderOptions,
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
    masks_report: dict[str, dict[str, float | int]],
    output_options: VirtualImageOutputOptions,
    *,
    image_limit: int | None,
) -> None:
    data = {
        "input_image_path": str(input_image_path),
        "image_limit": image_limit,
        "render_options": {
            "num_yaw_steps": options.num_yaw_steps,
            "pitches": list(options.pitches),
            "width": options.width,
            "height": options.height,
            "fov_x": options.fov_x,
            "fov_y": options.fov_y,
            "interpolation": options.interpolation,
            "virtual_image_format": output_options.image_format,
            "png_compression": output_options.png_compression,
            "jpeg_quality": output_options.jpeg_quality,
            "image_save_workers": output_options.save_workers,
            "render_backend": "precomputed_opencv_remap",
        },
        "camera": camera.report(),
        "masks": masks_report,
        "views": [
            {
                "index": view.index,
                "image_prefix": view.image_prefix,
                "yaw": view.yaw,
                "pitch": view.pitch,
                "cam_from_pano_rotation": view.cam_from_pano_rotation.tolist(),
            }
            for view in views
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_panorama_report(
    *,
    pycolmap_version: str,
    input_image_path: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
    panorama_count: int,
    matcher: str,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    ssl_cert_file: Path | None,
    feature_options: FeaturePipelineOptions,
    output_options: VirtualImageOutputOptions,
    masks_report: dict[str, dict[str, float | int]],
    feature_report: dict[str, dict[str, float | int]],
    feature_health: dict[str, Any],
    image_limit: int | None,
    reconstructions: dict[int, Any],
    best_model_id: int,
    best_reconstruction: Any,
) -> dict[str, Any]:
    model_stats = {
        str(model_id): panorama_reconstruction_stats(reconstruction)
        for model_id, reconstruction in sorted(reconstructions.items())
    }
    return {
        "status": "raw_colmap_panorama_rig_unscaled",
        "warning": (
            "This is a monocular COLMAP reconstruction in arbitrary COLMAP "
            "scale/frame. The virtual pinhole sensors are a fixed panorama rig, "
            "but the resulting world is not metric without an external scale."
        ),
        "pycolmap_version": pycolmap_version,
        "input_image_path": str(input_image_path),
        "workspace": str(paths["workspace"]),
        "image_dir": str(paths["image_dir"]),
        "mask_dir": str(paths["mask_dir"]),
        "database_path": str(paths["database"]),
        "sparse_path": str(paths["sparse"]),
        "best_sparse_path": str(paths["best_sparse"]),
        "raw_ply_path": str(paths["raw_ply"]),
        "rig_metadata_path": str(paths["rig_metadata"]),
        "camera": camera.report(),
        "options": {
            "matcher": matcher,
            "sequential_overlap": overlap,
            "loop_detection": loop_detection,
            "vocab_tree_path": (
                None if vocab_tree_path is None else str(vocab_tree_path)
            ),
            "ssl_cert_file": (
                os.environ.get("SSL_CERT_FILE")
                if ssl_cert_file is None
                else str(ssl_cert_file)
            ),
            "feature_extractor": feature_options.extractor,
            "feature_matcher": feature_options.matcher,
            "max_num_features": feature_options.max_num_features,
            "guided_matching": feature_options.guided_matching,
            "virtual_image_format": output_options.image_format,
            "png_compression": output_options.png_compression,
            "jpeg_quality": output_options.jpeg_quality,
            "image_save_workers": output_options.save_workers,
            "render_backend": "precomputed_opencv_remap",
            "image_limit": image_limit,
            "use_gpu": False,
            "camera_mode": "PER_FOLDER",
            "ba_refine_sensor_from_rig": False,
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
            "rig_verification": True,
            "skip_image_pairs_in_same_frame": True,
        },
        "panorama_count": panorama_count,
        "virtual_camera_count": len(views),
        "generated_image_count": panorama_count * len(views),
        "masks": masks_report,
        "features_by_virtual_camera": feature_report,
        "feature_health": feature_health,
        "database": read_database_stats(paths["database"]),
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": model_stats,
        "best_model": panorama_reconstruction_stats(best_reconstruction),
    }


def panorama_reconstruction_stats(reconstruction: Any) -> dict[str, Any]:
    stats = reconstruction_stats(reconstruction)
    stats["num_frames"] = reconstruction.num_frames()
    stats["num_registered_frames"] = reconstruction.num_reg_frames()
    return stats


def read_feature_stats_by_camera(
    database_path: Path,
) -> dict[str, dict[str, float | int]]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                substr(images.name, 1, instr(images.name, '/') - 1) AS folder,
                COUNT(*) AS image_count,
                SUM(keypoints.rows) AS total_features,
                AVG(keypoints.rows) AS mean_features,
                MIN(keypoints.rows) AS min_features,
                MAX(keypoints.rows) AS max_features
            FROM images
            JOIN keypoints USING(image_id)
            GROUP BY folder
            ORDER BY folder
            """
        ).fetchall()
    return {
        str(folder): {
            "image_count": int(image_count),
            "total_features": int(total_features),
            "mean_features": float(mean_features),
            "min_features": int(min_features),
            "max_features": int(max_features),
        }
        for folder, image_count, total_features, mean_features, min_features, max_features
        in rows
    }


def feature_health_summary(
    views: Sequence[PanoramaRigView],
    feature_report: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    expected = [view.image_prefix.rstrip("/") for view in views]
    missing = [name for name in expected if name not in feature_report]
    if missing:
        raise RuntimeError(
            f"Feature extraction did not create statistics for virtual cameras: {missing}"
        )
    zero_feature_cameras = [
        name
        for name in expected
        if int(feature_report[name]["total_features"]) == 0
    ]
    return {
        "virtual_camera_count": len(expected),
        "virtual_cameras_with_features": len(expected) - len(zero_feature_cameras),
        "zero_feature_virtual_cameras": zero_feature_cameras,
    }


def parse_pitches(value: str) -> tuple[float, ...]:
    pitches = tuple(radians(float(part.strip())) for part in value.split(",") if part.strip())
    if not pitches:
        raise argparse.ArgumentTypeError("at least one pitch is required")
    return pitches


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render 360 panoramas into a fixed virtual pinhole rig and run "
            "COLMAP reconstruction with rig constraints."
        )
    )
    parser.add_argument(
        "-i",
        "--input-image-path",
        type=Path,
        required=True,
        help="Directory containing input equirectangular panorama images.",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"COLMAP panorama-rig workspace (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--pano-render-type",
        choices=("overlapping", "non-overlapping"),
        default="overlapping",
        help="Preset virtual camera layout.",
    )
    parser.add_argument(
        "--num-yaw-steps",
        type=int,
        default=None,
        help="Override preset yaw step count.",
    )
    parser.add_argument(
        "--pitches-deg",
        type=parse_pitches,
        default=None,
        help="Comma-separated pitch angles in degrees, overriding the preset.",
    )
    parser.add_argument("--width", type=int, default=None, help="Virtual image width.")
    parser.add_argument("--height", type=int, default=None, help="Virtual image height.")
    parser.add_argument(
        "--fov-x",
        type=float,
        default=None,
        help="Virtual horizontal field of view in radians.",
    )
    parser.add_argument(
        "--fov-y",
        type=float,
        default=None,
        help="Virtual vertical field of view in radians.",
    )
    parser.add_argument(
        "--matcher",
        choices=("sequential", "exhaustive", "vocabtree"),
        default="sequential",
        help="Feature matching strategy.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=15,
        help="Sequential matcher overlap in neighboring frames.",
    )
    parser.add_argument(
        "--loop-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vocabulary-tree loop detection during sequential matching.",
    )
    parser.add_argument(
        "--vocab-tree-path",
        type=Path,
        default=None,
        help=(
            "Local COLMAP vocabulary tree for loop detection or vocabtree "
            "matching. If omitted, COLMAP may download its default tree."
        ),
    )
    parser.add_argument(
        "--ssl-cert-file",
        type=Path,
        default=None,
        help=(
            "CA certificate bundle for COLMAP downloads. Defaults to "
            "$SSL_CERT_FILE or /etc/ssl/certs/ca-certificates.crt."
        ),
    )
    parser.add_argument(
        "--feature-extractor",
        choices=("sift",),
        default="sift",
        help="Local feature extractor. The pipeline is structured for future backends.",
    )
    parser.add_argument(
        "--feature-matcher",
        choices=("sift-bruteforce",),
        default="sift-bruteforce",
        help="Local feature matcher. The pipeline is structured for future backends.",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="Maximum SIFT features per virtual camera image.",
    )
    parser.add_argument(
        "--guided-matching",
        action="store_true",
        help="Enable guided feature matching after geometric verification.",
    )
    parser.add_argument(
        "--virtual-image-format",
        choices=("png", "jpeg"),
        default="png",
        help="Format for generated virtual pinhole images.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        help="PNG compression level from 0 to 9 (default: 1).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100 (default: 95).",
    )
    parser.add_argument(
        "--image-save-workers",
        type=int,
        default=12,
        help="Concurrent workers used to encode virtual images (default: 12).",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=None,
        help="Use only the first N sorted panoramas for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the panorama-rig workspace.",
    )
    args = parser.parse_args(argv)
    if not args.input_image_path.is_dir():
        parser.error(f"--input-image-path must be an existing directory: {args.input_image_path}")
    if args.num_yaw_steps is not None and args.num_yaw_steps <= 0:
        parser.error("--num-yaw-steps must be positive")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive")
    if args.fov_x is not None and not 0.0 < args.fov_x < pi:
        parser.error("--fov-x must be greater than 0 and less than pi radians")
    if args.fov_y is not None and not 0.0 < args.fov_y < pi:
        parser.error("--fov-y must be greater than 0 and less than pi radians")
    if args.image_limit is not None and args.image_limit <= 0:
        parser.error("--image-limit must be positive")
    if args.max_num_features <= 0:
        parser.error("--max-num-features must be positive")
    if not 0 <= args.png_compression <= 9:
        parser.error("--png-compression must be between 0 and 9")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.image_save_workers <= 0:
        parser.error("--image-save-workers must be positive")
    if args.vocab_tree_path is not None and not args.vocab_tree_path.is_file():
        parser.error(f"--vocab-tree-path does not exist: {args.vocab_tree_path}")
    if args.ssl_cert_file is not None and not args.ssl_cert_file.is_file():
        parser.error(f"--ssl-cert-file does not exist: {args.ssl_cert_file}")
    return args


def render_options_from_args(args: argparse.Namespace) -> PanoramaRenderOptions:
    preset = default_render_options(args.pano_render_type)
    return PanoramaRenderOptions(
        num_yaw_steps=preset.num_yaw_steps
        if args.num_yaw_steps is None
        else args.num_yaw_steps,
        pitches=preset.pitches if args.pitches_deg is None else args.pitches_deg,
        width=preset.width if args.width is None else args.width,
        height=preset.height if args.height is None else args.height,
        fov_x=preset.fov_x if args.fov_x is None else args.fov_x,
        fov_y=preset.fov_y if args.fov_y is None else args.fov_y,
        interpolation=preset.interpolation,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run_panorama_reconstruction(
        input_image_path=args.input_image_path,
        workspace=args.workspace,
        render_options=render_options_from_args(args),
        matcher=args.matcher,
        overlap=args.overlap,
        loop_detection=args.loop_detection,
        vocab_tree_path=args.vocab_tree_path,
        ssl_cert_file=args.ssl_cert_file,
        feature_options=FeaturePipelineOptions(
            extractor=args.feature_extractor,
            matcher=args.feature_matcher,
            max_num_features=args.max_num_features,
            guided_matching=args.guided_matching,
        ),
        output_options=VirtualImageOutputOptions(
            image_format=args.virtual_image_format,
            png_compression=args.png_compression,
            jpeg_quality=args.jpeg_quality,
            save_workers=args.image_save_workers,
        ),
        image_limit=args.image_limit,
        overwrite=args.overwrite,
    )
    print(f"panoramas: {report['panorama_count']}")
    print(f"virtual cameras: {report['virtual_camera_count']}")
    print(f"generated images: {report['generated_image_count']}")
    print(
        "virtual cameras with features: "
        f"{report['feature_health']['virtual_cameras_with_features']}/"
        f"{report['feature_health']['virtual_camera_count']}"
    )
    print(f"best registered frames: {report['best_model']['num_registered_frames']}")
    print(f"best registered images: {report['best_model']['num_registered_images']}")
    print(f"best sparse points: {report['best_model']['num_points3D']}")
    print(f"report: {report['workspace']}/panorama_rig_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
