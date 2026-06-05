"""Run COLMAP rig SfM directly from equirectangular panoramas."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from math import cos, pi, radians, sin, tan
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from colmap.camera import ColmapCamera, camera_from_fov
from colmap.input_inspection import IMAGE_EXTENSIONS
from colmap.pipeline import write_best_model
from colmap.reporting import read_database_stats, reconstruction_stats
from ep_pinhole import EquirectangularToPinhole

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
    image_limit: int | None = None,
) -> tuple[list[Path], ColmapCamera, list[PanoramaRigView]]:
    panorama_paths = panorama_image_paths(input_image_path)
    if image_limit is not None:
        panorama_paths = panorama_paths[:image_limit]
    if not panorama_paths:
        raise RuntimeError(f"No panorama images found in {input_image_path}")

    views = build_rig_views(options)
    camera = camera_from_fov(options.width, options.height, options.fov_x, options.fov_y)
    masks = rig_view_masks(camera, views)

    for panorama_path in panorama_paths:
        renderer = EquirectangularToPinhole.from_file(panorama_path)
        relative_name = panorama_path.relative_to(input_image_path).as_posix()
        for view in views:
            image = renderer.render(
                height=options.height,
                width=options.width,
                longitude=view.yaw,
                latitude=view.pitch,
                fov_x=options.fov_x,
                fov_y=options.fov_y,
                interpolation=options.interpolation,
            )
            output_path = output_image_dir / view.image_prefix / relative_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(output_path)

            mask_path = mask_dir / f"{view.image_prefix}{relative_name}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(masks[view.index]).save(mask_path)

    return panorama_paths, camera, views


def rig_view_masks(
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
) -> dict[int, np.ndarray]:
    rays = virtual_camera_rays(camera)
    centers = np.vstack(
        [view.cam_from_pano_rotation @ np.asarray([0.0, 0.0, 1.0]) for view in views]
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


def run_panorama_reconstruction(
    input_image_path: Path,
    workspace: Path,
    *,
    render_options: PanoramaRenderOptions,
    matcher: str = "sequential",
    overlap: int = 15,
    image_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    pycolmap_module = require_pycolmap()
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
        image_limit=image_limit,
    )
    write_rig_metadata(
        paths["rig_metadata"],
        input_image_path,
        render_options,
        camera,
        views,
        image_limit=image_limit,
    )

    pycolmap_module.extract_features(
        paths["database"],
        paths["image_dir"],
        reader_options=pycolmap_module.ImageReaderOptions(mask_path=paths["mask_dir"]),
        camera_mode=pycolmap_module.CameraMode.PER_FOLDER,
        device=pycolmap_module.Device.cpu,
    )
    rig_config = create_rig_config(pycolmap_module, camera, views)
    with pycolmap_module.Database.open(paths["database"]) as database:
        pycolmap_module.apply_rig_config([rig_config], database)

    matching_options = pycolmap_module.FeatureMatchingOptions()
    matching_options.use_gpu = False
    matching_options.rig_verification = True
    matching_options.skip_image_pairs_in_same_frame = True
    if matcher == "sequential":
        pycolmap_module.match_sequential(
            paths["database"],
            matching_options=matching_options,
            pairing_options=pycolmap_module.SequentialPairingOptions(overlap=overlap),
            device=pycolmap_module.Device.cpu,
        )
    elif matcher == "exhaustive":
        pycolmap_module.match_exhaustive(
            paths["database"],
            matching_options=matching_options,
            device=pycolmap_module.Device.cpu,
        )
    else:
        raise ValueError("matcher must be 'sequential' or 'exhaustive'")

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
        },
        "camera": camera.report(),
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
    image_limit: int | None,
    reconstructions: dict[int, Any],
    best_model_id: int,
    best_reconstruction: Any,
) -> dict[str, Any]:
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
        "database": read_database_stats(paths["database"]),
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": {
            str(model_id): reconstruction_stats(reconstruction)
            for model_id, reconstruction in sorted(reconstructions.items())
        },
        "best_model": reconstruction_stats(best_reconstruction),
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
        choices=("sequential", "exhaustive"),
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
        image_limit=args.image_limit,
        overwrite=args.overwrite,
    )
    print(f"panoramas: {report['panorama_count']}")
    print(f"virtual cameras: {report['virtual_camera_count']}")
    print(f"generated images: {report['generated_image_count']}")
    print(f"best registered images: {report['best_model']['num_registered_images']}")
    print(f"best sparse points: {report['best_model']['num_points3D']}")
    print(f"report: {report['workspace']}/panorama_rig_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
