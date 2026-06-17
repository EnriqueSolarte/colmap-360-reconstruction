"""Initialize metadata for manually selected pinhole renders."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from importlib import metadata as package_metadata
from math import degrees, pi
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from colmap.camera import camera_from_fov
from render_pinhole.defaults import (
    DEFAULT_PINHOLE_FOV_X,
    DEFAULT_PINHOLE_FOV_Y,
    DEFAULT_PINHOLE_HEIGHT,
    DEFAULT_PINHOLE_INTERPOLATION,
    DEFAULT_PINHOLE_WIDTH,
)
from render_pinhole.renderer import EquirectangularToPinhole
from render_pinhole.tools.manual_pinhole_selection_store import (
    load_metadata_with_selections,
    write_metadata_with_sidecar,
)

try:  # Keep non-scene metadata initialization usable without pycolmap.
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None

SCHEMA_VERSION = 1
PROJECT_DISTRIBUTION = "colmap-360-reconstruction"
METADATA_FILENAME = "metadata.json"
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
POSE_STATUS_MISSING = "missing"
POSE_STATUS_MATCHED = "matched"
POSE_STATUS_UNRESOLVED = "unresolved"
SOURCE_STATUS_MATCHED = "matched"
SOURCE_STATUS_MISSING = "missing"


@dataclass(frozen=True)
class PanoramaRigViewMetadata:
    index: int
    image_prefix: str
    yaw: float
    pitch: float
    cam_from_pano_rotation: np.ndarray


@dataclass(frozen=True)
class ScenePanoramaGroup:
    name: str
    images: tuple[Any, ...]


@dataclass(frozen=True)
class SceneContext:
    workspace: Path
    model_path: Path
    rig_metadata_path: Path
    reconstruction: Any
    views: tuple[PanoramaRigViewMetadata, ...]
    groups: tuple[ScenePanoramaGroup, ...]
    image_frame_names: dict[int, str]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.render_selections:
        try:
            render_metadata_selections(args.output_dir)
        except (OSError, ValueError, KeyError) as error:
            print(error)
            return 1
        return 0

    pano_paths = _iter_pano_paths(args.pano_dir, show_progress=True)
    if not pano_paths:
        print(f"No panorama images found in {args.pano_dir}")
        return 1

    source_index = build_source_pano_index(pano_paths)
    scene_context = (
        None if args.scene_workspace is None else load_scene_context(args.scene_workspace)
    )
    if scene_context is None:
        validation_paths = pano_paths
    else:
        validation_paths = matched_source_paths(source_index, scene_context.groups)

    validation_errors = validate_panoramas(validation_paths, show_progress=True)
    if validation_errors:
        for error in validation_errors:
            print(error)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / METADATA_FILENAME
    if metadata_path.exists() and not args.overwrite:
        print(f"Metadata already exists: {metadata_path}")
        print("Pass --overwrite to rebuild it.")
        return 1

    metadata = build_metadata(
        pano_paths=pano_paths,
        pano_dir=args.pano_dir,
        output_dir=args.output_dir,
        scene_workspace=args.scene_workspace,
        width=args.width,
        height=args.height,
        fov_x=args.fov_x,
        fov_y=args.fov_y,
        interpolation=args.interpolation,
        output_extension=args.output_extension,
        source_index=source_index,
        scene_context=scene_context,
    )
    with tqdm(total=1, desc="Writing metadata", unit="file") as progress:
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        progress.update()
    print(f"Metadata: {metadata_path}")
    print(f"Initialized {metadata['panorama_count']} panoramas.")
    return 0


def render_metadata_selections(output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = load_metadata_with_selections(output_dir)
    render_defaults = metadata.get("render_defaults", {})
    panoramas = metadata.get("panoramas", [])
    selection_count = sum(len(panorama.get("selections", [])) for panorama in panoramas)
    rendered_count = 0

    progress = tqdm(
        total=selection_count,
        desc="Rendering selected pinhole views",
        unit="view",
    )
    try:
        for panorama in panoramas:
            source_path = panorama.get("source_pano_path") or panorama.get("pano_path")
            selections = panorama.get("selections", [])
            if not selections:
                continue
            if panorama.get("source_status") == SOURCE_STATUS_MISSING:
                raise ValueError(
                    "Cannot render selections for missing source panorama: "
                    f"{panorama.get('scene_pano_filename') or panorama.get('pano_filename')}"
                )
            if source_path is None:
                raise ValueError(
                    "Cannot render selections without source panorama path: "
                    f"{panorama.get('scene_pano_filename') or panorama.get('pano_filename')}"
                )

            source_path = Path(source_path)
            if not source_path.is_file():
                raise FileNotFoundError(f"Source panorama file not found: {source_path}")
            renderer = EquirectangularToPinhole.from_file(source_path)

            for fallback_index, selection in enumerate(selections):
                render_selection(
                    renderer,
                    output_dir,
                    panorama,
                    selection,
                    render_defaults,
                    fallback_index,
                )
                rendered_count += 1
                progress.update()
    finally:
        progress.close()

    metadata["render_summary"] = {
        "status": "rendered_selections",
        "selection_count": selection_count,
        "rendered_selection_count": rendered_count,
    }
    with tqdm(total=1, desc="Writing metadata", unit="file") as progress:
        write_metadata_with_sidecar(output_dir, metadata)
        progress.update()
    print(f"Metadata: {metadata_path}")
    print(f"Rendered {rendered_count} selected pinhole views.")
    return metadata


def render_selection(
    renderer: EquirectangularToPinhole,
    output_dir: Path,
    panorama: dict[str, Any],
    selection: dict[str, Any],
    render_defaults: dict[str, Any],
    fallback_index: int,
) -> None:
    longitude = wrap_longitude(float(selection["longitude"]))
    latitude = float(selection["latitude"])
    validate_latitude(latitude)
    selection_index = int(selection.get("selection_index", fallback_index))
    width = int(selection.get("width", render_defaults["width"]))
    height = int(selection.get("height", render_defaults["height"]))
    fov_x = float(selection.get("fov_x", render_defaults["fov_x"]))
    fov_y = float(selection.get("fov_y", render_defaults["fov_y"]))
    interpolation = str(selection.get("interpolation", render_defaults["interpolation"]))
    extension = str(
        selection.get("output_extension", render_defaults.get("output_extension", ".png"))
    )

    camera_from_fov(width=width, height=height, fov_x=fov_x, fov_y=fov_y)
    filename = rendered_filename(
        str(panorama["pano_stem"]),
        selection_index=selection_index,
        longitude_degrees=degrees(longitude),
        latitude_degrees=degrees(latitude),
        width=width,
        height=height,
        extension=extension,
    )
    output_path = output_dir / filename
    pinhole = renderer.render(
        height=height,
        width=width,
        longitude=longitude,
        latitude=latitude,
        fov_x=fov_x,
        fov_y=fov_y,
        interpolation=interpolation,
    )
    Image.fromarray(pinhole).save(output_path)

    selection.update(
        {
            "selection_index": selection_index,
            "longitude": longitude,
            "latitude": latitude,
            "longitude_degrees": degrees(longitude),
            "latitude_degrees": degrees(latitude),
            "width": width,
            "height": height,
            "fov_x": fov_x,
            "fov_y": fov_y,
            "interpolation": interpolation,
            "output_extension": _normalize_extension(extension),
            "rendered_filename": filename,
            "rendered_path": str(output_path),
            "render_status": "rendered",
            "pinhole_camera_pose": pinhole_camera_pose_report(
                panorama.get("pano_camera_pose"),
                longitude,
                latitude,
            ),
        }
    )


def build_metadata(
    *,
    pano_paths: list[Path],
    pano_dir: Path,
    output_dir: Path,
    scene_workspace: Path | None,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    interpolation: str,
    output_extension: str,
    source_index: dict[str, Path] | None = None,
    scene_context: SceneContext | None = None,
) -> dict[str, Any]:
    camera = camera_from_fov(width=width, height=height, fov_x=fov_x, fov_y=fov_y)
    source_index = build_source_pano_index(pano_paths) if source_index is None else source_index
    if scene_context is None and scene_workspace is not None:
        scene_context = load_scene_context(scene_workspace)
    if scene_context is not None:
        panoramas = scene_panorama_records(
            scene_context,
            source_index,
            show_progress=True,
        )
        pose_status = POSE_STATUS_MATCHED
    else:
        panoramas = source_panorama_records(pano_paths)
        pose_status = POSE_STATUS_MISSING

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "manual-pinhole-render",
        "project_version": _project_version(),
        "pano_dir": str(pano_dir),
        "scene_workspace": None if scene_workspace is None else str(scene_workspace),
        "output_dir": str(output_dir),
        "metadata_path": str(output_dir / METADATA_FILENAME),
        "pose_status": pose_status,
        "pose_status_description": _pose_status_description(pose_status),
        "source_pano_count": len(pano_paths),
        "scene": None if scene_context is None else scene_report(scene_context),
        "render_defaults": {
            "width": width,
            "height": height,
            "fov_x": fov_x,
            "fov_y": fov_y,
            "fov_x_degrees": degrees(fov_x),
            "fov_y_degrees": degrees(fov_y),
            "interpolation": interpolation,
            "output_extension": output_extension,
            "camera": camera.report(),
        },
        "panorama_count": len(panoramas),
        "panoramas": panoramas,
    }


def source_panorama_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "pano_index": index,
            "pano_filename": path.name,
            "pano_stem": path.stem,
            "pano_path": str(path),
            "scene_pano_filename": None,
            "source_pano_filename": path.name,
            "source_pano_path": str(path),
            "source_status": SOURCE_STATUS_MATCHED,
            "source_resolution_strategy": "direct",
            "pano_camera_pose": None,
            "pose_status": POSE_STATUS_MISSING,
            "colmap_quality": None,
            "selections": [],
        }
        for index, path in enumerate(paths)
    ]


def scene_panorama_records(
    scene_context: SceneContext,
    source_index: dict[str, Path],
    *,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    records = []
    progress = tqdm(
        scene_context.groups,
        desc="Computing COLMAP panorama stats",
        unit="pano",
        disable=not show_progress,
    )
    for index, group in enumerate(progress):
        source = resolve_source_panorama(group.name, source_index)
        records.append(
            {
                "pano_index": index,
                "pano_filename": source["source_pano_filename"] or group.name,
                "pano_stem": Path(source["source_pano_filename"] or group.name).stem,
                "pano_path": source["source_pano_path"],
                "scene_pano_filename": group.name,
                "source_pano_filename": source["source_pano_filename"],
                "source_pano_path": source["source_pano_path"],
                "source_status": source["source_status"],
                "source_resolution_strategy": source["source_resolution_strategy"],
                "pano_camera_pose": panorama_pose_report(group, scene_context.views),
                "pose_status": POSE_STATUS_MATCHED,
                "colmap_quality": panorama_quality_report(group, scene_context),
                "selections": [],
            }
        )
    return records


def validate_panoramas(
    paths: list[Path],
    *,
    show_progress: bool = False,
) -> list[str]:
    errors = []
    progress = tqdm(
        paths,
        desc="Validating source panoramas",
        unit="pano",
        disable=not show_progress,
    )
    for path in progress:
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError as error:
            errors.append(f"Invalid panorama image {path}: {error}")
    return errors


def load_scene_context(scene_workspace: Path) -> SceneContext:
    pycolmap_module = require_pycolmap()
    model_path = resolve_scene_model_path(scene_workspace)
    rig_metadata_path = resolve_rig_metadata_path(scene_workspace, model_path)
    views = panorama_views_from_metadata(rig_metadata_path)
    reconstruction = pycolmap_module.Reconstruction(model_path)
    groups, image_frame_names = scene_panorama_groups(
        reconstruction,
        views,
        show_progress=True,
    )
    return SceneContext(
        workspace=scene_workspace,
        model_path=model_path,
        rig_metadata_path=rig_metadata_path,
        reconstruction=reconstruction,
        views=views,
        groups=groups,
        image_frame_names=image_frame_names,
    )


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required when --scene-workspace is provided. Install "
            "the `colmap` extra or add pycolmap to the environment."
        )
    return pycolmap


def resolve_scene_model_path(scene_workspace: Path) -> Path:
    candidates = (scene_workspace / "sparse_best", scene_workspace)
    for candidate in candidates:
        if is_colmap_model_dir(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a readable COLMAP model. Expected sparse_best under "
        f"{scene_workspace}, or a model directory directly."
    )


def resolve_rig_metadata_path(scene_workspace: Path, model_path: Path) -> Path:
    candidates = (
        scene_workspace / "panorama_rig_config.json",
        model_path / "panorama_rig_config.json",
        model_path.parent / "panorama_rig_config.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find panorama_rig_config.json for scene workspace: "
        f"{scene_workspace}"
    )


def is_colmap_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_names = ("cameras", "images", "points3D")
    return all(
        any((path / f"{name}.{extension}").is_file() for extension in ("bin", "txt"))
        for name in required_names
    )


def panorama_views_from_metadata(path: Path) -> tuple[PanoramaRigViewMetadata, ...]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    views = []
    for view in metadata.get("views", []):
        prefix = view.get("image_prefix")
        rotation = view.get("cam_from_pano_rotation")
        if not isinstance(prefix, str) or rotation is None:
            continue
        rotation_array = np.asarray(rotation, dtype=np.float64)
        if rotation_array.shape != (3, 3):
            continue
        views.append(
            PanoramaRigViewMetadata(
                index=int(view["index"]),
                image_prefix=prefix,
                yaw=float(view["yaw"]),
                pitch=float(view["pitch"]),
                cam_from_pano_rotation=rotation_array,
            )
        )
    if not views:
        raise ValueError(f"No panorama rig views found in {path}")
    return tuple(sorted(views, key=lambda view: len(view.image_prefix), reverse=True))


def scene_panorama_groups(
    reconstruction: Any,
    views: tuple[PanoramaRigViewMetadata, ...],
    *,
    show_progress: bool = False,
) -> tuple[tuple[ScenePanoramaGroup, ...], dict[int, str]]:
    grouped: dict[str, list[Any]] = {}
    image_frame_names = {}
    reg_image_ids = list(reconstruction.reg_image_ids())
    progress = tqdm(
        reg_image_ids,
        desc="Grouping registered COLMAP images",
        unit="image",
        disable=not show_progress,
    )
    for image_id in progress:
        image = reconstruction.image(image_id)
        frame_name = panorama_frame_name(image.name, views)
        image_frame_names[int(image.image_id)] = frame_name
        grouped.setdefault(frame_name, []).append(image)
    groups = tuple(
        ScenePanoramaGroup(name=name, images=tuple(sorted(images, key=lambda image: image.name)))
        for name, images in sorted(grouped.items())
    )
    return groups, image_frame_names


def panorama_frame_name(
    image_name: str,
    views: tuple[PanoramaRigViewMetadata, ...],
) -> str:
    for view in views:
        if image_name.startswith(view.image_prefix):
            return image_name[len(view.image_prefix) :]
    return image_name


def panorama_view_for_image(
    image_name: str,
    views: tuple[PanoramaRigViewMetadata, ...],
) -> PanoramaRigViewMetadata | None:
    for view in views:
        if image_name.startswith(view.image_prefix):
            return view
    return None


def panorama_pose_report(
    group: ScenePanoramaGroup,
    views: tuple[PanoramaRigViewMetadata, ...],
) -> dict[str, Any] | None:
    centers = [np.asarray(image.projection_center(), dtype=np.float64) for image in group.images]
    center_world = np.vstack(centers).mean(axis=0) if centers else None
    if center_world is None:
        return None

    for image in group.images:
        view = panorama_view_for_image(image.name, views)
        if view is None:
            continue
        pose_cam_from_world = image.cam_from_world()
        rotation_cam_from_world = np.asarray(
            pose_cam_from_world.rotation.matrix(),
            dtype=np.float64,
        )
        rotation_pano_from_world = (
            view.cam_from_pano_rotation.T @ rotation_cam_from_world
        )
        rotation_world_from_pano = rotation_pano_from_world.T
        transform_world_from_pano = np.eye(4, dtype=np.float64)
        transform_world_from_pano[:3, :3] = rotation_world_from_pano
        transform_world_from_pano[:3, 3] = center_world
        return {
            "coordinate_frame": "colmap_world_unscaled",
            "source_image_name": image.name,
            "source_virtual_camera_index": view.index,
            "center_world": center_world.tolist(),
            "rotation_world_from_pano": rotation_world_from_pano.tolist(),
            "transform_world_from_pano": transform_world_from_pano.tolist(),
        }
    return None


def panorama_quality_report(
    group: ScenePanoramaGroup,
    scene_context: SceneContext,
) -> dict[str, Any]:
    point_ids = sorted(point3d_ids_for_images(group.images))
    points = safe_points3d(scene_context.reconstruction, point_ids)
    errors = np.asarray([float(point.error) for point in points], dtype=np.float64)
    track_lengths = np.asarray(
        [float(point.track.length()) for point in points],
        dtype=np.float64,
    )
    connected_counts = connected_panorama_counts(points, scene_context.image_frame_names, group.name)
    per_virtual_camera = [
        virtual_camera_quality_record(image, scene_context.views)
        for image in sorted(group.images, key=lambda image: image.name)
    ]
    registered_virtual_image_count = len(group.images)
    expected_virtual_image_count = len(scene_context.views)
    active_virtual_camera_count = sum(
        1 for image in group.images if int(image.num_points3D) > 0
    )
    return {
        "registered_virtual_image_count": registered_virtual_image_count,
        "expected_virtual_image_count": expected_virtual_image_count,
        "registered_virtual_image_ratio": safe_ratio(
            registered_virtual_image_count,
            expected_virtual_image_count,
        ),
        "active_virtual_camera_count": active_virtual_camera_count,
        "active_virtual_camera_ratio": safe_ratio(
            active_virtual_camera_count,
            expected_virtual_image_count,
        ),
        "triangulated_observation_count": int(
            sum(int(image.num_points3D) for image in group.images)
        ),
        "unique_point3D_count": len(point_ids),
        "mean_reprojection_error_px": optional_stat(errors, np.mean),
        "median_reprojection_error_px": optional_stat(errors, np.median),
        "p90_reprojection_error_px": optional_stat(
            errors,
            lambda values: np.percentile(values, 90),
        ),
        "mean_track_length": optional_stat(track_lengths, np.mean),
        "connected_panorama_count": len(connected_counts),
        "top_connected_panoramas": [
            {"scene_pano_filename": name, "shared_track_count": count}
            for name, count in connected_counts.most_common(10)
        ],
        "per_virtual_camera": per_virtual_camera,
    }


def virtual_camera_quality_record(
    image: Any,
    views: tuple[PanoramaRigViewMetadata, ...],
) -> dict[str, Any]:
    view = panorama_view_for_image(image.name, views)
    return {
        "image_id": int(image.image_id),
        "image_name": image.name,
        "camera_id": int(image.camera_id),
        "virtual_camera_index": None if view is None else view.index,
        "virtual_camera_prefix": None if view is None else view.image_prefix,
        "triangulated_observation_count": int(image.num_points3D),
    }


def point3d_ids_for_images(images: tuple[Any, ...]) -> set[int]:
    point_ids = set()
    for image in images:
        for point2d in image.points2D:
            if point2d.has_point3D():
                point_ids.add(int(point2d.point3D_id))
    return point_ids


def safe_points3d(reconstruction: Any, point_ids: list[int]) -> list[Any]:
    points = []
    for point_id in point_ids:
        try:
            points.append(reconstruction.point3D(point_id))
        except KeyError:
            continue
    return points


def connected_panorama_counts(
    points: list[Any],
    image_frame_names: dict[int, str],
    current_frame_name: str,
) -> Counter:
    counts: Counter = Counter()
    for point in points:
        connected_frames = {
            image_frame_names[int(element.image_id)]
            for element in point.track.elements
            if int(element.image_id) in image_frame_names
        }
        connected_frames.discard(current_frame_name)
        counts.update(connected_frames)
    return counts


def optional_stat(values: np.ndarray, reducer: Any) -> float | None:
    if values.size == 0:
        return None
    return float(reducer(values))


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def scene_report(scene_context: SceneContext) -> dict[str, Any]:
    return {
        "workspace": str(scene_context.workspace),
        "model_path": str(scene_context.model_path),
        "rig_metadata_path": str(scene_context.rig_metadata_path),
        "registered_virtual_image_count": int(scene_context.reconstruction.num_reg_images()),
        "registered_panorama_count": len(scene_context.groups),
        "expected_virtual_camera_count": len(scene_context.views),
    }


def build_source_pano_index(paths: list[Path]) -> dict[str, Path]:
    return {path.name: path for path in paths}


def matched_source_paths(
    source_index: dict[str, Path],
    groups: tuple[ScenePanoramaGroup, ...],
) -> list[Path]:
    paths = []
    for group in groups:
        source = resolve_source_panorama(group.name, source_index)
        if source["source_status"] == SOURCE_STATUS_MATCHED:
            paths.append(Path(source["source_pano_path"]))
    return paths


def resolve_source_panorama(
    scene_pano_filename: str,
    source_index: dict[str, Path],
) -> dict[str, str | None]:
    for strategy, candidate in source_filename_candidates(scene_pano_filename):
        path = source_index.get(candidate)
        if path is not None:
            return {
                "source_status": SOURCE_STATUS_MATCHED,
                "source_pano_filename": candidate,
                "source_pano_path": str(path),
                "source_resolution_strategy": strategy,
            }
    return {
        "source_status": SOURCE_STATUS_MISSING,
        "source_pano_filename": None,
        "source_pano_path": None,
        "source_resolution_strategy": "unresolved",
    }


def source_filename_candidates(scene_pano_filename: str) -> list[tuple[str, str]]:
    candidates = [("exact", scene_pano_filename)]
    if "_" in scene_pano_filename:
        candidates.append(("strip_first_prefix", scene_pano_filename.split("_", 1)[1]))
    return candidates


def pinhole_camera_pose_report(
    pano_camera_pose: dict[str, Any] | None,
    longitude: float,
    latitude: float,
) -> dict[str, Any] | None:
    if pano_camera_pose is None:
        return None
    rotation_world_from_pano = np.asarray(
        pano_camera_pose["rotation_world_from_pano"],
        dtype=np.float64,
    )
    center_world = np.asarray(pano_camera_pose["center_world"], dtype=np.float64)
    cam_from_pano = cam_from_pano_rotation(longitude, latitude)
    rotation_world_from_pinhole = rotation_world_from_pano @ cam_from_pano.T
    transform_world_from_pinhole = np.eye(4, dtype=np.float64)
    transform_world_from_pinhole[:3, :3] = rotation_world_from_pinhole
    transform_world_from_pinhole[:3, 3] = center_world
    return {
        "coordinate_frame": pano_camera_pose.get(
            "coordinate_frame",
            "colmap_world_unscaled",
        ),
        "longitude": longitude,
        "latitude": latitude,
        "longitude_degrees": degrees(longitude),
        "latitude_degrees": degrees(latitude),
        "center_world": center_world.tolist(),
        "rotation_world_from_pinhole": rotation_world_from_pinhole.tolist(),
        "transform_world_from_pinhole": transform_world_from_pinhole.tolist(),
        "cam_from_pano_rotation": cam_from_pano.tolist(),
    }


def cam_from_pano_rotation(longitude: float, latitude: float) -> np.ndarray:
    return rotation_x(-latitude) @ rotation_y(-longitude)


def rotation_x(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def rotation_y(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def wrap_longitude(longitude: float) -> float:
    return float((longitude + pi) % (2 * pi) - pi)


def validate_latitude(latitude: float) -> None:
    if not -pi / 2 <= latitude <= pi / 2:
        raise ValueError("latitude must be in [-pi / 2, pi / 2] radians")


def rendered_filename(
    pano_stem: str,
    *,
    selection_index: int,
    longitude_degrees: float,
    latitude_degrees: float,
    width: int,
    height: int,
    extension: str = ".png",
) -> str:
    if selection_index < 0:
        raise ValueError("selection_index must be non-negative")
    normalized_extension = _normalize_extension(extension)
    return (
        f"{pano_stem}__sel-{selection_index:03d}"
        f"__lon-{format_signed_degrees(longitude_degrees)}"
        f"__lat-{format_signed_degrees(latitude_degrees)}"
        f"__w{width}_h{height}{normalized_extension}"
    )


def format_signed_degrees(value: float) -> str:
    prefix = "p" if value >= 0 else "m"
    return f"{prefix}{abs(value):05.1f}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize metadata for manually selected pinhole renders. This "
            "slice does not render selected images yet."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--pano-dir",
        type=Path,
        help="Directory containing source equirectangular panorama images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where metadata.json and future renders will be written.",
    )
    parser.add_argument(
        "-w",
        "--scene-workspace",
        type=Path,
        help="Optional reconstructed scene workspace used for future pose lookup.",
    )
    parser.add_argument(
        "--width",
        default=DEFAULT_PINHOLE_WIDTH,
        type=int,
        help="Default rendered pinhole width in pixels.",
    )
    parser.add_argument(
        "--height",
        default=DEFAULT_PINHOLE_HEIGHT,
        type=int,
        help="Default rendered pinhole height in pixels.",
    )
    parser.add_argument(
        "--fov-x",
        default=DEFAULT_PINHOLE_FOV_X,
        type=float,
        help="Default horizontal field of view in radians.",
    )
    parser.add_argument(
        "--fov-y",
        default=DEFAULT_PINHOLE_FOV_Y,
        type=float,
        help="Default vertical field of view in radians.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("linear", "bilinear", "nearest"),
        default=DEFAULT_PINHOLE_INTERPOLATION,
        help="Default interpolation mode for future renders.",
    )
    parser.add_argument(
        "--output-extension",
        default=".png",
        help="Default output extension for future selected pinhole renders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing metadata.json in the output directory.",
    )
    parser.add_argument(
        "--render-selections",
        action="store_true",
        help=(
            "Render existing selections from output-dir/metadata.json and update "
            "the metadata in place."
        ),
    )
    args = parser.parse_args(argv)

    if not args.render_selections and args.pano_dir is None:
        parser.error("--pano-dir is required unless --render-selections is used")
    if args.pano_dir is not None and not args.pano_dir.is_dir():
        parser.error(f"--pano-dir must be an existing directory: {args.pano_dir}")
    if args.scene_workspace is not None and not args.scene_workspace.is_dir():
        parser.error(
            "--scene-workspace must be an existing directory: "
            f"{args.scene_workspace}"
        )
    if args.width <= 0:
        parser.error("--width must be positive")
    if args.height <= 0:
        parser.error("--height must be positive")
    camera_from_fov(
        width=args.width,
        height=args.height,
        fov_x=args.fov_x,
        fov_y=args.fov_y,
    )
    args.output_extension = _normalize_extension(args.output_extension)
    return args


def _iter_pano_paths(directory: Path, *, show_progress: bool = False) -> list[Path]:
    entries = tqdm(
        list(directory.iterdir()),
        desc="Scanning source panorama directory",
        unit="entry",
        disable=not show_progress,
    )
    return sorted(
        path
        for path in entries
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _normalize_extension(extension: str) -> str:
    return extension if extension.startswith(".") else f".{extension}"


def _project_version() -> str:
    try:
        return package_metadata.version(PROJECT_DISTRIBUTION)
    except package_metadata.PackageNotFoundError:
        return "0.1.0"


def _pose_status_description(pose_status: str) -> str:
    if pose_status == "missing":
        return "No scene workspace was provided, so panorama poses are unavailable."
    if pose_status == "unresolved":
        return (
            "A scene workspace was provided, but pose lookup is intentionally "
            "deferred to a later implementation slice."
        )
    if pose_status == "matched":
        return (
            "Panorama poses were derived from registered virtual-camera images "
            "in the COLMAP scene. Coordinates are raw unscaled COLMAP world "
            "coordinates."
        )
    return "Unknown pose status."


if __name__ == "__main__":
    raise SystemExit(main())
