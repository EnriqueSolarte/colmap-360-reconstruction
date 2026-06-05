"""Visualize a COLMAP sparse reconstruction with Open3D.

This viewer is for inspecting the raw COLMAP add-on result. If the model was
created from monocular images, its coordinate frame and scale are still COLMAP's
arbitrary reconstruction frame until marker-based scale recovery is applied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XDG_RUNTIME_DIR", "/tmp/open3d-runtime")

import numpy as np

from .reconstruct import DEFAULT_WORKSPACE

try:
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - environment dependent
    o3d = None


DEFAULT_MODEL = DEFAULT_WORKSPACE / "sparse_best"


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required to read COLMAP models. Install the `colmap` "
            "extra or add `pycolmap` to the environment."
        )
    return pycolmap


def require_open3d():
    if o3d is None:
        raise RuntimeError(
            "Open3D is required for point-cloud visualization. Install the "
            "`visualize` extra or add `open3d` to the environment."
        )
    return o3d


def point_cloud_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    *,
    min_track_length: int,
    max_reprojection_error: float,
    remove_outliers: bool,
) -> o3d.geometry.PointCloud:
    xyz: list[np.ndarray] = []
    rgb: list[np.ndarray] = []

    for point3d in reconstruction.points3D.values():
        if point3d.track.length() < min_track_length:
            continue
        if max_reprojection_error > 0.0 and point3d.error > max_reprojection_error:
            continue
        xyz.append(np.asarray(point3d.xyz, dtype=np.float64))
        rgb.append(np.asarray(point3d.color, dtype=np.float64) / 255.0)

    cloud = o3d.geometry.PointCloud()
    if xyz:
        cloud.points = o3d.utility.Vector3dVector(np.vstack(xyz))
        cloud.colors = o3d.utility.Vector3dVector(np.vstack(rgb))

    if remove_outliers and len(cloud.points) >= 20:
        cloud, _indices = cloud.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0,
        )
    return cloud


def camera_geometries_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    *,
    scale: float,
    max_cameras: int,
) -> tuple[list[o3d.geometry.Geometry], np.ndarray]:
    images = registered_images(reconstruction)
    if max_cameras > 0 and len(images) > max_cameras:
        sample_idxs = np.linspace(0, len(images) - 1, max_cameras, dtype=int)
        frustum_images = [images[i] for i in sample_idxs]
    else:
        frustum_images = images

    geometries: list[o3d.geometry.Geometry] = []
    centers = camera_centers(images)

    for image in frustum_images:
        camera = reconstruction.camera(image.camera_id)
        geometries.extend(draw_camera(image, camera, scale=scale))

    if centers.shape[0] >= 2:
        geometries.append(make_trajectory_lineset(centers))

    return geometries, centers


def registered_images(reconstruction: pycolmap.Reconstruction) -> list[pycolmap.Image]:
    images = [reconstruction.image(image_id) for image_id in reconstruction.reg_image_ids()]
    return sorted(images, key=lambda image: image.name)


def camera_centers(images: list[pycolmap.Image]) -> np.ndarray:
    if not images:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(
        [
            np.asarray(image.projection_center(), dtype=np.float64)
            for image in images
        ]
    )


def draw_camera(
    image: pycolmap.Image,
    camera: pycolmap.Camera,
    *,
    scale: float,
) -> list[o3d.geometry.Geometry]:
    pose_cam_from_world = image.cam_from_world()
    rotation_cam_from_world = np.asarray(
        pose_cam_from_world.rotation.matrix(),
        dtype=np.float64,
    )
    rotation_world_from_cam = rotation_cam_from_world.T
    center_world = np.asarray(image.projection_center(), dtype=np.float64)

    fx, fy, cx, cy = pinhole_params(camera)
    width = float(camera.width)
    height = float(camera.height)

    frustum_cam = np.array(
        [
            [0.0, 0.0, 0.0],
            [(0.0 - cx) * scale / fx, (0.0 - cy) * scale / fy, scale],
            [(width - cx) * scale / fx, (0.0 - cy) * scale / fy, scale],
            [(width - cx) * scale / fx, (height - cy) * scale / fy, scale],
            [(0.0 - cx) * scale / fx, (height - cy) * scale / fy, scale],
        ],
        dtype=np.float64,
    )
    frustum_world = (rotation_world_from_cam @ frustum_cam.T).T + center_world

    frustum_lines = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1],
        ],
        dtype=np.int32,
    )
    frustum_color = np.array([[0.9, 0.15, 0.8] for _ in frustum_lines])
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(frustum_world),
        lines=o3d.utility.Vector2iVector(frustum_lines),
    )
    frustum.colors = o3d.utility.Vector3dVector(frustum_color)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 0.5)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_world_from_cam
    transform[:3, 3] = center_world
    axis.transform(transform)

    return [frustum, axis]


def pinhole_params(camera: pycolmap.Camera) -> tuple[float, float, float, float]:
    model = camera.model_name
    params = np.asarray(camera.params, dtype=np.float64)
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    if model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE"}:
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])
    raise ValueError(f"Unsupported camera model for visualization: {model}")


def make_trajectory_lineset(centers: np.ndarray) -> o3d.geometry.LineSet:
    lines = np.column_stack(
        (
            np.arange(0, centers.shape[0] - 1, dtype=np.int32),
            np.arange(1, centers.shape[0], dtype=np.int32),
        )
    )
    colors = np.tile(np.array([[0.1, 0.8, 1.0]], dtype=np.float64), (lines.shape[0], 1))
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def auto_camera_scale(points: o3d.geometry.PointCloud, centers: np.ndarray) -> float:
    values: list[float] = []
    if len(points.points) > 0:
        point_array = np.asarray(points.points)
        extent = np.linalg.norm(point_array.max(axis=0) - point_array.min(axis=0))
        if np.isfinite(extent) and extent > 0.0:
            values.append(extent * 0.025)
    if centers.shape[0] >= 2:
        center_extent = np.linalg.norm(centers.max(axis=0) - centers.min(axis=0))
        if np.isfinite(center_extent) and center_extent > 0.0:
            values.append(center_extent * 0.04)
    return float(max(min(values, default=1.0), 1e-4))


def scale_from_report(report_path: Path) -> float:
    """Read metric scale in metres per COLMAP unit from a recovery report."""
    with report_path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    return scale_from_report_data(report, report_path)


def scale_from_report_data(report: dict, report_path: Path) -> float:
    """Read and validate metric scale from report data."""
    try:
        scale = float(report["scale_m_per_colmap_unit"])
    except KeyError as exc:
        raise ValueError(
            f"{report_path} does not contain 'scale_m_per_colmap_unit'"
        ) from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"{report_path} contains invalid scale_m_per_colmap_unit: {scale}"
        )
    return scale


def similarity_from_report(report_path: Path) -> tuple[float, np.ndarray, np.ndarray, bool]:
    """Read COLMAP-world to marker-frame similarity from a recovery report."""
    with report_path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    scale = scale_from_report_data(report, report_path)
    has_marker_transform = (
        "rotation_marker_from_colmap_world" in report
        and "translation_marker_from_colmap_world_m" in report
    )
    if not has_marker_transform:
        return scale, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), False

    rotation = np.asarray(
        report["rotation_marker_from_colmap_world"],
        dtype=np.float64,
    )
    translation = np.asarray(
        report["translation_marker_from_colmap_world_m"],
        dtype=np.float64,
    )
    if rotation.shape != (3, 3):
        raise ValueError(f"{report_path} has invalid rotation shape: {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(
            f"{report_path} has invalid translation shape: {translation.shape}"
        )
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise ValueError(f"{report_path} contains non-finite similarity values")
    return scale, rotation, translation, True


def apply_report_similarity(
    reconstruction: pycolmap.Reconstruction,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> None:
    """Apply report similarity to a reconstruction in-place."""
    pycolmap_module = require_pycolmap()
    transform = pycolmap_module.Sim3d(
        float(scale),
        pycolmap_module.Rotation3d(rotation),
        translation,
    )
    reconstruction.transform(transform)


def save_map_image(
    path: Path,
    geometries: list[o3d.geometry.Geometry],
    view_points: np.ndarray,
    *,
    point_size: float,
    width: int = 1600,
    height: int = 1100,
) -> None:
    """Save an Open3D offscreen rendering of the sparse map."""
    runtime_dir = Path(os.environ["XDG_RUNTIME_DIR"])
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(0o700)
    path.parent.mkdir(parents=True, exist_ok=True)
    visualizer = o3d.visualization.Visualizer()
    created = visualizer.create_window(
        window_name="COLMAP map screenshot",
        width=width,
        height=height,
        visible=False,
    )
    if not created:
        created = visualizer.create_window(
            window_name="COLMAP map screenshot",
            width=width,
            height=height,
            visible=True,
        )
    if not created:
        raise RuntimeError(
            "Open3D could not create a visualization window for screenshot "
            "capture. Check that this environment has GUI/OpenGL support."
        )

    try:
        for geometry in geometries:
            visualizer.add_geometry(geometry)
        render_options = visualizer.get_render_option()
        render_options.point_size = point_size
        render_options.background_color = np.asarray([0.02, 0.02, 0.025])
        configure_map_view(visualizer, view_points)
        for _ in range(3):
            visualizer.poll_events()
            visualizer.update_renderer()
        visualizer.capture_screen_image(str(path), do_render=True)
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Open3D failed to write screenshot: {path}")
    finally:
        visualizer.destroy_window()


def configure_map_view(visualizer, points: np.ndarray) -> None:
    """Set a stable Open3D camera view around the map."""
    point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if point_array.size == 0:
        center = np.zeros(3, dtype=np.float64)
    else:
        center = point_array.mean(axis=0)

    view = visualizer.get_view_control()
    view.set_lookat(center.tolist())
    view.set_front([0.55, -0.72, 0.42])
    view.set_up([0.0, 0.0, 1.0])
    view.set_zoom(0.72)


def visualize(
    model_path: Path,
    *,
    scale_report: Path | None,
    save_image: Path | None,
    camera_scale: float,
    max_cameras: int,
    min_track_length: int,
    max_reprojection_error: float,
    point_size: float,
    remove_outliers: bool,
    summary_only: bool,
) -> None:
    pycolmap_module = require_pycolmap()
    if not model_path.exists():
        raise FileNotFoundError(f"COLMAP model path does not exist: {model_path}")

    reconstruction = pycolmap_module.Reconstruction(model_path)
    metric_scale: float | None = None
    marker_frame_applied = False
    if scale_report is not None:
        if not scale_report.exists():
            raise FileNotFoundError(
                f"Scale recovery report does not exist: {scale_report}"
            )
        metric_scale, rotation, translation, marker_frame_applied = (
            similarity_from_report(scale_report)
        )
        apply_report_similarity(reconstruction, metric_scale, rotation, translation)

    if summary_only and save_image is None:
        summarize_reconstruction(
            reconstruction,
            metric_scale=metric_scale,
            marker_frame_applied=marker_frame_applied,
        )
        return

    open3d_module = require_open3d()
    point_cloud = point_cloud_from_reconstruction(
        reconstruction,
        min_track_length=min_track_length,
        max_reprojection_error=max_reprojection_error,
        remove_outliers=remove_outliers,
    )
    centers = camera_centers(registered_images(reconstruction))
    scale = camera_scale if camera_scale > 0.0 else auto_camera_scale(point_cloud, centers)
    camera_geometries, centers = camera_geometries_from_reconstruction(
        reconstruction,
        scale=scale,
        max_cameras=max_cameras,
    )
    geometries: list[o3d.geometry.Geometry] = [point_cloud]
    geometries.extend(camera_geometries)
    geometries.append(
        open3d_module.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    )
    view_points = np.vstack(
        [
            array
            for array in (np.asarray(point_cloud.points), centers, np.zeros((1, 3)))
            if array.size
        ]
    )

    summarize_reconstruction(
        reconstruction,
        visualized_points=len(point_cloud.points),
        registered_cameras=centers.shape[0],
        camera_scale=scale,
        metric_scale=metric_scale,
        marker_frame_applied=marker_frame_applied,
    )
    if save_image is not None:
        save_map_image(
            save_image,
            geometries,
            view_points,
            point_size=point_size,
        )
        print(f"saved map image: {save_image}")
    if summary_only:
        return

    visualizer = open3d_module.visualization.Visualizer()
    window_name = (
        "COLMAP metric sparse map"
        if metric_scale is not None
        else "COLMAP raw sparse map"
    )
    if not visualizer.create_window(window_name=window_name):
        raise RuntimeError("Open3D failed to create a visualization window")
    try:
        for geometry in geometries:
            visualizer.add_geometry(geometry)

        render_options = visualizer.get_render_option()
        render_options.point_size = point_size
        render_options.background_color = np.asarray([0.02, 0.02, 0.025])

        visualizer.run()
    finally:
        visualizer.destroy_window()


def summarize_reconstruction(
    reconstruction,
    *,
    visualized_points: int | None = None,
    registered_cameras: int | None = None,
    camera_scale: float | None = None,
    metric_scale: float | None,
    marker_frame_applied: bool,
) -> None:
    print(reconstruction.summary())
    if visualized_points is None:
        print(f"sparse points: {reconstruction.num_points3D()}")
    else:
        print(f"visualized points: {visualized_points}")
    if registered_cameras is None:
        print(f"registered cameras: {reconstruction.num_reg_images()}")
    else:
        print(f"registered cameras: {registered_cameras}")
    if camera_scale is not None:
        print(f"camera frustum scale: {camera_scale:.6g}")
    if metric_scale is None:
        print("warning: raw monocular COLMAP model is not metric.")
    else:
        print(f"metric scale applied: {metric_scale:.12g} m / COLMAP unit")
        if marker_frame_applied:
            print("frame: external metric report")
        else:
            print(
                "warning: report has no marker-frame transform; map is metric "
                "scale but still in COLMAP's coordinate frame."
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a COLMAP sparse model with Open3D."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"COLMAP model directory, binary or text (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--scale-report",
        type=Path,
        default=None,
        help=(
            "Scale recovery JSON report. When provided, the model is "
            "transformed in-memory before visualization."
        ),
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="Save an Open3D PNG/JPG map screenshot to this path.",
    )
    parser.add_argument(
        "--camera-scale",
        type=float,
        default=0.0,
        help="Camera frustum scale; 0 chooses an automatic scale (default: 0)",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=120,
        help="Maximum camera frustums to draw; 0 draws all cameras (default: 120)",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=3,
        help="Minimum COLMAP track length for displayed 3D points (default: 3)",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=0.0,
        help="Drop points above this error in px; 0 disables the filter (default: 0)",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Open3D point render size (default: 2.0)",
    )
    parser.add_argument(
        "--keep-outliers",
        action="store_true",
        help="Disable Open3D statistical outlier removal on displayed points.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Load and summarize the model without opening an Open3D window.",
    )
    args = parser.parse_args(argv)
    if args.camera_scale < 0.0:
        parser.error("--camera-scale must be >= 0")
    if args.max_cameras < 0:
        parser.error("--max-cameras must be >= 0")
    if args.min_track_length < 1:
        parser.error("--min-track-length must be >= 1")
    if args.max_reprojection_error < 0.0:
        parser.error("--max-reprojection-error must be >= 0")
    if args.point_size <= 0.0:
        parser.error("--point-size must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    visualize(
        args.model,
        scale_report=args.scale_report,
        save_image=args.save_image,
        camera_scale=args.camera_scale,
        max_cameras=args.max_cameras,
        min_track_length=args.min_track_length,
        max_reprojection_error=args.max_reprojection_error,
        point_size=args.point_size,
        remove_outliers=not args.keep_outliers,
        summary_only=args.summary_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
