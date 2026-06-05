"""Visualize reconstruction output from legacy_pano_reconstruction.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from colmap.visualize_map import (
    auto_camera_scale,
    make_trajectory_lineset,
    point_cloud_from_reconstruction,
    require_open3d,
    require_pycolmap,
    save_map_image,
)

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - environment dependent
    go = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "pano2"
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def discover_models(workspace: Path) -> list[Path]:
    sparse_dir = workspace / "sparse"
    if not sparse_dir.is_dir():
        return []
    return [
        path
        for path in sorted(sparse_dir.iterdir())
        if path.is_dir()
        and all((path / filename).is_file() for filename in MODEL_FILES)
    ]


def select_model(workspace: Path, model_path: Path | None = None) -> tuple[Path, Any]:
    pycolmap = require_pycolmap()
    if model_path is not None:
        if not model_path.is_dir():
            raise FileNotFoundError(f"COLMAP model directory not found: {model_path}")
        return model_path, pycolmap.Reconstruction(model_path)

    candidates = discover_models(workspace)
    if not candidates:
        raise FileNotFoundError(
            f"No COLMAP models found under legacy workspace: {workspace / 'sparse'}"
        )

    reconstructions = [
        (path, pycolmap.Reconstruction(path)) for path in candidates
    ]
    return max(
        reconstructions,
        key=lambda item: (
            item[1].num_reg_frames(),
            item[1].num_points3D(),
            item[1].num_reg_images(),
        ),
    )


def registered_frames(reconstruction: Any) -> list[Any]:
    return [
        reconstruction.frame(frame_id)
        for frame_id in sorted(reconstruction.reg_frame_ids())
    ]


def frame_centers(frames: list[Any]) -> np.ndarray:
    if not frames:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(
        [
            np.asarray(frame.rig_from_world.inverse().translation, dtype=np.float64)
            for frame in frames
        ]
    )


def frame_geometries(
    frames: list[Any],
    centers: np.ndarray,
    *,
    scale: float,
    max_frames: int,
) -> list[Any]:
    open3d = require_open3d()
    if max_frames > 0 and len(frames) > max_frames:
        sample_indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        displayed_frames = [frames[index] for index in sample_indices]
    else:
        displayed_frames = frames

    geometries: list[Any] = []
    for frame in displayed_frames:
        world_from_rig = frame.rig_from_world.inverse()
        axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            world_from_rig.rotation.matrix(), dtype=np.float64
        )
        transform[:3, 3] = np.asarray(
            world_from_rig.translation, dtype=np.float64
        )
        axis.transform(transform)
        geometries.append(axis)

    if centers.shape[0] >= 2:
        geometries.append(make_trajectory_lineset(centers))
    return geometries


def summarize(
    reconstruction: Any,
    model_path: Path,
    *,
    visualized_points: int | None = None,
) -> None:
    print(f"selected model: {model_path}")
    print(reconstruction.summary())
    print(f"registered panorama frames: {reconstruction.num_reg_frames()}")
    print(f"registered virtual-camera images: {reconstruction.num_reg_images()}")
    if visualized_points is not None:
        print(f"visualized sparse points: {visualized_points}")
    print("warning: raw monocular COLMAP model is not metric.")


def require_plotly():
    if go is None:
        raise RuntimeError(
            "Plotly is required for HTML visualization. Install the "
            "`visualize` extra or add `plotly` to the environment."
        )
    return go


def save_html_map(
    path: Path,
    point_cloud: Any,
    centers: np.ndarray,
    *,
    max_points: int,
    point_size: float,
) -> None:
    plotly = require_plotly()
    xyz = np.asarray(point_cloud.points)
    rgb = np.asarray(point_cloud.colors)
    if max_points > 0 and xyz.shape[0] > max_points:
        indices = np.linspace(0, xyz.shape[0] - 1, max_points, dtype=int)
        xyz = xyz[indices]
        if rgb.shape[0]:
            rgb = rgb[indices]

    traces: list[Any] = []
    if xyz.shape[0]:
        colors = None
        if rgb.shape == xyz.shape:
            colors = [
                f"rgb({red},{green},{blue})"
                for red, green, blue in np.clip(rgb * 255.0, 0, 255).astype(int)
            ]
        traces.append(
            plotly.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name="sparse points",
                marker={
                    "size": point_size,
                    "color": colors or "#d8dee9",
                    "opacity": 0.85,
                },
                hoverinfo="skip",
            )
        )
    if centers.shape[0]:
        traces.append(
            plotly.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="lines+markers",
                name="panorama trajectory",
                line={"color": "#19c3ff", "width": 5},
                marker={"color": "#f6c445", "size": 3},
                hovertemplate="panorama frame %{pointNumber}<extra></extra>",
            )
        )

    figure = plotly.Figure(data=traces)
    figure.update_layout(
        title="Legacy panorama COLMAP reconstruction",
        template="plotly_dark",
        scene={
            "aspectmode": "data",
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True)


def visualize(
    workspace: Path,
    *,
    model_path: Path | None,
    save_image: Path | None,
    save_html: Path | None,
    max_html_points: int,
    frame_scale: float,
    max_frames: int,
    min_track_length: int,
    max_reprojection_error: float,
    point_size: float,
    remove_outliers: bool,
    summary_only: bool,
) -> None:
    selected_path, reconstruction = select_model(workspace, model_path)
    if summary_only and save_image is None and save_html is None:
        summarize(reconstruction, selected_path)
        return

    open3d = require_open3d()
    point_cloud = point_cloud_from_reconstruction(
        reconstruction,
        min_track_length=min_track_length,
        max_reprojection_error=max_reprojection_error,
        remove_outliers=remove_outliers,
    )
    frames = registered_frames(reconstruction)
    centers = frame_centers(frames)
    scale = frame_scale if frame_scale > 0.0 else auto_camera_scale(
        point_cloud, centers
    )
    geometries: list[Any] = [point_cloud]
    geometries.extend(
        frame_geometries(
            frames,
            centers,
            scale=scale,
            max_frames=max_frames,
        )
    )
    geometries.append(
        open3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * 2.0)
    )
    point_array = np.asarray(point_cloud.points)
    view_arrays = [array for array in (point_array, centers) if array.size]
    view_points = (
        np.vstack(view_arrays)
        if view_arrays
        else np.zeros((1, 3), dtype=np.float64)
    )

    summarize(
        reconstruction,
        selected_path,
        visualized_points=len(point_cloud.points),
    )
    print(f"displayed panorama frames: {min(len(frames), max_frames or len(frames))}")
    print(f"frame axis scale: {scale:.6g}")

    if save_html is not None:
        save_html_map(
            save_html,
            point_cloud,
            centers,
            max_points=max_html_points,
            point_size=point_size,
        )
        print(f"saved interactive map: {save_html}")
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

    visualizer = open3d.visualization.Visualizer()
    if not visualizer.create_window(window_name="Legacy panorama COLMAP map"):
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


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a legacy panorama-rig COLMAP workspace with one pose "
            "per panorama frame."
        )
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help=f"Legacy reconstruction workspace (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Specific COLMAP model directory; otherwise select the best sparse model.",
    )
    parser.add_argument(
        "--save-image",
        type=Path,
        default=None,
        help="Save an Open3D PNG/JPG screenshot.",
    )
    parser.add_argument(
        "--save-html",
        type=Path,
        default=None,
        help="Save a headless-friendly interactive Plotly HTML visualization.",
    )
    parser.add_argument(
        "--max-html-points",
        type=int,
        default=100_000,
        help="Maximum sparse points in HTML output; 0 includes all (default: 100000)",
    )
    parser.add_argument(
        "--frame-scale",
        type=float,
        default=0.0,
        help="Panorama pose-axis scale; 0 chooses automatically (default: 0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum panorama pose axes to draw; 0 draws all (default: 120)",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=3,
        help="Minimum track length for displayed sparse points (default: 3)",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=0.0,
        help="Drop points above this error in px; 0 disables filtering (default: 0)",
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
        help="Disable statistical outlier removal on displayed points.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Select and summarize the model without opening Open3D.",
    )
    args = parser.parse_args(argv)
    if args.frame_scale < 0.0:
        parser.error("--frame-scale must be >= 0")
    if args.max_frames < 0:
        parser.error("--max-frames must be >= 0")
    if args.max_html_points < 0:
        parser.error("--max-html-points must be >= 0")
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
        args.workspace,
        model_path=args.model,
        save_image=args.save_image,
        save_html=args.save_html,
        max_html_points=args.max_html_points,
        frame_scale=args.frame_scale,
        max_frames=args.max_frames,
        min_track_length=args.min_track_length,
        max_reprojection_error=args.max_reprojection_error,
        point_size=args.point_size,
        remove_outliers=not args.keep_outliers,
        summary_only=args.summary_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
