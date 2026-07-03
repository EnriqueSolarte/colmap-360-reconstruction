"""Attach native COLMAP panorama poses to flat rendered pinhole selections."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from colmap.camera import camera_from_fov
from colmap.model_io import is_colmap_model_dir
from render_pinhole.geometry import pinhole_camera_pose_report

try:  # Keep import usable in test/dev environments without pycolmap.
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None

POSE_REPORT_FILENAME = "pose_report.json"
DEFAULT_OUTPUT_DIR_NAME = "posed-renders"
POSE_STATUS_MATCHED = "matched"
POSE_STATUS_MISSING = "missing"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    selections_store = _read_json(args.selections)
    render_report = _read_json(args.render_report)
    output_dir = args.output_dir or args.render_report.parent / DEFAULT_OUTPUT_DIR_NAME
    try:
        reconstruction = _load_reconstruction(args.sparse_best)
        pose_report = attach_pinhole_poses(
            selections_store=selections_store,
            render_report=render_report,
            selections_path=args.selections,
            render_report_path=args.render_report,
            sparse_best_path=args.sparse_best,
            output_dir=output_dir,
            reconstruction=reconstruction,
            show_progress=True,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(error)
        return 1

    pose_report_path = output_dir / POSE_REPORT_FILENAME
    export_report = write_colmap_text_export(
        pose_report=pose_report,
        render_report=render_report,
        reconstruction=reconstruction,
        output_dir=output_dir,
    )
    pose_report["colmap_text_export"] = export_report
    _write_json(pose_report_path, pose_report)
    print(f"Pose-attached views: {pose_report['pose_attached_view_count']}")
    if pose_report["missing_pose_view_count"]:
        print(f"Missing poses: {pose_report['missing_pose_view_count']}")
    print(f"Pose report: {pose_report_path}")
    if pose_report["missing_pose_view_count"] and not args.allow_missing_poses:
        return 1
    return 0


def attach_pinhole_poses(
    *,
    selections_store: dict[str, Any],
    render_report: dict[str, Any],
    selections_path: Path,
    render_report_path: Path,
    sparse_best_path: Path,
    output_dir: Path,
    reconstruction: Any,
    show_progress: bool = False,
) -> dict[str, Any]:
    selections = list(selections_store.get("selections", []))
    renders = list(render_report.get("renders", []))
    selection_by_id = {
        str(selection["selection_id"]): selection
        for selection in selections
        if selection.get("selection_id")
    }
    registered_poses = registered_native_panorama_poses(reconstruction)
    views = []
    progress = tqdm(
        renders,
        desc="Attaching pinhole poses",
        unit="view",
        disable=not show_progress,
    )
    for render_record in progress:
        selection_id = str(render_record.get("selection_id", ""))
        selection = selection_by_id.get(selection_id)
        if selection is None:
            views.append(
                {
                    "selection_id": selection_id or None,
                    "pano_image_name": render_record.get("pano_image_name"),
                    "output_file": render_record.get("output_file"),
                    "pose_status": POSE_STATUS_MISSING,
                    "missing_reason": "selection record not found",
                }
            )
            continue
        pano_image_name = str(selection.get("pano_image_name"))
        pano_pose = lookup_panorama_pose(registered_poses, pano_image_name)
        if pano_pose is None:
            views.append(
                {
                    "selection_id": selection_id,
                    "pano_image_name": pano_image_name,
                    "output_file": render_record.get("output_file"),
                    "pose_status": POSE_STATUS_MISSING,
                    "missing_reason": "panorama pose not registered in COLMAP model",
                }
            )
            continue
        pinhole_pose = pinhole_camera_pose_report(
            pano_pose,
            float(selection["longitude"]),
            float(selection["latitude"]),
        )
        views.append(
            {
                "selection_id": selection_id,
                "pano_image_name": pano_image_name,
                "output_file": render_record.get("output_file"),
                "output_name": render_record.get("output_name"),
                "pose_status": POSE_STATUS_MATCHED,
                "source_colmap_image_name": pano_pose["source_image_name"],
                "pinhole_camera_pose": pinhole_pose,
            }
        )

    pose_attached_count = sum(
        1 for view in views if view.get("pose_status") == POSE_STATUS_MATCHED
    )
    missing_pose_count = sum(
        1 for view in views if view.get("pose_status") == POSE_STATUS_MISSING
    )
    return {
        "schema_version": 1,
        "source": {
            "selections_path": str(selections_path),
            "render_report_path": str(render_report_path),
            "sparse_best_path": str(sparse_best_path),
        },
        "output": {
            "output_dir": str(output_dir),
            "pose_report": str(output_dir / POSE_REPORT_FILENAME),
            "sparse_dir": str(output_dir / "sparse" / "0"),
        },
        "selected_view_count": len(selections),
        "rendered_view_count": len(renders),
        "pose_attached_view_count": pose_attached_count,
        "missing_pose_view_count": missing_pose_count,
        "registered_panorama_count": registered_native_panorama_count(reconstruction),
        "views": views,
    }


def write_colmap_text_export(
    *,
    pose_report: dict[str, Any],
    render_report: dict[str, Any],
    reconstruction: Any,
    output_dir: Path,
) -> dict[str, Any]:
    sparse_dir = output_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    image_records, camera_records = colmap_image_and_camera_records(
        pose_report=pose_report,
        render_report=render_report,
    )
    point_records = colmap_point_records(reconstruction)
    write_cameras_text(sparse_dir / "cameras.txt", camera_records)
    write_images_text(sparse_dir / "images.txt", image_records)
    write_points3d_text(sparse_dir / "points3D.txt", point_records)
    return {
        "status": "exported",
        "format": "colmap_text",
        "coordinate_frame": "colmap_world_unscaled",
        "sparse_dir": str(sparse_dir),
        "camera_count": len(camera_records),
        "image_count": len(image_records),
        "point3D_count": len(point_records),
        "points3D_tracks": "omitted_for_derived_pinhole_views",
    }


def colmap_image_and_camera_records(
    *,
    pose_report: dict[str, Any],
    render_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    render_by_selection_id = {
        str(record["selection_id"]): record
        for record in render_report.get("renders", [])
        if record.get("selection_id")
    }
    camera_ids: dict[tuple[Any, ...], int] = {}
    camera_records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    for view in pose_report.get("views", []):
        if view.get("pose_status") != POSE_STATUS_MATCHED:
            continue
        render_record = render_by_selection_id.get(str(view.get("selection_id")))
        if render_record is None:
            raise ValueError(
                f"Missing render record for selection {view.get('selection_id')}"
            )
        camera = camera_record_from_render(render_record)
        key = camera_record_key(camera)
        if key not in camera_ids:
            camera_ids[key] = len(camera_records) + 1
            camera["camera_id"] = camera_ids[key]
            camera_records.append(camera)
        image_records.append(
            image_record_from_pose_view(
                image_id=len(image_records) + 1,
                camera_id=camera_ids[key],
                view=view,
            )
        )
    return image_records, camera_records


def camera_record_from_render(render_record: dict[str, Any]) -> dict[str, Any]:
    camera = camera_from_fov(
        width=int(render_record["width"]),
        height=int(render_record["height"]),
        fov_x=float(render_record["fov_x"]),
        fov_y=float(render_record["fov_y"]),
    )
    return {
        "model": camera.model,
        "width": camera.width,
        "height": camera.height,
        "params": camera.params,
    }


def camera_record_key(camera: dict[str, Any]) -> tuple[Any, ...]:
    return (
        camera["model"],
        int(camera["width"]),
        int(camera["height"]),
        tuple(round(float(value), 12) for value in camera["params"]),
    )


def image_record_from_pose_view(
    *,
    image_id: int,
    camera_id: int,
    view: dict[str, Any],
) -> dict[str, Any]:
    pose = view["pinhole_camera_pose"]
    rotation_world_from_pinhole = np.asarray(
        pose["rotation_world_from_pinhole"],
        dtype=np.float64,
    )
    center_world = np.asarray(pose["center_world"], dtype=np.float64)
    rotation_cam_from_world = rotation_world_from_pinhole.T
    translation = -rotation_cam_from_world @ center_world
    output_name = view.get("output_name") or Path(str(view["output_file"])).name
    return {
        "image_id": image_id,
        "qvec": quaternion_from_rotation(rotation_cam_from_world),
        "tvec": tuple(float(value) for value in translation),
        "camera_id": camera_id,
        "image_name": output_name,
    }


def registered_native_panorama_count(reconstruction: Any) -> int:
    if hasattr(reconstruction, "num_reg_images"):
        return int(reconstruction.num_reg_images())
    return len(list(reconstruction.reg_image_ids()))


def registered_native_panorama_poses(reconstruction: Any) -> dict[str, dict[str, Any]]:
    poses: dict[str, dict[str, Any]] = {}
    for image_id in sorted(int(value) for value in reconstruction.reg_image_ids()):
        image = reconstruction.image(image_id)
        pose = native_panorama_pose_record(image)
        for key in panorama_match_keys(image.name):
            poses.setdefault(key, pose)
    return poses


def native_panorama_pose_record(image: Any) -> dict[str, Any]:
    pose_cam_from_world = image.cam_from_world()
    rotation_cam_from_world = np.asarray(
        pose_cam_from_world.rotation.matrix(),
        dtype=np.float64,
    )
    rotation_world_from_pano = rotation_cam_from_world.T
    center_world = np.asarray(image.projection_center(), dtype=np.float64)
    transform_world_from_pano = np.eye(4, dtype=np.float64)
    transform_world_from_pano[:3, :3] = rotation_world_from_pano
    transform_world_from_pano[:3, 3] = center_world
    return {
        "coordinate_frame": "colmap_world_unscaled",
        "source_image_name": image.name,
        "center_world": center_world.tolist(),
        "rotation_world_from_pano": rotation_world_from_pano.tolist(),
        "transform_world_from_pano": transform_world_from_pano.tolist(),
    }


def lookup_panorama_pose(
    registered_poses: dict[str, dict[str, Any]],
    pano_image_name: str,
) -> dict[str, Any] | None:
    for key in panorama_match_keys(pano_image_name):
        pose = registered_poses.get(key)
        if pose is not None:
            return pose
    return None


def panorama_match_keys(image_name: str) -> tuple[str, ...]:
    path = Path(image_name)
    keys = [image_name, path.as_posix(), path.name]
    return tuple(dict.fromkeys(key for key in keys if key))


def colmap_point_records(reconstruction: Any) -> list[dict[str, Any]]:
    points = getattr(reconstruction, "points3D", {})
    return [
        {
            "point3D_id": int(point_id),
            "xyz": tuple(float(value) for value in point.xyz),
            "rgb": tuple(int(value) for value in point.color),
            "error": float(point.error),
        }
        for point_id, point in sorted(points.items())
    ]


def write_cameras_text(path: Path, cameras: list[dict[str, Any]]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(cameras)}",
    ]
    for camera in cameras:
        params = " ".join(format_float(value) for value in camera["params"])
        lines.append(
            f"{camera['camera_id']} {camera['model']} "
            f"{camera['width']} {camera['height']} {params}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_text(path: Path, images: list[dict[str, Any]]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(images)}, mean observations per image: 0",
    ]
    for image in images:
        qvec = " ".join(format_float(value) for value in image["qvec"])
        tvec = " ".join(format_float(value) for value in image["tvec"])
        lines.append(
            f"{image['image_id']} {qvec} {tvec} "
            f"{image['camera_id']} {image['image_name']}"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_points3d_text(path: Path, points: list[dict[str, Any]]) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
        f"# Number of points: {len(points)}, mean track length: 0",
    ]
    for point in points:
        xyz = " ".join(format_float(value) for value in point["xyz"])
        rgb = " ".join(str(int(value)) for value in point["rgb"])
        lines.append(
            f"{point['point3D_id']} {xyz} {rgb} {format_float(point['error'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quaternion_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach native COLMAP panorama poses to rendered pinhole views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-s",
        "--selections",
        required=True,
        type=Path,
        help="Flat selections.json path.",
    )
    parser.add_argument(
        "--render-report",
        required=True,
        type=Path,
        help="render_report.json path from manual-pinhole-render-selected.",
    )
    parser.add_argument(
        "--sparse-best",
        required=True,
        type=Path,
        help="Native equirectangular COLMAP sparse_best/model directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory where pose_report.json will be written.",
    )
    parser.add_argument(
        "--allow-missing-poses",
        action="store_true",
        help="Exit successfully even when some rendered views have no COLMAP pose.",
    )
    args = parser.parse_args(argv)
    if not args.selections.is_file():
        parser.error(f"--selections must be an existing file: {args.selections}")
    if not args.render_report.is_file():
        parser.error(
            f"--render-report must be an existing file: {args.render_report}"
        )
    if not is_colmap_model_dir(args.sparse_best):
        parser.error(f"--sparse-best must be a readable COLMAP model: {args.sparse_best}")
    return args


def _load_reconstruction(model_path: Path) -> Any:
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required to read --sparse-best. Install the `colmap` extra."
        )
    return pycolmap.Reconstruction(model_path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
