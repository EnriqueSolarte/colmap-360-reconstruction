"""Initial COLMAP model writing from panorama-derived pinhole poses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from colmap.camera import ColmapCamera
from colmap.panorama.types import GeneratedImageBatch
from colmap.pinhole.image_import import ImportedPinholeImage

INITIAL_POSE_SOURCE = "panorama_pose_report"
INITIAL_POSE_TYPE = "full_6dof"


@dataclass(frozen=True)
class PinholeInitialPose:
    selection_id: str
    output_name: str
    image_id: int
    camera_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    center_world: tuple[float, float, float]
    coordinate_frame: str
    source_colmap_image_name: str | None = None
    pano_image_name: str | None = None

    def report(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "output_name": self.output_name,
            "image_id": self.image_id,
            "camera_id": self.camera_id,
            "qvec": list(self.qvec),
            "tvec": list(self.tvec),
            "center_world": list(self.center_world),
            "coordinate_frame": self.coordinate_frame,
            "source_colmap_image_name": self.source_colmap_image_name,
            "pano_image_name": self.pano_image_name,
        }


@dataclass(frozen=True)
class PinholeInitialPoseSummary:
    pose_report_path: Path
    model_path: Path
    coordinate_frame: str | None
    poses: tuple[PinholeInitialPose, ...]
    missing: tuple[dict[str, Any], ...]
    unusable: tuple[dict[str, Any], ...]

    def report(self) -> dict[str, Any]:
        return {
            "source": INITIAL_POSE_SOURCE,
            "type": INITIAL_POSE_TYPE,
            "coordinate_frame": self.coordinate_frame,
            "pose_report_path": str(self.pose_report_path),
            "model_path": str(self.model_path),
            "initialized_count": len(self.poses),
            "missing_count": len(self.missing),
            "unusable_count": len(self.unusable),
            "fixed_poses": False,
            "refine_poses": True,
            "poses": [pose.report() for pose in self.poses],
            "missing": list(self.missing),
            "unusable": list(self.unusable),
        }


def write_initial_pose_model_from_pose_report(
    *,
    pose_report_path: Path,
    model_path: Path,
    imported_images: Sequence[ImportedPinholeImage],
    generated_image_batch: GeneratedImageBatch,
    camera: ColmapCamera,
) -> PinholeInitialPoseSummary:
    pose_report = json.loads(pose_report_path.read_text(encoding="utf-8"))
    views = pose_report.get("views")
    if not isinstance(views, list):
        raise ValueError("pose_report.json must contain a views array")
    by_selection_id, by_output_name = pose_views_by_identity(views)

    poses: list[PinholeInitialPose] = []
    missing: list[dict[str, Any]] = []
    unusable: list[dict[str, Any]] = []
    coordinate_frame: str | None = None
    for image in imported_images:
        render_record = image.record
        database_record = generated_image_batch.records_by_name.get(
            image.colmap_image_name,
        )
        if database_record is None:
            unusable.append(
                {
                    "selection_id": render_record.selection_id,
                    "output_name": render_record.output_name,
                    "reason": "image is missing from database batch",
                },
            )
            continue
        view = by_selection_id.get(render_record.selection_id)
        matched_by = "selection_id"
        if view is None:
            view = by_output_name.get(render_record.output_name)
            matched_by = "output_name"
        if view is None:
            missing.append(
                {
                    "selection_id": render_record.selection_id,
                    "output_name": render_record.output_name,
                    "reason": "pose view not found",
                },
            )
            continue
        pose, reason = initial_pose_from_view(
            view,
            fallback_selection_id=render_record.selection_id,
            fallback_output_name=render_record.output_name,
            image_id=database_record[0],
            camera_id=database_record[1],
        )
        if pose is None:
            unusable.append(
                {
                    "selection_id": render_record.selection_id,
                    "output_name": render_record.output_name,
                    "matched_by": matched_by,
                    "reason": reason,
                },
            )
            continue
        poses.append(pose)
        if coordinate_frame is None:
            coordinate_frame = pose.coordinate_frame
    if not poses:
        raise ValueError("No usable initial poses are available")

    write_colmap_text_model(
        model_path,
        camera=camera,
        camera_ids=tuple(sorted(set(generated_image_batch.camera_ids))),
        poses=tuple(poses),
    )
    return PinholeInitialPoseSummary(
        pose_report_path=pose_report_path,
        model_path=model_path,
        coordinate_frame=coordinate_frame,
        poses=tuple(poses),
        missing=tuple(missing),
        unusable=tuple(unusable),
    )


def initial_pose_from_view(
    view: dict[str, Any],
    *,
    fallback_selection_id: str,
    fallback_output_name: str,
    image_id: int,
    camera_id: int,
) -> tuple[PinholeInitialPose | None, str | None]:
    if view.get("pose_status") not in (None, "matched"):
        return None, f"pose_status is {view.get('pose_status')}"
    pose = view.get("pinhole_camera_pose")
    if not isinstance(pose, dict):
        return None, "pinhole_camera_pose is missing"
    center = pose.get("center_world")
    rotation = pose.get("rotation_world_from_pinhole")
    if not isinstance(center, list) or len(center) != 3:
        return None, "center_world must contain three values"
    if not isinstance(rotation, list):
        return None, "rotation_world_from_pinhole is missing"
    try:
        center_world = np.asarray(center, dtype=np.float64)
        rotation_world_from_pinhole = np.asarray(rotation, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "pose values must be numeric"
    if rotation_world_from_pinhole.shape != (3, 3):
        return None, "rotation_world_from_pinhole must be 3x3"
    rotation_cam_from_world = rotation_world_from_pinhole.T
    translation = -rotation_cam_from_world @ center_world
    coordinate_frame = optional_string(pose.get("coordinate_frame"))
    if coordinate_frame is None:
        coordinate_frame = "colmap_world_unscaled"
    return (
        PinholeInitialPose(
            selection_id=optional_string(view.get("selection_id"))
            or fallback_selection_id,
            output_name=optional_string(view.get("output_name"))
            or fallback_output_name,
            image_id=int(image_id),
            camera_id=int(camera_id),
            qvec=quaternion_from_rotation(rotation_cam_from_world),
            tvec=tuple(float(value) for value in translation),
            center_world=tuple(float(value) for value in center_world),
            coordinate_frame=coordinate_frame,
            source_colmap_image_name=optional_string(
                view.get("source_colmap_image_name"),
            ),
            pano_image_name=optional_string(view.get("pano_image_name")),
        ),
        None,
    )


def pose_views_by_identity(
    views: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_selection_id: dict[str, dict[str, Any]] = {}
    by_output_name: dict[str, dict[str, Any]] = {}
    for view in views:
        selection_id = optional_string(view.get("selection_id"))
        output_name = optional_string(view.get("output_name"))
        if selection_id is not None:
            by_selection_id.setdefault(selection_id, view)
        if output_name is not None:
            by_output_name.setdefault(output_name, view)
    return by_selection_id, by_output_name


def write_colmap_text_model(
    model_path: Path,
    *,
    camera: ColmapCamera,
    camera_ids: tuple[int, ...],
    poses: tuple[PinholeInitialPose, ...],
) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    write_cameras_text(model_path / "cameras.txt", camera, camera_ids)
    write_images_text(model_path / "images.txt", poses)
    write_points3d_text(model_path / "points3D.txt")


def write_cameras_text(
    path: Path,
    camera: ColmapCamera,
    camera_ids: tuple[int, ...],
) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(camera_ids)}",
    ]
    params = " ".join(format_float(value) for value in camera.params)
    for camera_id in camera_ids:
        lines.append(
            f"{camera_id} {camera.model} {camera.width} {camera.height} {params}",
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_images_text(path: Path, poses: tuple[PinholeInitialPose, ...]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(poses)}, mean observations per image: 0",
    ]
    for pose in poses:
        qvec = " ".join(format_float(value) for value in pose.qvec)
        tvec = " ".join(format_float(value) for value in pose.tvec)
        lines.append(
            f"{pose.image_id} {qvec} {tvec} {pose.camera_id} {pose.output_name}",
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_points3d_text(path: Path) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]",
        "# Number of points: 0, mean track length: 0",
    ]
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


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
