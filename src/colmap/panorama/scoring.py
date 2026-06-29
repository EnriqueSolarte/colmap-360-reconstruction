"""Keyframe anchor scoring for panorama reconstruction."""

from __future__ import annotations

from math import log
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from colmap.panorama.database import read_frame_connection_stats
from colmap.panorama.types import KeyframeAnchorScore


def log_capped_score(value: int | float, cap: int | float) -> float:
    if value <= 0 or cap <= 0:
        return 0.0
    return min(log(1.0 + float(value)) / log(1.0 + float(cap)), 1.0)


def linear_capped_score(value: int | float, cap: int | float) -> float:
    if value <= 0 or cap <= 0:
        return 0.0
    return min(float(value) / float(cap), 1.0)


def normalized_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    positive_counts = [count for count in counts if count > 0]
    if total <= 0 or len(positive_counts) <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * log(count / total)
        for count in positive_counts
    )
    return entropy / log(len(counts))


def score_keyframe_anchors(
    database_path: Path,
    reconstruction: Any,
    *,
    min_active_image_points: int = 30,
    connectivity_frame_cap: int = 10,
    verified_inlier_cap: int = 5000,
    triangulated_point_cap: int = 3000,
    reprojection_error_scale_px: float = 2.0,
) -> list[KeyframeAnchorScore]:
    frames: dict[int, list[Any]] = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        frame_id = colmap_frame_id_for_image(image)
        frames.setdefault(frame_id, []).append(image)

    frame_id_by_image_id = {
        int(image.image_id): frame_id
        for frame_id, images in frames.items()
        for image in images
    }
    connection_stats = read_frame_connection_stats(database_path, frame_id_by_image_id)
    scores = []
    for frame_id, images in frames.items():
        ordered_images = sorted(images, key=lambda image: int(image.image_id))
        triangulated_points_by_image = []
        point_ids: set[int] = set()
        for image in ordered_images:
            image_point_ids = {
                int(point2d.point3D_id)
                for point2d in image.points2D
                if point2d.has_point3D()
            }
            triangulated_points_by_image.append(len(image_point_ids))
            point_ids.update(image_point_ids)

        point_errors = [
            float(reconstruction.points3D[point_id].error)
            for point_id in point_ids
            if point_id in reconstruction.points3D
            and reconstruction.points3D[point_id].has_error()
        ]
        median_error = (
            None if not point_errors else float(np.median(np.asarray(point_errors)))
        )
        mean_error = (
            None if not point_errors else float(np.mean(np.asarray(point_errors)))
        )
        active_image_count = sum(
            count >= min_active_image_points
            for count in triangulated_points_by_image
        )
        total_image_count = max(len(ordered_images), 1)
        active_image_ratio = min(active_image_count / total_image_count, 1.0)
        image_entropy = normalized_entropy(triangulated_points_by_image)
        image_distribution_score = 0.5 * active_image_ratio + 0.5 * image_entropy

        frame_connection_stats = connection_stats.get(
            frame_id,
            {"connected_frame_ids": set(), "verified_inlier_count": 0},
        )
        connected_frame_ids = frame_connection_stats["connected_frame_ids"]
        if not isinstance(connected_frame_ids, set):
            raise AssertionError("invalid connected frame set")
        connected_frame_count = len(connected_frame_ids)
        verified_inlier_count = int(frame_connection_stats["verified_inlier_count"])
        connectivity_score = 0.5 * linear_capped_score(
            connected_frame_count,
            connectivity_frame_cap,
        ) + 0.5 * log_capped_score(verified_inlier_count, verified_inlier_cap)
        triangulated_support_score = log_capped_score(
            len(point_ids),
            triangulated_point_cap,
        )
        reprojection_quality_score = (
            0.0
            if median_error is None
            else 1.0 / (1.0 + median_error / reprojection_error_scale_px)
        )
        anchor_score = (
            connectivity_score
            * triangulated_support_score
            * image_distribution_score
            * reprojection_quality_score
        )
        scores.append(
            KeyframeAnchorScore(
                frame_id=frame_id,
                anchor_score=anchor_score,
                connectivity_score=connectivity_score,
                triangulated_support_score=triangulated_support_score,
                image_distribution_score=image_distribution_score,
                reprojection_quality_score=reprojection_quality_score,
                connected_frame_count=connected_frame_count,
                verified_inlier_count=verified_inlier_count,
                triangulated_point_count=len(point_ids),
                active_image_count=active_image_count,
                total_image_count=total_image_count,
                active_image_ratio=active_image_ratio,
                image_entropy=image_entropy,
                median_reprojection_error_px=median_error,
                mean_reprojection_error_px=mean_error,
                image_ids=tuple(int(image.image_id) for image in ordered_images),
                image_names=tuple(str(image.name) for image in ordered_images),
                triangulated_points_by_image=tuple(triangulated_points_by_image),
            )
        )
    return sorted(
        scores,
        key=lambda score: (
            score.anchor_score,
            score.connected_frame_count,
            score.triangulated_point_count,
            -score.frame_id,
        ),
        reverse=True,
    )


def colmap_frame_id_for_image(image: Any) -> int:
    frame_id = int(getattr(image, "frame_id", -1))
    if frame_id >= 0:
        return frame_id
    return int(image.image_id)
