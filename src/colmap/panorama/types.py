"""Shared data types for panorama reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def round_report_float(value: float) -> float:
    return round(float(value), 6)


def optional_report_float(value: float | None) -> float | None:
    return None if value is None else round_report_float(value)


@dataclass(frozen=True)
class FeaturePipelineOptions:
    extractor: str = "sift"
    matcher: str = "sift-bruteforce"
    max_num_features: int = 8192
    guided_matching: bool = False


@dataclass(frozen=True)
class PipelineExecutionOptions:
    device: str = "auto"
    gpu_index: str = "-1"
    gpu_bundle_adjustment: bool = True


@dataclass(frozen=True)
class GeneratedImageBatch:
    names: tuple[str, ...]
    records_by_name: dict[str, tuple[int, int]]

    @property
    def image_ids(self) -> list[int]:
        return [self.records_by_name[name][0] for name in self.names]

    @property
    def camera_ids(self) -> list[int]:
        return [self.records_by_name[name][1] for name in self.names]


@dataclass(frozen=True)
class KeyframeAnchorScore:
    frame_id: int
    anchor_score: float
    connectivity_score: float
    triangulated_support_score: float
    image_distribution_score: float
    reprojection_quality_score: float
    connected_frame_count: int
    verified_inlier_count: int
    triangulated_point_count: int
    active_image_count: int
    total_image_count: int
    active_image_ratio: float
    image_entropy: float
    median_reprojection_error_px: float | None
    mean_reprojection_error_px: float | None
    image_ids: tuple[int, ...]
    image_names: tuple[str, ...]
    triangulated_points_by_image: tuple[int, ...]

    def report(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "anchor_score": round_report_float(self.anchor_score),
            "connectivity_score": round_report_float(self.connectivity_score),
            "triangulated_support_score": round_report_float(
                self.triangulated_support_score,
            ),
            "image_distribution_score": round_report_float(
                self.image_distribution_score,
            ),
            "reprojection_quality_score": round_report_float(
                self.reprojection_quality_score,
            ),
            "connected_frame_count": self.connected_frame_count,
            "verified_inlier_count": self.verified_inlier_count,
            "triangulated_point_count": self.triangulated_point_count,
            "active_image_count": self.active_image_count,
            "total_image_count": self.total_image_count,
            "active_image_ratio": round_report_float(self.active_image_ratio),
            "image_entropy": round_report_float(self.image_entropy),
            "median_reprojection_error_px": optional_report_float(
                self.median_reprojection_error_px,
            ),
            "mean_reprojection_error_px": optional_report_float(
                self.mean_reprojection_error_px,
            ),
            "image_ids": list(self.image_ids),
            "image_names": list(self.image_names),
            "triangulated_points_by_image": list(self.triangulated_points_by_image),
        }
