"""Position-prior parsing for rendered pinhole reconstructions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from colmap.pinhole.image_import import ImportedPinholeImage

POSE_PRIOR_SOURCE = "panorama_pose_report"
POSE_PRIOR_TYPE = "camera_center"


@dataclass(frozen=True)
class PinholePositionPrior:
    selection_id: str
    output_name: str
    center_world: tuple[float, float, float]
    coordinate_frame: str
    source_colmap_image_name: str | None = None
    pano_image_name: str | None = None

    def report(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "output_name": self.output_name,
            "center_world": list(self.center_world),
            "coordinate_frame": self.coordinate_frame,
            "source_colmap_image_name": self.source_colmap_image_name,
            "pano_image_name": self.pano_image_name,
        }


@dataclass(frozen=True)
class PinholePositionPriorSummary:
    pose_report_path: Path
    coordinate_frame: str | None
    priors: tuple[PinholePositionPrior, ...]
    missing: tuple[dict[str, Any], ...]
    unusable: tuple[dict[str, Any], ...]

    def report(
        self,
        *,
        use_position_priors: bool = False,
        written_prior_count: int | None = None,
    ) -> dict[str, Any]:
        return {
            "source": POSE_PRIOR_SOURCE,
            "type": POSE_PRIOR_TYPE,
            "coordinate_frame": self.coordinate_frame,
            "pose_report_path": str(self.pose_report_path),
            "matched_count": len(self.priors),
            "missing_count": len(self.missing),
            "unusable_count": len(self.unusable),
            "use_position_priors": use_position_priors,
            "written_prior_count": written_prior_count,
            "priors": [prior.report() for prior in self.priors],
            "missing": list(self.missing),
            "unusable": list(self.unusable),
        }


def load_pose_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_position_priors_to_database(
    pycolmap_module: Any,
    *,
    database_path: Path,
    prior_summary: PinholePositionPriorSummary,
) -> int:
    if not prior_summary.priors:
        raise ValueError("No usable position priors are available to write")
    database = pycolmap_module.Database.open(database_path)
    try:
        written_count = 0
        for prior in prior_summary.priors:
            image = database.read_image_with_name(prior.output_name)
            if image is None:
                raise ValueError(
                    f"Expected image in database for position prior: {prior.output_name}",
                )
            pose_prior = pycolmap_module.PosePrior()
            pose_prior.position = np.asarray(prior.center_world, dtype=np.float64)
            pose_prior.coordinate_system = (
                pycolmap_module.PosePriorCoordinateSystem.CARTESIAN
            )
            pose_prior.corr_data_id = image.data_id
            database.write_pose_prior(pose_prior)
            written_count += 1
        return written_count
    finally:
        close = getattr(database, "close", None)
        if close is not None:
            close()


def build_position_prior_summary(
    *,
    pose_report_path: Path,
    imported_images: Sequence[ImportedPinholeImage],
) -> PinholePositionPriorSummary:
    pose_report = load_pose_report(pose_report_path)
    views = pose_report.get("views")
    if not isinstance(views, list):
        raise ValueError("pose_report.json must contain a views array")
    by_selection_id, by_output_name = pose_views_by_identity(views)

    priors: list[PinholePositionPrior] = []
    missing: list[dict[str, Any]] = []
    unusable: list[dict[str, Any]] = []
    coordinate_frame: str | None = None
    for image in imported_images:
        render_record = image.record
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
        prior, reason = position_prior_from_view(
            view,
            fallback_selection_id=render_record.selection_id,
            fallback_output_name=render_record.output_name,
        )
        if prior is None:
            unusable.append(
                {
                    "selection_id": render_record.selection_id,
                    "output_name": render_record.output_name,
                    "matched_by": matched_by,
                    "reason": reason,
                },
            )
            continue
        priors.append(prior)
        if coordinate_frame is None:
            coordinate_frame = prior.coordinate_frame
    return PinholePositionPriorSummary(
        pose_report_path=pose_report_path,
        coordinate_frame=coordinate_frame,
        priors=tuple(priors),
        missing=tuple(missing),
        unusable=tuple(unusable),
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


def position_prior_from_view(
    view: dict[str, Any],
    *,
    fallback_selection_id: str,
    fallback_output_name: str,
) -> tuple[PinholePositionPrior | None, str | None]:
    if view.get("pose_status") not in (None, "matched"):
        return None, f"pose_status is {view.get('pose_status')}"
    pose = view.get("pinhole_camera_pose")
    if not isinstance(pose, dict):
        return None, "pinhole_camera_pose is missing"
    center = pose.get("center_world")
    if not isinstance(center, list) or len(center) != 3:
        return None, "center_world must contain three values"
    try:
        center_world = tuple(float(value) for value in center)
    except (TypeError, ValueError):
        return None, "center_world must contain numeric values"
    coordinate_frame = optional_string(pose.get("coordinate_frame"))
    if coordinate_frame is None:
        coordinate_frame = "colmap_world_unscaled"
    return (
        PinholePositionPrior(
            selection_id=optional_string(view.get("selection_id"))
            or fallback_selection_id,
            output_name=optional_string(view.get("output_name"))
            or fallback_output_name,
            center_world=center_world,  # type: ignore[arg-type]
            coordinate_frame=coordinate_frame,
            source_colmap_image_name=optional_string(
                view.get("source_colmap_image_name"),
            ),
            pano_image_name=optional_string(view.get("pano_image_name")),
        ),
        None,
    )


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
