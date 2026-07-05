"""Parsing and validation helpers for pinhole render reports."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isclose
from pathlib import Path
from typing import Any


FOV_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PinholeRenderRecord:
    selection_id: str
    output_name: str
    image_path: Path
    reported_output_file: Path | None
    reported_width: int | None
    reported_height: int | None
    fov_x: float
    fov_y: float
    pano_image_name: str | None = None
    selection_index: int | None = None
    source_file: str | None = None
    longitude: float | None = None
    latitude: float | None = None
    longitude_degrees: float | None = None
    latitude_degrees: float | None = None

    def provenance_report(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "pano_image_name": self.pano_image_name,
            "selection_index": self.selection_index,
            "source_file": self.source_file,
            "reported_output_file": (
                None
                if self.reported_output_file is None
                else str(self.reported_output_file)
            ),
            "input_image_path": str(self.image_path),
            "output_name": self.output_name,
            "reported_width": self.reported_width,
            "reported_height": self.reported_height,
            "fov_x": self.fov_x,
            "fov_y": self.fov_y,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "longitude_degrees": self.longitude_degrees,
            "latitude_degrees": self.latitude_degrees,
        }


def load_render_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def render_records_from_report(
    report: dict[str, Any],
    *,
    render_report_path: Path,
    image_dir: Path | None = None,
) -> tuple[PinholeRenderRecord, ...]:
    renders = report.get("renders")
    if not isinstance(renders, list) or not renders:
        raise ValueError("render_report.json must contain a non-empty renders array")

    records = tuple(
        render_record_from_report(
            render_record,
            report=report,
            render_report_path=render_report_path,
            image_dir=image_dir,
        )
        for render_record in renders
    )
    validate_unique_output_names(records)
    validate_uniform_fov(records)
    return records


def render_record_from_report(
    render_record: dict[str, Any],
    *,
    report: dict[str, Any],
    render_report_path: Path,
    image_dir: Path | None,
) -> PinholeRenderRecord:
    selection_id = required_string(render_record, "selection_id")
    output_name = required_string(render_record, "output_name")
    fov_x = optional_float(render_record.get("fov_x"))
    fov_y = optional_float(render_record.get("fov_y"))
    if fov_x is None:
        fov_x = optional_float(report.get("render_params", {}).get("fov_x"))
    if fov_y is None:
        fov_y = optional_float(report.get("render_params", {}).get("fov_y"))
    if fov_x is None or fov_y is None:
        raise ValueError(f"Render record {selection_id} is missing fov_x/fov_y")

    reported_output_file = optional_path(render_record.get("output_file"))
    return PinholeRenderRecord(
        selection_id=selection_id,
        output_name=output_name,
        image_path=resolve_rendered_image_path(
            output_name=output_name,
            reported_output_file=reported_output_file,
            report=report,
            render_report_path=render_report_path,
            image_dir=image_dir,
        ),
        reported_output_file=reported_output_file,
        reported_width=optional_int(render_record.get("width")),
        reported_height=optional_int(render_record.get("height")),
        fov_x=fov_x,
        fov_y=fov_y,
        pano_image_name=optional_string(render_record.get("pano_image_name")),
        selection_index=optional_int(render_record.get("selection_index")),
        source_file=optional_string(render_record.get("source_file")),
        longitude=optional_float(render_record.get("longitude")),
        latitude=optional_float(render_record.get("latitude")),
        longitude_degrees=optional_float(render_record.get("longitude_degrees")),
        latitude_degrees=optional_float(render_record.get("latitude_degrees")),
    )


def resolve_rendered_image_path(
    *,
    output_name: str,
    reported_output_file: Path | None,
    report: dict[str, Any],
    render_report_path: Path,
    image_dir: Path | None,
) -> Path:
    if image_dir is not None:
        return image_dir / output_name
    if reported_output_file is not None:
        if reported_output_file.is_absolute():
            return reported_output_file
        return render_report_path.parent / reported_output_file
    output_dir = optional_path(report.get("output", {}).get("output_dir"))
    if output_dir is not None:
        return output_dir / output_name
    raise ValueError(
        f"Render record {output_name} is missing output_file and no --image-dir was set",
    )


def validate_unique_output_names(records: tuple[PinholeRenderRecord, ...]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.output_name in seen:
            raise ValueError(f"Duplicate rendered output_name: {record.output_name}")
        seen.add(record.output_name)


def validate_uniform_fov(records: tuple[PinholeRenderRecord, ...]) -> None:
    first = records[0]
    for record in records[1:]:
        if not isclose(record.fov_x, first.fov_x, rel_tol=0.0, abs_tol=FOV_TOLERANCE):
            raise ValueError(
                f"Render {record.output_name} has fov_x={record.fov_x}; "
                f"expected {first.fov_x}",
            )
        if not isclose(record.fov_y, first.fov_y, rel_tol=0.0, abs_tol=FOV_TOLERANCE):
            raise ValueError(
                f"Render {record.output_name} has fov_y={record.fov_y}; "
                f"expected {first.fov_y}",
            )


def required_string(record: dict[str, Any], key: str) -> str:
    value = optional_string(record.get(key))
    if value is None:
        raise ValueError(f"Render record is missing {key}")
    return value


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def optional_path(value: Any) -> Path | None:
    text = optional_string(value)
    return None if text is None else Path(text)

