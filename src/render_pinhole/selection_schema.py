"""Flat selection artifact helpers for pinhole rendering stages."""

from __future__ import annotations

from math import degrees, pi, radians
from pathlib import Path
from typing import Any, Sequence

SELECTION_SCHEMA_VERSION = 1
DEFAULT_AUTO_SELECTION_DELTA_DEGREES = 45.0


def selection_record(
    *,
    pano_image_name: str,
    selection_index: int,
    longitude: float,
    latitude: float,
    selected_by: str,
) -> dict[str, Any]:
    """Build one flat pinhole-view selection record."""
    if selection_index < 0:
        raise ValueError("selection_index must be non-negative")
    validate_longitude_latitude(longitude, latitude)
    return {
        "selection_id": selection_id(pano_image_name, selection_index),
        "pano_image_name": pano_image_name,
        "selection_index": selection_index,
        "longitude": float(longitude),
        "latitude": float(latitude),
        "longitude_degrees": degrees(longitude),
        "latitude_degrees": degrees(latitude),
        "selected_by": selected_by,
    }


def build_selection_store(
    *,
    image_dir: Path,
    selections: Sequence[dict[str, Any]],
    selection_mode: str,
    auto_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stage-1 flat selections.json payload."""
    store = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "source": {
            "image_dir": str(image_dir),
        },
        "selection_mode": selection_mode,
        "selection_summary": selection_summary(selections),
        "selections": list(selections),
    }
    if auto_selection is not None:
        store["auto_selection"] = dict(auto_selection)
    return store


def auto_grid_selection_store(
    *,
    image_dir: Path,
    pano_image_names: Sequence[str],
    delta_degrees: float = DEFAULT_AUTO_SELECTION_DELTA_DEGREES,
) -> dict[str, Any]:
    """Build flat selections for every panorama using a lon/lat grid."""
    grid = auto_selection_longitude_latitude_grid(delta_degrees=delta_degrees)
    selections = [
        selection_record(
            pano_image_name=pano_image_name,
            selection_index=selection_index,
            longitude=longitude,
            latitude=latitude,
            selected_by="auto_grid",
        )
        for pano_image_name in pano_image_names
        for selection_index, (longitude, latitude) in enumerate(grid)
    ]
    return build_selection_store(
        image_dir=image_dir,
        selections=selections,
        selection_mode="auto_grid",
        auto_selection={
            "strategy": "longitude_latitude_grid",
            "delta_degrees": float(delta_degrees),
            "longitude_sample_count": len(
                auto_selection_axis_values(-pi, pi, radians(delta_degrees))
            ),
            "latitude_sample_count": len(
                auto_selection_axis_values(-pi / 2.0, pi / 2.0, radians(delta_degrees))
            ),
            "views_per_panorama": len(grid),
            "selection_count": len(selections),
        },
    )


def selection_summary(selections: Sequence[dict[str, Any]]) -> dict[str, int]:
    pano_names = {
        str(selection["pano_image_name"])
        for selection in selections
        if selection.get("pano_image_name")
    }
    return {
        "selected_panorama_count": len(pano_names),
        "selection_count": len(selections),
    }


def auto_selection_longitude_latitude_grid(
    *,
    delta_degrees: float = DEFAULT_AUTO_SELECTION_DELTA_DEGREES,
) -> list[tuple[float, float]]:
    if delta_degrees <= 0.0:
        raise ValueError("auto selection delta must be positive")
    delta = radians(delta_degrees)
    longitudes = auto_selection_axis_values(-pi, pi, delta)
    latitudes = list(
        reversed(auto_selection_axis_values(-pi / 2.0, pi / 2.0, delta))
    )
    return [
        (longitude, latitude)
        for latitude in latitudes
        for longitude in longitudes
    ]


def auto_selection_axis_values(
    minimum: float,
    maximum: float,
    delta: float,
) -> list[float]:
    span = maximum - minimum
    if span <= delta:
        return [(minimum + maximum) / 2.0]
    value = minimum + delta / 2.0
    values = []
    while value < maximum:
        values.append(value)
        value += delta
    return values


def selection_id(pano_image_name: str, selection_index: int) -> str:
    if selection_index < 0:
        raise ValueError("selection_index must be non-negative")
    stem = Path(pano_image_name).with_suffix("").as_posix()
    safe_stem = stem.replace("/", "__")
    return f"{safe_stem}__sel-{selection_index:03d}"


def validate_longitude_latitude(longitude: float, latitude: float) -> None:
    if not -pi <= longitude <= pi:
        raise ValueError("longitude must be in [-pi, pi] radians")
    if not -pi / 2 <= latitude <= pi / 2:
        raise ValueError("latitude must be in [-pi / 2, pi / 2] radians")
