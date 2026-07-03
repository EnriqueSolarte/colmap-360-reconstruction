"""Render pinhole images from flat selections.json without COLMAP."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from math import degrees, radians
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from render_pinhole import EquirectangularToPinhole
from render_pinhole.defaults import (
    DEFAULT_PINHOLE_FOV_X,
    DEFAULT_PINHOLE_FOV_Y,
    DEFAULT_PINHOLE_HEIGHT,
    DEFAULT_PINHOLE_INTERPOLATION,
    DEFAULT_PINHOLE_WIDTH,
)

RENDER_REPORT_FILENAME = "render_report.json"
DEFAULT_OUTPUT_DIR_NAME = "renders"
DEFAULT_OUTPUT_EXTENSION = ".png"
DEFAULT_FOV_X_DEGREES = round(degrees(DEFAULT_PINHOLE_FOV_X), 6)
DEFAULT_FOV_Y_DEGREES = round(degrees(DEFAULT_PINHOLE_FOV_Y), 6)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    store = _load_selection_store(args.selections)
    image_dir = args.image_dir or _image_dir_from_store(store)
    if image_dir is None:
        print("Image directory is missing. Pass --image-dir or set source.image_dir.")
        return 1
    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}")
        return 1

    output_dir = args.output_dir or args.selections.parent / DEFAULT_OUTPUT_DIR_NAME
    report = render_selected_pinhole_images(
        selections_store=store,
        selections_path=args.selections,
        image_dir=image_dir,
        output_dir=output_dir,
        width=args.width,
        height=args.height,
        fov_x=args.fov_x,
        fov_y=args.fov_y,
        interpolation=args.interpolation,
        output_extension=args.output_extension,
        workers=args.workers,
    )
    report_path = output_dir / RENDER_REPORT_FILENAME
    _write_json(report_path, report)

    print(f"Rendered: {report['rendered_count']}")
    if report["skipped_count"]:
        print(f"Skipped: {report['skipped_count']}")
    if report["failed_count"]:
        print(f"Failed: {report['failed_count']}")
    print(f"Render report: {report_path}")
    return 1 if report["skipped_count"] or report["failed_count"] else 0


def render_selected_pinhole_images(
    *,
    selections_store: dict[str, Any],
    selections_path: Path,
    image_dir: Path,
    output_dir: Path,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    interpolation: str,
    output_extension: str,
    workers: int = 1,
    show_progress: bool = True,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    selections = list(selections_store.get("selections", []))
    render_params = {
        "width": int(width),
        "height": int(height),
        "fov_x": float(fov_x),
        "fov_y": float(fov_y),
        "fov_x_degrees": degrees(float(fov_x)),
        "fov_y_degrees": degrees(float(fov_y)),
        "interpolation": interpolation,
        "output_extension": output_extension,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "source": {
            "image_dir": str(image_dir),
            "selections_path": str(selections_path),
        },
        "output": {
            "output_dir": str(output_dir),
            "render_report": str(output_dir / RENDER_REPORT_FILENAME),
        },
        "options": {
            "workers": int(workers),
        },
        "render_params": render_params,
        "selection_summary": selections_store.get("selection_summary", {}),
        "selected_count": len(selections),
        "rendered_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "renders": [],
        "skipped": [],
        "failed": [],
    }

    jobs = _render_jobs(
        selections=selections,
        image_dir=image_dir,
        output_dir=output_dir,
        output_extension=output_extension,
    )
    outcomes = _render_jobs_with_progress(
        jobs,
        width=width,
        height=height,
        fov_x=fov_x,
        fov_y=fov_y,
        interpolation=interpolation,
        workers=workers,
        show_progress=show_progress,
    )
    for outcome in sorted(outcomes, key=lambda record: int(record["order"])):
        kind = outcome.pop("kind")
        outcome.pop("order")
        if kind == "rendered":
            report["renders"].append(outcome)
            report["rendered_count"] += 1
        elif kind == "skipped":
            report["skipped"].append(outcome)
            report["skipped_count"] += 1
        elif kind == "failed":
            report["failed"].append(outcome)
            report["failed_count"] += 1
        else:  # pragma: no cover - defensive branch
            raise ValueError(f"Unsupported render outcome kind: {kind}")

    return report


def _render_jobs(
    *,
    selections: list[dict[str, Any]],
    image_dir: Path,
    output_dir: Path,
    output_extension: str,
) -> list[dict[str, Any]]:
    return [
        {
            "order": order,
            "selection": selection,
            "source_path": image_dir / str(selection.get("pano_image_name", "")),
            "output_path": output_dir / _rendered_filename(selection, output_extension),
        }
        for order, selection in enumerate(selections)
    ]


def _render_jobs_with_progress(
    jobs: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    interpolation: str,
    workers: int,
    show_progress: bool,
) -> list[dict[str, Any]]:
    job_groups = _jobs_by_source_path(jobs)
    progress_options = {
        "total": len(jobs),
        "desc": "Rendering selected pinholes",
        "unit": "view",
        "disable": not show_progress,
    }
    group_kwargs = {
        "width": width,
        "height": height,
        "fov_x": fov_x,
        "fov_y": fov_y,
        "interpolation": interpolation,
    }

    outcomes: list[dict[str, Any]] = []
    with tqdm(**progress_options) as progress:
        if workers == 1:
            for group in job_groups:
                group_outcomes = _render_source_group(group, **group_kwargs)
                outcomes.extend(group_outcomes)
                progress.update(len(group))
            return outcomes

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_render_source_group, group, **group_kwargs)
                for group in job_groups
            ]
            for future in as_completed(futures):
                group_outcomes = future.result()
                outcomes.extend(group_outcomes)
                progress.update(len(group_outcomes))
    return outcomes


def _jobs_by_source_path(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups_by_source: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        groups_by_source[job["source_path"]].append(job)
    return list(groups_by_source.values())


def _render_source_group(
    jobs: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    interpolation: str,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    source_path = jobs[0]["source_path"]
    if not source_path.is_file():
        return [
            {
                "kind": "skipped",
                "order": job["order"],
                **_selection_issue_record(
                    job["selection"],
                    reason=f"source panorama not found: {source_path}",
                ),
            }
            for job in jobs
        ]

    try:
        renderer = EquirectangularToPinhole.from_file(source_path)
    except Exception as error:  # pragma: no cover - defensive report path
        return [
            {
                "kind": "failed",
                "order": job["order"],
                **_selection_issue_record(job["selection"], reason=str(error)),
            }
            for job in jobs
        ]

    source_width, source_height = _resolution(renderer.image)
    outcomes = []
    for job in jobs:
        selection = job["selection"]
        output_path = job["output_path"]
        try:
            pinhole = renderer.render(
                height=height,
                width=width,
                longitude=float(selection["longitude"]),
                latitude=float(selection["latitude"]),
                fov_x=fov_x,
                fov_y=fov_y,
                interpolation=interpolation,
            )
            _write_image(output_path, pinhole)
        except Exception as error:  # pragma: no cover - defensive report path
            outcomes.append(
                {
                    "kind": "failed",
                    "order": job["order"],
                    **_selection_issue_record(selection, reason=str(error)),
                }
            )
            continue

        output_width, output_height = _resolution(pinhole)
        outcomes.append(
            {
                "kind": "rendered",
                "order": job["order"],
                "selection_id": selection.get("selection_id"),
                "pano_image_name": selection.get("pano_image_name"),
                "selection_index": selection.get("selection_index"),
                "source_file": str(source_path),
                "source_width": source_width,
                "source_height": source_height,
                "output_file": str(output_path),
                "output_name": output_path.name,
                "width": output_width,
                "height": output_height,
                "longitude": selection.get("longitude"),
                "latitude": selection.get("latitude"),
                "longitude_degrees": selection.get("longitude_degrees"),
                "latitude_degrees": selection.get("latitude_degrees"),
                "fov_x": fov_x,
                "fov_y": fov_y,
                "interpolation": interpolation,
            }
        )
    return outcomes


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render pinhole images from flat selections.json without COLMAP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--help",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "-s",
        "--selections",
        required=True,
        type=Path,
        help="Flat selections.json path from manual-pinhole-select.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing source equirectangular panorama images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "Directory where rendered pinhole images and render_report.json go. "
            "Defaults to a renders/ directory beside --selections."
        ),
    )
    parser.add_argument(
        "-h",
        "--height",
        default=DEFAULT_PINHOLE_HEIGHT,
        type=int,
        help="Output image height in pixels.",
    )
    parser.add_argument(
        "-w",
        "--width",
        default=DEFAULT_PINHOLE_WIDTH,
        type=int,
        help="Output image width in pixels.",
    )
    parser.add_argument(
        "--fov-x-deg",
        default=DEFAULT_FOV_X_DEGREES,
        type=float,
        help="Horizontal field of view in degrees.",
    )
    parser.add_argument(
        "--fov-y-deg",
        default=DEFAULT_FOV_Y_DEGREES,
        type=float,
        help="Vertical field of view in degrees.",
    )
    parser.add_argument(
        "--interpolation",
        choices=("linear", "bilinear", "nearest"),
        default=DEFAULT_PINHOLE_INTERPOLATION,
        help="Interpolation mode.",
    )
    parser.add_argument(
        "--output-extension",
        default=DEFAULT_OUTPUT_EXTENSION,
        help="Output image extension such as .png or .jpg.",
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help=(
            "Number of worker threads. Each worker renders one source panorama "
            "group at a time."
        ),
    )
    args = parser.parse_args(argv)

    if not args.selections.is_file():
        parser.error(f"--selections must be an existing file: {args.selections}")
    if args.width <= 0:
        parser.error("--width must be positive")
    if args.height <= 0:
        parser.error("--height must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    args.fov_x = radians(args.fov_x_deg)
    args.fov_y = radians(args.fov_y_deg)
    if args.fov_x <= 0.0:
        parser.error("--fov-x-deg must be positive")
    if args.fov_y <= 0.0:
        parser.error("--fov-y-deg must be positive")
    args.output_extension = _normalize_extension(args.output_extension)
    return args


def _image_dir_from_store(store: dict[str, Any]) -> Path | None:
    image_dir = store.get("source", {}).get("image_dir")
    return Path(str(image_dir)) if image_dir else None


def _load_selection_store(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rendered_filename(selection: dict[str, Any], extension: str) -> str:
    selection_id = selection.get("selection_id")
    if selection_id:
        stem = str(selection_id)
    else:
        pano_stem = Path(str(selection.get("pano_image_name", "selection"))).stem
        selection_index = int(selection.get("selection_index", 0))
        stem = f"{pano_stem}__sel-{selection_index:03d}"
    return f"{stem}{extension}"


def _selection_issue_record(
    selection: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "selection_id": selection.get("selection_id"),
        "pano_image_name": selection.get("pano_image_name"),
        "selection_index": selection.get("selection_index"),
        "reason": reason,
    }


def _normalize_extension(extension: str) -> str:
    return extension if extension.startswith(".") else f".{extension}"


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolution(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return width, height


if __name__ == "__main__":
    raise SystemExit(main())
