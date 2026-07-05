"""Import rendered pinhole images into a COLMAP workspace."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

from colmap.camera import ColmapCamera, camera_from_fov
from colmap.panorama.image_import import hardlink_or_symlink_file
from colmap.pinhole.render_report import (
    PinholeRenderRecord,
    load_render_report,
    render_records_from_report,
)


@dataclass(frozen=True)
class ImportedPinholeImage:
    record: PinholeRenderRecord
    source_path: Path
    colmap_image_name: str
    actual_width: int
    actual_height: int

    def report(self) -> dict[str, Any]:
        return {
            **self.record.provenance_report(),
            "source_path": str(self.source_path),
            "colmap_image_name": self.colmap_image_name,
            "actual_width": self.actual_width,
            "actual_height": self.actual_height,
            "resolution_matches_report": (
                self.record.reported_width == self.actual_width
                and self.record.reported_height == self.actual_height
            ),
        }


@dataclass(frozen=True)
class PinholeImageImportResult:
    render_report_path: Path
    image_dir_override: Path | None
    camera: ColmapCamera
    imported_images: tuple[ImportedPinholeImage, ...]

    @property
    def image_names(self) -> tuple[str, ...]:
        return tuple(image.colmap_image_name for image in self.imported_images)

    @property
    def resolution_mismatch_count(self) -> int:
        return sum(
            1
            for image in self.imported_images
            if (
                image.record.reported_width,
                image.record.reported_height,
            )
            != (image.actual_width, image.actual_height)
        )


def import_pinhole_images_from_render_report(
    *,
    render_report_path: Path,
    output_image_dir: Path,
    image_dir: Path | None = None,
    fail_on_existing: bool = False,
    import_workers: int = 1,
    show_progress: bool = True,
) -> PinholeImageImportResult:
    if import_workers <= 0:
        raise ValueError("import_workers must be positive")
    report = load_render_report(render_report_path)
    records = render_records_from_report(
        report,
        render_report_path=render_report_path,
        image_dir=image_dir,
    )
    output_image_dir.mkdir(parents=True, exist_ok=True)
    images = import_pinhole_records(
        records,
        output_image_dir=output_image_dir,
        fail_on_existing=fail_on_existing,
        import_workers=import_workers,
        show_progress=show_progress,
    )
    camera = shared_camera_from_imported_images(images)
    return PinholeImageImportResult(
        render_report_path=render_report_path,
        image_dir_override=image_dir,
        camera=camera,
        imported_images=images,
    )


def import_pinhole_records(
    records: tuple[PinholeRenderRecord, ...],
    *,
    output_image_dir: Path,
    fail_on_existing: bool,
    import_workers: int,
    show_progress: bool,
) -> tuple[ImportedPinholeImage, ...]:
    def import_record(record: PinholeRenderRecord) -> ImportedPinholeImage:
        if not record.image_path.is_file():
            raise FileNotFoundError(f"Rendered pinhole image not found: {record.image_path}")
        actual_width, actual_height = image_resolution(record.image_path)
        output_path = output_image_dir / record.output_name
        if fail_on_existing and output_path.exists():
            raise FileExistsError(
                "New pinhole image would overwrite a previous session file: "
                f"{output_path}",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hardlink_or_symlink_file(str(record.image_path), str(output_path))
        return ImportedPinholeImage(
            record=record,
            source_path=record.image_path,
            colmap_image_name=Path(record.output_name).as_posix(),
            actual_width=actual_width,
            actual_height=actual_height,
        )

    progress_options = {
        "desc": "Importing rendered pinholes",
        "unit": "image",
        "total": len(records),
        "disable": not show_progress,
    }
    if import_workers == 1:
        imported = [import_record(record) for record in tqdm(records, **progress_options)]
    else:
        with ThreadPoolExecutor(max_workers=import_workers) as executor:
            imported = list(
                tqdm(
                    executor.map(import_record, records),
                    **progress_options,
                ),
            )
    validate_uniform_actual_resolution(tuple(imported))
    return tuple(imported)


def shared_camera_from_imported_images(
    images: tuple[ImportedPinholeImage, ...],
) -> ColmapCamera:
    if not images:
        raise ValueError("at least one imported pinhole image is required")
    first = images[0]
    return camera_from_fov(
        width=first.actual_width,
        height=first.actual_height,
        fov_x=first.record.fov_x,
        fov_y=first.record.fov_y,
    )


def validate_uniform_actual_resolution(
    images: tuple[ImportedPinholeImage, ...],
) -> None:
    first = images[0]
    for image in images[1:]:
        if (image.actual_width, image.actual_height) != (
            first.actual_width,
            first.actual_height,
        ):
            raise ValueError(
                f"Rendered image {image.source_path} has size "
                f"{image.actual_width}x{image.actual_height}; expected "
                f"{first.actual_width}x{first.actual_height}",
            )


def image_resolution(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size

