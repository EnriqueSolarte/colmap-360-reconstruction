"""Command-line interface for pinhole image import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colmap.pinhole.import_pipeline import run_pinhole_image_import


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-pinhole"


class PinholeImportArgumentDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.required and action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_import_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import rendered pinhole images into a COLMAP workspace using "
            "render_report.json as the metadata and FOV manifest."
        ),
        formatter_class=PinholeImportArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--render-report",
        type=Path,
        required=True,
        help="render_report.json path from manual-pinhole-render-selected.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing the actual rendered pinhole images. "
            "Files are matched by render_report renders[].output_name."
        ),
    )
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="COLMAP pinhole workspace.",
    )
    parser.add_argument(
        "--import-workers",
        type=int,
        default=1,
        help="Number of worker threads for importing rendered pinhole images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the import workspace.",
    )
    args = parser.parse_args(argv)
    if not args.render_report.is_file():
        parser.error(f"--render-report must be an existing file: {args.render_report}")
    if args.image_dir is not None and not args.image_dir.is_dir():
        parser.error(f"--image-dir must be an existing directory: {args.image_dir}")
    if args.import_workers <= 0:
        parser.error("--import-workers must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_import_args(sys.argv[1:] if argv is None else argv)
    report = run_pinhole_image_import(
        render_report_path=args.render_report,
        workspace=args.workspace,
        image_dir=args.image_dir,
        import_workers=args.import_workers,
        overwrite=args.overwrite,
    )
    print(f"workspace: {report['workspace']}")
    print(f"camera setup: {report['camera_setup']}")
    print(f"camera model: {report['camera']['model']}")
    print(f"pinhole images: {report['rendered_image_count']}")
    print(f"resolution mismatches: {report['resolution_mismatch_count']}")
    print(f"elapsed: {report['elapsed_time']}")
    print(f"metadata: {report['metadata_path']}")
    print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
