"""Command-line interface for panorama image import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colmap.panorama.import_pipeline import run_panorama_image_import


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-equirectangular"


class PanoramaImportArgumentDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.required and action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_import_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import equirectangular panorama images into a COLMAP workspace "
            "without feature extraction, matching, or mapping."
        ),
        formatter_class=PanoramaImportArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-image-path",
        type=Path,
        required=True,
        help="Directory containing input equirectangular panorama images.",
    )
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="COLMAP equirectangular workspace.",
    )
    parser.add_argument(
        "--image-limit",
        type=int,
        default=None,
        help="Use only the first N sorted panoramas for smoke testing.",
    )
    parser.add_argument(
        "--import-workers",
        type=int,
        default=1,
        help="Number of worker threads for importing panorama images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the import workspace.",
    )
    args = parser.parse_args(argv)
    if not args.input_image_path.is_dir():
        parser.error(
            f"--input-image-path must be an existing directory: {args.input_image_path}"
        )
    if args.image_limit is not None and args.image_limit <= 0:
        parser.error("--image-limit must be positive")
    if args.import_workers <= 0:
        parser.error("--import-workers must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_import_args(sys.argv[1:] if argv is None else argv)
    report = run_panorama_image_import(
        input_image_path=args.input_image_path,
        workspace=args.workspace,
        image_limit=args.image_limit,
        import_workers=args.import_workers,
        overwrite=args.overwrite,
    )
    print(f"workspace: {report['workspace']}")
    print(f"camera setup: {report['camera_setup']}")
    print(f"camera model: {report['camera']['model']}")
    print(f"panorama images: {report['panorama_count']}")
    print(f"elapsed: {report['elapsed_time']}")
    print(f"metadata: {report['metadata_path']}")
    print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
