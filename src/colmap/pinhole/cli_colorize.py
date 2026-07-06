"""Command-line repair tool for colorizing pinhole COLMAP sparse models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from colmap.panorama.pycolmap_adapter import require_pycolmap
from colmap.pinhole.colorize import (
    colorize_reconstruction_model,
    default_output_model_path,
    infer_image_dir,
)


class PinholeColorizeDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.required and action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Colorize an existing rendered-pinhole COLMAP sparse model by "
            "sampling RGB from its registered image observations."
        ),
        formatter_class=PinholeColorizeDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        required=True,
        help="Input COLMAP sparse model directory, for example workspace/sparse_best.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing registered rendered pinhole images. Defaults "
            "to a nearby images/ directory next to the model or workspace."
        ),
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=None,
        help=(
            "Output COLMAP model directory. Defaults to MODEL_colorized unless "
            "--in-place is passed."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input model directory with the colorized model.",
    )
    parser.add_argument(
        "--ply",
        type=Path,
        default=None,
        help="Optional PLY path to export after colorizing.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of image-loading workers for colorization.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing --output-model directory.",
    )
    args = parser.parse_args(argv)
    if not args.model.is_dir():
        parser.error(
            f"--model must be an existing COLMAP model directory: {args.model}",
        )
    if args.image_dir is None:
        args.image_dir = infer_image_dir(args.model)
    if not args.image_dir.is_dir():
        parser.error(
            f"--image-dir must be an existing image directory: {args.image_dir}",
        )
    if args.in_place and args.output_model is not None:
        parser.error("--in-place cannot be combined with --output-model")
    if args.in_place:
        args.output_model = args.model
        args.overwrite = True
    elif args.output_model is None:
        args.output_model = default_output_model_path(args.model)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    summary = colorize_reconstruction_model(
        require_pycolmap(),
        model_path=args.model,
        image_dir=args.image_dir,
        output_model_path=args.output_model,
        output_ply_path=args.ply,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
        workers=args.workers,
    )
    print(json.dumps(summary.report(), indent=2))
    print(f"colorized model: {args.output_model}")
    if args.ply is not None:
        print(f"colorized ply: {args.ply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
