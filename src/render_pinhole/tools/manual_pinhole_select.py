"""CLI entry point for the manual pinhole selection server."""

from __future__ import annotations

import argparse
from pathlib import Path

from render_pinhole.tools.manual_pinhole_select_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    serve_image_dir_selector,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        serve_image_dir_selector(
            args.image_dir,
            args.output_dir,
            host=args.host,
            port=args.port,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(error)
        return 1
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a local browser UI for saving manual pinhole selections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "Directory for selector outputs. Defaults to the parent directory "
            "of --image-dir."
        ),
    )
    parser.add_argument(
        "-i",
        "--image-dir",
        type=Path,
        required=True,
        help=(
            "Directory of equirectangular panorama images for the new flat "
            "selections.json workflow."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host interface for the local selector server.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help="Port for the local selector server. Use 0 for any free port.",
    )
    args = parser.parse_args(argv)
    if not args.image_dir.is_dir():
        parser.error(f"--image-dir must be an existing directory: {args.image_dir}")
    if args.output_dir is None:
        args.output_dir = args.image_dir.parent
    return args


if __name__ == "__main__":
    raise SystemExit(main())
