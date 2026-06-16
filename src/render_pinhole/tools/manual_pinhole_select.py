"""CLI entry point for the manual pinhole selection server."""

from __future__ import annotations

import argparse
from pathlib import Path

from render_pinhole.tools.manual_pinhole_select_server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    METADATA_FILENAME,
    serve_selector,
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata_path = args.output_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        print(f"Metadata file not found: {metadata_path}")
        return 1

    serve_selector(args.output_dir, host=args.host, port=args.port)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a local browser UI for saving manual pinhole selections.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=Path,
        help="Directory containing metadata.json from manual-pinhole-render.",
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
    if not args.output_dir.is_dir():
        parser.error(f"--output-dir must be an existing directory: {args.output_dir}")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
