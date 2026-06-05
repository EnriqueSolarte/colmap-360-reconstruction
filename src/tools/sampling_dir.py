#!/usr/bin/env python3
"""Sample a directory of images by creating symbolic links in a target directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a sampled subset of images by linking files from the source "
            "directory into the destination directory."
        )
    )
    parser.add_argument(
        "-s",
        "--source",
        help="Directory containing the original images.",
    )
    parser.add_argument(
        "-d",
        "--destination",
        help="Directory where symbolic links will be created.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=4,
        help=(
            "Number of images to skip between each link. "
            "For example, --skip 2 links every third image. Default: 0 (link every image)."
        ),
    )
    return parser.parse_args()


def list_images(directory: Path) -> List[Path]:
    """Return a sorted list of files inside ``directory``."""
    files: Iterable[Path] = directory.iterdir()
    return sorted(file for file in files if file.is_file())


def create_symlink(src: Path, dst: Path) -> None:
    """
    Create a symbolic link ``dst`` pointing to ``src``.

    If the link already exists and points to the correct source, nothing happens.
    Otherwise ``FileExistsError`` is raised to avoid overwriting existing files.
    """
    try:
        os.symlink(src, dst)
        return
    except FileExistsError:
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source).expanduser().resolve()
    dest_dir = Path(args.destination).expanduser()

    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    if args.skip < 0:
        raise SystemExit("--skip must be >= 0")

    dest_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_images(source_dir)
    if not image_files:
        raise SystemExit(f"No files found in {source_dir}")

    step = args.skip + 1
    sampled_files = image_files[::step]

    created = 0
    for file in sampled_files:
        target_path = dest_dir / file.name
        create_symlink(file, target_path)
        created += 1

    print(
        f"Linked {created} file(s) from {source_dir} "
        f"into {dest_dir.resolve()} with skip={args.skip}"
    )


if __name__ == "__main__":
    main()
