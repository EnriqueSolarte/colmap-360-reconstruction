#!/usr/bin/env python3
"""Horizontally shift equirectangular panorama images."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
DEFAULT_THETA_DEGREES = 180.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Horizontally shift equirectangular panorama image pixels with "
            "wraparound. Positive theta shifts pixels to the right."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input panorama image or directory of panorama images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output image path or directory. If omitted, source image files are "
            "shifted in place after confirmation."
        ),
    )
    parser.add_argument(
        "-t",
        "--theta",
        default=DEFAULT_THETA_DEGREES,
        type=float,
        help="Horizontal shift angle in degrees.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output image files.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=1,
        type=int,
        help="Number of parallel worker threads for directory inputs.",
    )
    return parser.parse_args(argv)


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def shift_image_array(image: np.ndarray, theta_degrees: float) -> tuple[np.ndarray, int]:
    if image.ndim < 2:
        raise ValueError("Image array must have height and width dimensions.")
    width = image.shape[1]
    if width <= 0:
        raise ValueError("Image width must be positive.")
    pixel_shift = int(round(width * theta_degrees / 360.0)) % width
    return np.roll(image, pixel_shift, axis=1), pixel_shift


def shift_image_file(
    input_path: Path,
    output_path: Path,
    theta_degrees: float,
    *,
    overwrite: bool = False,
) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output image already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as image:
        shifted_array, pixel_shift = shift_image_array(np.asarray(image), theta_degrees)
        shifted_image = Image.fromarray(shifted_array, mode=image.mode)
    shifted_image.save(output_path)
    return pixel_shift


def confirm_in_place(input_path: Path, image_count: int, theta_degrees: float) -> bool:
    print(
        "WARNING: no --output was provided. "
        f"This will overwrite {image_count} source panorama image(s) in place "
        f"under {input_path} with theta={theta_degrees:g} degrees."
    )
    response = input("Continue? Type 'yes' to overwrite in place: ")
    return response.strip().lower() == "yes"


def shift_image_files(
    paths: list[Path],
    output_dir: Path,
    theta_degrees: float,
    *,
    overwrite: bool,
    workers: int,
) -> set[int]:
    if workers <= 0:
        raise ValueError("workers must be positive")

    def shift_path(path: Path) -> int:
        return shift_image_file(
            path,
            output_dir / path.name,
            theta_degrees,
            overwrite=overwrite,
        )

    if workers == 1:
        return {
            shift_path(path)
            for path in tqdm(paths, desc="Shifting panoramas", unit="image")
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(shift_path, path) for path in paths]
        return set(
            tqdm(
                (future.result() for future in as_completed(futures)),
                total=len(paths),
                desc="Shifting panoramas",
                unit="image",
            )
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not math.isfinite(args.theta):
        raise SystemExit("--theta must be a finite number of degrees")
    if args.workers <= 0:
        raise SystemExit("--workers must be >= 1")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser() if args.output is not None else input_path
    in_place = args.output is None

    if input_path.is_file():
        if in_place and not confirm_in_place(input_path, 1, args.theta):
            print("Aborted; no images were changed.")
            return 1
        pixel_shift = shift_image_file(
            input_path,
            output_path,
            args.theta,
            overwrite=args.overwrite or in_place,
        )
        print(
            f"Shifted {input_path} by theta={args.theta:g} degrees "
            f"({pixel_shift} px) into {output_path.resolve()}"
        )
        return 0

    if not input_path.is_dir():
        raise SystemExit(f"Input path does not exist: {input_path}")

    paths = image_paths(input_path)
    if not paths:
        raise SystemExit(f"No panorama images found in {input_path}")

    if in_place and not confirm_in_place(input_path, len(paths), args.theta):
        print("Aborted; no images were changed.")
        return 1

    if output_path.exists() and not output_path.is_dir():
        raise SystemExit(f"Output path exists and is not a directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    pixel_shifts = shift_image_files(
        paths,
        output_path,
        args.theta,
        overwrite=args.overwrite or in_place,
        workers=args.workers,
    )
    shift_summary = (
        f"{next(iter(pixel_shifts))} px"
        if len(pixel_shifts) == 1
        else f"varied pixel shifts: {sorted(pixel_shifts)}"
    )

    print(
        f"Shifted {len(paths)} panorama image(s) from {input_path} "
        f"by theta={args.theta:g} degrees ({shift_summary}) "
        f"into {output_path.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
