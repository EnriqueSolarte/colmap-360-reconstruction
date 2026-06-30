#!/usr/bin/env python3
"""Download prebuilt COLMAP vocabulary tree files."""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


DEFAULT_RELEASE = "3.11.1"
DEFAULT_DESTINATION_DIR = Path("data") / "colmap-vocab-tree"
GITHUB_RELEASE_URL = "https://github.com/colmap/colmap/releases/download"
DEFAULT_PROGRESS_INTERVAL_BYTES = 64 * 1024 * 1024
DEFAULT_TREE_FORMAT = "faiss"
LEGACY_FORMAT_WARNING = (
    "compatibility note: legacy prebuilt Flickr100K vocabulary trees "
    "without the 'faiss' filename prefix may be FLANN-format files. "
    "Newer COLMAP/pycolmap builds that require FAISS vocabulary trees can "
    "reject them with 'Failed to read faiss index'."
)
FAISS_FORMAT_NOTE = (
    "compatibility note: downloading FAISS-format Flickr100K vocabulary trees "
    "for newer COLMAP/pycolmap builds."
)

VOCAB_TREE_FILENAMES_BY_FORMAT = {
    "legacy": {
        "32k": "vocab_tree_flickr100K_words32K.bin",
        "256k": "vocab_tree_flickr100K_words256K.bin",
        "1m": "vocab_tree_flickr100K_words1M.bin",
    },
    "faiss": {
        "32k": "vocab_tree_faiss_flickr100K_words32K.bin",
        "256k": "vocab_tree_faiss_flickr100K_words256K.bin",
        "1m": "vocab_tree_faiss_flickr100K_words1M.bin",
    },
}
VOCAB_TREE_FILENAMES = VOCAB_TREE_FILENAMES_BY_FORMAT[DEFAULT_TREE_FORMAT]


@dataclass(frozen=True)
class DownloadResult:
    variant: str
    url: str
    path: Path
    status: str
    bytes_written: int = 0


def format_byte_count(byte_count: int) -> str:
    if byte_count < 1024:
        return f"{byte_count} bytes"
    mib = byte_count / (1024 * 1024)
    if mib < 1024:
        return f"{mib:.1f} MiB"
    return f"{mib / 1024:.2f} GiB"


def vocab_tree_url(
    variant: str,
    *,
    tree_format: str = DEFAULT_TREE_FORMAT,
    release: str = DEFAULT_RELEASE,
    base_url: str = GITHUB_RELEASE_URL,
) -> str:
    if tree_format not in VOCAB_TREE_FILENAMES_BY_FORMAT:
        raise ValueError(f"unknown vocab tree format: {tree_format}")
    filenames = VOCAB_TREE_FILENAMES_BY_FORMAT[tree_format]
    if variant not in filenames:
        raise ValueError(f"unknown vocab tree variant: {variant}")
    return f"{base_url}/{release}/{filenames[variant]}"


def selected_variants(variant: str) -> tuple[str, ...]:
    if variant == "all":
        return tuple(VOCAB_TREE_FILENAMES)
    if variant not in VOCAB_TREE_FILENAMES:
        raise ValueError(f"unknown vocab tree variant: {variant}")
    return (variant,)


def download_file(
    url: str,
    destination: Path,
    *,
    overwrite: bool = False,
    timeout_seconds: float = 30.0,
    chunk_size: int = 1024 * 1024,
    progress_interval_bytes: int = DEFAULT_PROGRESS_INTERVAL_BYTES,
    progress: Callable[[str], None] | None = None,
    urlopen: Callable[..., object] = urllib.request.urlopen,
) -> int:
    if destination.exists() and not overwrite:
        if progress is not None:
            progress(f"skipping existing file: {destination}")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "colmap-360-reconstruction"},
    )
    bytes_written = 0
    next_progress_bytes = max(1, progress_interval_bytes)
    if progress is not None:
        progress(f"opening: {url}")
    with urlopen(request, timeout=timeout_seconds) as response:
        with temporary_path.open("wb") as output:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                output.write(chunk)
                bytes_written += len(chunk)
                if bytes_written >= next_progress_bytes:
                    if progress is not None:
                        progress(
                            "downloaded "
                            f"{format_byte_count(bytes_written)} "
                            f"to {temporary_path}",
                        )
                    while next_progress_bytes <= bytes_written:
                        next_progress_bytes += max(1, progress_interval_bytes)
    temporary_path.replace(destination)
    if progress is not None:
        progress(
            f"finished: {destination} "
            f"({format_byte_count(bytes_written)})",
        )
    return bytes_written


def download_vocab_trees(
    *,
    variant: str,
    destination_dir: Path,
    tree_format: str = DEFAULT_TREE_FORMAT,
    release: str = DEFAULT_RELEASE,
    overwrite: bool = False,
    dry_run: bool = False,
    timeout_seconds: float = 30.0,
    progress: Callable[[str], None] | None = None,
    urlopen: Callable[..., object] = urllib.request.urlopen,
) -> tuple[DownloadResult, ...]:
    if tree_format not in VOCAB_TREE_FILENAMES_BY_FORMAT:
        raise ValueError(f"unknown vocab tree format: {tree_format}")
    filenames = VOCAB_TREE_FILENAMES_BY_FORMAT[tree_format]
    results: list[DownloadResult] = []
    for selected_variant in selected_variants(variant):
        filename = filenames[selected_variant]
        destination = destination_dir / filename
        url = vocab_tree_url(
            selected_variant,
            tree_format=tree_format,
            release=release,
        )
        if dry_run:
            if progress is not None:
                progress(f"planned download: {url} -> {destination}")
            status = "planned"
            bytes_written = 0
        elif destination.exists() and not overwrite:
            if progress is not None:
                progress(f"skipping existing file: {destination}")
            status = "exists"
            bytes_written = 0
        else:
            if progress is not None:
                progress(
                    f"downloading {selected_variant} vocabulary tree: "
                    f"{url} -> {destination}",
                )
            bytes_written = download_file(
                url,
                destination,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
                progress=progress,
                urlopen=urlopen,
            )
            status = "downloaded"
        results.append(
            DownloadResult(
                variant=selected_variant,
                url=url,
                path=destination,
                status=status,
                bytes_written=bytes_written,
            ),
        )
    return tuple(results)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download prebuilt COLMAP Flickr100K vocabulary tree files.",
    )
    parser.add_argument(
        "--variant",
        choices=(*VOCAB_TREE_FILENAMES.keys(), "all"),
        default="256k",
        help="Vocabulary tree size to download.",
    )
    parser.add_argument(
        "--format",
        choices=tuple(VOCAB_TREE_FILENAMES_BY_FORMAT),
        default=DEFAULT_TREE_FORMAT,
        help="Vocabulary tree file format to download.",
    )
    parser.add_argument(
        "-d",
        "--destination-dir",
        type=Path,
        default=DEFAULT_DESTINATION_DIR,
        help="Directory where vocabulary tree files will be stored.",
    )
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help="COLMAP GitHub release tag containing the vocabulary tree assets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing vocabulary tree file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing files.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP timeout per request.",
    )
    return parser.parse_args(argv)


def print_results(results: Iterable[DownloadResult]) -> None:
    for result in results:
        detail = (
            f" ({result.bytes_written} bytes)"
            if result.status == "downloaded"
            else ""
        )
        print(f"{result.status}: {result.path}{detail}")
        print(f"source: {result.url}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.format == "legacy":
        print(LEGACY_FORMAT_WARNING)
    else:
        print(FAISS_FORMAT_NOTE)
    results = download_vocab_trees(
        variant=args.variant,
        destination_dir=args.destination_dir.expanduser(),
        tree_format=args.format,
        release=args.release,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        timeout_seconds=args.timeout_seconds,
        progress=print,
    )
    print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
