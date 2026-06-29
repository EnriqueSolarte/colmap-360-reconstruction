"""Command-line interface for panorama feature extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colmap.panorama.feature_pipeline import run_panorama_feature_extraction
from colmap.panorama.pycolmap_adapter import parse_gpu_index
from colmap.panorama.types import FeaturePipelineOptions, PipelineExecutionOptions


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-equirectangular"


class PanoramaFeaturesArgumentDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.required and action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_feature_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import equirectangular panorama images and extract COLMAP "
            "features without matching or mapping."
        ),
        formatter_class=PanoramaFeaturesArgumentDefaultsHelpFormatter,
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
        "--ssl-cert-file",
        type=Path,
        default=None,
        help=(
            "CA certificate bundle for COLMAP downloads. Defaults to "
            "$SSL_CERT_FILE or /etc/ssl/certs/ca-certificates.crt."
        ),
    )
    parser.add_argument(
        "--feature-extractor",
        choices=("sift",),
        default="sift",
        help="Local feature extractor. The pipeline is structured for future backends.",
    )
    parser.add_argument(
        "--feature-matcher",
        choices=("sift-bruteforce",),
        default="sift-bruteforce",
        help="Feature matcher option recorded for downstream compatibility.",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="Maximum SIFT features per panorama image.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=-1,
        help=(
            "Maximum image size before feature extraction. Use -1 for "
            "COLMAP's automatic setting."
        ),
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=-1,
        help=(
            "Number of threads for feature extraction. Use -1 for COLMAP's "
            "automatic setting."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="COLMAP compute device. Auto uses CUDA when pycolmap supports it.",
    )
    parser.add_argument(
        "--gpu-index",
        type=parse_gpu_index,
        default="-1",
        help=(
            "CUDA GPU index, or comma-separated indices for multi-GPU feature "
            "extraction. Use -1 for all GPUs."
        ),
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
        help="Delete and rebuild the feature workspace.",
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
    if args.max_num_features <= 0:
        parser.error("--max-num-features must be positive")
    if args.max_image_size < -1 or args.max_image_size == 0:
        parser.error("--max-image-size must be -1 or positive")
    if args.num_threads < -1 or args.num_threads == 0:
        parser.error("--num-threads must be -1 or positive")
    if args.ssl_cert_file is not None and not args.ssl_cert_file.is_file():
        parser.error(f"--ssl-cert-file does not exist: {args.ssl_cert_file}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_feature_args(sys.argv[1:] if argv is None else argv)
    report = run_panorama_feature_extraction(
        input_image_path=args.input_image_path,
        workspace=args.workspace,
        ssl_cert_file=args.ssl_cert_file,
        feature_options=FeaturePipelineOptions(
            extractor=args.feature_extractor,
            matcher=args.feature_matcher,
            max_num_features=args.max_num_features,
            max_image_size=args.max_image_size,
            num_threads=args.num_threads,
        ),
        execution_options=PipelineExecutionOptions(
            device=args.device,
            gpu_index=args.gpu_index,
        ),
        image_limit=args.image_limit,
        import_workers=args.import_workers,
        overwrite=args.overwrite,
    )
    print(f"workspace: {report['workspace']}")
    print(f"camera setup: {report['camera_setup']}")
    print(f"camera model: {report['camera']['model']}")
    print(f"panorama images: {report['panorama_count']}")
    print(f"compute device: {report['options']['resolved_device']}")
    print(
        "panorama images with features: "
        f"{report['feature_health']['panorama_images_with_features']}/"
        f"{report['feature_health']['expected_panorama_images']}"
    )
    print(f"elapsed: {report['elapsed_time']}")
    print(f"metadata: {report['metadata_path']}")
    print(f"database: {report['database_path']}")
    print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
