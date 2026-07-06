"""Command-line interface for pinhole reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colmap.panorama.pycolmap_adapter import parse_gpu_index
from colmap.panorama.types import FeaturePipelineOptions, PipelineExecutionOptions
from colmap.pinhole.pipeline import run_pinhole_reconstruction


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-pinhole"


class PinholeArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.required and action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run COLMAP reconstruction from rendered pinhole images using "
            "render_report.json as the metadata and FOV manifest."
        ),
        formatter_class=PinholeArgumentDefaultsHelpFormatter,
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
        "--overlap",
        type=int,
        default=10,
        help="Sequential matcher overlap in neighboring pinhole images.",
    )
    parser.add_argument(
        "--loop-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vocabulary-tree loop detection during sequential matching.",
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
        help="Local feature matcher. The pipeline is structured for future backends.",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=8192,
        help="Maximum SIFT features per pinhole image.",
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
        "--guided-matching",
        action="store_true",
        help="Enable guided feature matching after geometric verification.",
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
            "extraction and matching. Use -1 for all GPUs."
        ),
    )
    parser.add_argument(
        "--gpu-bundle-adjustment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use GPU bundle adjustment when CUDA is selected. Bundle adjustment "
            "uses the first selected GPU."
        ),
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
        help="Delete and rebuild the pinhole workspace.",
    )
    args = parser.parse_args(argv)
    if not args.render_report.is_file():
        parser.error(f"--render-report must be an existing file: {args.render_report}")
    if args.image_dir is not None and not args.image_dir.is_dir():
        parser.error(f"--image-dir must be an existing directory: {args.image_dir}")
    if args.overlap <= 0:
        parser.error("--overlap must be positive")
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
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run_pinhole_reconstruction(
        render_report_path=args.render_report,
        workspace=args.workspace,
        image_dir=args.image_dir,
        overlap=args.overlap,
        loop_detection=args.loop_detection,
        ssl_cert_file=args.ssl_cert_file,
        feature_options=FeaturePipelineOptions(
            extractor=args.feature_extractor,
            matcher=args.feature_matcher,
            max_num_features=args.max_num_features,
            max_image_size=args.max_image_size,
            num_threads=args.num_threads,
            guided_matching=args.guided_matching,
        ),
        execution_options=PipelineExecutionOptions(
            device=args.device,
            gpu_index=args.gpu_index,
            gpu_bundle_adjustment=args.gpu_bundle_adjustment,
        ),
        import_workers=args.import_workers,
        overwrite=args.overwrite,
    )
    print(f"workspace: {report['workspace']}")
    print(f"camera setup: {report['camera_setup']}")
    print(f"camera model: {report['camera']['model']}")
    print(f"pinhole images: {report['rendered_image_count']}")
    print(f"compute device: {report['options']['resolved_device']}")
    print(
        "pinhole images with features: "
        f"{report['feature_health']['pinhole_images_with_features']}/"
        f"{report['feature_health']['expected_pinhole_images']}"
    )
    print(
        "best sparse model: "
        f"{report['best_sparse_path']} "
        f"({report['best_model']['num_registered_images']} registered images, "
        f"{report['best_model']['num_points3D']} points)"
    )
    print(f"elapsed: {report['elapsed_time']}")
    print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
