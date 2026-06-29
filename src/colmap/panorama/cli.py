"""Command-line interface for panorama reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colmap.panorama.pipeline import run_panorama_reconstruction
from colmap.panorama.pycolmap_adapter import parse_gpu_index
from colmap.panorama.types import FeaturePipelineOptions, PipelineExecutionOptions


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-equirectangular"


class PanoramaArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.required and action.default is None:
            return action.help
        return super()._get_help_string(action)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run COLMAP reconstruction directly from equirectangular panorama "
            "images using COLMAP's native EQUIRECTANGULAR camera model."
        ),
        formatter_class=PanoramaArgumentDefaultsHelpFormatter,
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
        "--matcher",
        choices=("sequential", "exhaustive", "vocabtree"),
        default="sequential",
        help="Feature matching strategy.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=15,
        help="Sequential matcher overlap in neighboring images.",
    )
    parser.add_argument(
        "--loop-detection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable vocabulary-tree loop detection during sequential matching.",
    )
    parser.add_argument(
        "--vocab-tree-path",
        type=Path,
        default=None,
        help=(
            "Local COLMAP vocabulary tree for loop detection or vocabtree "
            "matching. If omitted, COLMAP may download its default tree."
        ),
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
        help="Maximum SIFT features per panorama image.",
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
        "--image-limit",
        type=int,
        default=None,
        help="Use only the first N sorted panoramas for smoke testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the equirectangular workspace.",
    )
    args = parser.parse_args(argv)
    if not args.input_image_path.is_dir():
        parser.error(f"--input-image-path must be an existing directory: {args.input_image_path}")
    if args.image_limit is not None and args.image_limit <= 0:
        parser.error("--image-limit must be positive")
    if args.max_num_features <= 0:
        parser.error("--max-num-features must be positive")
    if args.vocab_tree_path is not None and not args.vocab_tree_path.is_file():
        parser.error(f"--vocab-tree-path does not exist: {args.vocab_tree_path}")
    if args.ssl_cert_file is not None and not args.ssl_cert_file.is_file():
        parser.error(f"--ssl-cert-file does not exist: {args.ssl_cert_file}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run_panorama_reconstruction(
        input_image_path=args.input_image_path,
        workspace=args.workspace,
        matcher=args.matcher,
        overlap=args.overlap,
        loop_detection=args.loop_detection,
        vocab_tree_path=args.vocab_tree_path,
        ssl_cert_file=args.ssl_cert_file,
        feature_options=FeaturePipelineOptions(
            extractor=args.feature_extractor,
            matcher=args.feature_matcher,
            max_num_features=args.max_num_features,
            guided_matching=args.guided_matching,
        ),
        execution_options=PipelineExecutionOptions(
            device=args.device,
            gpu_index=args.gpu_index,
            gpu_bundle_adjustment=args.gpu_bundle_adjustment,
        ),
        image_limit=args.image_limit,
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
    print(
        "best sparse model: "
        f"{report['best_sparse_path']} "
        f"({report['best_model']['num_registered_images']} registered images, "
        f"{report['best_model']['num_points3D']} points)"
    )
    print(f"elapsed: {report['elapsed_time']}")
    print(f"report: {report['workspace']}/panorama_equirectangular_report.json")
    return 0
