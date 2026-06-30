"""Command-line interface for panorama reconstruction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colmap.panorama.pipeline import run_panorama_reconstruction
from colmap.panorama.pycolmap_adapter import parse_gpu_index
from colmap.panorama.keyframe_matching import KeyframeSelectionOptions
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
        choices=("sequential", "exhaustive", "vocabtree", "keyframes"),
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
        "--keyframe-selection",
        choices=("interval", "sequential-vocabtree"),
        default="sequential-vocabtree",
        help="Keyframe selection strategy when --matcher keyframes is used.",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=10,
        help="Interval between deterministic keyframes.",
    )
    parser.add_argument(
        "--keyframe-overlap",
        type=int,
        default=3,
        help="Number of neighboring keyframes to connect.",
    )
    parser.add_argument(
        "--keyframe-backbone-policy",
        choices=("none", "gaps", "all"),
        default="gaps",
        help="Sequential fallback policy for keyframe matching.",
    )
    parser.add_argument(
        "--keyframe-backbone-overlap",
        type=int,
        default=1,
        help="Sequential fallback overlap for keyframe matching.",
    )
    parser.add_argument(
        "--keyframe-cluster-min-size",
        type=int,
        default=2,
        help="Minimum keyframe cluster size target.",
    )
    parser.add_argument(
        "--keyframe-cluster-max-size",
        type=int,
        default=30,
        help="Maximum ordered keyframe cluster size.",
    )
    parser.add_argument(
        "--keyframe-cluster-recent-window",
        type=int,
        default=3,
        help="Recent images checked when extending an ordered vocabtree cluster.",
    )
    parser.add_argument(
        "--keyframe-vocabtree-neighbors",
        type=int,
        default=10,
        help="Nearest vocabtree neighbors for keyframe candidate generation.",
    )
    parser.add_argument(
        "--keyframe-vocabtree-checks",
        type=int,
        default=64,
        help="Vocabtree checks for keyframe candidate generation.",
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
        "--feature-workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads for sharded feature extraction. Values "
            "greater than 1 extract into shard databases and merge them."
        ),
    )
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=None,
        help=(
            "Images per feature shard. If omitted, batches are split across "
            "--feature-workers."
        ),
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
    if args.import_workers <= 0:
        parser.error("--import-workers must be positive")
    if args.feature_workers <= 0:
        parser.error("--feature-workers must be positive")
    if args.feature_batch_size is not None and args.feature_batch_size <= 0:
        parser.error("--feature-batch-size must be positive")
    if args.max_num_features <= 0:
        parser.error("--max-num-features must be positive")
    if args.max_image_size < -1 or args.max_image_size == 0:
        parser.error("--max-image-size must be -1 or positive")
    if args.num_threads < -1 or args.num_threads == 0:
        parser.error("--num-threads must be -1 or positive")
    if args.keyframe_interval <= 0:
        parser.error("--keyframe-interval must be positive")
    if args.keyframe_overlap < 0:
        parser.error("--keyframe-overlap must be non-negative")
    if args.keyframe_backbone_overlap < 0:
        parser.error("--keyframe-backbone-overlap must be non-negative")
    if args.keyframe_cluster_min_size <= 0:
        parser.error("--keyframe-cluster-min-size must be positive")
    if args.keyframe_cluster_max_size <= 0:
        parser.error("--keyframe-cluster-max-size must be positive")
    if args.keyframe_cluster_min_size > args.keyframe_cluster_max_size:
        parser.error("--keyframe-cluster-min-size cannot exceed max size")
    if args.keyframe_cluster_recent_window < 0:
        parser.error("--keyframe-cluster-recent-window must be non-negative")
    if args.keyframe_vocabtree_neighbors <= 0:
        parser.error("--keyframe-vocabtree-neighbors must be positive")
    if args.keyframe_vocabtree_checks <= 0:
        parser.error("--keyframe-vocabtree-checks must be positive")
    if (
        args.matcher == "keyframes"
        and args.keyframe_selection == "sequential-vocabtree"
        and args.vocab_tree_path is None
    ):
        parser.error(
            "--vocab-tree-path is required for "
            "--matcher keyframes --keyframe-selection sequential-vocabtree"
        )
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
            max_image_size=args.max_image_size,
            num_threads=args.num_threads,
            guided_matching=args.guided_matching,
        ),
        execution_options=PipelineExecutionOptions(
            device=args.device,
            gpu_index=args.gpu_index,
            gpu_bundle_adjustment=args.gpu_bundle_adjustment,
        ),
        image_limit=args.image_limit,
        import_workers=args.import_workers,
        feature_workers=args.feature_workers,
        feature_batch_size=args.feature_batch_size,
        keyframe_options=KeyframeSelectionOptions(
            selection=args.keyframe_selection,
            interval=args.keyframe_interval,
            overlap=args.keyframe_overlap,
            backbone_policy=args.keyframe_backbone_policy,
            backbone_overlap=args.keyframe_backbone_overlap,
            cluster_min_size=args.keyframe_cluster_min_size,
            cluster_max_size=args.keyframe_cluster_max_size,
            cluster_recent_window=args.keyframe_cluster_recent_window,
            vocabtree_neighbors=args.keyframe_vocabtree_neighbors,
            vocabtree_checks=args.keyframe_vocabtree_checks,
        ),
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
