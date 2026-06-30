"""Command-line interface for keyframe cluster pair planning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from colmap.panorama.cluster_pipeline import build_keyframe_pair_plan
from colmap.panorama.constants import NATIVE_CAMERA_SETUP
from colmap.panorama.database import read_generated_image_batch
from colmap.panorama.keyframe_matching import (
    KeyframePairPlan,
    KeyframeSelectionOptions,
)
from colmap.panorama.paths import panorama_workspace_paths
from colmap.panorama.pycolmap_adapter import require_pycolmap
from colmap.panorama.reports import write_image_pair_list


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-equirectangular"


class PanoramaClustersArgumentDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.required and action.default is None:
            return action.help
        return super()._get_help_string(action)


def read_metadata_image_names(metadata_path: Path) -> tuple[str, ...]:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    panorama_images = data.get("panorama_images")
    if not isinstance(panorama_images, list):
        raise ValueError(f"metadata does not contain panorama_images: {metadata_path}")
    image_names = tuple(
        str(image_data["colmap_image_name"])
        for image_data in panorama_images
        if isinstance(image_data, dict) and "colmap_image_name" in image_data
    )
    if not image_names:
        raise ValueError(f"metadata does not contain COLMAP image names: {metadata_path}")
    return image_names


def keyframe_options_from_cluster_args(
    args: argparse.Namespace,
) -> KeyframeSelectionOptions:
    return KeyframeSelectionOptions(
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
    )


def build_keyframe_cluster_report(
    *,
    paths: dict[str, Path],
    report_path: Path,
    match_pairs_path: Path,
    image_names: tuple[str, ...],
    keyframe_options: KeyframeSelectionOptions,
    keyframe_pair_plan: KeyframePairPlan,
    vocab_tree_path: Path | None,
) -> dict[str, Any]:
    pair_counts = keyframe_pair_plan.pair_counts_by_source
    return {
        "status": f"{NATIVE_CAMERA_SETUP}_keyframe_clusters",
        "camera_setup": NATIVE_CAMERA_SETUP,
        "workspace": str(paths["workspace"]),
        "database_path": str(paths["database"]),
        "metadata_path": str(paths["metadata"]),
        "match_pairs_path": str(match_pairs_path),
        "report_path": str(report_path),
        "vocab_tree_path": None if vocab_tree_path is None else str(vocab_tree_path),
        "options": {
            "keyframe_selection": keyframe_options.selection,
            "effective_pairing": keyframe_pair_plan.effective_selection,
            "keyframe_interval": keyframe_options.interval,
            "keyframe_overlap": keyframe_options.overlap,
            "keyframe_backbone_policy": keyframe_options.backbone_policy,
            "keyframe_backbone_overlap": keyframe_options.backbone_overlap,
            "keyframe_cluster_min_size": keyframe_options.cluster_min_size,
            "keyframe_cluster_max_size": keyframe_options.cluster_max_size,
            "keyframe_cluster_recent_window": (
                keyframe_options.cluster_recent_window
            ),
            "keyframe_vocabtree_neighbors": keyframe_options.vocabtree_neighbors,
            "keyframe_vocabtree_checks": keyframe_options.vocabtree_checks,
            "keyframe_pair_count": pair_counts["keyframe"],
            "vocabtree_pair_count": pair_counts["vocabtree"],
            "sequential_backbone_pair_count": pair_counts["sequential_backbone"],
            "matching_pair_count": pair_counts["all"],
        },
        "generated_image_count": len(image_names),
        "generated_image_names": list(image_names),
        "keyframe_count": len(keyframe_pair_plan.keyframes),
        "cluster_count": len(keyframe_pair_plan.clusters),
        "keyframes": keyframe_pair_plan.report_keyframes(),
        "keyframe_clusters": keyframe_pair_plan.report_clusters(),
        "pair_counts_by_source": pair_counts,
    }


def parse_cluster_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build keyframe/vocabtree cluster image pairs from an existing "
            "panorama workspace. Run this after image import and feature "
            "extraction, before matching."
        ),
        formatter_class=PanoramaClustersArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-w",
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="Existing COLMAP equirectangular workspace.",
    )
    parser.add_argument(
        "--vocab-tree-path",
        type=Path,
        default=None,
        help="Local COLMAP vocabulary tree for sequential-vocabtree clustering.",
    )
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=None,
        help="Image-pair list to write for COLMAP matching.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Cluster planning report path.",
    )
    parser.add_argument(
        "--keyframe-selection",
        choices=("interval", "sequential-vocabtree"),
        default="sequential-vocabtree",
        help="Keyframe selection strategy.",
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
    args = parser.parse_args(argv)
    paths = panorama_workspace_paths(args.workspace)
    if not args.workspace.is_dir():
        parser.error(f"--workspace must be an existing directory: {args.workspace}")
    if not paths["metadata"].is_file():
        parser.error(f"workspace metadata does not exist: {paths['metadata']}")
    if not paths["database"].is_file():
        parser.error(f"workspace database does not exist: {paths['database']}")
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
        args.keyframe_selection == "sequential-vocabtree"
        and args.vocab_tree_path is None
    ):
        parser.error(
            "--vocab-tree-path is required for "
            "--keyframe-selection sequential-vocabtree"
        )
    if args.vocab_tree_path is not None and not args.vocab_tree_path.is_file():
        parser.error(f"--vocab-tree-path does not exist: {args.vocab_tree_path}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_cluster_args(sys.argv[1:] if argv is None else argv)
    paths = panorama_workspace_paths(args.workspace)
    output_pairs = args.output_pairs or paths["match_pairs"]
    report_path = args.report_path or paths["cluster_report"]
    image_names = read_metadata_image_names(paths["metadata"])
    generated_image_batch = read_generated_image_batch(paths["database"], image_names)
    keyframe_options = keyframe_options_from_cluster_args(args)
    pycolmap_module = (
        require_pycolmap()
        if keyframe_options.selection == "sequential-vocabtree"
        else None
    )
    keyframe_pair_plan = build_keyframe_pair_plan(
        pycolmap_module,
        database_path=paths["database"],
        generated_image_batch=generated_image_batch,
        keyframe_options=keyframe_options,
        vocab_tree_path=args.vocab_tree_path,
    )
    write_image_pair_list(output_pairs, keyframe_pair_plan.all_pairs)
    report = build_keyframe_cluster_report(
        paths=paths,
        report_path=report_path,
        match_pairs_path=output_pairs,
        image_names=image_names,
        keyframe_options=keyframe_options,
        keyframe_pair_plan=keyframe_pair_plan,
        vocab_tree_path=args.vocab_tree_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"workspace: {report['workspace']}")
    print(f"panorama images: {report['generated_image_count']}")
    print(f"keyframes: {report['keyframe_count']}")
    print(f"clusters: {report['cluster_count']}")
    print(f"matching pairs: {report['options']['matching_pair_count']}")
    print(f"effective pairing: {report['options']['effective_pairing']}")
    print(f"pairs: {report['match_pairs_path']}")
    print(f"report: {report['report_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
