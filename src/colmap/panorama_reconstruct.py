"""Run COLMAP SfM directly from equirectangular panoramas."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from colmap.camera import ColmapCamera, camera_from_equirectangular
try:  # Keep argument parsing and tests usable before pycolmap is installed.
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-equirectangular"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
COLMAP_IMAGE_PAIR_ID_PRIME = 2_147_483_647
FILENAME_TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{10,})(?!\d)")
NATIVE_CAMERA_SETUP = "native_equirectangular"


class PanoramaArgumentDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.required and action.default is None:
            return action.help
        return super()._get_help_string(action)


@dataclass(frozen=True)
class FeaturePipelineOptions:
    extractor: str = "sift"
    matcher: str = "sift-bruteforce"
    max_num_features: int = 8192
    guided_matching: bool = False


@dataclass(frozen=True)
class PipelineExecutionOptions:
    device: str = "auto"
    gpu_index: str = "-1"
    gpu_bundle_adjustment: bool = True


@dataclass(frozen=True)
class GeneratedImageBatch:
    names: tuple[str, ...]
    records_by_name: dict[str, tuple[int, int]]

    @property
    def image_ids(self) -> list[int]:
        return [self.records_by_name[name][0] for name in self.names]

    @property
    def camera_ids(self) -> list[int]:
        return [self.records_by_name[name][1] for name in self.names]


@dataclass(frozen=True)
class KeyframeAnchorScore:
    frame_id: int
    anchor_score: float
    connectivity_score: float
    triangulated_support_score: float
    image_distribution_score: float
    reprojection_quality_score: float
    connected_frame_count: int
    verified_inlier_count: int
    triangulated_point_count: int
    active_image_count: int
    total_image_count: int
    active_image_ratio: float
    image_entropy: float
    median_reprojection_error_px: float | None
    mean_reprojection_error_px: float | None
    image_ids: tuple[int, ...]
    image_names: tuple[str, ...]
    triangulated_points_by_image: tuple[int, ...]

    def report(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "anchor_score": round_report_float(self.anchor_score),
            "connectivity_score": round_report_float(self.connectivity_score),
            "triangulated_support_score": round_report_float(
                self.triangulated_support_score,
            ),
            "image_distribution_score": round_report_float(
                self.image_distribution_score,
            ),
            "reprojection_quality_score": round_report_float(
                self.reprojection_quality_score,
            ),
            "connected_frame_count": self.connected_frame_count,
            "verified_inlier_count": self.verified_inlier_count,
            "triangulated_point_count": self.triangulated_point_count,
            "active_image_count": self.active_image_count,
            "total_image_count": self.total_image_count,
            "active_image_ratio": round_report_float(self.active_image_ratio),
            "image_entropy": round_report_float(self.image_entropy),
            "median_reprojection_error_px": optional_report_float(
                self.median_reprojection_error_px,
            ),
            "mean_reprojection_error_px": optional_report_float(
                self.mean_reprojection_error_px,
            ),
            "image_ids": list(self.image_ids),
            "image_names": list(self.image_names),
            "triangulated_points_by_image": list(self.triangulated_points_by_image),
        }


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required for panorama reconstruction. Install the "
            "`colmap` extra or add `pycolmap` to the environment."
        )
    return pycolmap


def configure_ssl_cert_file(ssl_cert_file: Path | None = None) -> None:
    """Configure the CA bundle used by COLMAP/libcurl for optional downloads."""
    if ssl_cert_file is not None:
        if not ssl_cert_file.is_file():
            raise FileNotFoundError(f"SSL certificate bundle not found: {ssl_cert_file}")
        os.environ["SSL_CERT_FILE"] = str(ssl_cert_file)
        return

    if os.environ.get("SSL_CERT_FILE"):
        return

    default_cert_file = Path("/etc/ssl/certs/ca-certificates.crt")
    if default_cert_file.is_file():
        os.environ["SSL_CERT_FILE"] = str(default_cert_file)


def hardlink_or_symlink_file(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except FileExistsError:
        if os.path.samefile(source, destination):
            return destination
        raise
    except OSError:
        try:
            os.symlink(os.path.abspath(source), destination)
        except FileExistsError:
            if os.path.samefile(source, destination):
                return destination
            raise
    return destination


def write_best_model(best_reconstruction: Any, paths: dict[str, Path]) -> None:
    if paths["best_sparse"].exists():
        shutil.rmtree(paths["best_sparse"])
    paths["best_sparse"].mkdir(parents=True, exist_ok=True)
    best_reconstruction.write(paths["best_sparse"])
    best_reconstruction.export_PLY(paths["raw_ply"])


def reconstruction_stats(reconstruction: Any) -> dict[str, Any]:
    return {
        "num_images": reconstruction.num_images(),
        "num_registered_images": reconstruction.num_reg_images(),
        "num_points3D": reconstruction.num_points3D(),
        "num_observations": reconstruction.compute_num_observations(),
        "mean_observations_per_registered_image": (
            reconstruction.compute_mean_observations_per_reg_image()
        ),
        "mean_track_length": reconstruction.compute_mean_track_length(),
        "mean_reprojection_error_px": reconstruction.compute_mean_reprojection_error(),
    }


def read_database_stats(database_path: Path) -> dict[str, int]:
    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        tables = {
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

        def count(table: str) -> int:
            if table not in tables:
                return 0
            return int(cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

        return {
            "cameras": count("cameras"),
            "images": count("images"),
            "keypoints": count("keypoints"),
            "descriptors": count("descriptors"),
            "matches": count("matches"),
            "two_view_geometries": count("two_view_geometries"),
        }


def read_image_records_by_name(
    database_path: Path,
    image_names: Sequence[str],
) -> dict[str, tuple[int, int]]:
    records: dict[str, tuple[int, int]] = {}
    with sqlite3.connect(database_path) as connection:
        for image_name in image_names:
            row = connection.execute(
                """
                SELECT image_id, camera_id
                FROM images
                WHERE name = ?
                """,
                (image_name,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Expected image in database: {image_name}")
            records[image_name] = (int(row[0]), int(row[1]))
    return records


def read_generated_image_batch(
    database_path: Path,
    image_names: Sequence[str],
) -> GeneratedImageBatch:
    return GeneratedImageBatch(
        names=tuple(image_names),
        records_by_name=read_image_records_by_name(database_path, image_names),
    )


def write_image_pair_list(
    path: Path,
    pairs: Sequence[tuple[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{image_name_a} {image_name_b}\n"
            for image_name_a, image_name_b in pairs
        ),
        encoding="utf-8",
    )


def round_report_float(value: float) -> float:
    return round(float(value), 6)


def optional_report_float(value: float | None) -> float | None:
    return None if value is None else round_report_float(value)


def log_capped_score(value: int | float, cap: int | float) -> float:
    if value <= 0 or cap <= 0:
        return 0.0
    return min(log(1.0 + float(value)) / log(1.0 + float(cap)), 1.0)


def linear_capped_score(value: int | float, cap: int | float) -> float:
    if value <= 0 or cap <= 0:
        return 0.0
    return min(float(value) / float(cap), 1.0)


def decode_colmap_image_pair_id(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % COLMAP_IMAGE_PAIR_ID_PRIME
    image_id1 = (pair_id - image_id2) // COLMAP_IMAGE_PAIR_ID_PRIME
    return int(image_id1), int(image_id2)


def read_frame_connection_stats(
    database_path: Path,
    frame_id_by_image_id: dict[int, int],
) -> dict[int, dict[str, int | set[int]]]:
    stats: dict[int, dict[str, int | set[int]]] = {
        frame_id: {"connected_frame_ids": set(), "verified_inlier_count": 0}
        for frame_id in set(frame_id_by_image_id.values())
    }
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "two_view_geometries" not in tables:
            return stats
        rows = connection.execute(
            """
            SELECT pair_id, rows
            FROM two_view_geometries
            WHERE rows > 0
            """
        ).fetchall()
    for pair_id, inlier_count in rows:
        image_id1, image_id2 = decode_colmap_image_pair_id(int(pair_id))
        frame_id1 = frame_id_by_image_id.get(image_id1)
        frame_id2 = frame_id_by_image_id.get(image_id2)
        if frame_id1 is None or frame_id2 is None or frame_id1 == frame_id2:
            continue
        frame1_connected = stats[frame_id1]["connected_frame_ids"]
        frame2_connected = stats[frame_id2]["connected_frame_ids"]
        if not isinstance(frame1_connected, set) or not isinstance(frame2_connected, set):
            raise AssertionError("invalid frame connection stats")
        frame1_connected.add(frame_id2)
        frame2_connected.add(frame_id1)
        stats[frame_id1]["verified_inlier_count"] = int(
            stats[frame_id1]["verified_inlier_count"],
        ) + int(inlier_count)
        stats[frame_id2]["verified_inlier_count"] = int(
            stats[frame_id2]["verified_inlier_count"],
        ) + int(inlier_count)
    return stats


def normalized_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    positive_counts = [count for count in counts if count > 0]
    if total <= 0 or len(positive_counts) <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * log(count / total)
        for count in positive_counts
    )
    return entropy / log(len(counts))


def score_keyframe_anchors(
    database_path: Path,
    reconstruction: Any,
    *,
    min_active_image_points: int = 30,
    connectivity_frame_cap: int = 10,
    verified_inlier_cap: int = 5000,
    triangulated_point_cap: int = 3000,
    reprojection_error_scale_px: float = 2.0,
) -> list[KeyframeAnchorScore]:
    frames: dict[int, list[Any]] = {}
    for image_id in reconstruction.reg_image_ids():
        image = reconstruction.image(image_id)
        frame_id = colmap_frame_id_for_image(image)
        frames.setdefault(frame_id, []).append(image)

    frame_id_by_image_id = {
        int(image.image_id): frame_id
        for frame_id, images in frames.items()
        for image in images
    }
    connection_stats = read_frame_connection_stats(database_path, frame_id_by_image_id)
    scores = []
    for frame_id, images in frames.items():
        ordered_images = sorted(images, key=lambda image: int(image.image_id))
        triangulated_points_by_image = []
        point_ids: set[int] = set()
        for image in ordered_images:
            image_point_ids = {
                int(point2d.point3D_id)
                for point2d in image.points2D
                if point2d.has_point3D()
            }
            triangulated_points_by_image.append(len(image_point_ids))
            point_ids.update(image_point_ids)

        point_errors = [
            float(reconstruction.points3D[point_id].error)
            for point_id in point_ids
            if point_id in reconstruction.points3D
            and reconstruction.points3D[point_id].has_error()
        ]
        median_error = (
            None if not point_errors else float(np.median(np.asarray(point_errors)))
        )
        mean_error = (
            None if not point_errors else float(np.mean(np.asarray(point_errors)))
        )
        active_image_count = sum(
            count >= min_active_image_points
            for count in triangulated_points_by_image
        )
        total_image_count = max(len(ordered_images), 1)
        active_image_ratio = min(active_image_count / total_image_count, 1.0)
        image_entropy = normalized_entropy(triangulated_points_by_image)
        image_distribution_score = 0.5 * active_image_ratio + 0.5 * image_entropy

        frame_connection_stats = connection_stats.get(
            frame_id,
            {"connected_frame_ids": set(), "verified_inlier_count": 0},
        )
        connected_frame_ids = frame_connection_stats["connected_frame_ids"]
        if not isinstance(connected_frame_ids, set):
            raise AssertionError("invalid connected frame set")
        connected_frame_count = len(connected_frame_ids)
        verified_inlier_count = int(frame_connection_stats["verified_inlier_count"])
        connectivity_score = 0.5 * linear_capped_score(
            connected_frame_count,
            connectivity_frame_cap,
        ) + 0.5 * log_capped_score(verified_inlier_count, verified_inlier_cap)
        triangulated_support_score = log_capped_score(
            len(point_ids),
            triangulated_point_cap,
        )
        reprojection_quality_score = (
            0.0
            if median_error is None
            else 1.0 / (1.0 + median_error / reprojection_error_scale_px)
        )
        anchor_score = (
            connectivity_score
            * triangulated_support_score
            * image_distribution_score
            * reprojection_quality_score
        )
        scores.append(
            KeyframeAnchorScore(
                frame_id=frame_id,
                anchor_score=anchor_score,
                connectivity_score=connectivity_score,
                triangulated_support_score=triangulated_support_score,
                image_distribution_score=image_distribution_score,
                reprojection_quality_score=reprojection_quality_score,
                connected_frame_count=connected_frame_count,
                verified_inlier_count=verified_inlier_count,
                triangulated_point_count=len(point_ids),
                active_image_count=active_image_count,
                total_image_count=total_image_count,
                active_image_ratio=active_image_ratio,
                image_entropy=image_entropy,
                median_reprojection_error_px=median_error,
                mean_reprojection_error_px=mean_error,
                image_ids=tuple(int(image.image_id) for image in ordered_images),
                image_names=tuple(str(image.name) for image in ordered_images),
                triangulated_points_by_image=tuple(triangulated_points_by_image),
            )
        )
    return sorted(
        scores,
        key=lambda score: (
            score.anchor_score,
            score.connected_frame_count,
            score.triangulated_point_count,
            -score.frame_id,
        ),
        reverse=True,
    )


def colmap_frame_id_for_image(image: Any) -> int:
    frame_id = int(getattr(image, "frame_id", -1))
    if frame_id >= 0:
        return frame_id
    return int(image.image_id)


def panorama_image_paths(input_image_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_image_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=panorama_timestamp_sort_key,
    )


def panorama_timestamp_sort_key(path: Path) -> tuple[int, int | str, str]:
    timestamp = timestamp_from_image_name(path.name)
    if timestamp is None:
        return (1, path.name, path.name)
    return (0, timestamp, path.name)


def import_equirectangular_panorama_images(
    input_image_path: Path,
    output_image_dir: Path,
    *,
    image_limit: int | None = None,
    fail_on_existing: bool = False,
    relative_name_prefix: Path | None = None,
) -> tuple[list[Path], ColmapCamera, list[str]]:
    panorama_paths = panorama_image_paths(input_image_path)
    if image_limit is not None:
        panorama_paths = panorama_paths[:image_limit]
    if not panorama_paths:
        raise RuntimeError(f"No panorama images found in {input_image_path}")

    first_width, first_height = image_resolution(panorama_paths[0])
    camera = camera_from_equirectangular(first_width, first_height)
    image_names: list[str] = []
    for panorama_path in tqdm(
        panorama_paths,
        desc="Importing equirectangular panoramas",
        unit="image",
    ):
        width, height = image_resolution(panorama_path)
        if (width, height) != (first_width, first_height):
            raise ValueError(
                f"Panorama {panorama_path} has size {width}x{height}; "
                f"expected {first_width}x{first_height}"
            )
        relative_name = prefixed_panorama_relative_name(
            input_image_path,
            panorama_path,
            relative_name_prefix,
        )
        output_path = output_image_dir / relative_name
        if fail_on_existing and output_path.exists():
            raise FileExistsError(
                "New panorama image would overwrite a previous session file: "
                f"{output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        hardlink_or_symlink_file(str(panorama_path), str(output_path))
        image_names.append(relative_name.as_posix())

    return panorama_paths, camera, image_names


def image_resolution(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def prefixed_panorama_relative_name(
    input_image_path: Path,
    panorama_path: Path,
    relative_name_prefix: Path | None = None,
) -> Path:
    relative_name = panorama_path.relative_to(input_image_path)
    if relative_name_prefix is None:
        return relative_name
    return Path(f"{relative_name_prefix.as_posix()}_{relative_name.as_posix()}")


def timestamp_from_image_name(name: str) -> int | None:
    match = FILENAME_TIMESTAMP_PATTERN.search(name)
    if match is None:
        return None
    return int(match.group(1))


def create_feature_extraction_options(
    pycolmap_module: Any,
    options: FeaturePipelineOptions,
    *,
    use_gpu: bool = False,
    gpu_index: str = "-1",
) -> Any:
    if options.extractor != "sift":
        raise ValueError(f"Unsupported feature extractor: {options.extractor}")
    extraction_options = pycolmap_module.FeatureExtractionOptions()
    extraction_options.type = pycolmap_module.FeatureExtractorType.SIFT
    extraction_options.use_gpu = use_gpu
    extraction_options.gpu_index = gpu_index
    extraction_options.sift.max_num_features = options.max_num_features
    return extraction_options


def create_feature_matching_options(
    pycolmap_module: Any,
    options: FeaturePipelineOptions,
    *,
    use_gpu: bool = False,
    gpu_index: str = "-1",
) -> Any:
    if options.matcher != "sift-bruteforce":
        raise ValueError(f"Unsupported feature matcher: {options.matcher}")
    matching_options = pycolmap_module.FeatureMatchingOptions()
    matching_options.type = pycolmap_module.FeatureMatcherType.SIFT_BRUTEFORCE
    matching_options.use_gpu = use_gpu
    matching_options.gpu_index = gpu_index
    matching_options.guided_matching = options.guided_matching
    matching_options.rig_verification = False
    matching_options.skip_image_pairs_in_same_frame = False
    return matching_options


def create_mapper_options(
    pycolmap_module: Any,
    execution: dict[str, Any],
) -> Any:
    mapper_options = pycolmap_module.IncrementalPipelineOptions()
    mapper_options.ba_refine_sensor_from_rig = False
    mapper_options.ba_refine_focal_length = False
    mapper_options.ba_refine_principal_point = False
    mapper_options.ba_refine_extra_params = False
    mapper_options.ba_use_gpu = execution["gpu_bundle_adjustment"]
    mapper_options.ba_gpu_index = execution["bundle_adjustment_gpu_index"]
    return mapper_options


def validate_vocab_tree_path(vocab_tree_path: Path | None) -> None:
    if vocab_tree_path is not None and not vocab_tree_path.is_file():
        raise FileNotFoundError(f"Vocabulary tree not found: {vocab_tree_path}")


def resolve_pipeline_execution(
    pycolmap_module: Any,
    options: PipelineExecutionOptions,
) -> dict[str, Any]:
    if options.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    validate_gpu_index(options.gpu_index)

    cuda_available = bool(pycolmap_module.has_cuda)
    if options.device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but this pycolmap build has no CUDA support. "
            "Replace the CPU wheel with the CUDA wheel using "
            "`python -m pip uninstall -y pycolmap` followed by "
            "`python -m pip install -e '.[colmap-cuda12]'`, or install another "
            "CUDA-enabled pycolmap build."
        )
    use_gpu = cuda_available and options.device != "cpu"
    resolved_device_name = "cuda" if use_gpu else "cpu"
    return {
        "requested_device": options.device,
        "resolved_device": resolved_device_name,
        "device": getattr(pycolmap_module.Device, resolved_device_name),
        "cuda_available": cuda_available,
        "gpu_index": options.gpu_index,
        "gpu_bundle_adjustment": use_gpu and options.gpu_bundle_adjustment,
        "bundle_adjustment_gpu_index": first_gpu_index(options.gpu_index),
    }


def validate_gpu_index(value: str) -> None:
    if value == "-1":
        return
    parts = value.split(",")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(
            "gpu_index must be -1 or comma-separated non-negative indices"
        )


def parse_gpu_index(value: str) -> str:
    try:
        validate_gpu_index(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if value == "-1":
        return value
    return ",".join(str(int(part)) for part in value.split(","))


def first_gpu_index(value: str) -> str:
    return value if value == "-1" else value.split(",", maxsplit=1)[0]


def format_elapsed_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    rounded_seconds = int(round(seconds))
    minutes, seconds_part = divmod(rounded_seconds, 60)
    hours, minutes_part = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes_part:02d}m {seconds_part:02d}s"
    return f"{minutes_part}m {seconds_part:02d}s"


def record_elapsed_stage(
    elapsed_seconds_by_stage: dict[str, float],
    stage: str,
    stage_started_at: float,
) -> None:
    elapsed_seconds_by_stage[stage] = elapsed_seconds_by_stage.get(
        stage,
        0.0,
    ) + (time.perf_counter() - stage_started_at)


def format_elapsed_seconds_by_stage(
    elapsed_seconds_by_stage: dict[str, float],
) -> dict[str, str]:
    return {
        stage: format_elapsed_seconds(seconds)
        for stage, seconds in elapsed_seconds_by_stage.items()
    }


def match_features(
    pycolmap_module: Any,
    database_path: Path,
    *,
    matcher: str,
    matching_options: Any,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    image_pairs_path: Path | None = None,
    device: Any | None = None,
) -> None:
    validate_vocab_tree_path(vocab_tree_path)
    if device is None:
        device = pycolmap_module.Device.cpu
    if image_pairs_path is not None:
        pairing_options = pycolmap_module.ImportedPairingOptions()
        pairing_options.match_list_path = image_pairs_path
        pycolmap_module.match_image_pairs(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
        return
    if matcher == "sequential":
        pairing_options = pycolmap_module.SequentialPairingOptions()
        pairing_options.overlap = overlap
        pairing_options.expand_rig_images = False
        pairing_options.loop_detection = loop_detection
        if vocab_tree_path is not None:
            pairing_options.vocab_tree_path = vocab_tree_path
        pycolmap_module.match_sequential(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    elif matcher == "exhaustive":
        pycolmap_module.match_exhaustive(
            database_path,
            matching_options=matching_options,
            device=device,
        )
    elif matcher == "vocabtree":
        pairing_options = pycolmap_module.VocabTreePairingOptions()
        if vocab_tree_path is not None:
            pairing_options.vocab_tree_path = vocab_tree_path
        pycolmap_module.match_vocabtree(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
    else:
        raise ValueError("matcher must be 'sequential', 'exhaustive', or 'vocabtree'")


def run_incremental_mapping(
    pycolmap_module: Any,
    paths: dict[str, Path],
    *,
    image_dir: Path,
    mapper_options: Any,
) -> dict[int, Any]:
    return pycolmap_module.incremental_mapping(
        paths["database"],
        image_dir,
        paths["sparse"],
        options=mapper_options,
    )


def run_panorama_reconstruction(
    input_image_path: Path,
    workspace: Path,
    *,
    matcher: str = "sequential",
    overlap: int = 15,
    loop_detection: bool = True,
    vocab_tree_path: Path | None = None,
    ssl_cert_file: Path | None = None,
    feature_options: FeaturePipelineOptions = FeaturePipelineOptions(),
    execution_options: PipelineExecutionOptions = PipelineExecutionOptions(),
    image_limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    elapsed_seconds_by_stage: dict[str, float] = {}
    stage_started_at = started_at
    pycolmap_module = require_pycolmap()
    pycolmap_module.set_random_seed(0)
    configure_ssl_cert_file(ssl_cert_file)
    execution = resolve_pipeline_execution(pycolmap_module, execution_options)
    paths = panorama_workspace_paths(workspace)
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    paths["image_dir"].mkdir(parents=True, exist_ok=True)
    paths["sparse"].mkdir(parents=True, exist_ok=True)
    if paths["database"].exists():
        raise FileExistsError(f"{paths['database']} already exists. Pass --overwrite.")
    record_elapsed_stage(elapsed_seconds_by_stage, "setup", stage_started_at)

    stage_started_at = time.perf_counter()
    panorama_paths, camera, generated_image_names = import_equirectangular_panorama_images(
        input_image_path=input_image_path,
        output_image_dir=paths["image_dir"],
        image_limit=image_limit,
    )
    write_equirectangular_metadata(
        paths["metadata"],
        input_image_path,
        camera,
        panorama_paths,
        generated_image_names,
        image_limit=image_limit,
        relative_name_prefix=None,
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "import_equirectangular_images",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    extraction_options = create_feature_extraction_options(
        pycolmap_module,
        feature_options,
        use_gpu=execution["resolved_device"] == "cuda",
        gpu_index=execution["gpu_index"],
    )
    reader_options = pycolmap_module.ImageReaderOptions()
    reader_options.camera_model = camera.model
    reader_options.camera_params = camera.params_string
    pycolmap_module.extract_features(
        paths["database"],
        paths["image_dir"],
        reader_options=reader_options,
        camera_mode=pycolmap_module.CameraMode.SINGLE,
        extraction_options=extraction_options,
        device=execution["device"],
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "feature_extraction",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    generated_image_batch = read_generated_image_batch(
        paths["database"],
        generated_image_names,
    )
    feature_report = read_feature_stats_by_database_camera(paths["database"])
    feature_health = equirectangular_feature_health_summary(
        feature_report,
        expected_image_count=len(generated_image_names),
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "feature_stats",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    matching_options = create_feature_matching_options(
        pycolmap_module,
        feature_options,
        use_gpu=execution["resolved_device"] == "cuda",
        gpu_index=execution["gpu_index"],
    )
    match_features(
        pycolmap_module,
        paths["database"],
        matcher=matcher,
        matching_options=matching_options,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        device=execution["device"],
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "matching",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    mapper_options = create_mapper_options(pycolmap_module, execution)
    reconstructions = run_incremental_mapping(
        pycolmap_module,
        paths,
        image_dir=paths["image_dir"],
        mapper_options=mapper_options,
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "incremental_mapping",
        stage_started_at,
    )
    if not reconstructions:
        raise RuntimeError("COLMAP did not produce any sparse reconstruction")

    stage_started_at = time.perf_counter()
    best_model_id, best_reconstruction = max(
        reconstructions.items(),
        key=lambda item: (item[1].num_reg_images(), item[1].num_points3D()),
    )
    write_best_model(best_reconstruction, paths)
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "best_model_write",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    keyframe_scores = score_keyframe_anchors(paths["database"], best_reconstruction)
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "keyframe_scoring",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    report = build_panorama_report(
        pycolmap_version=pycolmap_module.__version__,
        input_image_path=input_image_path,
        paths=paths,
        camera=camera,
        panorama_count=len(panorama_paths),
        matcher=matcher,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        ssl_cert_file=ssl_cert_file,
        feature_options=feature_options,
        execution=execution,
        feature_report=feature_report,
        feature_health=feature_health,
        image_limit=image_limit,
        generated_image_batch=generated_image_batch,
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
        keyframe_scores=keyframe_scores,
    )
    elapsed_seconds_by_stage["report_writing"] = 0.0
    elapsed_seconds = time.perf_counter() - started_at
    report["elapsed_seconds"] = elapsed_seconds
    report["elapsed_time"] = format_elapsed_seconds(elapsed_seconds)
    report["elapsed_seconds_by_stage"] = elapsed_seconds_by_stage
    report["elapsed_time_by_stage"] = format_elapsed_seconds_by_stage(
        elapsed_seconds_by_stage,
    )
    json.dumps(report, indent=2)
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "report_writing",
        stage_started_at,
    )
    elapsed_seconds = time.perf_counter() - started_at
    report["elapsed_seconds"] = elapsed_seconds
    report["elapsed_time"] = format_elapsed_seconds(elapsed_seconds)
    report["elapsed_seconds_by_stage"] = elapsed_seconds_by_stage
    report["elapsed_time_by_stage"] = format_elapsed_seconds_by_stage(
        elapsed_seconds_by_stage,
    )
    paths["report"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def panorama_workspace_paths(workspace: Path) -> dict[str, Path]:
    return {
        "workspace": workspace,
        "image_dir": workspace / "images",
        "database": workspace / "database.db",
        "sparse": workspace / "sparse",
        "best_sparse": workspace / "sparse_best",
        "raw_ply": workspace / "cloud_colmap_raw.ply",
        "report": workspace / "panorama_equirectangular_report.json",
        "metadata": workspace / "panorama_equirectangular_config.json",
        "match_pairs": workspace / "match_pairs.txt",
    }


def write_equirectangular_metadata(
    path: Path,
    input_image_path: Path,
    camera: ColmapCamera,
    panorama_paths: Sequence[Path],
    image_names: Sequence[str],
    *,
    image_limit: int | None,
    relative_name_prefix: Path | None,
) -> None:
    data = {
        "schema_version": 2,
        "camera_setup": NATIVE_CAMERA_SETUP,
        "input_image_path": str(input_image_path),
        "image_limit": image_limit,
        "camera": camera.report(),
        "import_options": {
            "copy_backend": "hardlink_or_symlink",
            "new_image_prefix": (
                None
                if relative_name_prefix is None
                else relative_name_prefix.as_posix()
            ),
        },
        "panorama_images": [
            {
                "source_path": str(panorama_path),
                "colmap_image_name": image_name,
            }
            for panorama_path, image_name in zip(panorama_paths, image_names)
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_panorama_report(
    *,
    pycolmap_version: str,
    input_image_path: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    panorama_count: int,
    matcher: str,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    ssl_cert_file: Path | None,
    feature_options: FeaturePipelineOptions,
    execution: dict[str, Any],
    feature_report: dict[str, dict[str, float | int]],
    feature_health: dict[str, Any],
    image_limit: int | None,
    generated_image_batch: GeneratedImageBatch,
    reconstructions: dict[int, Any],
    best_model_id: int,
    best_reconstruction: Any,
    keyframe_scores: Sequence[KeyframeAnchorScore],
) -> dict[str, Any]:
    database_report = read_database_stats(paths["database"])
    model_stats = {
        str(model_id): reconstruction_stats(reconstruction)
        for model_id, reconstruction in sorted(reconstructions.items())
    }
    return {
        "status": f"raw_colmap_{NATIVE_CAMERA_SETUP}_unscaled",
        "warning": (
            "This is a monocular COLMAP reconstruction in arbitrary COLMAP "
            "scale/frame. The resulting world is not metric without an "
            "external scale."
        ),
        "camera_setup": NATIVE_CAMERA_SETUP,
        "pycolmap_version": pycolmap_version,
        "input_image_path": str(input_image_path),
        "workspace": str(paths["workspace"]),
        "image_dir": str(paths["image_dir"]),
        "database_path": str(paths["database"]),
        "sparse_path": str(paths["sparse"]),
        "best_sparse_path": str(paths["best_sparse"]),
        "raw_ply_path": str(paths["raw_ply"]),
        "metadata_path": str(paths["metadata"]),
        "camera": camera.report(),
        "options": {
            "matcher": matcher,
            "effective_pairing": matcher,
            "sequential_overlap": overlap,
            "loop_detection": loop_detection,
            "vocab_tree_path": (
                None if vocab_tree_path is None else str(vocab_tree_path)
            ),
            "ssl_cert_file": (
                os.environ.get("SSL_CERT_FILE")
                if ssl_cert_file is None
                else str(ssl_cert_file)
            ),
            "feature_extractor": feature_options.extractor,
            "feature_matcher": feature_options.matcher,
            "max_num_features": feature_options.max_num_features,
            "guided_matching": feature_options.guided_matching,
            "render_backend": "source_equirectangular_images",
            "image_limit": image_limit,
            "requested_device": execution["requested_device"],
            "resolved_device": execution["resolved_device"],
            "cuda_available": execution["cuda_available"],
            "gpu_index": execution["gpu_index"],
            "gpu_bundle_adjustment": execution["gpu_bundle_adjustment"],
            "bundle_adjustment_gpu_index": execution[
                "bundle_adjustment_gpu_index"
            ],
            "camera_mode": "SINGLE",
            "ba_refine_sensor_from_rig": False,
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
            "rig_verification": False,
            "skip_image_pairs_in_same_frame": False,
        },
        "panorama_count": panorama_count,
        "generated_image_count": panorama_count,
        "database_image_count": database_report["images"],
        "new_database_image_count": len(generated_image_batch.names),
        "new_database_image_ids": generated_image_batch.image_ids,
        "new_database_camera_ids": generated_image_batch.camera_ids,
        "new_database_image_names": list(generated_image_batch.names),
        "database": database_report,
        "features_by_camera": feature_report,
        "feature_health": feature_health,
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": model_stats,
        "best_model": reconstruction_stats(best_reconstruction),
        "keyframe_score_count": len(keyframe_scores),
        "keyframe_scores": [score.report() for score in keyframe_scores],
    }


def read_feature_stats_by_camera(
    database_path: Path,
) -> dict[int, dict[str, float | int]]:
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "images" not in tables:
            return {}
        keypoint_join = (
            "LEFT JOIN keypoints ON keypoints.image_id = images.image_id"
            if "keypoints" in tables
            else ""
        )
        feature_expr = "COALESCE(keypoints.rows, 0)" if "keypoints" in tables else "0"
        rows = connection.execute(
            f"""
            SELECT
                images.camera_id,
                COUNT(images.image_id) AS image_count,
                SUM({feature_expr}) AS total_features
            FROM images
            {keypoint_join}
            GROUP BY images.camera_id
            ORDER BY images.camera_id
            """
        ).fetchall()
    report: dict[int, dict[str, float | int]] = {}
    for camera_id, image_count, total_features in rows:
        total = int(total_features or 0)
        count = int(image_count)
        report[int(camera_id)] = {
            "image_count": count,
            "total_features": total,
            "mean_features_per_image": 0.0 if count == 0 else total / count,
        }
    return report


def read_feature_stats_by_database_camera(
    database_path: Path,
) -> dict[str, dict[str, float | int]]:
    return {
        f"camera_{camera_id}": stats
        for camera_id, stats in read_feature_stats_by_camera(database_path).items()
    }


def equirectangular_feature_health_summary(
    feature_report: dict[str, dict[str, float | int]],
    *,
    expected_image_count: int,
) -> dict[str, Any]:
    image_count = sum(int(stats["image_count"]) for stats in feature_report.values())
    zero_feature_cameras = [
        camera_name
        for camera_name, stats in sorted(feature_report.items())
        if int(stats["total_features"]) == 0
    ]
    return {
        "camera_setup": NATIVE_CAMERA_SETUP,
        "expected_panorama_images": expected_image_count,
        "panorama_images_with_features": image_count,
        "camera_count": len(feature_report),
        "cameras_with_features": len(feature_report) - len(zero_feature_cameras),
        "zero_feature_cameras": zero_feature_cameras,
    }


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
