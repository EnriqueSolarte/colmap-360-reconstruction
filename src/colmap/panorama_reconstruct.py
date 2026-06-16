"""Run COLMAP rig SfM directly from equirectangular panoramas."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import cos, pi, radians, sin, tan
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image
from py360convert.utils import uv2coor, xyz2uv, xyzpers
from tqdm import tqdm

from colmap.camera import ColmapCamera, camera_from_fov
from colmap.input_inspection import IMAGE_EXTENSIONS
from colmap.pipeline import write_best_model
from colmap.reporting import read_database_stats, reconstruction_stats

try:  # Keep argument parsing and tests usable before pycolmap is installed.
    import pycolmap
except ImportError:  # pragma: no cover - environment dependent
    pycolmap = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "colmap-panorama-rig"


@dataclass(frozen=True)
class PanoramaRigView:
    index: int
    yaw: float
    pitch: float
    cam_from_pano_rotation: np.ndarray

    @property
    def image_prefix(self) -> str:
        return f"pano_camera{self.index:02d}/"


@dataclass(frozen=True)
class PanoramaRenderOptions:
    num_yaw_steps: int
    pitches: tuple[float, ...]
    width: int
    height: int
    fov_x: float
    fov_y: float
    interpolation: str = "bilinear"


@dataclass(frozen=True)
class FeaturePipelineOptions:
    extractor: str = "sift"
    matcher: str = "sift-bruteforce"
    max_num_features: int = 8192
    guided_matching: bool = False


@dataclass(frozen=True)
class VirtualImageOutputOptions:
    image_format: str = "png"
    png_compression: int = 1
    jpeg_quality: int = 95
    save_workers: int = 12


@dataclass(frozen=True)
class PipelineExecutionOptions:
    device: str = "auto"
    gpu_index: str = "-1"
    gpu_bundle_adjustment: bool = True


@dataclass(frozen=True)
class InitialPanoramaSession:
    workspace: Path
    model_path: Path
    image_dir: Path
    mask_dir: Path
    database_path: Path
    rig_metadata_path: Path
    report_path: Path | None = None


@dataclass(frozen=True)
class GeneratedVirtualImageBatch:
    names: tuple[str, ...]
    records_by_name: dict[str, tuple[int, int]]

    @property
    def image_ids(self) -> list[int]:
        return [self.records_by_name[name][0] for name in self.names]

    @property
    def camera_ids(self) -> list[int]:
        return [self.records_by_name[name][1] for name in self.names]


@dataclass(frozen=True)
class RigFrameImageSet:
    frame_id: int
    image_ids: tuple[int, ...]
    image_names: tuple[str, ...]


def default_render_options(render_type: str) -> PanoramaRenderOptions:
    if render_type == "overlapping":
        return PanoramaRenderOptions(
            num_yaw_steps=4,
            pitches=tuple(radians(value) for value in (-35.0, 0.0, 35.0)),
            width=1024,
            height=1024,
            fov_x=pi / 2,
            fov_y=pi / 2,
        )
    if render_type == "non-overlapping":
        return PanoramaRenderOptions(
            num_yaw_steps=4,
            pitches=(0.0,),
            width=1024,
            height=1024,
            fov_x=pi / 2,
            fov_y=pi / 2,
        )
    raise ValueError("render_type must be 'overlapping' or 'non-overlapping'")


def require_pycolmap():
    if pycolmap is None:
        raise RuntimeError(
            "pycolmap is required for panorama rig reconstruction. Install the "
            "`colmap` extra or add `pycolmap` to the environment."
        )
    return pycolmap


def configure_ssl_cert_file(ssl_cert_file: Path | None = None) -> None:
    """Configure the CA bundle used by COLMAP/libcurl for optional downloads.

    Prefer an explicit bundle, preserve an existing environment configuration,
    and otherwise use the common system bundle when available. This lets
    vocabulary-tree downloads verify TLS certificates without overriding a
    caller's custom trust store.
    """
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


def resolve_initial_panorama_session(path: Path) -> InitialPanoramaSession:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Initial panorama session path is not a directory: {path}"
        )

    if (path / "panorama_rig_config.json").is_file():
        workspace = path
        model_path = workspace / "sparse_best"
    elif (path.parent / "panorama_rig_config.json").is_file():
        workspace = path.parent
        model_path = path
    else:
        raise ValueError(
            "Initial model must come from a previous colmap-panorama-reconstruct "
            "workspace, or its sparse_best model directory."
        )

    session = InitialPanoramaSession(
        workspace=workspace,
        model_path=model_path,
        image_dir=workspace / "images",
        mask_dir=workspace / "masks",
        database_path=workspace / "database.db",
        rig_metadata_path=workspace / "panorama_rig_config.json",
        report_path=workspace / "panorama_rig_report.json",
    )
    validate_initial_panorama_session(session)
    return session


def validate_initial_panorama_session(session: InitialPanoramaSession) -> None:
    if not session.rig_metadata_path.is_file():
        raise FileNotFoundError(
            "Initial panorama session is missing panorama_rig_config.json: "
            f"{session.rig_metadata_path}"
        )
    if not session.database_path.is_file():
        raise FileNotFoundError(
            f"Initial panorama session is missing database.db: {session.database_path}"
        )
    if not session.image_dir.is_dir():
        raise FileNotFoundError(
            f"Initial panorama session is missing images/: {session.image_dir}"
        )
    if not session.mask_dir.is_dir():
        raise FileNotFoundError(
            f"Initial panorama session is missing masks/: {session.mask_dir}"
        )
    if not is_colmap_model_dir(session.model_path):
        raise FileNotFoundError(
            "Initial panorama session is missing a readable COLMAP model at "
            f"{session.model_path}"
        )


def is_colmap_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    required_names = ("cameras", "images", "points3D")
    return all(
        any((path / f"{name}.{extension}").is_file() for extension in ("bin", "txt"))
        for name in required_names
    )


def validate_initial_session_compatible(
    session: InitialPanoramaSession,
    render_options: PanoramaRenderOptions,
    output_options: VirtualImageOutputOptions,
) -> dict[str, Any]:
    metadata = json.loads(session.rig_metadata_path.read_text(encoding="utf-8"))
    previous_render = metadata.get("render_options", {})
    expected = {
        "num_yaw_steps": render_options.num_yaw_steps,
        "pitches": list(render_options.pitches),
        "width": render_options.width,
        "height": render_options.height,
        "fov_x": render_options.fov_x,
        "fov_y": render_options.fov_y,
        "interpolation": render_options.interpolation,
        "virtual_image_format": output_options.image_format,
    }
    mismatches = {
        key: {"previous": previous_render.get(key), "current": value}
        for key, value in expected.items()
        if not metadata_values_match(previous_render.get(key), value)
    }
    if mismatches:
        raise ValueError(
            "Initial panorama session render settings do not match the current "
            f"run: {mismatches}"
        )
    return metadata


def metadata_values_match(previous: Any, current: Any) -> bool:
    if isinstance(current, float):
        return previous is not None and abs(float(previous) - current) <= 1e-12
    if isinstance(current, list) and all(isinstance(value, float) for value in current):
        if previous is None or len(previous) != len(current):
            return False
        return all(
            abs(float(left) - right) <= 1e-12
            for left, right in zip(previous, current)
        )
    return previous == current


def copy_initial_panorama_session(
    session: InitialPanoramaSession,
    paths: dict[str, Path],
    _pycolmap_module: Any | None = None,
) -> None:
    copytree_hardlink_or_copy(session.image_dir, paths["image_dir"])
    copytree_hardlink_or_copy(session.mask_dir, paths["mask_dir"])
    shutil.copy2(session.database_path, paths["database"])
    if paths["initial_sparse"].exists():
        shutil.rmtree(paths["initial_sparse"])
    copy_colmap_model(session.model_path, paths["initial_sparse"])


def copytree_hardlink_or_copy(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        copy_function=hardlink_or_copy_file,
        dirs_exist_ok=True,
    )


def hardlink_or_copy_file(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except FileExistsError:
        if os.path.samefile(source, destination):
            return destination
        shutil.copy2(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def copy_colmap_model(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("cameras", "images", "points3D", "rigs", "frames"):
        for extension in ("bin", "txt"):
            source_file = source / f"{name}.{extension}"
            if source_file.is_file():
                shutil.copy2(source_file, destination / source_file.name)


def read_camera_ids_by_prefix(
    database_path: Path,
    views: Sequence[PanoramaRigView],
) -> dict[str, int]:
    camera_ids: dict[str, int] = {}
    with sqlite3.connect(database_path) as connection:
        for view in views:
            prefix = view.image_prefix
            rows = connection.execute(
                """
                SELECT DISTINCT camera_id
                FROM images
                WHERE name LIKE ?
                ORDER BY camera_id
                """,
                (f"{prefix}%",),
            ).fetchall()
            if len(rows) != 1:
                raise ValueError(
                    f"Expected one existing camera_id for prefix {prefix}, "
                    f"found {[row[0] for row in rows]}"
                )
            camera_ids[prefix] = int(rows[0][0])
    return camera_ids


def assign_camera_ids_by_prefix(
    database_path: Path,
    camera_ids: dict[str, int],
) -> None:
    with sqlite3.connect(database_path) as connection:
        for prefix, camera_id in camera_ids.items():
            connection.execute(
                """
                UPDATE images
                SET camera_id = ?
                WHERE name LIKE ?
                """,
                (camera_id, f"{prefix}%"),
            )


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


def read_generated_virtual_image_batch(
    database_path: Path,
    image_names: Sequence[str],
) -> GeneratedVirtualImageBatch:
    return GeneratedVirtualImageBatch(
        names=tuple(image_names),
        records_by_name=read_image_records_by_name(database_path, image_names),
    )


def read_rig_frame_image_sets(
    database_path: Path,
    frame_ids: Sequence[int],
) -> list[RigFrameImageSet]:
    if not frame_ids:
        return []
    frame_id_set = {int(frame_id) for frame_id in frame_ids}
    placeholders = ",".join("?" for _value in frame_id_set)
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT frame_data.frame_id, images.image_id, images.name
            FROM frame_data
            JOIN images ON images.image_id = frame_data.data_id
            WHERE frame_data.frame_id IN ({placeholders})
            ORDER BY frame_data.frame_id, frame_data.sensor_id, images.name
            """,
            tuple(sorted(frame_id_set)),
        ).fetchall()
    frames: dict[int, list[tuple[int, str]]] = {
        frame_id: [] for frame_id in sorted(frame_id_set)
    }
    for frame_id, image_id, image_name in rows:
        frames[int(frame_id)].append((int(image_id), str(image_name)))
    return [
        RigFrameImageSet(
            frame_id=frame_id,
            image_ids=tuple(image_id for image_id, _image_name in images),
            image_names=tuple(image_name for _image_id, image_name in images),
        )
        for frame_id, images in frames.items()
        if images
    ]


def read_rig_frame_image_sets_for_images(
    database_path: Path,
    image_ids: Sequence[int],
) -> list[RigFrameImageSet]:
    if not image_ids:
        return []
    image_id_set = {int(image_id) for image_id in image_ids}
    placeholders = ",".join("?" for _value in image_id_set)
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            f"""
            SELECT DISTINCT frame_id
            FROM frame_data
            WHERE data_id IN ({placeholders})
            ORDER BY frame_id
            """,
            tuple(sorted(image_id_set)),
        ).fetchall()
    return read_rig_frame_image_sets(
        database_path,
        [int(row[0]) for row in rows],
    )


def read_last_old_rig_frame_image_sets(
    database_path: Path,
    *,
    excluded_image_ids: Sequence[int],
    limit: int,
) -> list[RigFrameImageSet]:
    if limit <= 0:
        return []
    excluded_image_id_set = {int(image_id) for image_id in excluded_image_ids}
    with sqlite3.connect(database_path) as connection:
        if excluded_image_id_set:
            placeholders = ",".join("?" for _value in excluded_image_id_set)
            rows = connection.execute(
                f"""
                SELECT frames.frame_id
                FROM frames
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM frame_data
                    WHERE frame_data.frame_id = frames.frame_id
                    AND frame_data.data_id IN ({placeholders})
                )
                ORDER BY frames.frame_id DESC
                LIMIT ?
                """,
                (*tuple(sorted(excluded_image_id_set)), limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT frame_id
                FROM frames
                ORDER BY frame_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return read_rig_frame_image_sets(
        database_path,
        sorted(int(row[0]) for row in rows),
    )


def add_frame_image_pairs(
    pairs: list[tuple[str, str]],
    seen_pair_keys: set[tuple[int, int]],
    frame_a: RigFrameImageSet,
    frame_b: RigFrameImageSet,
) -> None:
    for image_id_a, image_name_a in zip(frame_a.image_ids, frame_a.image_names):
        for image_id_b, image_name_b in zip(frame_b.image_ids, frame_b.image_names):
            if image_id_a == image_id_b:
                continue
            key = tuple(sorted((image_id_a, image_id_b)))
            if key in seen_pair_keys:
                continue
            seen_pair_keys.add(key)
            pairs.append((image_name_a, image_name_b))


def build_last_k_match_pairs(
    new_frames: Sequence[RigFrameImageSet],
    selected_old_frames: Sequence[RigFrameImageSet],
    *,
    new_frame_overlap: int,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen_pair_keys: set[tuple[int, int]] = set()
    ordered_new_frames = sorted(new_frames, key=lambda frame: frame.frame_id)
    for index, frame in enumerate(ordered_new_frames):
        for neighbor in ordered_new_frames[index + 1 : index + 1 + new_frame_overlap]:
            add_frame_image_pairs(pairs, seen_pair_keys, frame, neighbor)
    for new_frame in ordered_new_frames:
        for old_frame in selected_old_frames:
            add_frame_image_pairs(pairs, seen_pair_keys, old_frame, new_frame)
    return pairs


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


def generated_virtual_image_names(
    input_image_path: Path,
    panorama_paths: Sequence[Path],
    views: Sequence[PanoramaRigView],
    output_options: VirtualImageOutputOptions,
    relative_name_prefix: Path | None,
) -> list[str]:
    names: list[str] = []
    for panorama_path in panorama_paths:
        relative_name = prefixed_virtual_image_relative_name(
            input_image_path,
            panorama_path,
            output_options.image_format,
            relative_name_prefix,
        ).as_posix()
        names.extend(f"{view.image_prefix}{relative_name}" for view in views)
    return names


def replace_generated_frames_with_seed_rig_frames(
    database_path: Path,
    *,
    input_image_path: Path,
    panorama_paths: Sequence[Path],
    views: Sequence[PanoramaRigView],
    output_options: VirtualImageOutputOptions,
    relative_name_prefix: Path | None,
    seed_camera_ids: dict[str, int],
    generated_records: dict[str, tuple[int, int]],
) -> None:
    generated_image_ids = {
        image_id for image_id, _camera_id in generated_records.values()
    }
    generated_camera_ids = {
        camera_id for _image_id, camera_id in generated_records.values()
    } - set(seed_camera_ids.values())
    with sqlite3.connect(database_path) as connection:
        seed_rig_id = find_seed_rig_id(connection, set(seed_camera_ids.values()))
        cleanup_generated_singleton_rigs(
            connection,
            generated_image_ids,
            generated_camera_ids,
            seed_rig_id,
        )
        for panorama_path in panorama_paths:
            relative_name = prefixed_virtual_image_relative_name(
                input_image_path,
                panorama_path,
                output_options.image_format,
                relative_name_prefix,
            ).as_posix()
            frame_images = []
            for view in views:
                image_name = f"{view.image_prefix}{relative_name}"
                image_id, _camera_id = generated_records[image_name]
                frame_images.append((image_id, seed_camera_ids[view.image_prefix]))
            cursor = connection.execute(
                "INSERT INTO frames(rig_id) VALUES (?)",
                (seed_rig_id,),
            )
            frame_id = int(cursor.lastrowid)
            connection.executemany(
                """
                INSERT INTO frame_data(frame_id, data_id, sensor_id, sensor_type)
                VALUES (?, ?, ?, 0)
                """,
                [
                    (frame_id, image_id, camera_id)
                    for image_id, camera_id in frame_images
                ],
            )


def find_seed_rig_id(
    connection: sqlite3.Connection,
    seed_camera_ids: set[int],
) -> int:
    rows = connection.execute(
        """
        SELECT rig_id, ref_sensor_id
        FROM rigs
        WHERE ref_sensor_type = 0
        ORDER BY rig_id
        """
    ).fetchall()
    for rig_id, ref_sensor_id in rows:
        sensor_rows = connection.execute(
            """
            SELECT sensor_id
            FROM rig_sensors
            WHERE rig_id = ? AND sensor_type = 0
            """,
            (rig_id,),
        ).fetchall()
        sensor_ids = {int(ref_sensor_id)} | {int(row[0]) for row in sensor_rows}
        if sensor_ids == seed_camera_ids:
            return int(rig_id)
    raise ValueError(
        "Initial session database does not contain a rig matching the seed cameras"
    )


def cleanup_generated_singleton_rigs(
    connection: sqlite3.Connection,
    generated_image_ids: set[int],
    generated_camera_ids: set[int],
    seed_rig_id: int,
) -> None:
    if not generated_image_ids:
        return
    placeholders = ",".join("?" for _value in generated_image_ids)
    frame_rows = connection.execute(
        f"""
        SELECT DISTINCT frame_id
        FROM frame_data
        WHERE data_id IN ({placeholders})
        """,
        tuple(generated_image_ids),
    ).fetchall()
    frame_ids = {int(row[0]) for row in frame_rows}
    if frame_ids:
        frame_placeholders = ",".join("?" for _value in frame_ids)
        rig_rows = connection.execute(
            f"""
            SELECT DISTINCT rig_id
            FROM frames
            WHERE frame_id IN ({frame_placeholders})
            """,
            tuple(frame_ids),
        ).fetchall()
        rig_ids = {int(row[0]) for row in rig_rows} - {seed_rig_id}
        connection.execute(
            f"DELETE FROM frame_data WHERE frame_id IN ({frame_placeholders})",
            tuple(frame_ids),
        )
        connection.execute(
            f"DELETE FROM frames WHERE frame_id IN ({frame_placeholders})",
            tuple(frame_ids),
        )
        if rig_ids:
            rig_placeholders = ",".join("?" for _value in rig_ids)
            connection.execute(
                f"DELETE FROM rig_sensors WHERE rig_id IN ({rig_placeholders})",
                tuple(rig_ids),
            )
            connection.execute(
                f"DELETE FROM rigs WHERE rig_id IN ({rig_placeholders})",
                tuple(rig_ids),
            )
    if generated_camera_ids:
        camera_placeholders = ",".join("?" for _value in generated_camera_ids)
        connection.execute(
            f"DELETE FROM cameras WHERE camera_id IN ({camera_placeholders})",
            tuple(generated_camera_ids),
        )


def panorama_image_paths(input_image_path: Path) -> list[Path]:
    return sorted(
        path
        for path in input_image_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_rig_views(options: PanoramaRenderOptions) -> list[PanoramaRigView]:
    if options.num_yaw_steps <= 0:
        raise ValueError("num_yaw_steps must be positive")
    if not options.pitches:
        raise ValueError("at least one pitch is required")

    views: list[PanoramaRigView] = []
    base_yaws = np.linspace(0.0, 2 * pi, options.num_yaw_steps, endpoint=False)
    for pitch in options.pitches:
        yaw_offset = (pi / options.num_yaw_steps) if pitch > 0 else 0.0
        for yaw in base_yaws + yaw_offset:
            views.append(
                PanoramaRigView(
                    index=len(views),
                    yaw=float(wrap_radians(yaw)),
                    pitch=float(pitch),
                    cam_from_pano_rotation=cam_from_pano_rotation(yaw, pitch),
                )
            )
    return views


def cam_from_pano_rotation(yaw: float, pitch: float) -> np.ndarray:
    """Rotation matching COLMAP's panorama rig example convention."""
    return rotation_x(-pitch) @ rotation_y(-yaw)


def rotation_x(angle: float) -> np.ndarray:
    c = cos(angle)
    s = sin(angle)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def rotation_y(angle: float) -> np.ndarray:
    c = cos(angle)
    s = sin(angle)
    return np.asarray(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def wrap_radians(value: float) -> float:
    return float((value + pi) % (2 * pi) - pi)


def render_panorama_rig_images(
    input_image_path: Path,
    output_image_dir: Path,
    mask_dir: Path,
    options: PanoramaRenderOptions,
    *,
    output_options: VirtualImageOutputOptions = VirtualImageOutputOptions(),
    image_limit: int | None = None,
    fail_on_existing: bool = False,
    relative_name_prefix: Path | None = None,
) -> tuple[list[Path], ColmapCamera, list[PanoramaRigView]]:
    validate_virtual_image_output_options(output_options)
    panorama_paths = panorama_image_paths(input_image_path)
    if image_limit is not None:
        panorama_paths = panorama_paths[:image_limit]
    if not panorama_paths:
        raise RuntimeError(f"No panorama images found in {input_image_path}")

    views = build_rig_views(options)
    camera = camera_from_fov(options.width, options.height, options.fov_x, options.fov_y)
    masks = rig_view_masks(camera, views)
    validate_rig_masks(camera, views, masks)
    if fail_on_existing:
        validate_virtual_views_do_not_exist(
            input_image_path,
            output_image_dir,
            mask_dir,
            panorama_paths,
            views,
            output_options,
            relative_name_prefix,
        )

    first_panorama = read_panorama_image(panorama_paths[0])
    source_shape = first_panorama.shape[:2]
    remap_maps = build_rig_remap_maps(source_shape, options, views)
    total_views = len(panorama_paths) * len(views)

    with (
        ThreadPoolExecutor(max_workers=output_options.save_workers) as save_pool,
        tqdm(
            total=total_views,
            desc="Rendering virtual pinhole views",
            unit="view",
        ) as progress,
    ):
        for panorama_index, panorama_path in enumerate(panorama_paths):
            panorama = (
                first_panorama
                if panorama_index == 0
                else read_panorama_image(panorama_path)
            )
            if panorama.shape[:2] != source_shape:
                raise ValueError(
                    f"Panorama {panorama_path} has size {panorama.shape[1]}x"
                    f"{panorama.shape[0]}; expected {source_shape[1]}x{source_shape[0]}"
                )
            padded_panorama = pad_equirectangular_image(panorama)
            relative_name = prefixed_virtual_image_relative_name(
                input_image_path,
                panorama_path,
                output_options.image_format,
                relative_name_prefix,
            )
            futures = []
            for view in views:
                image = render_remapped_view(padded_panorama, remap_maps[view.index])
                output_path = output_image_dir / view.image_prefix / relative_name
                output_path.parent.mkdir(parents=True, exist_ok=True)

                mask_path = mask_dir / f"{view.image_prefix}{relative_name}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                futures.append(
                    save_pool.submit(
                        save_virtual_view,
                        image,
                        output_path,
                        masks[view.index],
                        mask_path,
                        output_options,
                    )
                )

            for future in as_completed(futures):
                future.result()
                progress.update()

    return panorama_paths, camera, views


def validate_virtual_views_do_not_exist(
    input_image_path: Path,
    output_image_dir: Path,
    mask_dir: Path,
    panorama_paths: Sequence[Path],
    views: Sequence[PanoramaRigView],
    output_options: VirtualImageOutputOptions,
    relative_name_prefix: Path | None = None,
) -> None:
    collisions = []
    for panorama_path in panorama_paths:
        relative_name = prefixed_virtual_image_relative_name(
            input_image_path,
            panorama_path,
            output_options.image_format,
            relative_name_prefix,
        )
        for view in views:
            image_path = output_image_dir / view.image_prefix / relative_name
            mask_path = mask_dir / f"{view.image_prefix}{relative_name}.png"
            if image_path.exists():
                collisions.append(str(image_path))
            if mask_path.exists():
                collisions.append(str(mask_path))
            if len(collisions) >= 5:
                raise FileExistsError(
                    "New panorama views would overwrite previous session files: "
                    f"{collisions}"
                )
    if collisions:
        raise FileExistsError(
            "New panorama views would overwrite previous session files: "
            f"{collisions}"
        )


def validate_virtual_image_output_options(options: VirtualImageOutputOptions) -> None:
    if options.image_format not in {"png", "jpeg"}:
        raise ValueError("image_format must be 'png' or 'jpeg'")
    if not 0 <= options.png_compression <= 9:
        raise ValueError("png_compression must be between 0 and 9")
    if not 1 <= options.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if options.save_workers <= 0:
        raise ValueError("save_workers must be positive")


def read_panorama_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def build_rig_remap_maps(
    source_shape: tuple[int, int],
    options: PanoramaRenderOptions,
    views: Sequence[PanoramaRigView],
) -> dict[int, tuple[np.ndarray, np.ndarray | None, int]]:
    source_height, source_width = source_shape
    interpolation = cv2_interpolation(options.interpolation)
    nearest = interpolation == cv2.INTER_NEAREST
    maps = {}
    for view in views:
        xyz = xyzpers(
            options.fov_x,
            options.fov_y,
            -view.yaw,
            view.pitch,
            (options.height, options.width),
            0.0,
        )
        u, v = xyz2uv(xyz)
        x, y = uv2coor(u, v, source_height, source_width)
        map1, map2 = cv2.convertMaps(
            x + 1,
            y + 1,
            cv2.CV_16SC2,
            nninterpolation=nearest,
        )
        maps[view.index] = (map1, map2, interpolation)
    return maps


def cv2_interpolation(interpolation: str) -> int:
    if interpolation in {"linear", "bilinear"}:
        return cv2.INTER_LINEAR
    if interpolation == "nearest":
        return cv2.INTER_NEAREST
    raise ValueError("interpolation must be 'linear', 'bilinear', or 'nearest'")


def pad_equirectangular_image(image: np.ndarray) -> np.ndarray:
    # Match py360convert's seam and pole padding for pixel-identical remapping.
    padding = ((1, 1), (1, 1)) + ((0, 0),) * (image.ndim - 2)
    padded = np.pad(image, padding, mode="empty")
    width = image.shape[1]
    padded[0, 1:-1] = np.roll(image[0], width // 2, axis=0)
    padded[-1, 1:-1] = np.roll(image[-1], width // 2, axis=0)
    padded[:, 0] = padded[:, -2]
    padded[:, -1] = padded[:, 1]
    return padded


def render_remapped_view(
    padded_panorama: np.ndarray,
    remap: tuple[np.ndarray, np.ndarray | None, int],
) -> np.ndarray:
    map1, map2, interpolation = remap
    return cv2.remap(padded_panorama, map1, map2, interpolation=interpolation)


def virtual_image_relative_name(relative_path: Path, image_format: str) -> Path:
    suffix = ".png" if image_format == "png" else ".jpg"
    return relative_path.with_suffix(suffix)


def prefixed_virtual_image_relative_name(
    input_image_path: Path,
    panorama_path: Path,
    image_format: str,
    relative_name_prefix: Path | None = None,
) -> Path:
    relative_name = virtual_image_relative_name(
        panorama_path.relative_to(input_image_path),
        image_format,
    )
    if relative_name_prefix is None:
        return relative_name
    prefix = relative_name_prefix.as_posix().strip("/").replace("/", "_")
    return relative_name.with_name(f"{prefix}_{relative_name.name}")


def save_virtual_view(
    image: np.ndarray,
    output_path: Path,
    mask: np.ndarray,
    mask_path: Path,
    options: VirtualImageOutputOptions,
) -> None:
    if options.image_format == "png":
        Image.fromarray(image).save(
            output_path,
            format="PNG",
            compress_level=options.png_compression,
        )
    else:
        Image.fromarray(image).save(
            output_path,
            format="JPEG",
            quality=options.jpeg_quality,
            subsampling=0,
        )
    Image.fromarray(mask).save(mask_path, format="PNG", compress_level=1)


def read_rendered_rig_masks(
    mask_dir: Path,
    input_image_path: Path,
    panorama_path: Path,
    views: Sequence[PanoramaRigView],
    output_options: VirtualImageOutputOptions = VirtualImageOutputOptions(),
    relative_name_prefix: Path | None = None,
) -> dict[int, np.ndarray]:
    relative_name = prefixed_virtual_image_relative_name(
        input_image_path,
        panorama_path,
        output_options.image_format,
        relative_name_prefix,
    )
    masks: dict[int, np.ndarray] = {}
    for view in views:
        path = mask_dir / f"{view.image_prefix}{relative_name}.png"
        with Image.open(path) as mask:
            masks[view.index] = np.asarray(mask).copy()
    return masks


def rig_view_masks(
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
) -> dict[int, np.ndarray]:
    rays = virtual_camera_rays(camera)
    forward = np.asarray([0.0, 0.0, 1.0])
    centers = np.vstack(
        [view.cam_from_pano_rotation.T @ forward for view in views]
    )
    masks: dict[int, np.ndarray] = {}
    for view in views:
        rays_in_pano = rays @ view.cam_from_pano_rotation
        closest_camera = np.argmax(rays_in_pano @ centers.T, axis=1)
        masks[view.index] = (
            (closest_camera == view.index)
            .astype(np.uint8)
            .reshape(camera.height, camera.width)
            * 255
        )
    return masks


def validate_rig_masks(
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
    masks: dict[int, np.ndarray],
) -> None:
    expected_indices = {view.index for view in views}
    if set(masks) != expected_indices:
        raise ValueError(
            f"Rig mask indices {sorted(masks)} do not match views "
            f"{sorted(expected_indices)}"
        )

    principal_x = min(max(int(camera.params[2]), 0), camera.width - 1)
    principal_y = min(max(int(camera.params[3]), 0), camera.height - 1)
    for view in views:
        mask = masks[view.index]
        if mask.shape != (camera.height, camera.width):
            raise ValueError(
                f"Mask {view.index} has shape {mask.shape}; expected "
                f"{(camera.height, camera.width)}"
            )
        if mask.dtype != np.uint8 or not set(np.unique(mask)).issubset({0, 255}):
            raise ValueError(f"Mask {view.index} must be binary uint8")
        if not np.any(mask):
            raise ValueError(f"Mask {view.index} is empty")
        if mask[principal_y, principal_x] == 0:
            raise ValueError(
                f"Mask {view.index} excludes its virtual camera principal ray"
            )


def mask_statistics(masks: dict[int, np.ndarray]) -> dict[str, dict[str, float | int]]:
    return {
        str(index): {
            "active_pixels": int(np.count_nonzero(mask)),
            "active_fraction": float(np.count_nonzero(mask) / mask.size),
        }
        for index, mask in sorted(masks.items())
    }


def virtual_camera_rays(camera: ColmapCamera) -> np.ndarray:
    fx, fy, cx, cy = camera.params
    x, y = np.meshgrid(
        np.arange(camera.width, dtype=np.float64) + 0.5,
        np.arange(camera.height, dtype=np.float64) + 0.5,
    )
    rays = np.stack(((x - cx) / fx, (y - cy) / fy, np.ones_like(x)), axis=-1)
    rays = rays.reshape(-1, 3)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def create_pycolmap_camera(pycolmap_module: Any, camera: ColmapCamera) -> Any:
    return pycolmap_module.Camera(
        camera_id=0,
        model=camera.model,
        width=camera.width,
        height=camera.height,
        params=list(camera.params),
    )


def create_rig_config(
    pycolmap_module: Any,
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
    *,
    ref_index: int = 0,
) -> Any:
    if not views:
        raise ValueError("at least one rig view is required")
    ref_rotation = views[ref_index].cam_from_pano_rotation
    pycolmap_camera = create_pycolmap_camera(pycolmap_module, camera)
    rig_cameras = []
    zero_translation = np.zeros((3, 1), dtype=np.float64)
    for view in views:
        if view.index == ref_index:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = view.cam_from_pano_rotation @ ref_rotation.T
            cam_from_rig = pycolmap_module.Rigid3d(
                pycolmap_module.Rotation3d(cam_from_ref_rotation),
                zero_translation,
            )
        rig_camera = pycolmap_module.RigConfigCamera(
            ref_sensor=view.index == ref_index,
            image_prefix=view.image_prefix,
            cam_from_rig=cam_from_rig,
        )
        rig_camera.camera = pycolmap_camera
        rig_cameras.append(rig_camera)
    return pycolmap_module.RigConfig(cameras=rig_cameras)


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
    matching_options.rig_verification = True
    matching_options.skip_image_pairs_in_same_frame = True
    return matching_options


def create_mapper_options(
    pycolmap_module: Any,
    execution: dict[str, Any],
    *,
    continue_existing_model: bool = False,
) -> Any:
    mapper_options = pycolmap_module.IncrementalPipelineOptions()
    if continue_existing_model:
        mapper_options.multiple_models = False
        mapper_options.max_num_models = 1
        mapper_options.fix_existing_frames = True
        mapper_options.mapper.fix_existing_frames = True
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
    match_scope: str,
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
    if match_scope == "last-k":
        if image_pairs_path is None:
            raise ValueError("image_pairs_path is required for last-k matching")
        pairing_options = pycolmap_module.ImportedPairingOptions()
        pairing_options.match_list_path = image_pairs_path
        pycolmap_module.match_image_pairs(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
        return
    if match_scope != "all":
        raise ValueError("match_scope must be 'all' or 'last-k'")
    if matcher == "sequential":
        pairing_options = pycolmap_module.SequentialPairingOptions()
        pairing_options.overlap = overlap
        pairing_options.expand_rig_images = True
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
    initial_model_path: Path | None = None,
) -> dict[int, Any]:
    kwargs: dict[str, Any] = {"options": mapper_options}
    if initial_model_path is not None:
        kwargs["input_path"] = initial_model_path
    return pycolmap_module.incremental_mapping(
        paths["database"],
        image_dir,
        paths["sparse"],
        **kwargs,
    )


def run_panorama_reconstruction(
    input_image_path: Path,
    workspace: Path,
    *,
    render_options: PanoramaRenderOptions,
    matcher: str = "sequential",
    match_scope: str = "all",
    last_k_frames: int = 20,
    overlap: int = 15,
    loop_detection: bool = True,
    vocab_tree_path: Path | None = None,
    ssl_cert_file: Path | None = None,
    feature_options: FeaturePipelineOptions = FeaturePipelineOptions(),
    output_options: VirtualImageOutputOptions = VirtualImageOutputOptions(),
    execution_options: PipelineExecutionOptions = PipelineExecutionOptions(),
    image_limit: int | None = None,
    initial_session_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    elapsed_seconds_by_stage: dict[str, float] = {}
    stage_started_at = started_at
    pycolmap_module = require_pycolmap()
    pycolmap_module.set_random_seed(0)
    configure_ssl_cert_file(ssl_cert_file)
    execution = resolve_pipeline_execution(pycolmap_module, execution_options)
    initial_session = (
        None
        if initial_session_path is None
        else resolve_initial_panorama_session(initial_session_path)
    )
    initial_metadata = (
        None
        if initial_session is None
        else validate_initial_session_compatible(
            initial_session,
            render_options,
            output_options,
        )
    )
    paths = panorama_workspace_paths(workspace)
    if (
        initial_session is not None
        and workspace.resolve() == initial_session.workspace.resolve()
    ):
        raise ValueError("Workspace must be different from the initial session workspace")
    if workspace.exists() and overwrite:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    paths["image_dir"].mkdir(parents=True, exist_ok=True)
    paths["mask_dir"].mkdir(parents=True, exist_ok=True)
    paths["sparse"].mkdir(parents=True, exist_ok=True)
    if paths["database"].exists():
        raise FileExistsError(f"{paths['database']} already exists. Pass --overwrite.")
    record_elapsed_stage(elapsed_seconds_by_stage, "setup", stage_started_at)
    if initial_session is not None:
        stage_started_at = time.perf_counter()
        copy_initial_panorama_session(initial_session, paths, pycolmap_module)
        record_elapsed_stage(
            elapsed_seconds_by_stage,
            "initial_session_copy",
            stage_started_at,
        )
    else:
        elapsed_seconds_by_stage["initial_session_copy"] = 0.0
    new_image_prefix = (
        None if initial_session is None else Path(input_image_path.resolve().name)
    )

    stage_started_at = time.perf_counter()
    panorama_paths, camera, views = render_panorama_rig_images(
        input_image_path=input_image_path,
        output_image_dir=paths["image_dir"],
        mask_dir=paths["mask_dir"],
        options=render_options,
        output_options=output_options,
        image_limit=image_limit,
        fail_on_existing=initial_session is not None,
        relative_name_prefix=new_image_prefix,
    )
    masks = read_rendered_rig_masks(
        paths["mask_dir"],
        input_image_path,
        panorama_paths[0],
        views,
        output_options,
        relative_name_prefix=new_image_prefix,
    )
    validate_rig_masks(camera, views, masks)
    masks_report = mask_statistics(masks)
    write_rig_metadata(
        paths["rig_metadata"],
        input_image_path,
        render_options,
        camera,
        views,
        masks_report,
        output_options,
        image_limit=image_limit,
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "render_virtual_views",
        stage_started_at,
    )
    generated_image_names = generated_virtual_image_names(
        input_image_path,
        panorama_paths,
        views,
        output_options,
        new_image_prefix,
    )

    stage_started_at = time.perf_counter()
    seed_camera_ids = (
        None
        if initial_session is None
        else read_camera_ids_by_prefix(paths["database"], views)
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "initial_camera_lookup",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    extraction_options = create_feature_extraction_options(
        pycolmap_module,
        feature_options,
        use_gpu=execution["resolved_device"] == "cuda",
        gpu_index=execution["gpu_index"],
    )
    pycolmap_module.extract_features(
        paths["database"],
        paths["image_dir"],
        reader_options=pycolmap_module.ImageReaderOptions(mask_path=paths["mask_dir"]),
        camera_mode=pycolmap_module.CameraMode.PER_FOLDER,
        extraction_options=extraction_options,
        device=execution["device"],
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "feature_extraction",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    generated_image_batch = read_generated_virtual_image_batch(
        paths["database"],
        generated_image_names,
    )
    if seed_camera_ids is not None:
        assign_camera_ids_by_prefix(paths["database"], seed_camera_ids)
    feature_report = read_feature_stats_by_camera(paths["database"])
    feature_health = feature_health_summary(views, feature_report)
    rig_config = create_rig_config(pycolmap_module, camera, views)
    if initial_session is None:
        with pycolmap_module.Database.open(paths["database"]) as database:
            pycolmap_module.apply_rig_config([rig_config], database)
    else:
        if seed_camera_ids is None:
            raise AssertionError(
                "seeded reconstruction is missing generated image data"
            )
        replace_generated_frames_with_seed_rig_frames(
            paths["database"],
            input_image_path=input_image_path,
            panorama_paths=panorama_paths,
            views=views,
            output_options=output_options,
            relative_name_prefix=new_image_prefix,
            seed_camera_ids=seed_camera_ids,
            generated_records=generated_image_batch.records_by_name,
        )
        generated_image_batch = read_generated_virtual_image_batch(
            paths["database"],
            generated_image_names,
        )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "seed_camera_frame_database_rewrite",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    if match_scope == "last-k" and initial_session is None:
        raise ValueError("--match-scope last-k requires --initial-session")
    if match_scope not in {"all", "last-k"}:
        raise ValueError("match_scope must be 'all' or 'last-k'")
    new_frame_sets = read_rig_frame_image_sets_for_images(
        paths["database"],
        generated_image_batch.image_ids,
    )
    selected_old_frame_sets: list[RigFrameImageSet] = []
    match_pairs: list[tuple[str, str]] = []
    if match_scope == "last-k":
        selected_old_frame_sets = read_last_old_rig_frame_image_sets(
            paths["database"],
            excluded_image_ids=generated_image_batch.image_ids,
            limit=last_k_frames,
        )
        if not selected_old_frame_sets:
            raise RuntimeError(
                "--match-scope last-k could not find old rig frames in the "
                "initial session database"
            )
        match_pairs = build_last_k_match_pairs(
            new_frame_sets,
            selected_old_frame_sets,
            new_frame_overlap=overlap,
        )
        if not match_pairs:
            raise RuntimeError("--match-scope last-k produced no image pairs")
        write_image_pair_list(paths["match_pairs"], match_pairs)

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
        match_scope=match_scope,
        matching_options=matching_options,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        image_pairs_path=paths["match_pairs"] if match_scope == "last-k" else None,
        device=execution["device"],
    )
    record_elapsed_stage(
        elapsed_seconds_by_stage,
        "matching_and_rig_verification",
        stage_started_at,
    )

    stage_started_at = time.perf_counter()
    mapper_options = create_mapper_options(
        pycolmap_module,
        execution,
        continue_existing_model=initial_session is not None,
    )
    reconstructions = run_incremental_mapping(
        pycolmap_module,
        paths,
        image_dir=paths["image_dir"],
        mapper_options=mapper_options,
        initial_model_path=None
        if initial_session is None
        else paths["initial_sparse"],
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
    report = build_panorama_report(
        pycolmap_version=pycolmap_module.__version__,
        input_image_path=input_image_path,
        paths=paths,
        camera=camera,
        views=views,
        panorama_count=len(panorama_paths),
        matcher=matcher,
        match_scope=match_scope,
        last_k_frames=last_k_frames,
        overlap=overlap,
        loop_detection=loop_detection,
        vocab_tree_path=vocab_tree_path,
        ssl_cert_file=ssl_cert_file,
        feature_options=feature_options,
        output_options=output_options,
        execution=execution,
        masks_report=masks_report,
        feature_report=feature_report,
        feature_health=feature_health,
        image_limit=image_limit,
        new_image_prefix=new_image_prefix,
        generated_image_batch=generated_image_batch,
        new_frame_sets=new_frame_sets,
        selected_old_frame_sets=selected_old_frame_sets,
        matching_pair_count=len(match_pairs) if match_scope == "last-k" else None,
        match_pairs_path=paths["match_pairs"] if match_scope == "last-k" else None,
        initial_session=initial_session,
        initial_metadata=initial_metadata,
        reconstructions=reconstructions,
        best_model_id=best_model_id,
        best_reconstruction=best_reconstruction,
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
        "mask_dir": workspace / "masks",
        "database": workspace / "database.db",
        "sparse": workspace / "sparse",
        "initial_sparse": workspace / "initial_sparse",
        "best_sparse": workspace / "sparse_best",
        "raw_ply": workspace / "cloud_colmap_raw.ply",
        "report": workspace / "panorama_rig_report.json",
        "rig_metadata": workspace / "panorama_rig_config.json",
        "match_pairs": workspace / "last_k_match_pairs.txt",
    }


def write_rig_metadata(
    path: Path,
    input_image_path: Path,
    options: PanoramaRenderOptions,
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
    masks_report: dict[str, dict[str, float | int]],
    output_options: VirtualImageOutputOptions,
    *,
    image_limit: int | None,
) -> None:
    data = {
        "input_image_path": str(input_image_path),
        "image_limit": image_limit,
        "render_options": {
            "num_yaw_steps": options.num_yaw_steps,
            "pitches": list(options.pitches),
            "width": options.width,
            "height": options.height,
            "fov_x": options.fov_x,
            "fov_y": options.fov_y,
            "interpolation": options.interpolation,
            "virtual_image_format": output_options.image_format,
            "png_compression": output_options.png_compression,
            "jpeg_quality": output_options.jpeg_quality,
            "image_save_workers": output_options.save_workers,
            "render_backend": "precomputed_opencv_remap",
        },
        "camera": camera.report(),
        "masks": masks_report,
        "views": [
            {
                "index": view.index,
                "image_prefix": view.image_prefix,
                "yaw": view.yaw,
                "pitch": view.pitch,
                "cam_from_pano_rotation": view.cam_from_pano_rotation.tolist(),
            }
            for view in views
        ],
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_panorama_report(
    *,
    pycolmap_version: str,
    input_image_path: Path,
    paths: dict[str, Path],
    camera: ColmapCamera,
    views: Sequence[PanoramaRigView],
    panorama_count: int,
    matcher: str,
    match_scope: str,
    last_k_frames: int,
    overlap: int,
    loop_detection: bool,
    vocab_tree_path: Path | None,
    ssl_cert_file: Path | None,
    feature_options: FeaturePipelineOptions,
    output_options: VirtualImageOutputOptions,
    execution: dict[str, Any],
    masks_report: dict[str, dict[str, float | int]],
    feature_report: dict[str, dict[str, float | int]],
    feature_health: dict[str, Any],
    image_limit: int | None,
    new_image_prefix: Path | None,
    generated_image_batch: GeneratedVirtualImageBatch,
    new_frame_sets: Sequence[RigFrameImageSet],
    selected_old_frame_sets: Sequence[RigFrameImageSet],
    matching_pair_count: int | None,
    match_pairs_path: Path | None,
    initial_session: InitialPanoramaSession | None,
    initial_metadata: dict[str, Any] | None,
    reconstructions: dict[int, Any],
    best_model_id: int,
    best_reconstruction: Any,
) -> dict[str, Any]:
    database_report = read_database_stats(paths["database"])
    new_database_image_count = len(generated_image_batch.names)
    model_stats = {
        str(model_id): panorama_reconstruction_stats(reconstruction)
        for model_id, reconstruction in sorted(reconstructions.items())
    }
    return {
        "status": "raw_colmap_panorama_rig_unscaled",
        "warning": (
            "This is a monocular COLMAP reconstruction in arbitrary COLMAP "
            "scale/frame. The virtual pinhole sensors are a fixed panorama rig, "
            "but the resulting world is not metric without an external scale."
        ),
        "pycolmap_version": pycolmap_version,
        "input_image_path": str(input_image_path),
        "workspace": str(paths["workspace"]),
        "image_dir": str(paths["image_dir"]),
        "mask_dir": str(paths["mask_dir"]),
        "database_path": str(paths["database"]),
        "sparse_path": str(paths["sparse"]),
        "initial_sparse_path": (
            None if initial_session is None else str(paths["initial_sparse"])
        ),
        "best_sparse_path": str(paths["best_sparse"]),
        "raw_ply_path": str(paths["raw_ply"]),
        "rig_metadata_path": str(paths["rig_metadata"]),
        "initial_session": build_initial_session_report(
            initial_session,
            paths,
            initial_metadata,
        ),
        "camera": camera.report(),
        "options": {
            "matcher": matcher,
            "match_scope": match_scope,
            "effective_pairing": (
                "imported_last_k" if match_scope == "last-k" else matcher
            ),
            "last_k_frames": last_k_frames,
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
            "virtual_image_format": output_options.image_format,
            "png_compression": output_options.png_compression,
            "jpeg_quality": output_options.jpeg_quality,
            "image_save_workers": output_options.save_workers,
            "render_backend": "precomputed_opencv_remap",
            "image_limit": image_limit,
            "new_virtual_image_prefix": (
                None if new_image_prefix is None else new_image_prefix.as_posix()
            ),
            "requested_device": execution["requested_device"],
            "resolved_device": execution["resolved_device"],
            "cuda_available": execution["cuda_available"],
            "gpu_index": execution["gpu_index"],
            "gpu_bundle_adjustment": execution["gpu_bundle_adjustment"],
            "bundle_adjustment_gpu_index": execution[
                "bundle_adjustment_gpu_index"
            ],
            "camera_mode": "PER_FOLDER",
            "reuse_initial_camera_ids_by_prefix": initial_session is not None,
            "continue_existing_model": initial_session is not None,
            "multiple_models": initial_session is None,
            "fix_existing_frames": initial_session is not None,
            "ba_refine_sensor_from_rig": False,
            "ba_refine_focal_length": False,
            "ba_refine_principal_point": False,
            "ba_refine_extra_params": False,
            "rig_verification": True,
            "skip_image_pairs_in_same_frame": True,
        },
        "panorama_count": panorama_count,
        "virtual_camera_count": len(views),
        "generated_image_count": panorama_count * len(views),
        "database_image_count": database_report["images"],
        "new_database_image_count": new_database_image_count,
        "old_database_image_count": max(
            0,
            database_report["images"] - new_database_image_count,
        ),
        "new_database_image_ids": generated_image_batch.image_ids,
        "new_database_camera_ids": generated_image_batch.camera_ids,
        "new_database_image_names": list(generated_image_batch.names),
        "new_frame_count": len(new_frame_sets),
        "new_frame_ids": [frame.frame_id for frame in new_frame_sets],
        "selected_old_frame_count": len(selected_old_frame_sets),
        "selected_old_frame_ids": [
            frame.frame_id for frame in selected_old_frame_sets
        ],
        "selected_old_image_count": sum(
            len(frame.image_ids) for frame in selected_old_frame_sets
        ),
        "selected_old_image_ids": [
            image_id
            for frame in selected_old_frame_sets
            for image_id in frame.image_ids
        ],
        "selected_old_image_names": [
            image_name
            for frame in selected_old_frame_sets
            for image_name in frame.image_names
        ],
        "matching_pair_count": matching_pair_count,
        "match_pairs_path": None if match_pairs_path is None else str(match_pairs_path),
        "masks": masks_report,
        "features_by_virtual_camera": feature_report,
        "feature_health": feature_health,
        "database": database_report,
        "model_count": len(reconstructions),
        "best_model_id": best_model_id,
        "models": model_stats,
        "best_model": panorama_reconstruction_stats(best_reconstruction),
    }


def build_initial_session_report(
    initial_session: InitialPanoramaSession | None,
    paths: dict[str, Path],
    initial_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if initial_session is None:
        return {"enabled": False}
    previous_views = [] if initial_metadata is None else initial_metadata.get("views", [])
    return {
        "enabled": True,
        "mode": "previous_colmap_panorama_reconstruct_session",
        "workspace": str(initial_session.workspace),
        "model_path": str(initial_session.model_path),
        "database_path": str(initial_session.database_path),
        "image_dir": str(initial_session.image_dir),
        "mask_dir": str(initial_session.mask_dir),
        "rig_metadata_path": str(initial_session.rig_metadata_path),
        "report_path": (
            None
            if initial_session.report_path is None
            or not initial_session.report_path.is_file()
            else str(initial_session.report_path)
        ),
        "copied_model_path": str(paths["initial_sparse"]),
        "copied_model_preserves_rig_frames": True,
        "existing_camera_poses_refinable": False,
        "intrinsics_refinable": False,
        "virtual_camera_count": len(previous_views),
    }


def panorama_reconstruction_stats(reconstruction: Any) -> dict[str, Any]:
    stats = reconstruction_stats(reconstruction)
    stats["num_frames"] = reconstruction.num_frames()
    stats["num_registered_frames"] = reconstruction.num_reg_frames()
    return stats


def read_feature_stats_by_camera(
    database_path: Path,
) -> dict[str, dict[str, float | int]]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                substr(images.name, 1, instr(images.name, '/') - 1) AS folder,
                COUNT(*) AS image_count,
                SUM(keypoints.rows) AS total_features,
                AVG(keypoints.rows) AS mean_features,
                MIN(keypoints.rows) AS min_features,
                MAX(keypoints.rows) AS max_features
            FROM images
            JOIN keypoints USING(image_id)
            GROUP BY folder
            ORDER BY folder
            """
        ).fetchall()
    return {
        str(folder): {
            "image_count": int(image_count),
            "total_features": int(total_features),
            "mean_features": float(mean_features),
            "min_features": int(min_features),
            "max_features": int(max_features),
        }
        for folder, image_count, total_features, mean_features, min_features, max_features
        in rows
    }


def feature_health_summary(
    views: Sequence[PanoramaRigView],
    feature_report: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    expected = [view.image_prefix.rstrip("/") for view in views]
    missing = [name for name in expected if name not in feature_report]
    if missing:
        raise RuntimeError(
            f"Feature extraction did not create statistics for virtual cameras: {missing}"
        )
    zero_feature_cameras = [
        name
        for name in expected
        if int(feature_report[name]["total_features"]) == 0
    ]
    return {
        "virtual_camera_count": len(expected),
        "virtual_cameras_with_features": len(expected) - len(zero_feature_cameras),
        "zero_feature_virtual_cameras": zero_feature_cameras,
    }


def parse_pitches(value: str) -> tuple[float, ...]:
    pitches = tuple(radians(float(part.strip())) for part in value.split(",") if part.strip())
    if not pitches:
        raise argparse.ArgumentTypeError("at least one pitch is required")
    return pitches


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render 360 panoramas into a fixed virtual pinhole rig and run "
            "COLMAP reconstruction with rig constraints."
        )
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
        help=f"COLMAP panorama-rig workspace (default: {DEFAULT_WORKSPACE})",
    )
    parser.add_argument(
        "--pano-render-type",
        choices=("overlapping", "non-overlapping"),
        default="overlapping",
        help="Preset virtual camera layout.",
    )
    parser.add_argument(
        "--num-yaw-steps",
        type=int,
        default=None,
        help="Override preset yaw step count.",
    )
    parser.add_argument(
        "--pitches-deg",
        type=parse_pitches,
        default=None,
        help="Comma-separated pitch angles in degrees, overriding the preset.",
    )
    parser.add_argument("--width", type=int, default=None, help="Virtual image width.")
    parser.add_argument("--height", type=int, default=None, help="Virtual image height.")
    parser.add_argument(
        "--fov-x",
        type=float,
        default=None,
        help="Virtual horizontal field of view in radians.",
    )
    parser.add_argument(
        "--fov-y",
        type=float,
        default=None,
        help="Virtual vertical field of view in radians.",
    )
    parser.add_argument(
        "--matcher",
        choices=("sequential", "exhaustive", "vocabtree"),
        default="sequential",
        help="Feature matching strategy.",
    )
    parser.add_argument(
        "--match-scope",
        choices=("all", "last-k"),
        default="all",
        help=(
            "Image-pair scope. 'all' uses the selected matcher over the full "
            "database. 'last-k' matches current images against current images "
            "and the last K old rig frames from --initial-session."
        ),
    )
    parser.add_argument(
        "--last-k-frames",
        type=int,
        default=20,
        help="Number of old rig frames to anchor when --match-scope=last-k.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=15,
        help="Sequential matcher overlap in neighboring frames.",
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
        help="Maximum SIFT features per virtual camera image.",
    )
    parser.add_argument(
        "--guided-matching",
        action="store_true",
        help="Enable guided feature matching after geometric verification.",
    )
    parser.add_argument(
        "--virtual-image-format",
        choices=("png", "jpeg"),
        default="png",
        help="Format for generated virtual pinhole images.",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        default=1,
        help="PNG compression level from 0 to 9 (default: 1).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100 (default: 95).",
    )
    parser.add_argument(
        "--image-save-workers",
        type=int,
        default=12,
        help="Concurrent workers used to encode virtual images (default: 12).",
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
            "extraction and matching (default: -1, all GPUs)."
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
        "--initial-session",
        type=Path,
        default=None,
        help=(
            "Previous colmap-panorama-reconstruct workspace, or its sparse_best "
            "model directory, to use as the initial map."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the panorama-rig workspace.",
    )
    args = parser.parse_args(argv)
    if not args.input_image_path.is_dir():
        parser.error(f"--input-image-path must be an existing directory: {args.input_image_path}")
    if args.num_yaw_steps is not None and args.num_yaw_steps <= 0:
        parser.error("--num-yaw-steps must be positive")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be positive")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be positive")
    if args.fov_x is not None and not 0.0 < args.fov_x < pi:
        parser.error("--fov-x must be greater than 0 and less than pi radians")
    if args.fov_y is not None and not 0.0 < args.fov_y < pi:
        parser.error("--fov-y must be greater than 0 and less than pi radians")
    if args.image_limit is not None and args.image_limit <= 0:
        parser.error("--image-limit must be positive")
    if args.last_k_frames <= 0:
        parser.error("--last-k-frames must be positive")
    if args.match_scope == "last-k" and args.initial_session is None:
        parser.error("--match-scope last-k requires --initial-session")
    if args.max_num_features <= 0:
        parser.error("--max-num-features must be positive")
    if not 0 <= args.png_compression <= 9:
        parser.error("--png-compression must be between 0 and 9")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.image_save_workers <= 0:
        parser.error("--image-save-workers must be positive")
    if args.initial_session is not None and not args.initial_session.is_dir():
        parser.error(
            f"--initial-session must be an existing directory: {args.initial_session}"
        )
    if args.vocab_tree_path is not None and not args.vocab_tree_path.is_file():
        parser.error(f"--vocab-tree-path does not exist: {args.vocab_tree_path}")
    if args.ssl_cert_file is not None and not args.ssl_cert_file.is_file():
        parser.error(f"--ssl-cert-file does not exist: {args.ssl_cert_file}")
    return args


def render_options_from_args(args: argparse.Namespace) -> PanoramaRenderOptions:
    preset = default_render_options(args.pano_render_type)
    return PanoramaRenderOptions(
        num_yaw_steps=preset.num_yaw_steps
        if args.num_yaw_steps is None
        else args.num_yaw_steps,
        pitches=preset.pitches if args.pitches_deg is None else args.pitches_deg,
        width=preset.width if args.width is None else args.width,
        height=preset.height if args.height is None else args.height,
        fov_x=preset.fov_x if args.fov_x is None else args.fov_x,
        fov_y=preset.fov_y if args.fov_y is None else args.fov_y,
        interpolation=preset.interpolation,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run_panorama_reconstruction(
        input_image_path=args.input_image_path,
        workspace=args.workspace,
        render_options=render_options_from_args(args),
        matcher=args.matcher,
        match_scope=args.match_scope,
        last_k_frames=args.last_k_frames,
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
        output_options=VirtualImageOutputOptions(
            image_format=args.virtual_image_format,
            png_compression=args.png_compression,
            jpeg_quality=args.jpeg_quality,
            save_workers=args.image_save_workers,
        ),
        execution_options=PipelineExecutionOptions(
            device=args.device,
            gpu_index=args.gpu_index,
            gpu_bundle_adjustment=args.gpu_bundle_adjustment,
        ),
        image_limit=args.image_limit,
        initial_session_path=args.initial_session,
        overwrite=args.overwrite,
    )
    print(f"panoramas: {report['panorama_count']}")
    print(f"virtual cameras: {report['virtual_camera_count']}")
    print(f"generated images: {report['generated_image_count']}")
    if report["initial_session"]["enabled"]:
        print(f"initial session: {report['initial_session']['workspace']}")
    print(f"compute device: {report['options']['resolved_device']}")
    print(f"match scope: {report['options']['match_scope']}")
    print(
        "virtual cameras with features: "
        f"{report['feature_health']['virtual_cameras_with_features']}/"
        f"{report['feature_health']['virtual_camera_count']}"
    )
    print(f"best registered frames: {report['best_model']['num_registered_frames']}")
    print(f"best registered images: {report['best_model']['num_registered_images']}")
    print(f"best sparse points: {report['best_model']['num_points3D']}")
    print(f"elapsed time: {report['elapsed_time']}")
    print(f"report: {report['workspace']}/panorama_rig_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
