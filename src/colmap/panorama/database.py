"""SQLite read helpers for COLMAP panorama reconstruction."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

from colmap.panorama.constants import COLMAP_IMAGE_PAIR_ID_PRIME
from colmap.panorama.types import GeneratedImageBatch


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


def read_feature_counts_by_image_name(
    database_path: Path,
    image_names: Sequence[str],
) -> dict[str, int]:
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if "images" not in tables:
            return {image_name: 0 for image_name in image_names}
        keypoint_join = (
            "LEFT JOIN keypoints ON keypoints.image_id = images.image_id"
            if "keypoints" in tables
            else ""
        )
        feature_expr = "COALESCE(keypoints.rows, 0)" if "keypoints" in tables else "0"
        rows = connection.execute(
            f"""
            SELECT images.name, {feature_expr}
            FROM images
            {keypoint_join}
            """
        ).fetchall()
    counts_by_name = {str(name): int(count or 0) for name, count in rows}
    return {
        image_name: counts_by_name.get(image_name, 0)
        for image_name in image_names
    }
