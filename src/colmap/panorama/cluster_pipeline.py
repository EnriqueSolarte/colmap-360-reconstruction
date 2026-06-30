"""Keyframe clustering orchestration for panorama reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from colmap.panorama.database import read_feature_counts_by_image_name
from colmap.panorama.keyframe_matching import (
    ImagePair,
    KeyframePairPlan,
    KeyframeSelectionOptions,
    build_interval_keyframe_pair_plan,
    build_sequential_vocabtree_pair_plan,
    deduplicate_image_pairs,
    validate_keyframe_selection_options,
)
from colmap.panorama.pycolmap_adapter import build_vocabtree_candidate_pairs


def image_id_pairs_to_name_pairs(
    image_id_pairs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    generated_image_batch: Any,
) -> tuple[ImagePair, ...]:
    image_name_by_id = {
        image_id: image_name
        for image_name, (
            image_id,
            _camera_id,
        ) in generated_image_batch.records_by_name.items()
    }
    return deduplicate_image_pairs(
        (
            (image_name_by_id[image_id1], image_name_by_id[image_id2])
            for image_id1, image_id2 in image_id_pairs
            if image_id1 in image_name_by_id and image_id2 in image_name_by_id
        ),
    )


def build_keyframe_pair_plan(
    pycolmap_module: Any,
    *,
    database_path: Path,
    generated_image_batch: Any,
    keyframe_options: KeyframeSelectionOptions,
    vocab_tree_path: Path | None,
) -> KeyframePairPlan:
    validate_keyframe_selection_options(keyframe_options)
    image_names = generated_image_batch.names
    image_ids_by_name = {
        image_name: image_id
        for image_name, (
            image_id,
            _camera_id,
        ) in generated_image_batch.records_by_name.items()
    }
    if keyframe_options.selection == "interval":
        return build_interval_keyframe_pair_plan(
            image_names,
            keyframe_options,
            image_ids_by_name=image_ids_by_name,
        )
    if vocab_tree_path is None:
        raise ValueError(
            "--vocab-tree-path is required for vocabtree-backed keyframe selection",
        )

    candidate_image_id_pairs = build_vocabtree_candidate_pairs(
        pycolmap_module,
        database_path,
        generated_image_batch.image_ids,
        keyframe_options,
        vocab_tree_path=vocab_tree_path,
    )
    candidate_pairs = image_id_pairs_to_name_pairs(
        candidate_image_id_pairs,
        generated_image_batch,
    )
    if not candidate_pairs:
        return build_interval_keyframe_pair_plan(
            image_names,
            keyframe_options,
            image_ids_by_name=image_ids_by_name,
            effective_selection="interval_fallback",
        )

    feature_counts_by_name = read_feature_counts_by_image_name(
        database_path,
        image_names,
    )
    return build_sequential_vocabtree_pair_plan(
        image_names,
        candidate_pairs,
        keyframe_options,
        image_ids_by_name=image_ids_by_name,
        feature_counts_by_name=feature_counts_by_name,
    )
