"""Keyframe-based image-pair planning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

ImagePair = tuple[str, str]


@dataclass(frozen=True)
class KeyframeSelectionOptions:
    selection: str = "interval"
    interval: int = 10
    overlap: int = 3
    backbone_policy: str = "gaps"
    backbone_overlap: int = 1
    cluster_min_size: int = 2
    cluster_max_size: int = 30
    cluster_recent_window: int = 3
    vocabtree_neighbors: int = 10
    vocabtree_checks: int = 64


@dataclass(frozen=True)
class KeyframeCluster:
    cluster_id: int
    image_names: tuple[str, ...]
    keyframe_image_name: str
    source: str
    image_ids: tuple[int, ...] = ()
    keyframe_image_id: int | None = None

    def report(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "image_count": len(self.image_names),
            "image_names": list(self.image_names),
            "keyframe_image_name": self.keyframe_image_name,
            "keyframe_image_id": self.keyframe_image_id,
            "source": self.source,
        }


@dataclass(frozen=True)
class KeyframePairPlan:
    selection: str
    effective_selection: str
    keyframe_pairs: tuple[ImagePair, ...] = ()
    vocabtree_pairs: tuple[ImagePair, ...] = ()
    sequential_backbone_pairs: tuple[ImagePair, ...] = ()
    keyframes: tuple[str, ...] = ()
    clusters: tuple[KeyframeCluster, ...] = ()

    @property
    def all_pairs(self) -> tuple[ImagePair, ...]:
        return deduplicate_image_pairs(
            (
                *self.keyframe_pairs,
                *self.vocabtree_pairs,
                *self.sequential_backbone_pairs,
            ),
        )

    @property
    def pair_counts_by_source(self) -> dict[str, int]:
        return {
            "keyframe": len(deduplicate_image_pairs(self.keyframe_pairs)),
            "vocabtree": len(deduplicate_image_pairs(self.vocabtree_pairs)),
            "sequential_backbone": len(
                deduplicate_image_pairs(self.sequential_backbone_pairs),
            ),
            "all": len(self.all_pairs),
        }

    def report_keyframes(self) -> list[dict[str, object]]:
        return [
            {"image_name": image_name, "selection": self.effective_selection}
            for image_name in self.keyframes
        ]

    def report_clusters(self) -> list[dict[str, object]]:
        return [cluster.report() for cluster in self.clusters]


def validate_keyframe_selection_options(options: KeyframeSelectionOptions) -> None:
    if options.selection not in {"interval", "sequential-vocabtree", "vocabtree"}:
        raise ValueError(
            "keyframe selection must be 'interval', 'sequential-vocabtree', "
            "or 'vocabtree'",
        )
    if options.interval <= 0:
        raise ValueError("keyframe interval must be positive")
    if options.overlap < 0:
        raise ValueError("keyframe overlap must be non-negative")
    if options.backbone_policy not in {"none", "gaps", "all"}:
        raise ValueError("keyframe backbone policy must be 'none', 'gaps', or 'all'")
    if options.backbone_overlap < 0:
        raise ValueError("keyframe backbone overlap must be non-negative")
    if options.cluster_min_size <= 0:
        raise ValueError("keyframe cluster minimum size must be positive")
    if options.cluster_max_size <= 0:
        raise ValueError("keyframe cluster maximum size must be positive")
    if options.cluster_min_size > options.cluster_max_size:
        raise ValueError(
            "keyframe cluster minimum size cannot exceed maximum size",
        )
    if options.cluster_recent_window < 0:
        raise ValueError("keyframe cluster recent window must be non-negative")
    if options.vocabtree_neighbors <= 0:
        raise ValueError("keyframe vocabtree neighbors must be positive")
    if options.vocabtree_checks <= 0:
        raise ValueError("keyframe vocabtree checks must be positive")


def image_pair_adjacency(pairs: Iterable[ImagePair]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for pair in deduplicate_image_pairs(pairs):
        image_name_a, image_name_b = pair
        adjacency.setdefault(image_name_a, set()).add(image_name_b)
        adjacency.setdefault(image_name_b, set()).add(image_name_a)
    return adjacency


def image_pair_connected(
    adjacency: dict[str, set[str]],
    image_name_a: str,
    image_name_b: str,
) -> bool:
    return image_name_b in adjacency.get(image_name_a, set())


def build_sequential_backbone_pairs(
    image_names: Iterable[str],
    overlap: int,
) -> tuple[ImagePair, ...]:
    if overlap < 0:
        raise ValueError("sequential backbone overlap must be non-negative")
    ordered_image_names = tuple(image_names)
    pairs: list[ImagePair] = []
    for image_index, image_name in enumerate(ordered_image_names):
        for neighbor_index in range(
            image_index + 1,
            min(len(ordered_image_names), image_index + overlap + 1),
        ):
            pairs.append((image_name, ordered_image_names[neighbor_index]))
    return deduplicate_image_pairs(pairs)


def select_cluster_keyframe(
    image_names: Iterable[str],
    candidate_pairs: Iterable[ImagePair],
    feature_counts_by_name: dict[str, int] | None = None,
    image_order: dict[str, int] | None = None,
) -> str:
    cluster_image_names = tuple(image_names)
    if not cluster_image_names:
        raise ValueError("cannot select a keyframe from an empty cluster")
    feature_counts = feature_counts_by_name or {}
    order = image_order or {
        image_name: image_index
        for image_index, image_name in enumerate(cluster_image_names)
    }
    cluster_set = set(cluster_image_names)
    adjacency = image_pair_adjacency(candidate_pairs)
    center_index = (len(cluster_image_names) - 1) / 2.0
    cluster_position = {
        image_name: image_index
        for image_index, image_name in enumerate(cluster_image_names)
    }

    def score(image_name: str) -> tuple[int, int, float, int]:
        degree = sum(
            1
            for neighbor_name in adjacency.get(image_name, set())
            if neighbor_name in cluster_set
        )
        feature_count = feature_counts.get(image_name, 0)
        middle_distance = abs(cluster_position[image_name] - center_index)
        return (
            degree,
            feature_count,
            -middle_distance,
            -order.get(image_name, cluster_position[image_name]),
        )

    return max(cluster_image_names, key=score)


def build_keyframe_pairs_from_clusters(
    clusters: Iterable[KeyframeCluster],
    keyframe_overlap: int,
) -> tuple[ImagePair, ...]:
    if keyframe_overlap < 0:
        raise ValueError("keyframe overlap must be non-negative")
    cluster_list = tuple(clusters)
    pairs: list[ImagePair] = []
    for cluster in cluster_list:
        for image_name in cluster.image_names:
            pairs.append((image_name, cluster.keyframe_image_name))
    keyframe_names = tuple(cluster.keyframe_image_name for cluster in cluster_list)
    for keyframe_index, keyframe_name in enumerate(keyframe_names):
        for neighbor_index in range(
            keyframe_index + 1,
            min(len(keyframe_names), keyframe_index + keyframe_overlap + 1),
        ):
            pairs.append((keyframe_name, keyframe_names[neighbor_index]))
    return deduplicate_image_pairs(pairs)


def build_backbone_pairs_for_policy(
    image_names: Iterable[str],
    *,
    existing_pairs: Iterable[ImagePair],
    policy: str,
    overlap: int,
) -> tuple[ImagePair, ...]:
    if policy == "none" or overlap == 0:
        return ()
    if policy == "all":
        return build_sequential_backbone_pairs(image_names, overlap)
    if policy != "gaps":
        raise ValueError("keyframe backbone policy must be 'none', 'gaps', or 'all'")

    adjacency = image_pair_adjacency(existing_pairs)
    gap_pairs: list[ImagePair] = []
    ordered_image_names = tuple(image_names)
    for image_index, image_name in enumerate(ordered_image_names):
        for neighbor_index in range(
            image_index + 1,
            min(len(ordered_image_names), image_index + overlap + 1),
        ):
            neighbor_name = ordered_image_names[neighbor_index]
            if not image_pair_connected(adjacency, image_name, neighbor_name):
                gap_pairs.append((image_name, neighbor_name))
    return deduplicate_image_pairs(gap_pairs)


def build_sequential_vocabtree_clusters(
    image_names: Iterable[str],
    candidate_pairs: Iterable[ImagePair],
    options: KeyframeSelectionOptions,
    *,
    image_ids_by_name: dict[str, int] | None = None,
    feature_counts_by_name: dict[str, int] | None = None,
) -> tuple[KeyframeCluster, ...]:
    validate_keyframe_selection_options(options)
    ordered_image_names = tuple(image_names)
    if not ordered_image_names:
        return ()

    candidate_pair_list = deduplicate_image_pairs(candidate_pairs)
    adjacency = image_pair_adjacency(candidate_pair_list)
    image_order = {
        image_name: image_index
        for image_index, image_name in enumerate(ordered_image_names)
    }
    ids_by_name = image_ids_by_name or {}
    clusters: list[KeyframeCluster] = []
    current_cluster: list[str] = [ordered_image_names[0]]

    def connected_to_current_cluster(image_name: str) -> bool:
        temporary_keyframe = current_cluster[0]
        recent_window = options.cluster_recent_window
        recent_images = (
            current_cluster[-recent_window:]
            if recent_window > 0
            else []
        )
        references = (temporary_keyframe, *recent_images)
        return any(
            image_pair_connected(adjacency, image_name, reference_name)
            for reference_name in references
        )

    def close_current_cluster() -> None:
        cluster_image_names = tuple(current_cluster)
        keyframe_image_name = select_cluster_keyframe(
            cluster_image_names,
            candidate_pair_list,
            feature_counts_by_name=feature_counts_by_name,
            image_order=image_order,
        )
        clusters.append(
            KeyframeCluster(
                cluster_id=len(clusters),
                image_names=cluster_image_names,
                image_ids=tuple(
                    ids_by_name[image_name]
                    for image_name in cluster_image_names
                    if image_name in ids_by_name
                ),
                keyframe_image_name=keyframe_image_name,
                keyframe_image_id=ids_by_name.get(keyframe_image_name),
                source="sequential-vocabtree",
            ),
        )

    for image_name in ordered_image_names[1:]:
        cluster_is_full = len(current_cluster) >= options.cluster_max_size
        if cluster_is_full or not connected_to_current_cluster(image_name):
            close_current_cluster()
            current_cluster = [image_name]
            continue
        current_cluster.append(image_name)

    close_current_cluster()
    return tuple(clusters)


def build_sequential_vocabtree_pair_plan(
    image_names: Iterable[str],
    candidate_pairs: Iterable[ImagePair],
    options: KeyframeSelectionOptions,
    *,
    image_ids_by_name: dict[str, int] | None = None,
    feature_counts_by_name: dict[str, int] | None = None,
) -> KeyframePairPlan:
    selection_options = KeyframeSelectionOptions(
        selection="sequential-vocabtree",
        interval=options.interval,
        overlap=options.overlap,
        backbone_policy=options.backbone_policy,
        backbone_overlap=options.backbone_overlap,
        cluster_min_size=options.cluster_min_size,
        cluster_max_size=options.cluster_max_size,
        cluster_recent_window=options.cluster_recent_window,
        vocabtree_neighbors=options.vocabtree_neighbors,
        vocabtree_checks=options.vocabtree_checks,
    )
    clusters = build_sequential_vocabtree_clusters(
        image_names,
        candidate_pairs,
        selection_options,
        image_ids_by_name=image_ids_by_name,
        feature_counts_by_name=feature_counts_by_name,
    )
    keyframe_pairs = build_keyframe_pairs_from_clusters(
        clusters,
        selection_options.overlap,
    )
    ordered_image_names = tuple(image_names)
    vocabtree_pairs = deduplicate_image_pairs(candidate_pairs)
    sequential_backbone_pairs = build_backbone_pairs_for_policy(
        ordered_image_names,
        existing_pairs=(*keyframe_pairs, *vocabtree_pairs),
        policy=selection_options.backbone_policy,
        overlap=selection_options.backbone_overlap,
    )
    return KeyframePairPlan(
        selection=selection_options.selection,
        effective_selection=selection_options.selection,
        keyframe_pairs=keyframe_pairs,
        vocabtree_pairs=vocabtree_pairs,
        sequential_backbone_pairs=sequential_backbone_pairs,
        keyframes=tuple(cluster.keyframe_image_name for cluster in clusters),
        clusters=clusters,
    )


def build_interval_keyframes(
    image_names: Iterable[str],
    interval: int,
) -> tuple[str, ...]:
    if interval <= 0:
        raise ValueError("keyframe interval must be positive")
    ordered_image_names = tuple(image_names)
    if not ordered_image_names:
        return ()
    keyframe_indices = set(range(0, len(ordered_image_names), interval))
    keyframe_indices.add(0)
    keyframe_indices.add(len(ordered_image_names) - 1)
    return tuple(
        ordered_image_names[index]
        for index in sorted(keyframe_indices)
    )


def nearest_keyframe_name(
    image_name: str,
    keyframes: Iterable[str],
    image_order: dict[str, int],
) -> str:
    keyframe_names = tuple(keyframes)
    if not keyframe_names:
        raise ValueError("cannot find nearest keyframe without keyframes")
    image_index = image_order[image_name]
    return min(
        keyframe_names,
        key=lambda keyframe_name: (
            abs(image_order[keyframe_name] - image_index),
            image_order[keyframe_name],
        ),
    )


def build_interval_keyframe_pair_plan(
    image_names: Iterable[str],
    options: KeyframeSelectionOptions,
    *,
    image_ids_by_name: dict[str, int] | None = None,
    effective_selection: str = "interval",
) -> KeyframePairPlan:
    selection_options = KeyframeSelectionOptions(
        selection="interval",
        interval=options.interval,
        overlap=options.overlap,
        backbone_policy=options.backbone_policy,
        backbone_overlap=options.backbone_overlap,
        cluster_min_size=options.cluster_min_size,
        cluster_max_size=options.cluster_max_size,
        cluster_recent_window=options.cluster_recent_window,
        vocabtree_neighbors=options.vocabtree_neighbors,
        vocabtree_checks=options.vocabtree_checks,
    )
    validate_keyframe_selection_options(selection_options)
    ordered_image_names = tuple(image_names)
    keyframes = build_interval_keyframes(
        ordered_image_names,
        selection_options.interval,
    )
    image_order = {
        image_name: image_index
        for image_index, image_name in enumerate(ordered_image_names)
    }
    ids_by_name = image_ids_by_name or {}
    images_by_keyframe = {keyframe: [] for keyframe in keyframes}
    for image_name in ordered_image_names:
        nearest_keyframe = nearest_keyframe_name(image_name, keyframes, image_order)
        images_by_keyframe[nearest_keyframe].append(image_name)

    clusters = tuple(
        KeyframeCluster(
            cluster_id=cluster_id,
            image_names=tuple(images_by_keyframe[keyframe]),
            image_ids=tuple(
                ids_by_name[image_name]
                for image_name in images_by_keyframe[keyframe]
                if image_name in ids_by_name
            ),
            keyframe_image_name=keyframe,
            keyframe_image_id=ids_by_name.get(keyframe),
            source=effective_selection,
        )
        for cluster_id, keyframe in enumerate(keyframes)
        if images_by_keyframe[keyframe]
    )
    keyframe_pairs = build_keyframe_pairs_from_clusters(
        clusters,
        selection_options.overlap,
    )
    sequential_backbone_pairs = build_backbone_pairs_for_policy(
        ordered_image_names,
        existing_pairs=keyframe_pairs,
        policy=selection_options.backbone_policy,
        overlap=selection_options.backbone_overlap,
    )
    return KeyframePairPlan(
        selection=options.selection,
        effective_selection=effective_selection,
        keyframe_pairs=keyframe_pairs,
        sequential_backbone_pairs=sequential_backbone_pairs,
        keyframes=keyframes,
        clusters=clusters,
    )


def normalize_image_pair(pair: ImagePair) -> ImagePair | None:
    image_name_a, image_name_b = pair
    if image_name_a == image_name_b:
        return None
    if image_name_a < image_name_b:
        return image_name_a, image_name_b
    return image_name_b, image_name_a


def deduplicate_image_pairs(pairs: Iterable[ImagePair]) -> tuple[ImagePair, ...]:
    unique_pairs: set[ImagePair] = set()
    deduplicated_pairs: list[ImagePair] = []
    for pair in pairs:
        normalized_pair = normalize_image_pair(pair)
        if normalized_pair is None or normalized_pair in unique_pairs:
            continue
        unique_pairs.add(normalized_pair)
        deduplicated_pairs.append(normalized_pair)
    return tuple(deduplicated_pairs)
