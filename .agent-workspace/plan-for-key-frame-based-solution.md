Date: 2026-06-29
Project version: v0.1.0
Description: Plans keyframe-based matching for the native equirectangular COLMAP pipeline.

# Plan For Keyframe-Based 360 Reconstruction

## Goal

Add keyframe-based matching to the native `EQUIRECTANGULAR` COLMAP pipeline.
The feature extraction and camera setup remain unchanged: every panorama is
imported directly, all panoramas use one shared native `EQUIRECTANGULAR`
camera, and features are extracted for every panorama image.

Keyframes affect only the matching-pair selection stage.

## Core Idea

Use a sparse but connected image-pair graph:

1. Keep a sequential local-neighbor backbone so ordered capture continuity is
   preserved.
2. Use keyframes to add longer-range structure without matching all images.
3. Prefer vocabtree-based visual clusters as the main keyframe source:
   panoramas that retrieve each other through vocabtree form visual clusters,
   and each cluster gets one representative keyframe.

This gives us a scalable matching strategy for large 360 datasets while still
keeping enough local edges for incremental mapping to start and grow reliably.

## Proposed CLI

Extend the existing `--matcher` choices:

```bash
--matcher {sequential,exhaustive,vocabtree,keyframes}
```

Add keyframe-specific options:

```bash
--keyframe-selection {vocabtree,interval}
--keyframe-interval 10
--keyframe-overlap 3
--keyframe-cluster-min-size 2
--keyframe-cluster-max-size 30
--keyframe-vocabtree-neighbors 10
--keyframe-vocabtree-checks 64
```

Defaults:

- `--matcher sequential` remains the default to avoid changing current runs.
- If `--matcher keyframes`, default to `--keyframe-selection vocabtree`.
- `--overlap` continues to mean local sequential neighbor overlap.
- `--vocab-tree-path` is required for `--matcher keyframes --keyframe-selection vocabtree`.
- `--loop-detection` is ignored in keyframe mode because keyframe mode writes
  its own imported pair list.

Example:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-equirectangular-keyframes \
  --matcher keyframes \
  --keyframe-selection vocabtree \
  --vocab-tree-path /path/to/vocab_tree.bin \
  --overlap 5 \
  --keyframe-vocabtree-neighbors 10 \
  --keyframe-cluster-max-size 30 \
  --keyframe-overlap 3 \
  --overwrite
```

## Matching Strategy

### Sequential Backbone

Always add local continuity pairs in keyframe mode:

- For each image `i`, match to the next `--overlap` images.
- Deduplicate unordered pairs.
- Never match an image to itself.

This protects against vocabtree failures in visually repetitive or weakly
textured areas.

### Interval Keyframes

This is the fallback and test-friendly selector:

- Select every `--keyframe-interval` image.
- Always include the first and last image.
- Add image-to-nearest-keyframe pairs.
- Add keyframe-to-keyframe pairs to the next `--keyframe-overlap` keyframes.

### Vocabtree Cluster Keyframes

This is the preferred selector:

1. After feature extraction, open the COLMAP database with pycolmap.
2. Use `pycolmap.VocabTreePairGenerator` with `VocabTreePairingOptions`.
3. Generate candidate visual-neighbor pairs for the panorama image ids.
4. Build an undirected graph:
   - node = image id/name
   - edge = vocabtree candidate visual neighbor
5. Split graph components into clusters.
6. If a cluster is larger than `--keyframe-cluster-max-size`, split it by image
   order into smaller chunks.
7. If a cluster is smaller than `--keyframe-cluster-min-size`, merge it into
   the nearest ordered cluster when possible; otherwise keep it as a singleton.
8. Pick one representative keyframe per cluster.

Representative selection for v1:

- Choose the node with the highest graph degree.
- Tie-break by larger feature count from the `keypoints` table.
- Final tie-break by earlier image order for determinism.

Pairs generated from clusters:

- Every image matches its cluster keyframe.
- Cluster keyframes match the next `--keyframe-overlap` keyframes in image
  order.
- Optional cross-cluster vocabtree edges are kept only when either endpoint is
  a cluster keyframe.

## Implementation Changes

Main file: `src/colmap/panorama_reconstruct.py`.

Add small data structures:

- `KeyframeSelectionOptions`
- `KeyframeCluster`
- `KeyframePairPlan`

Add helper functions:

- `build_local_sequential_pairs(image_names, overlap)`
- `build_interval_keyframes(image_names, keyframe_interval)`
- `build_interval_keyframe_pair_plan(image_names, options)`
- `build_vocabtree_candidate_pairs(pycolmap_module, database_path, image_ids, options)`
- `cluster_vocabtree_pairs(image_records, candidate_pairs, options)`
- `select_cluster_keyframe(cluster, feature_counts_by_image_id, image_order)`
- `build_vocabtree_keyframe_pair_plan(...)`
- `deduplicate_image_pairs(pairs)`

Update the pipeline:

1. Import images and extract features as today.
2. Read generated image ids/names from the database.
3. If `matcher != "keyframes"`, keep the current matching path.
4. If `matcher == "keyframes"`:
   - build a `KeyframePairPlan`
   - write `workspace/match_pairs.txt`
   - call `match_features(..., image_pairs_path=paths["match_pairs"])`
5. Run incremental mapping exactly as today.

Update report fields:

- `options.matcher`
- `options.effective_pairing`
- `options.keyframe_selection`
- `options.keyframe_interval`
- `options.keyframe_overlap`
- `options.keyframe_cluster_min_size`
- `options.keyframe_cluster_max_size`
- `options.keyframe_vocabtree_neighbors`
- `options.keyframe_vocabtree_checks`
- `options.matching_pair_count`
- `options.match_pairs_path`
- `keyframes`
- `keyframe_clusters`

Report cluster records should include:

- `cluster_id`
- `image_count`
- `image_names`
- `keyframe_image_name`
- `keyframe_image_id`
- `source`

## Edge Cases

- 0 images: existing import path already raises.
- 1 image: no pairs; fail before matching with a clear error because COLMAP
  cannot reconstruct from one panorama.
- 2 images: local sequential pair only.
- Missing vocab tree in vocabtree keyframe mode: CLI/parser error.
- Vocabtree returns no candidate pairs: fall back to interval keyframes and
  record `keyframe_selection_effective: interval_fallback`.
- Very large connected component: split by image order using
  `--keyframe-cluster-max-size`.
- Repetitive environment: sequential backbone remains active even if vocabtree
  clusters are noisy.

## Test Plan

Unit tests:

- local sequential pair generation is deterministic and deduplicated.
- interval keyframes include first/last and respect interval.
- vocabtree graph clustering handles isolated nodes, multiple components, and
  oversized components.
- representative selection uses degree, then feature count, then image order.
- pair plan combines local, image-to-keyframe, and keyframe-to-keyframe pairs
  without duplicates.

CLI tests:

- `--matcher keyframes` is accepted.
- invalid keyframe intervals/overlaps/cluster sizes are rejected.
- vocabtree keyframe mode requires `--vocab-tree-path`.
- interval keyframe mode does not require `--vocab-tree-path`.

Pipeline tests with fake pycolmap:

- keyframe interval mode writes `match_pairs.txt` and calls imported pair
  matching.
- keyframe vocabtree mode uses fake vocabtree candidate pairs and writes the
  expected pair list.
- report includes keyframe settings, selected keyframes, clusters, pair count,
  and pair-list path.
- non-keyframe matchers keep existing behavior.

Validation:

```bash
python -m pytest
python -m colmap.panorama_reconstruct --help
```

## Rollout Plan

1. Implement interval keyframe mode first as a deterministic baseline.
2. Add vocabtree candidate-pair generation and clustering.
3. Add report details and README examples.
4. Run smoke reconstruction with `--image-limit`.
5. Compare pair count, registered images, sparse points, and runtime against:
   - `--matcher sequential`
   - `--matcher vocabtree`
   - `--matcher keyframes --keyframe-selection interval`
   - `--matcher keyframes --keyframe-selection vocabtree`

## Open Follow-Ups

- Consider using verified inlier counts as cluster edge weights after an
  initial lightweight matching pass.
- Consider geospatial or timestamp-aware clustering if capture metadata becomes
  available.
- Consider exporting the keyframe graph as GraphML/JSON for inspection.
- Consider a visual debug report that draws clusters over capture order.
