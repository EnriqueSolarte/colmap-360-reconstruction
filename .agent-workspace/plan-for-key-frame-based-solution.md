Date: 2026-06-30
Project version: v0.1.0
Description: Plan and implementation record for keyframe/vocabtree panorama matching.

# Keyframe-Based Matching Plan And Implementation

## Goal

Add keyframe-based matching to the native `EQUIRECTANGULAR` panorama COLMAP
pipeline.

The import and feature stages stay unchanged:

- every panorama is imported directly into the COLMAP workspace;
- all panoramas use one shared native `EQUIRECTANGULAR` camera;
- features are extracted for every panorama image.

Keyframes only change the matching-pair planning stage. Instead of matching all
images or relying only on sequential neighbors, the pipeline builds a sparse
pair graph, writes it to `match_pairs.txt`, and asks COLMAP to match only those
imported pairs.

## Plan We Chose

The final design is keyframe/vocabtree first, sequential fallback second.

The planned pair graph order is:

1. Select keyframes from ordered panorama images.
2. Add image-to-keyframe pairs inside each cluster.
3. Add keyframe-to-neighbor-keyframe pairs.
4. In vocabtree mode, add vocabulary-tree visual-neighbor candidate pairs.
5. Add local sequential fallback pairs according to a policy.
6. Deduplicate all unordered image-name pairs.
7. Write the final list to `workspace/match_pairs.txt`.

The reason for this order: vocabtree/keyframes give the runtime reduction, while
the sequential fallback is a safety net for weak or disconnected visual
retrieval.

## What Was Implemented

Implemented modules:

- `src/colmap/panorama/keyframe_matching.py`
- `src/colmap/panorama/cluster_pipeline.py`
- `src/colmap/panorama/cli_clusters.py`
- updates to `src/colmap/panorama/pipeline.py`
- updates to `src/colmap/panorama/cli.py`
- updates to `src/colmap/panorama/pycolmap_adapter.py`
- updates to `src/colmap/panorama/database.py`
- updates to `src/colmap/panorama/reports.py`
- compatibility exports in `src/colmap/panorama_reconstruct.py`
- vocabulary-tree downloader in `src/tools/download_colmap_vocab_tree.py`

Implemented command-line entry points:

```bash
colmap-panorama-reconstruct --matcher keyframes ...
colmap-panorama-clusters ...
tools-download-colmap-vocab-tree ...
```

## Implemented Pair Planning API

`keyframe_matching.py` owns the pure planning logic.

Main data structures:

- `KeyframeSelectionOptions`
- `KeyframeCluster`
- `KeyframePairPlan`

Main helpers:

- `normalize_image_pair`
- `deduplicate_image_pairs`
- `build_sequential_backbone_pairs`
- `build_backbone_pairs_for_policy`
- `build_interval_keyframes`
- `build_interval_keyframe_pair_plan`
- `build_sequential_vocabtree_clusters`
- `build_sequential_vocabtree_pair_plan`
- `select_cluster_keyframe`

`KeyframePairPlan` preserves pair provenance:

- `keyframe_pairs`
- `vocabtree_pairs`
- `sequential_backbone_pairs`
- `all_pairs`
- `pair_counts_by_source`
- `keyframes`
- `clusters`

The final pair list is deterministic, unordered, deduplicated, and stored as
image-name pairs.

## Implemented Modes

### Interval Keyframes

This is the deterministic baseline.

Behavior:

1. Select every `--keyframe-interval` image.
2. Always include the first and last image.
3. Assign each image to its nearest keyframe.
4. Add image-to-keyframe pairs.
5. Add keyframe-to-next-keyframe pairs using `--keyframe-overlap`.
6. Add sequential fallback pairs according to `--keyframe-backbone-policy`.

This mode does not require a vocabulary tree.

Example:

```bash
colmap-panorama-clusters \
  --workspace /workspace/data/colmap-equirectangular \
  --keyframe-selection interval \
  --keyframe-interval 10
```

### Sequential Vocabtree Keyframes

This is the preferred mode for ordered panorama captures.

Behavior:

1. Features are already extracted for all images.
2. `pycolmap.VocabTreePairGenerator` produces visual-neighbor candidate pairs
   from the COLMAP database.
3. Images are scanned in capture order.
4. The first image starts the current cluster as a temporary anchor.
5. Each next image joins the current cluster if it is connected by vocabtree to
   the temporary anchor or one of the recent images in the cluster.
6. If it is not visually connected, the current cluster closes and a new cluster
   starts.
7. After a cluster closes, the final keyframe is selected from that cluster.

The first image in a cluster is only a temporary anchor. The final keyframe is
selected after the cluster is known.

Implemented keyframe selection score:

1. highest vocabtree degree inside the cluster;
2. higher feature count from the `keypoints` table;
3. image closest to the cluster middle;
4. earlier image order for deterministic ties.

Example:

```bash
colmap-panorama-clusters \
  --workspace /workspace/data/colmap-equirectangular \
  --vocab-tree-path /workspace/data/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin \
  --keyframe-selection sequential-vocabtree
```

## Sequential Fallback Policy

Implemented policy flag:

```bash
--keyframe-backbone-policy {none,gaps,all}
--keyframe-backbone-overlap 1
```

Policies:

- `none`: add no local sequential fallback pairs.
- `all`: add local sequential pairs for every image to the next
  `--keyframe-backbone-overlap` images.
- `gaps`: add only local sequential pairs missing from the keyframe/vocabtree
  pair graph.

Default behavior for keyframe mode is `gaps` with overlap `1`.

## Full Reconstruction Integration

The full reconstruction path now supports:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-equirectangular-keyframes \
  --matcher keyframes \
  --keyframe-selection sequential-vocabtree \
  --vocab-tree-path /workspace/data/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin \
  --keyframe-backbone-policy gaps \
  --keyframe-backbone-overlap 1 \
  --overwrite
```

Pipeline flow:

1. Import panorama images.
2. Extract features for every panorama.
3. Read generated image names and ids from the COLMAP database.
4. If `--matcher keyframes`, call `cluster_pipeline.build_keyframe_pair_plan`.
5. Write `workspace/match_pairs.txt`.
6. Call COLMAP matching with imported image pairs.
7. Run incremental mapping as before.
8. Write the normal panorama reconstruction report.

Non-keyframe matchers still use their previous matching paths.

## Standalone Cluster Pipeline

The standalone cluster-stage CLI was added because this stage needs to be
inspectable without running full reconstruction:

```bash
colmap-panorama-clusters \
  --workspace /workspace/data/colmap-equirectangular-keyframes \
  --vocab-tree-path /workspace/data/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin \
  --keyframe-selection sequential-vocabtree
```

This command assumes the workspace already contains:

- `panorama_equirectangular_config.json`
- `database.db`
- extracted features in the database

It writes:

- `match_pairs.txt`
- `panorama_keyframe_cluster_report.json`

The cluster report records:

- selected mode and effective mode;
- selected keyframes;
- keyframe clusters;
- pair counts by source;
- final matching pair count;
- paths to metadata, database, pair list, and report.

## Vocab Tree File

Sequential-vocabtree mode requires a local COLMAP vocabulary tree file. A helper
was added:

```bash
tools-download-colmap-vocab-tree \
  --variant 256k \
  --destination-dir /workspace/data/colmap-vocab-tree
```

Supported variants:

- `32k`
- `256k`
- `1m`
- `all`

The downloader prints progress, skips existing files by default, and supports
`--overwrite` and `--dry-run`.

## Report Updates

The normal panorama report now records keyframe information when keyframe mode
is used.

Added report fields include:

- `options.effective_pairing`
- `options.keyframe_selection`
- `options.keyframe_interval`
- `options.keyframe_overlap`
- `options.keyframe_backbone_policy`
- `options.keyframe_backbone_overlap`
- `options.keyframe_cluster_min_size`
- `options.keyframe_cluster_max_size`
- `options.keyframe_cluster_recent_window`
- `options.keyframe_vocabtree_neighbors`
- `options.keyframe_vocabtree_checks`
- `options.keyframe_pair_count`
- `options.vocabtree_pair_count`
- `options.sequential_backbone_pair_count`
- `options.matching_pair_count`
- `options.match_pairs_path`
- `keyframes`
- `keyframe_clusters`

## Important Edge Behavior

- Missing `--vocab-tree-path` in sequential-vocabtree mode is a CLI error.
- Interval mode does not require a vocabulary tree.
- If vocabtree candidate generation returns no pairs, the implementation falls
  back to interval keyframes and records `effective_selection` as
  `interval_fallback`.
- Pair counts are tracked separately for keyframe, vocabtree, and sequential
  fallback pairs.
- Final `all_pairs` are deduplicated across all sources.

## Implemented Tests

Tests cover:

- pair normalization and deduplication;
- pair source accounting;
- interval keyframe selection;
- interval pair planning;
- sequential-vocabtree ordered clustering;
- keyframe selection tie-breaking;
- fallback policies `none`, `gaps`, and `all`;
- pycolmap vocabtree candidate-pair adapter with fake pycolmap;
- full reconstruction keyframe integration with fake pycolmap;
- standalone cluster CLI;
- vocabulary-tree downloader without real network access;
- report fields and pair-list writing.

Latest verified command:

```bash
python -m pytest
```

Last observed result:

```text
194 passed
```

## Open Follow-Ups

- Run a real smoke reconstruction with `--image-limit`.
- Compare pair count, runtime, registered images, sparse points, and reprojection
  error against:
  - `--matcher sequential`;
  - `--matcher vocabtree`;
  - `--matcher keyframes --keyframe-selection interval`;
  - `--matcher keyframes --keyframe-selection sequential-vocabtree`.
- Inspect whether `gaps`, `none`, or `all` is the best fallback policy on real
  data.
- Consider exporting the keyframe graph as JSON or GraphML for visual debugging.
