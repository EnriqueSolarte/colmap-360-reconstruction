Date: 2026-06-30
Project version: v0.1.0
Description: Describes the current native equirectangular COLMAP implementation and runtime flow

# Current Implementation Summary

## Project Shape

The project is packaged as `colmap-360-reconstruction` version `0.1.0`.

The maintained reconstruction path uses COLMAP's native `EQUIRECTANGULAR`
camera model directly. It no longer renders virtual pinhole views for the main
panorama reconstruction command.

Primary package areas:

- `src/render_pinhole/`: standalone equirectangular-to-pinhole rendering tools.
- `src/colmap/panorama/`: maintained native panorama COLMAP pipeline.
- `src/colmap/panorama_reconstruct.py`: compatibility facade and command entry
  target.
- `src/tools/`: utility commands such as video processing, sampling, and
  vocabulary-tree download/build helpers.

Main console commands:

- `colmap-panorama-reconstruct`
- `colmap-panorama-import`
- `colmap-panorama-features`
- `colmap-panorama-clusters`
- `colmap-visualize`
- `tools-download-colmap-vocab-tree`

## Workspace Layout

The panorama workspace paths are centralized in `panorama_workspace_paths`.

For a workspace such as `/path/to/workspace`, the pipeline uses:

```text
workspace/
  images/
  database.db
  feature_shards/
  sparse/
  sparse_best/
  cloud_colmap_raw.ply
  panorama_equirectangular_config.json
  panorama_equirectangular_import_report.json
  panorama_equirectangular_feature_report.json
  panorama_keyframe_cluster_report.json
  panorama_equirectangular_report.json
  match_pairs.txt
```

Not every command writes every file. Import-only, feature-only, cluster-only,
and full reconstruction each write the subset relevant to that stage.

## Full Reconstruction Flow

The full path is owned by `src/colmap/panorama/pipeline.py` and exposed through
`colmap-panorama-reconstruct`.

High-level flow:

1. Resolve pycolmap and execution options.
2. Prepare workspace directories.
3. Import panorama images into `workspace/images/`.
4. Extract features into `database.db`.
5. Read database image ids and feature statistics.
6. Build matching options.
7. Optionally build a keyframe pair plan and write `match_pairs.txt`.
8. Match features.
9. Run incremental mapping.
10. Select the best sparse reconstruction.
11. Write `sparse_best/` and `cloud_colmap_raw.ply`.
12. Score registered keyframe anchors.
13. Write `panorama_equirectangular_report.json`.

The command refuses to overwrite an existing `database.db` unless the workspace
is rebuilt with `--overwrite`.

## Image Import

Image import is split between:

- `image_import.py`: file discovery, sorting, resolution checks, and link
  creation;
- `import_pipeline.py`: workspace-level import orchestration and metadata
  writing.

Behavior:

1. Recursively discover supported image files under `--input-image-path`.
2. Sort by timestamp when a timestamp is present in the filename; otherwise use
   filename ordering.
3. Apply `--image-limit` when present.
4. Read the first image resolution.
5. Require all selected panoramas to share that resolution.
6. Create one native `EQUIRECTANGULAR` camera from width and height.
7. Import files into `workspace/images/` using a hard link when possible and an
   absolute symbolic link fallback across filesystems.
8. Write `panorama_equirectangular_config.json`.

Import can run serially or with `--import-workers`.

The import-only command is `colmap-panorama-import`. It performs discovery,
resolution validation, linking, metadata writing, and an import report without
requiring pycolmap feature extraction or mapping.

## Camera Model

The current reconstruction camera model is native COLMAP `EQUIRECTANGULAR`.

For each selected dataset:

```text
camera model: EQUIRECTANGULAR
camera params: width,height
camera mode: SINGLE
```

The main pipeline assumes all imported panoramas have the same resolution.

Legacy virtual pinhole rig fields and CLI options have been removed from the
main reconstruction path.

## Feature Extraction

Feature extraction is owned by `feature_pipeline.py` and pycolmap option
construction lives in `pycolmap_adapter.py`.

Current feature backend:

```text
feature extractor: sift
feature matcher: sift-bruteforce
```

Feature extraction options include:

- `--max-num-features`
- `--max-image-size`
- `--num-threads`
- `--device {auto,cpu,cuda}`
- `--gpu-index`

When CUDA is available and `--device` is not `cpu`, feature extraction and
matching use GPU execution. Bundle adjustment can also request GPU through
`--gpu-bundle-adjustment`, using the first selected GPU index.

The feature-only command is `colmap-panorama-features`. It imports panoramas,
extracts features, reads feature health statistics, and writes
`panorama_equirectangular_feature_report.json` without matching or mapping.

## Sharded Feature Extraction

Feature extraction has an opt-in sharded mode controlled by `--feature-workers`.

Default behavior:

```text
feature_workers = 1
feature_extraction_mode = single_database
```

When `feature_workers > 1`:

1. imported image names are split into deterministic batches;
2. batch size is computed automatically:

```text
feature_batch_size = ceil(image_count / feature_workers)
```

3. each batch extracts features into its own shard database under
   `feature_shards/`;
4. shard databases are merged into the final `database.db`;
5. matching and mapping run only after merge completes.

The report records:

- feature worker count;
- computed feature batch size;
- shard batch contents;
- shard database paths;
- shard elapsed times;
- merge elapsed time;
- camera policy.

The accepted camera policy in sharded mode is:

```text
multiple_identical_equirectangular_cameras_are_expected
```

This exists because pycolmap database merging can preserve separate but
identical equirectangular camera rows from each shard.

## Matching Modes

The full reconstruction CLI supports:

```text
--matcher sequential
--matcher exhaustive
--matcher vocabtree
--matcher keyframes
```

### Sequential

Sequential matching uses COLMAP sequential pairing with configurable
`--overlap`. Rig expansion is disabled. Loop detection can be controlled with
`--loop-detection` or `--no-loop-detection`.

### Exhaustive

Exhaustive matching sends all database images through COLMAP exhaustive
matching. It is useful for small datasets but expensive for large captures.

### Vocabtree

Vocabtree matching uses COLMAP vocabulary-tree pairing. A local tree can be
provided with `--vocab-tree-path`; otherwise COLMAP may use/download its own
default tree depending on pycolmap behavior.

### Keyframes

Keyframe mode builds an imported image-pair list before matching and then calls
COLMAP's image-pair matcher.

The supported keyframe selection modes are:

- `interval`
- `sequential-vocabtree`

The keyframe code lives in:

- `keyframe_matching.py`: pure pair-planning logic;
- `cluster_pipeline.py`: database-aware orchestration;
- `cli_clusters.py`: standalone cluster command.

## Keyframe Pair Planning

`KeyframeSelectionOptions` currently includes:

- `selection`
- `interval`
- `overlap`
- `backbone_policy`
- `backbone_overlap`
- `cluster_max_size`
- `cluster_recent_window`
- `vocabtree_neighbors`
- `vocabtree_checks`

The previously exposed minimum cluster-size option has been removed because it
was not used by the clustering algorithm.

### Interval Keyframes

Interval mode:

1. selects every `--keyframe-interval` image;
2. always includes first and last images;
3. assigns each image to its nearest keyframe;
4. adds image-to-keyframe pairs;
5. connects neighboring keyframes according to `--keyframe-overlap`;
6. adds sequential fallback pairs according to `--keyframe-backbone-policy`.

This mode does not require a vocabulary tree.

### Sequential Vocabtree Keyframes

Sequential-vocabtree mode:

1. requires extracted features and a compatible vocabulary tree;
2. asks `pycolmap.VocabTreePairGenerator` for visual-neighbor candidate pairs;
3. scans panorama images in capture order;
4. keeps a current cluster as long as new images connect to the temporary
   anchor or recent images;
5. closes a cluster when connectivity breaks or `cluster_max_size` is reached;
6. selects a final keyframe per cluster.

Keyframe selection score prefers:

1. higher vocabtree degree within the cluster;
2. higher feature count;
3. image closer to the cluster middle;
4. earlier image order as deterministic tie-break.

The final pair plan includes:

- image-to-keyframe pairs;
- keyframe-to-neighbor-keyframe pairs;
- vocabtree candidate pairs;
- sequential fallback pairs according to policy;
- a final deduplicated unordered pair list.

This means sequential-vocabtree mode is best understood as
vocabtree/keyframe-guided pair planning, not a purely keyframe-only matcher.

## Sequential Fallback Policy

Keyframe mode supports:

```text
--keyframe-backbone-policy {none,gaps,all}
--keyframe-backbone-overlap N
```

Policy meanings:

- `none`: add no local sequential fallback pairs.
- `gaps`: add only local sequential pairs missing from the keyframe/vocabtree
  graph.
- `all`: add all local sequential pairs up to the requested overlap.

The default is `gaps` with overlap `1`.

## Vocabulary Tree Handling

The repository includes `tools-download-colmap-vocab-tree`.

The current default tree format is FAISS:

```bash
tools-download-colmap-vocab-tree \
  --format faiss \
  --variant 256k \
  --destination-dir /workspace/data/colmap-vocab-tree
```

The expected FAISS-prefixed file for the common `256k` variant is:

```text
vocab_tree_faiss_flickr100K_words256K.bin
```

Legacy FLANN-format files can fail with newer pycolmap/COLMAP builds that
expect FAISS vocabulary indexes. The pycolmap adapter catches the known
`Failed to read faiss index` failure and rewrites it into a clearer error that
recommends a FAISS tree, a COLMAP vocab-tree upgrade, or interval keyframes as
a temporary fallback.

## Reports

The implementation writes structured JSON reports to make long COLMAP runs
auditable.

Import report:

- input path;
- workspace paths;
- camera model and params;
- image count;
- imported image names;
- import worker count;
- skipped later stages.

Feature report:

- pycolmap version;
- database path;
- feature options;
- device resolution;
- feature worker count;
- feature batch size if sharded;
- feature health;
- database image/camera ids;
- shard details when used.

Cluster report:

- selected and effective keyframe strategy;
- keyframes;
- clusters;
- pair counts by source;
- match pair path.

Full reconstruction report:

- status and raw-unscaled warning;
- camera setup;
- pycolmap version;
- workspace paths;
- matcher and effective pairing;
- feature and execution options;
- database counts;
- feature health;
- feature shard report when used;
- reconstruction stats for all models;
- best model id and stats;
- registered image ids/names;
- keyframe scores;
- elapsed time by stage.

## Current Public CLI Highlights

Full reconstruction:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-equirectangular \
  --matcher sequential \
  --overlap 15 \
  --overwrite
```

Keyframe reconstruction:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-equirectangular-keyframes \
  --matcher keyframes \
  --keyframe-selection sequential-vocabtree \
  --vocab-tree-path /workspace/data/colmap-vocab-tree/vocab_tree_faiss_flickr100K_words256K.bin \
  --keyframe-backbone-policy gaps \
  --overwrite
```

Cluster-only planning:

```bash
colmap-panorama-clusters \
  --workspace /workspace/data/colmap-equirectangular-keyframes \
  --vocab-tree-path /workspace/data/colmap-vocab-tree/vocab_tree_faiss_flickr100K_words256K.bin \
  --keyframe-selection sequential-vocabtree
```

Feature-only with sharded extraction:

```bash
colmap-panorama-features \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-equirectangular-features \
  --feature-workers 4 \
  --overwrite
```

## Important Limitations

The sparse reconstruction is monocular and unscaled. It should not be treated as
metric geometry until aligned with an external scale source.

The native pipeline assumes all selected panoramas share one resolution.

Sharded feature extraction is opt-in. It avoids SQLite contention by writing
separate shard databases, but it should still be benchmarked on representative
datasets before being treated as a default speedup.

Sequential-vocabtree keyframe planning requires a compatible vocabulary tree.
For current FAISS-based pycolmap builds, use FAISS-prefixed tree assets.

Keyframe mode reduces and structures matching through pair planning, but
sequential-vocabtree mode currently includes vocabtree candidate pairs in the
final pair list. It is not an aggressively keyframe-only matcher.

The public compatibility facade still exports many internal helpers for older
tests and scripts. The focused modules under `src/colmap/panorama/` should be
used as the ownership map for future implementation work.
