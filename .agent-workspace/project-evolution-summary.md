Date: 2026-06-30
Project version: v0.1.0
Description: Summarizes how the COLMAP panorama reconstruction project evolved into the current native equirectangular pipeline

# Project Evolution Summary

## Purpose

This project evolved from a collection of pinhole rendering and imported COLMAP
scripts into a repo-native pipeline for raw sparse reconstruction from
equirectangular panoramas.

The current project goal is to reconstruct monocular, unscaled COLMAP sparse
models directly from 360-degree images while keeping the pipeline inspectable,
testable, and practical for large ordered panorama captures.

Evidence used for this summary:

- `.archived/plan-for-colmap.md`
- `.archived/re-structuring-colmap-for-pano-images.md`
- `.archived/native-equirectangular-pipeline-summary.md`
- `.archived/panorama-module-restructure-plan.md`
- `.archived/parallel-feature-extraction-plan.md`
- `.archived/plan-for-key-frame-based-solution.md`
- `.archived/vocab-tree-faiss-compatibility-report.md`
- Current code under `src/colmap/panorama/`
- Current `README.md`

## Phase 1: Pinhole Reconstruction From Rendered Views

The earliest direction was to reconstruct from pinhole images rendered out of
equirectangular panoramas.

The project already had rendering utilities that could convert an
equirectangular image into perspective views. The first COLMAP plan treated
those generated pinhole images as the reconstruction input and focused on
making imported COLMAP scripts work inside this repository.

Important assumptions at that stage:

- Generated pinhole images used shared renderer defaults:
  - width `1280`
  - height `960`
  - horizontal FOV about `70` degrees
  - vertical FOV about `55.4` degrees
- All generated pinhole views could share one `PINHOLE` camera model.
- The first output should be labeled raw and unscaled, not metric geometry.

That phase identified useful pieces from copied COLMAP scripts:

- image discovery;
- workspace setup;
- feature extraction;
- sequential matching;
- incremental mapping;
- best model selection;
- PLY export;
- JSON reporting.

It also identified inherited problems:

- dependencies on missing `starter.*` modules;
- unclear input metadata;
- no reliable link from rendered pinhole images back to source panoramas;
- no scale source for metric reconstruction.

The practical next step at that time was an input-inspection and intrinsics
reporting path before running expensive COLMAP jobs.

## Phase 2: Virtual Pinhole Rig Direction

The next direction was to make panorama reconstruction more geometrically
correct by treating each panorama as a rigid rig of virtual pinhole cameras.

This was motivated by COLMAP's recommended handling of spherical imagery:
render several perspective views from a panorama, assign known rotations, and
force those views to share one panorama camera center instead of letting COLMAP
treat every crop as an independent physical camera.

The intended rig behavior was:

- render fixed view directions per panorama;
- use known rigid rotations between virtual cameras;
- preserve one shared camera center per panorama frame;
- apply feature ownership masks so crops do not claim features outside their
  valid field;
- use matching behavior that understands panorama frames;
- keep loop detection and temporal matching available.

This was a sound geometric plan, but it made the maintained pipeline more
complex. It required virtual rendering, rig metadata, frame-scoped matching,
masks, and compatibility layers.

## Phase 3: Pivot To Native Equirectangular COLMAP

The major simplification came when the pipeline pivoted to COLMAP's native
`EQUIRECTANGULAR` camera model.

Instead of rendering virtual pinhole images, the maintained pipeline now imports
the original panorama images directly and asks pycolmap/COLMAP to treat them as
native equirectangular cameras.

This removed a large amount of legacy complexity:

- virtual pinhole rendering in the reconstruction path;
- virtual camera rig options;
- panorama rig view models;
- mask ownership helpers;
- frame-scoped legacy matching helpers;
- initial-session continuation helpers;
- report fields tied to virtual camera counts.

The output names also changed from rig-oriented names to native equirectangular
names:

- `panorama_equirectangular_config.json`
- `panorama_equirectangular_report.json`

The native pipeline still keeps the same big warning: COLMAP output is
monocular and unscaled unless aligned with an external scale source.

## Phase 4: Module Restructure

After the native pipeline was working, `panorama_reconstruct.py` was split into
a focused package under `src/colmap/panorama/`.

The reason was maintainability. The old module mixed too many responsibilities:

- CLI parsing;
- workspace path construction;
- image discovery and import;
- SQLite database reads;
- pycolmap option construction;
- COLMAP execution calls;
- report writing;
- reconstruction scoring;
- pipeline orchestration.

The current package separates those concerns into modules such as:

- `image_import.py`
- `import_pipeline.py`
- `feature_pipeline.py`
- `database.py`
- `pycolmap_adapter.py`
- `keyframe_matching.py`
- `cluster_pipeline.py`
- `reports.py`
- `pipeline.py`
- `cli.py`
- `cli_import.py`
- `cli_features.py`
- `cli_clusters.py`

`src/colmap/panorama_reconstruct.py` remains as a compatibility facade so older
imports and the public `colmap-panorama-reconstruct` entry point can keep
working.

## Phase 5: Performance And Observability

Once the native equirectangular baseline existed, the project shifted toward
runtime and debugging ergonomics.

The performance plan separated low-risk improvements from risky ones:

1. Expose existing COLMAP feature-extraction knobs:
   - `--max-image-size`
   - `--num-threads`
2. Add parallel panorama import with `--import-workers`.
3. Add isolated import-only and feature-only commands.
4. Validate sharded feature extraction before using it in reconstruction.
5. Integrate opt-in sharded feature extraction into full reconstruction.

The important database constraint was that multiple feature extractors should
not write to the same COLMAP SQLite database. The chosen design became:

1. split imported image names into batches;
2. extract each batch into its own shard database;
3. merge shard databases into the final `database.db`;
4. run matching and mapping only after the merge.

Current behavior computes shard batch size automatically:

```text
feature_batch_size = ceil(image_count / feature_workers)
```

The public control is therefore worker count, not manual batch size.

## Phase 6: Keyframe And Vocabtree Pair Planning

For larger ordered panorama captures, matching all image pairs is too expensive
and sequential matching alone can miss visual loops or reconnects. The project
therefore added keyframe-based pair planning on top of the native
equirectangular import and feature stages.

The key idea is that every panorama is still imported and feature-extracted, but
the matching stage can use a planned pair list.

Implemented keyframe modes:

- `interval`: deterministic keyframes at a fixed interval, always including the
  first and last image.
- `sequential-vocabtree`: scan images in capture order, use vocabulary-tree
  candidate connectivity to form clusters, then select one keyframe per
  cluster.

The keyframe pair plan records provenance:

- image-to-keyframe pairs;
- vocabtree candidate pairs;
- sequential fallback pairs;
- final deduplicated pair list;
- selected keyframes;
- clusters.

The standalone `colmap-panorama-clusters` command exists because pair planning
needs to be inspectable without rerunning import, feature extraction, matching,
and mapping.

## Phase 7: FAISS Vocabulary Tree Compatibility

A real sequential-vocabtree run exposed a compatibility issue: current
`pycolmap 4.1.0` expects FAISS-format vocabulary trees, while the originally
downloaded prebuilt tree was legacy FLANN format.

The failure was not in the keyframe clustering algorithm. It happened inside
COLMAP when `pycolmap.VocabTreePairGenerator` tried to read the incompatible
tree.

The repository response was:

- catch the specific `"Failed to read faiss index"` error and raise a clearer
  message;
- update the vocabulary-tree downloader to default to FAISS-prefixed assets;
- keep `--format legacy` for older COLMAP builds;
- update the README examples to use `vocab_tree_faiss_...` files.

The current recommended helper command is:

```bash
tools-download-colmap-vocab-tree \
  --format faiss \
  --variant 256k \
  --destination-dir /workspace/data/colmap-vocab-tree
```

## Current State

The current solution is a native equirectangular COLMAP pipeline with:

- direct panorama import by hardlink or symlink;
- shared native `EQUIRECTANGULAR` camera parameters from panorama resolution;
- SIFT feature extraction through pycolmap;
- sequential, exhaustive, vocabtree, and keyframe matchers;
- optional parallel import;
- optional sharded feature extraction;
- import-only, feature-only, cluster-only, and full reconstruction commands;
- FAISS-aware vocabulary-tree download support;
- structured JSON reports for import, feature extraction, clustering, and full
  reconstruction;
- package-level separation of pipeline, CLI, reports, pycolmap boundary,
  database reads, image import, scoring, and keyframe planning.

The remaining strategic caution is that performance-oriented paths such as
sharded feature extraction and sequential-vocabtree keyframe matching should be
benchmarked on representative real datasets before being treated as universally
better defaults.
