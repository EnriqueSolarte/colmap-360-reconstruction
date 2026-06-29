Date: 2026-06-29
Project version: v0.1.0
Description: Plans staged standalone and parallel COLMAP panorama feature extraction.

# Parallel Feature Extraction Plan

## Goal

Reduce the excessive runtime spent computing features in the native
equirectangular COLMAP panorama pipeline while keeping reconstruction behavior
reliable and easy to debug.

The target pipeline is `src/colmap/panorama_reconstruct.py`, which currently
imports panoramas, extracts features, matches images, runs incremental mapping,
and writes a reconstruction report in one command.

## Current State

`colmap-panorama-reconstruct` already:

1. Imports original equirectangular panoramas into `workspace/images/` by
   hardlinking or symlinking.
2. Uses COLMAP's native `EQUIRECTANGULAR` camera model.
3. Calls `pycolmap.extract_features(...)` once for all imported images.
4. Uses GPU extraction when CUDA pycolmap is available and `--device` is not
   `cpu`.
5. Exposes `--gpu-index`, including comma-separated values such as `0,1` and
   `-1`.
6. Records `feature_extraction` elapsed time in
   `panorama_equirectangular_report.json`.

Current gaps:

1. There is no standalone feature-only command.
2. Feature extraction tuning options such as `num_threads` and `max_image_size`
   are not exposed.
3. There is no safe batch-worker orchestration.
4. There is no validated database-sharding and merge path for parallel
   extraction.

## Important Constraints

Feature extraction writes cameras, images, keypoints, and descriptors into a
COLMAP SQLite database. Running multiple extractors against the same database is
likely to cause SQLite locking or nondeterministic failure.

The safer design is:

1. Split images into disjoint batches.
2. Let each worker extract features into its own shard database.
3. Merge shard databases into one final COLMAP database.
4. Run matching and mapping only after the merged database is complete.

`pycolmap.Database.merge(...)` appears to provide a supported path for merging
databases, but this must be tested before it becomes part of the main
reconstruction path.

## Stage 1: Standalone Feature-Only CLI

Add a new command, tentatively:

```text
colmap-panorama-features
```

Behavior:

1. Import native equirectangular panoramas into a feature workspace.
2. Write `panorama_equirectangular_config.json`.
3. Extract SIFT features into `database.db`.
4. Write a feature-only report with image counts, database counts, feature
   health, compute device, GPU index, and elapsed stage timings.
5. Do not run matching.
6. Do not run incremental mapping.

Recommended implementation shape:

1. Put reusable feature-extraction helpers in a new module such as
   `src/colmap/panorama_features.py`.
2. Keep `panorama_reconstruct.py` unchanged except for shared imports if needed.
3. Add a console script in `pyproject.toml`.

CLI options:

```text
--input-image-path
--workspace
--ssl-cert-file
--feature-extractor {sift}
--max-num-features
--max-image-size
--num-threads
--device {auto,cpu,cuda}
--gpu-index
--image-limit
--overwrite
```

Success criteria:

1. The command can produce a valid COLMAP database with cameras, images,
   keypoints, and descriptors.
2. The report clearly shows feature extraction runtime.
3. Existing `colmap-panorama-reconstruct` behavior is not changed.
4. Tests cover argument parsing, option propagation, report shape, and failure
   on an existing database without `--overwrite`.

## Stage 2: Share Feature Extraction With Reconstruction

Refactor `panorama_reconstruct.py` to call the reusable feature-extraction
helper from Stage 1.

Success criteria:

1. Reconstruction output remains behaviorally unchanged.
2. Existing reconstruction tests still pass.
3. Feature extraction options are represented consistently across both commands.
4. The reconstruction report records any newly exposed extraction options.

This stage should not add parallel workers yet.

## Stage 3: Parallel Batch Workers

Add an experimental parallel feature extraction mode to the feature-only CLI.

Tentative CLI options:

```text
--feature-workers N
--feature-batch-size N
--worker-gpu-indices 0,1
--keep-feature-shards
```

Behavior:

1. Partition imported image names into deterministic batches.
2. Create one shard workspace or shard database per worker/batch.
3. Run `pycolmap.extract_features(...)` separately per shard.
4. Merge successful shard databases into the final `database.db`.
5. Write shard reports and a top-level merge report.

GPU policy:

1. If one GPU is selected, either run one GPU worker or allow an explicit
   oversubscription mode after testing.
2. If multiple GPU indices are selected, assign workers round-robin.
3. Record per-worker GPU assignment in the report.

Success criteria:

1. A small synthetic image set produces the same image count and nonzero feature
   rows as single-process extraction.
2. The merged database can be consumed by COLMAP matching.
3. Failed workers leave enough logs/report data to identify the failed batch.
4. Parallel mode is opt-in and does not affect default reconstruction behavior.

## Stage 4: Reconstruction Integration

After the feature-only CLI and parallel merge path are validated, add an opt-in
path to `colmap-panorama-reconstruct`.

Possible options:

```text
--feature-workers N
--feature-batch-size N
--reuse-feature-database PATH
```

Behavior:

1. Default reconstruction remains single-call extraction.
2. When `--feature-workers` is greater than one, reconstruction uses the same
   shard-and-merge feature extraction path as the standalone command.
3. When `--reuse-feature-database` is provided, reconstruction skips extraction
   and validates that the database image names match the imported panoramas.

Success criteria:

1. Reconstruction can start from a database produced by
   `colmap-panorama-features`.
2. Report output distinguishes fresh extraction, parallel extraction, and reused
   feature databases.
3. Matching and mapping behavior remain unchanged after feature extraction.

## Evaluation

Measure before and after each performance stage:

1. `feature_extraction` elapsed seconds.
2. Total command elapsed seconds.
3. Database image count.
4. Keypoint and descriptor row counts.
5. Mean features per image.
6. Matching success.
7. Registered image count.
8. Sparse point count.
9. Mean reprojection error.

Representative runs should include:

1. A small smoke-test subset.
2. A medium local sequence that currently feels slow.
3. A large sequence where GPU memory remains available during extraction.

## Risks

1. Multiple workers on one GPU may reduce throughput if COLMAP already saturates
   the device internally.
2. SQLite database merging may not preserve every schema detail needed by the
   current pycolmap version.
3. Shard databases may create duplicate camera ids or image ids unless the merge
   path handles remapping correctly.
4. More workers can increase CPU RAM, VRAM, disk IO, and temporary disk usage.
5. Parallel extraction can make failure diagnosis harder unless shard reports
   are written carefully.

## Recommended Next Slice

Implement Stage 1 only:

1. Add the standalone feature-only command.
2. Expose `num_threads` and `max_image_size`.
3. Write a focused feature extraction report.
4. Add tests for the new command and options.

Do not implement database sharding or parallel workers until the standalone
feature-only path is working and validated.
