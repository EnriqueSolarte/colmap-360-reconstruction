Date: 2026-06-29
Project version: v0.1.0
Description: Captures the preferred staged path for faster panorama import and feature extraction.

# Parallel Feature Extraction Plan

## Goal

Reduce the excessive runtime spent importing panoramas and computing features in
the native equirectangular COLMAP panorama pipeline while keeping reconstruction
behavior reliable and easy to debug.

The target pipeline is `src/colmap/panorama/pipeline.py`, called through the
compatibility facade in `src/colmap/panorama_reconstruct.py`. It currently
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

1. Feature extraction tuning options such as `num_threads` and `max_image_size`
   are not exposed.
2. Image import is serial even though resolution checks and hardlink/symlink
   creation can be parallelized safely with ordered result collection.
3. There is no isolated import-only path for validating image discovery,
   resolution checks, link creation, and metadata without running COLMAP.
4. There is no reusable feature-only or batch-extraction workflow for testing
   extraction independently from matching and mapping.
5. There is no safe batch-worker orchestration.
6. There is no validated database-sharding and merge path for parallel
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

The local pycolmap stubs also show that `pycolmap.extract_features(...)` accepts
an `image_names` sequence. That means each shard can point at the same
`workspace/images/` directory and process only its assigned image names, instead
of creating temporary image directories.

## Stage 1: Expose Low-Risk Feature Extraction Options

Expose COLMAP's existing feature extraction tuning knobs in the current
reconstruction command before introducing worker orchestration.

CLI options:

```text
--max-image-size
--num-threads
```

Behavior:

1. Add `max_image_size` and `num_threads` to `FeaturePipelineOptions`.
2. Pass them through `create_feature_extraction_options(...)`.
3. Record them in `panorama_equirectangular_report.json`.
4. Keep defaults equivalent to pycolmap defaults:
   - `max_image_size = -1`
   - `num_threads = -1`

Success criteria:

1. Existing reconstruction behavior remains unchanged when the options are not
   provided.
2. CLI validation rejects invalid nonzero negative values where appropriate,
   while preserving `-1` as COLMAP's automatic setting.
3. Tests cover option parsing, option propagation, and report fields.

## Stage 2: Parallel Panorama Import

Improve `import_equirectangular_panorama_images(...)` so image resolution checks
and hardlink/symlink creation can run with multiple workers.

Possible option:

```text
--import-workers N
```

Behavior:

1. Discover and sort panorama paths exactly as today.
2. Read the first image resolution before starting workers so camera creation
   remains deterministic.
3. Use workers to validate each image resolution, create the output parent
   directory, and hardlink or symlink the source image.
4. Collect results in the original sorted order so `panorama_paths`,
   `generated_image_names`, metadata, and reports remain deterministic.
5. Default to one worker to preserve current behavior.

Implementation notes:

1. A `ThreadPoolExecutor` is sufficient because most import work is image
   header IO and filesystem calls.
2. Do not copy source images. Preserve the current hardlink-then-absolute-symlink
   behavior.
3. If any worker sees a mixed resolution, fail the import and surface the same
   kind of error as the serial path.

Success criteria:

1. Existing image import tests pass with the default one-worker path.
2. New tests cover multi-worker import, deterministic ordering, mixed-size
   rejection, and existing-output failure behavior.
3. Metadata remains byte-shape compatible except for any explicitly added import
   worker reporting.

## Stage 2.1: Isolated Image Import

Add an import-only execution path that stops after panorama discovery,
resolution validation, hardlink/symlink import, metadata writing, and a small
import report. It must not run feature extraction, feature matching, or
incremental mapping.

Purpose:

1. Validate input image discovery and timestamp ordering without paying COLMAP
   feature extraction cost.
2. Validate mixed-resolution rejection and workspace image layout before longer
   reconstruction runs.
3. Give `--import-workers` a direct smoke-test path on real datasets.
4. Create a clean foundation for later feature-only and sharded extraction
   helpers to reuse imported images.

Possible shape:

1. Add a small import-only helper that shares the same import implementation as
   reconstruction.
2. Expose it as either:
   - a dedicated command such as `colmap-panorama-import`, or
   - an explicit reconstruction flag such as `--import-only`.
3. Write `panorama_equirectangular_config.json`.
4. Write an import report containing image count, camera model/params,
   `image_limit`, `import_workers`, elapsed import time, and generated image
   names.
5. Exit before `pycolmap.extract_features(...)`.

Preferred default:

1. Use a dedicated command if this starts to grow its own reporting or user
   workflow.
2. Use `--import-only` only if the implementation can stay very small and does
   not make the reconstruction pipeline branchy.

Success criteria:

1. Import-only execution creates `workspace/images/` and
   `panorama_equirectangular_config.json`.
2. No COLMAP database is required and no feature extraction, matching, sparse
   mapping, or best-model writing is attempted.
3. Tests prove pycolmap feature extraction is not called in this mode.
4. The import report records enough information to compare serial and
   multi-worker import behavior.
5. The normal reconstruction path remains unchanged when import-only mode is not
   requested.

## Stage 3: Feature-Only Batch Extraction Helper And Merge Validation

Add a reusable feature-only/batch extraction helper that can produce shard
databases and merge them behind tests. This stage proves the risky database
behavior before adding any reconstruction-facing worker option.

Preferred implementation shape:

1. Keep the main reconstruction path unchanged while the helper is validated.
2. Import all panoramas once into the normal `workspace/images/` directory.
3. Split `generated_image_names` into deterministic batches.
4. Run each batch with:

```text
pycolmap.extract_features(
    shard_database,
    image_dir,
    image_names=batch_names,
    ...
)
```

5. Merge shard databases into the final `database.db`.
6. Run matching and mapping unchanged only after the merged database is valid.

Validation target:

1. Import a small deterministic image set.
2. Split image names into two or more batches.
3. Run `pycolmap.extract_features(...)` for each batch into a separate shard
   database, using `image_names=batch_names`.
4. Merge shard databases into a final database with `pycolmap.Database.merge`.
5. Confirm the merged database can be consumed by COLMAP matching.

Open questions to answer before Stage 4:

1. Does `Database.merge` deduplicate identical cameras, or does each shard keep
   its own camera id?
2. If multiple camera ids remain, does this affect matching, mapping, reporting,
   or the expectation that the native panorama pipeline uses one shared camera?
3. Are image ids stable enough for reporting, or should reports treat merged
   image ids as final database-assigned ids only?
4. Does merging preserve every table required by the current pycolmap version?

Success criteria:

1. A small synthetic image set produces the expected image count and nonzero
   feature rows after merge.
2. The merged database can run at least one matching strategy.
3. Camera id behavior is documented in the test or a short note before the main
   reconstruction path depends on it.
4. Failed shard or merge attempts leave enough evidence to diagnose the failure.

Optional follow-up:

1. A standalone feature-only CLI can still be added later if repeated manual
   benchmarking needs it.
2. It should reuse this helper rather than drive the design.

## Stage 4: Reconstruction Integration For Parallel Feature Workers

Only after the merge validation passes, expose opt-in parallel feature workers
in `colmap-panorama-reconstruct`.

Possible options:

```text
--feature-workers N
--feature-batch-size N
--worker-gpu-indices 0,1
--keep-feature-shards
--reuse-feature-database PATH
```

Behavior:

1. Default reconstruction remains single-call extraction.
2. When `--feature-workers` is greater than one, reconstruction uses the
   validated shard-and-merge helper.
3. Each shard extraction passes
   `image_names=batch_names` and the shared `workspace/images/` directory.
4. Matching and mapping remain unchanged after feature extraction.
5. When `--reuse-feature-database` is provided, reconstruction skips extraction
   and validates that the database image names match the imported panoramas.
6. Report output distinguishes fresh extraction, parallel extraction, and reused
   feature databases.

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
10. Import elapsed seconds.
11. Shard extraction elapsed seconds by worker/batch.
12. Merge elapsed seconds.
13. Import-only elapsed seconds and imported image count.

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
6. Parallel import may offer only modest speedups if hardlink/symlink creation
   is not the runtime bottleneck.
7. A merged database with multiple identical cameras may be acceptable for
   reconstruction but could weaken report assumptions if not handled explicitly.

## Recommended Next Slice

Implement Stage 1 only, then Stage 2, then Stage 2.1 if each slice is clean:

1. Expose `num_threads` and `max_image_size` through
   `FeaturePipelineOptions`, the CLI, pycolmap option construction, and the
   reconstruction report.
2. Add tests for option parsing, option propagation, and report output.
3. After that is verified, add opt-in multi-worker import with deterministic
   result ordering and focused import tests.
4. After multi-worker import is verified, add an isolated import-only path that
   writes metadata/report output and explicitly skips feature extraction,
   matching, and mapping.

Do not expose `--feature-workers` in reconstruction until the Stage 3 helper
proves camera/image identity and matching behavior after database merge.
