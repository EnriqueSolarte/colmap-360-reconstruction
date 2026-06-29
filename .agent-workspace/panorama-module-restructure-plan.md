Date: 2026-06-29
Project version: v0.1.0
Description: Staged plan to split panorama_reconstruct.py into focused modules without changing behavior.

# Panorama Module Restructure Plan

## Objective

Split `src/colmap/panorama_reconstruct.py` into a small `src/colmap/panorama/`
package with clear responsibilities, while preserving the current CLI behavior,
test imports, and native equirectangular reconstruction behavior.

This is a structure-first refactor. It should not change reconstruction
semantics, COLMAP options, report schema, scoring math, matching behavior, or
workspace layout unless a later stage explicitly requests that behavior change.

## Current Problem

`panorama_reconstruct.py` currently mixes too many reasons to change:

- CLI argument parsing and console output.
- Pipeline orchestration.
- Workspace path construction.
- Panorama image discovery, sorting, validation, and linking.
- SQLite reads from the COLMAP database.
- pycolmap option construction and COLMAP execution calls.
- Reconstruction scoring.
- Metadata and report construction/writing.

The file is difficult to scan because these concerns are interleaved. The risk
is not just aesthetics: future work on the rig-based panorama pipeline could
accidentally modify unrelated reporting, database, or CLI behavior.

## Target Structure

```text
src/colmap/panorama/
  __init__.py
  constants.py
  types.py
  paths.py
  image_import.py
  database.py
  pycolmap_adapter.py
  scoring.py
  reports.py
  pipeline.py
  cli.py
```

`src/colmap/panorama_reconstruct.py` should remain as a compatibility facade
at first. Tests and external callers currently import internals from this module,
so deleting it immediately would create noisy breakage unrelated to the actual
refactor.

## Proposed Ownership

`constants.py`

- `IMAGE_EXTENSIONS`
- `COLMAP_IMAGE_PAIR_ID_PRIME`
- `FILENAME_TIMESTAMP_PATTERN`
- `NATIVE_CAMERA_SETUP`

`types.py`

- `FeaturePipelineOptions`
- `PipelineExecutionOptions`
- `GeneratedImageBatch`
- `KeyframeAnchorScore`

`paths.py`

- `panorama_workspace_paths`

`image_import.py`

- `hardlink_or_symlink_file`
- `panorama_image_paths`
- `panorama_timestamp_sort_key`
- `timestamp_from_image_name`
- `image_resolution`
- `prefixed_panorama_relative_name`
- `import_equirectangular_panorama_images`

`database.py`

- `decode_colmap_image_pair_id`
- `read_database_stats`
- `read_image_records_by_name`
- `read_generated_image_batch`
- `read_feature_stats_by_camera`
- `read_feature_stats_by_database_camera`
- frame connection database reads used by scoring, unless scoring ownership is
  clearer after extraction

`pycolmap_adapter.py`

- `require_pycolmap`
- `configure_ssl_cert_file`
- `create_feature_extraction_options`
- `create_feature_matching_options`
- `create_mapper_options`
- `validate_vocab_tree_path`
- `resolve_pipeline_execution`
- `validate_gpu_index`
- `parse_gpu_index`
- `first_gpu_index`
- `match_features`
- `run_incremental_mapping`
- `write_best_model`

`scoring.py`

- `round_report_float`
- `optional_report_float`
- `log_capped_score`
- `linear_capped_score`
- `normalized_entropy`
- `colmap_frame_id_for_image`
- `score_keyframe_anchors`

`reports.py`

- `reconstruction_stats`
- `write_image_pair_list`
- `write_equirectangular_metadata`
- `build_panorama_report`
- `equirectangular_feature_health_summary`
- elapsed formatting helpers if they are only report-facing

`pipeline.py`

- `run_panorama_reconstruction`

`cli.py`

- `PanoramaArgumentDefaultsHelpFormatter`
- `parse_args`
- `main`

## Compatibility Policy

For the first refactor pass, keep `colmap.panorama_reconstruct` import-compatible.
It should re-export the functions and dataclasses currently used by tests and
scripts.

Only after the new modules are stable should tests be gradually updated to import
from the focused modules directly.

## Staged Implementation

### Stage 1: Create Package Shell And Move Low-Risk Pieces

Create `src/colmap/panorama/` with `__init__.py`, `constants.py`, `types.py`,
`paths.py`, and a compatibility facade in `panorama_reconstruct.py`.

Move only constants, dataclasses, path construction, and pure formatting helpers.

Verification:

- `pytest tests/test_colmap_panorama_reconstruct.py`
- Confirm `colmap-panorama-reconstruct` entry point still resolves.

Why first:

- These pieces have low runtime coupling.
- This validates the facade approach before moving pycolmap or SQLite code.

### Stage 2: Move Image Import Code

Move panorama file discovery, timestamp sorting, image resolution checks,
relative naming, and hardlink/symlink import into `image_import.py`.

Verification:

- Existing image import tests still pass.
- Confirm monkeypatch expectations are updated carefully if tests patch
  `colmap.panorama_reconstruct.os.link`.

Risk:

- Monkeypatching through the old facade may stop affecting the moved function.
  Prefer updating tests to patch the new module after the compatibility layer is
  proven, or preserve a simple wrapper during this stage.

### Stage 3: Move Database Read Code

Move COLMAP SQLite read helpers into `database.py`.

Verification:

- Database identity tests pass.
- Feature stats tests pass.
- Keyframe scoring tests pass if frame connection reads are moved here.

Risk:

- SQL helpers are small but central to reporting and scoring. Keep queries
  byte-for-byte equivalent unless a bug is being intentionally fixed.

### Stage 4: Move Reporting And Metadata Code

Move report construction, metadata writing, feature health summary, image pair
list writing, and reconstruction stats into `reports.py`.

Verification:

- Report schema tests pass.
- Compare generated report keys before and after the move in tests or by manual
  inspection.

Risk:

- This stage can accidentally change JSON field names or option values. Treat
  the current report schema as frozen during this refactor.

### Stage 5: Move pycolmap Boundary

Move pycolmap import handling, SSL setup, option construction, matching, mapping,
GPU resolution, and best-model writing into `pycolmap_adapter.py`.

Verification:

- Existing option propagation tests pass.
- Tests still import without pycolmap installed.
- pycolmap-dependent tests continue to skip cleanly when pycolmap is missing.

Risk:

- This is the most sensitive extraction because `pycolmap` is optional at import
  time. Keep lazy import behavior intact.

### Stage 6: Move Scoring

Move keyframe scoring and score math into `scoring.py`.

Verification:

- Keyframe anchor scoring tests pass.
- No scoring constants or default caps change.

Risk:

- This code depends on reconstruction object shape and database pair IDs. Do not
  make it generic during the move.

### Stage 7: Thin Pipeline And CLI

Move `run_panorama_reconstruction` into `pipeline.py`. Move argument parsing and
console printing into `cli.py`. Keep `panorama_reconstruct.py` as a thin facade
that imports `main` from `colmap.panorama.cli`.

Verification:

- `pytest tests/test_colmap_panorama_reconstruct.py`
- `python -m colmap.panorama_reconstruct --help`
- `colmap-panorama-reconstruct --help` if the package is installed editable in
  the active environment.

Risk:

- The high-level function is long because it is doing real orchestration. Do not
  split it further until the module extraction is complete and tests are green.

## What Not To Do In This Refactor

- Do not implement the future rig-based pipeline as part of this structural
  cleanup.
- Do not change report schema fields.
- Do not change default matching, GPU, feature, or mapper options.
- Do not rename the console script yet unless the final CLI stage explicitly
  approves it.
- Do not remove `panorama_reconstruct.py` until external imports have been
  migrated.
- Do not introduce broad abstract base classes or plugin systems.

## Success Criteria

- `panorama_reconstruct.py` becomes a small compatibility facade.
- Each new module has one clear responsibility.
- Existing tests pass after every stage.
- CLI behavior remains stable.
- The refactor makes later rig-based panorama work easier without implementing
  that behavioral change prematurely.

## Suggested First Slice

Start with Stage 1 only:

1. Add `src/colmap/panorama/__init__.py`.
2. Add `constants.py`, `types.py`, and `paths.py`.
3. Re-export the moved names from `panorama_reconstruct.py`.
4. Run the targeted tests.

Do not proceed to Stage 2 until Stage 1 is reviewed and accepted.
