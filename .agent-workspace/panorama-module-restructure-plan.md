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

## Implementation Summary

Status as of 2026-06-29: all planned restructuring stages have been executed.

The former `src/colmap/panorama_reconstruct.py` implementation has been split
into the `src/colmap/panorama/` package. The old module remains as a
compatibility facade that re-exports the previous public helpers and preserves
the `python -m colmap.panorama_reconstruct` execution path.

Implemented modules:

- `constants.py`: common constants and filename timestamp pattern.
- `types.py`: shared dataclasses and score report float formatting helpers.
- `paths.py`: workspace path construction.
- `image_import.py`: panorama discovery, timestamp sorting, resolution checks,
  relative naming, and hardlink/symlink import.
- `database.py`: COLMAP SQLite reads, database image identity reads, feature
  statistics, pair-id decoding, and frame connection reads.
- `reports.py`: metadata writing, report construction, model statistics,
  feature-health summary, image-pair-list writing, and elapsed-time formatting.
- `pycolmap_adapter.py`: optional `pycolmap` import boundary, SSL setup, feature
  and matcher option construction, mapper option construction, GPU resolution,
  feature matching, incremental mapping, and best-model writing.
- `scoring.py`: keyframe anchor scoring and score math.
- `pipeline.py`: high-level `run_panorama_reconstruction` orchestration.
- `cli.py`: argument parsing, defaults, and console output.

The compatibility facade `panorama_reconstruct.py` is intentionally still
present. It is no longer the implementation owner; it is the backwards-compatible
import and execution surface.

## Why This Is Better

The implementation is materially easier to reason about than the original
single-file script. The original file mixed CLI parsing, image import, SQLite
queries, pycolmap calls, scoring, report writing, timing, and pipeline control
in one place. That meant a change to one concern required scanning unrelated
code and increased the chance of accidentally modifying behavior outside the
intended area.

The new layout improves practical maintainability in these ways:

- Changing image import behavior now starts in `image_import.py`, not in a
  1,000+ line script.
- Changing COLMAP database reads now starts in `database.py`.
- Changing feature extraction, matching, mapping, CUDA selection, or pycolmap
  compatibility now starts in `pycolmap_adapter.py`.
- Changing report schema or metadata writing now starts in `reports.py`.
- Changing keyframe anchor scoring now starts in `scoring.py`.
- Changing CLI flags or printed output now starts in `cli.py`.
- The high-level pipeline now reads as a stage sequence instead of being
  interrupted by unrelated helper definitions.

This is especially useful for the future rig-based panorama work. That work
will likely need to touch image generation/import, database identity, pycolmap
rig options, matching, and reports. With the split, those changes can be made
and reviewed in the module that owns the relevant concern.

The refactor also keeps behavior stable:

- The console script target can continue pointing at
  `colmap.panorama_reconstruct:main`.
- Existing imports from `colmap.panorama_reconstruct` still work.
- The report schema was not intentionally changed.
- Matching, mapping, feature, GPU, and reconstruction defaults were not
  intentionally changed.

## What Is Still Not Ideal

This implementation is better, but it is not finished architecture.

The most important remaining weaknesses are:

- `pipeline.py` is still long. It is now long for a better reason: it contains
  the actual orchestration sequence. But it still mixes stage timing, report
  finalization, pycolmap calls, and workspace setup. Do not split it further
  casually; if it is split, the next split should be around real pipeline stages
  with tests, not just smaller files.
- Tests still import most helpers through `colmap.panorama_reconstruct`. That
  protects backwards compatibility, but it does not strongly document the new
  module-level APIs. Future cleanup should migrate tests gradually to the owning
  modules while keeping a smaller facade-compatibility test.
- `types.py` contains `round_report_float` and `optional_report_float` because
  `KeyframeAnchorScore.report()` uses them. That is pragmatic, but conceptually
  these helpers are report formatting. A later cleanup could move score report
  serialization out of the dataclass, or leave this as acceptable coupling.
- `reports.py` builds large nested dictionaries directly. That preserves the
  existing report schema, but it is still easy to accidentally change fields.
  If report stability becomes important, add focused tests for required keys and
  report-schema compatibility.
- `database.py` returns loose dictionaries such as
  `dict[int, dict[str, int | set[int]]]` for frame connection stats. That
  preserves behavior, but a small dataclass could make scoring safer later.
  Do not add that abstraction unless scoring changes require it.
- `pycolmap_adapter.py` includes both pycolmap option/call wrappers and
  `write_best_model`, which performs filesystem writes around pycolmap
  reconstruction output. This is acceptable for now because it is still at the
  COLMAP boundary, but it is not a pure adapter.
- `pipeline.py` still performs a `json.dumps(report, indent=2)` before writing
  the final report. This appears to be a validation step inherited from the
  original implementation. It is harmless but awkward; if report writing is
  cleaned up later, make that intent explicit or remove it with tests.

## Verification Performed

After the restructuring, the targeted panorama tests passed:

```text
python -m pytest tests/test_colmap_panorama_reconstruct.py
38 passed
```

The full test suite also passed:

```text
python -m pytest
111 passed
```

The old module execution path was checked:

```text
python -m colmap.panorama_reconstruct --help
```

The help output rendered successfully with the existing CLI options.

## Practical Next Steps

Recommended next cleanup order:

1. Add or migrate tests to import from the owning modules directly, while keeping
   one compatibility test for `colmap.panorama_reconstruct`.
2. Add report-schema-focused tests before changing `reports.py`.
3. Consider extracting explicit pipeline stage helpers only if a new feature
   needs to modify one stage independently.
4. Revisit `types.py` versus `reports.py` ownership for score report formatting
   only if scoring/report serialization changes.
5. Keep `panorama_reconstruct.py` as the facade until external callers and tests
   have clearly moved to the new module paths.
