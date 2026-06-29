Date: 2026-06-29
Project version: v0.1.0
Description: Summarizes the completed native equirectangular COLMAP pipeline rewrite.

# Native Equirectangular Pipeline Summary

## Goal

Update the panorama reconstruction pipeline to use pycolmap/COLMAP's native
`EQUIRECTANGULAR` camera model directly, instead of rendering a virtual
pinhole camera rig from each panorama.

## Current Pipeline Shape

`src/colmap/panorama_reconstruct.py` now runs the main reconstruction path as
a native equirectangular image pipeline:

1. Create the COLMAP workspace directories.
2. Discover panorama images under `--input-image-path`, sorted by timestamp
   when possible.
3. Validate that all selected panoramas share the same resolution.
4. Create one shared `EQUIRECTANGULAR` camera with parameters
   `(width, height)`.
5. Import each original panorama into `workspace/images/` using a hard link
   when possible, with an absolute symbolic-link fallback across filesystems.
   The native import does not fall back to copying.
6. Write `panorama_equirectangular_config.json` metadata with
   `camera_setup: native_equirectangular`.
7. Extract features with:
   - `reader_options.camera_model = "EQUIRECTANGULAR"`
   - `reader_options.camera_params = "<width>,<height>"`
   - `CameraMode.SINGLE`
8. Read feature statistics by database camera/image.
9. Match features with the requested matcher, while disabling rig-specific
   behavior:
   - `rig_verification = False`
   - `expand_rig_images = False`
   - no legacy `match_scope` selection
10. Run incremental mapping.
11. Select and write the best sparse model.
12. Build `panorama_equirectangular_report.json` with native equirectangular status,
    feature health, matching options, image ids, and reconstruction stats.

## User-Facing CLI

The reconstruction CLI is now native-only. Current relevant arguments are:

- `--input-image-path`
- `--workspace`
- `--matcher {sequential,exhaustive,vocabtree}`
- `--overlap`
- `--loop-detection` / `--no-loop-detection`
- `--vocab-tree-path`
- `--ssl-cert-file`
- `--feature-extractor {sift}`
- `--feature-matcher {sift-bruteforce}`
- `--max-num-features`
- `--guided-matching`
- `--device {auto,cpu,cuda}`
- `--gpu-index`
- `--gpu-bundle-adjustment` / `--no-gpu-bundle-adjustment`
- `--image-limit`
- `--overwrite`

Removed from the CLI: virtual pinhole rendering options, rig options,
initial-session continuation options, and old match-scope/keyframe-frame
selection arguments from the previous virtual-rig version.

## File Changes

- `src/colmap/camera.py`
  - `ColmapCamera.params` now supports variable-length parameter tuples.
  - `camera_from_equirectangular(width, height)` returns a native
    `EQUIRECTANGULAR` camera with params `(float(width), float(height))`.
  - Camera reports label equirectangular params as `["width", "height"]`.

- `src/colmap/panorama_reconstruct.py`
  - Main `run_panorama_reconstruction()` path imports original panoramas
    directly and no longer renders virtual pinhole views.
  - Added native import helpers:
    - `panorama_image_paths`
    - `import_equirectangular_panorama_images`
    - `image_resolution`
    - `hardlink_or_symlink_file`
  - Added native metadata writer:
    - `write_equirectangular_metadata`
  - Matching can skip rig expansion and rig verification.
  - Reports now assume `native_equirectangular`; the legacy
    `virtual_pinhole_rig` branch was removed.

- `src/colmap/visualize_map.py`
  - `EQUIRECTANGULAR` cameras are visualized as a sphere/axis camera marker
    instead of pinhole frustums.

- `src/colmap/workspace.py`
  - Deleted after checking usage; it was not used by the current pipeline.

- `src/colmap/input_inspection.py`, `src/colmap/pipeline.py`,
  `src/colmap/reporting.py`
  - Deleted after folding the remaining useful constants/helpers into the
    native equirectangular reconstruction module.

- `README.md`
  - Updated reconstruction documentation to describe native
    `EQUIRECTANGULAR`, hardlink/symlink imports, and removed virtual-rig
    arguments.

- Tests
  - Added/updated coverage for native equirectangular import, hardlink to
    symlink fallback, mixed-resolution rejection, native reports, CLI argument
    cleanup, and visualization support.

## Legacy Code Removed

The previous virtual-rig compatibility layer has been removed from
`panorama_reconstruct.py` and its tests. Removed symbols include:

- `PanoramaRenderOptions`
- `VirtualImageOutputOptions`
- `PanoramaRigView`
- `InitialPanoramaSession`
- virtual camera/view rendering helpers
- virtual mask helpers
- initial-session continuation helpers
- frame-scoped legacy matching helpers
- report fields with names like `virtual_camera_count`

The reconstruction output filenames were also renamed from the old
`panorama_rig_*` names to `panorama_equirectangular_config.json` and
`panorama_equirectangular_report.json`.

## Validation Status

Latest local validation after removing the legacy virtual-rig layer:

- Full test suite passed: `111 passed`
- `python -m colmap.panorama_reconstruct --help` showed only native relevant
  reconstruction arguments.
- Removed CLI flag search found old virtual arguments only in the summary and
  tests that assert the old CLI arguments/report fields are absent.

## Matching Notes

Current matcher choices:

- `sequential`: best default for ordered captures; matches neighboring
  panoramas and optional loop detection.
- `exhaustive`: all image pairs; useful for small datasets only.
- `vocabtree`: retrieval-style matching for large or unordered image sets,
  requiring a vocabulary tree.

For future keyframe-based matching, the clean next step is to add a
deterministic image-pair generator that chooses keyframes from the native
panorama image sequence, writes a pair list, and calls pycolmap pair matching.
This should be treated as pair selection on top of normal feature extraction,
not as a new feature matcher.
