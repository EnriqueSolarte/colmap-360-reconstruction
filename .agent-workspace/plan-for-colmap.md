Date: 2026-06-03
Project version: v0.1.0
Description: Adapt imported COLMAP scripts into a repo-native pinhole reconstruction stage

# Plan For COLMAP

## Goal

Build a reliable COLMAP reconstruction stage for pinhole images generated from
equirectangular panoramas.

The immediate target dataset is:

```text
/data/itri-3d-reconstruction/itri-3d-reconstruction/pinhole_1780034936312
```

This folder currently contains many flat `*_pinhole_00.png` images rather than
a complete COLMAP workspace. The COLMAP stage should therefore make the input
self-describing before it attempts reconstruction.

## Current Evidence

- The existing `ep_pinhole` package renders pinhole images from EP images.
- Project reconstruction defaults are:

```text
width: 1280
height: 960
fov_x: 1.221730476 radians / 70.0 degrees
fov_y: 0.967145867 radians / 55.4 degrees
```

- The sample pinhole dataset has RGB PNG files at `1280x960`.
- No nearby metadata or calibration sidecar was found under the provided
  `/data/itri-3d-reconstruction/itri-3d-reconstruction` tree.
- `src/colmap` scripts were copied from another project and still depend on
  missing `starter.*` modules.
- `pycolmap` and `open3d` are not currently installed in the active environment.

## Important Assumptions

For the first reconstruction experiment, assume all images in the pinhole
folder were rendered with the project defaults above.

This means the virtual camera can use:

```text
model: PINHOLE
width: 1280
height: 960
fx: 914.0147
fy: 914.0060
cx: 640.0
cy: 480.0
```

These intrinsics come from:

```text
fx = (width / 2) / tan(fov_x / 2)
fy = (height / 2) / tan(fov_y / 2)
cx = width / 2
cy = height / 2
```

If the images were rendered with different FOVs or resizing, this assumption is
wrong and COLMAP will receive bad intrinsics. The first implementation slice
must make this assumption visible in a report.

## Imported Script Assessment

### `src/colmap/reconstruct.py`

Useful pieces:

- Image discovery.
- COLMAP workspace preparation.
- Feature extraction with a shared camera.
- Sequential matching.
- Incremental mapping.
- Best sparse model selection.
- PLY export.
- JSON reporting.

Required changes:

- Remove dependency on `starter.camera_calibration.CameraCalibration`.
- Derive pinhole intrinsics from CLI FOV/resolution or a local metadata file.
- Make default paths point to the current pinhole reconstruction workflow, not
  `/workspace/data/rgb`.
- Add an inspection/dry-run mode before running expensive COLMAP steps.
- Support a small image subset for smoke testing.
- Keep the output clearly labeled as raw, unscaled COLMAP geometry.

### `src/colmap/scale_recover.py`

Mostly out of scope for this stage.

It depends on old ArUco marker code and assumes there are marker observations
that can recover metric scale. Keep the high-level idea, but do not wire this
into the current pipeline until the dataset has a known metric reference such
as marker poses, GPS/INS, LiDAR, or ground-truth camera trajectory.

### `src/colmap/visualize_map.py`

Useful after reconstruction succeeds.

Required changes:

- Ensure dependency handling is clear for `open3d` and `pycolmap`.
- Keep `--summary-only` useful in headless environments.
- Avoid assuming a metric scale report exists.

## Implementation Slices

### Slice 1: Input Inspection And Intrinsics

Create a repo-native input inspection path for pinhole COLMAP data.

Target behavior:

- Accept `--image-dir`.
- Count supported images.
- Verify all sampled or all images share one resolution.
- Compute intrinsics from `--width`, `--height`, `--fov-x`, and `--fov-y`.
- Print first and last filenames.
- Write `colmap_input_report.json` to the workspace.
- Do not run COLMAP yet.

Acceptance criteria:

- The report records image count, image resolution, FOVs, computed intrinsics,
  image name examples, and the explicit assumption that all images share one
  virtual camera.
- The command works without `pycolmap`.

### Slice 2: Raw COLMAP Reconstruction CLI

Adapt `reconstruct.py` so it can run raw sparse reconstruction from the
pinhole folder.

Target behavior:

- Accept explicit camera intrinsics or compute them from FOV/resolution.
- Use COLMAP `PINHOLE` with a single shared camera by default.
- Use sequential matching for temporally ordered filenames.
- Disable intrinsic refinement for the first run.
- Write:

```text
database.db
sparse/
sparse_best/
cloud_colmap_raw.ply
validation_report.json
```

Acceptance criteria:

- No import from `starter.*`.
- The JSON report records COLMAP options and reconstruction stats.
- Failure modes are explicit: no images, inconsistent resolution, no model,
  missing `pycolmap`, existing workspace without `--overwrite`.

### Slice 3: Smoke Subset

Run a small reconstruction before the full dataset.

Recommended first smoke:

```text
100-300 sequential images
overlap: 15-30
max_num_features: 8192
CPU first, GPU optional later
```

Acceptance criteria:

- COLMAP registers a useful fraction of the subset.
- Mean reprojection error and sparse point count are reported.
- Output can be loaded by the visualization summary path.

### Slice 4: Full Sequence Reconstruction

Run the adapted pipeline on the full `12,124` image folder.

Target behavior:

- Keep sequential matching as the first strategy.
- Tune overlap only after inspecting smoke results.
- Keep intrinsics fixed unless there is strong evidence refinement is needed.

Acceptance criteria:

- A best sparse model is written.
- The report clearly states registered image count, total image count, sparse
  point count, matching stats, and mean reprojection error.
- Output remains labeled `raw_colmap_unscaled`.

### Slice 5: Visualization And Quality Review

Use the visualization script only after reconstruction output exists.

Target behavior:

- Provide a summary-only mode for headless checks.
- Optionally save a screenshot when Open3D can create a window.
- Draw sparse points and camera frustums.

Acceptance criteria:

- The model can be summarized without requiring metric scale.
- The report and visualization agree on registered image count.

### Slice 6: Metric Scale Decision

Do not implement metric scale recovery until the scale source is known.

Possible scale sources:

- Dataset-provided camera poses.
- Vehicle GPS/INS.
- LiDAR map alignment.
- AprilTag/ArUco observations, if actually present.
- Known scene measurements.

Acceptance criteria:

- A separate decision note exists before implementing metric recovery.
- The raw COLMAP model is never presented as metric output by default.

## Risks

- The pinhole images may not have been rendered with the assumed FOV.
- The current flat folder has no metadata linking pinhole images back to source
  panoramas or view angles.
- Sequential image filenames may not guarantee smooth camera motion.
- 12k images can be expensive for feature extraction and matching.
- COLMAP may struggle if the pinhole crop direction changes too aggressively
  between frames.
- Imported scale recovery code is misleading until the current dataset has a
  confirmed metric reference.

## Recommended Next Action

Implement Slice 1 first: input inspection and intrinsics reporting.

This is the safest next step because it turns the dataset and camera assumptions
into an explicit artifact before running expensive reconstruction.
