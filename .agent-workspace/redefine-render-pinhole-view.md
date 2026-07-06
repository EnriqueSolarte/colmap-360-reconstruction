Date: 2026-07-05
Project version: v0.1.0
Description: Completed pinhole rendering workflow and planned pinhole COLMAP reconstruction

# Redefine Pinhole View Rendering

## Purpose

Separate pinhole view rendering from COLMAP pose optimization.

The pipeline should allow users to select longitude/latitude directions from
panorama images and render pinhole crops without requiring `pycolmap`,
`sparse_best`, or registered panorama poses. COLMAP reconstruction should be
used only when a later stage needs pose-aware outputs, such as a COLMAP text
handoff for Gaussian Splatting.

## Motivation

The current manual pinhole workflow still carries assumptions from the older
virtual-pinhole-rig reconstruction path. In particular, scene-aware rendering
expects optimized scene metadata before it can attach poses, even though image
selection and pinhole rendering are pure image-processing operations.

Decoupling the stages makes the workflow faster to iterate on, easier to test,
and better aligned with the maintained native equirectangular reconstruction
pipeline.

## Final Status

Implemented and verified.

The maintained pinhole workflow is now:

```text
images/ -> selections.json -> renders/render_report.json -> pose-renders/sparse/0
```

Public commands:

```text
manual-pinhole-select
manual-pinhole-render-selected
manual-pinhole-attach-poses
colmap-visualize
```

Removed legacy workflow:

- `manual-pinhole-render` console script was removed.
- `src/render_pinhole/tools/manual_pinhole_render.py` was deleted.
- `src/render_pinhole/tools/manual_pinhole_selection_store.py` was deleted.
- selector `metadata.json` legacy mode was removed.
- selector now requires `--image-dir` and writes flat `selections.json`.

Verification:

```text
pytest -q
197 passed
```

## Target Pipeline

### Stage 1: Select Panorama Directions

Input:

```text
workspace/images/
```

Behavior:

- Launch the selection web server from a directory of equirectangular panorama
  images.
- Support manual selection of one or more longitude/latitude directions per
  panorama.
- Support automatic grid selection from the browser, using the same flat
  selection schema as manual selections.
- Do not require COLMAP, `pycolmap`, `sparse_best`, or optimized poses.

Output:

```text
selections.json
```

The selection artifact is a flat collection of selected views. Manual and
automatic selection must write the same schema.

Example:

```json
{
  "schema_version": 1,
  "source": {
    "image_dir": "/data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/images"
  },
  "selection_mode": "manual",
  "selections": [
    {
      "selection_id": "1782890858308__sel-000",
      "pano_image_name": "1782890858308.png",
      "selection_index": 0,
      "longitude": 0.52,
      "latitude": -0.12,
      "longitude_degrees": 29.7938,
      "latitude_degrees": -6.8755,
      "selected_by": "manual-pinhole-select"
    }
  ]
}
```

Automatic selection should add an `auto_selection` block while preserving the
same `selections` array shape:

```json
{
  "selection_mode": "auto_grid",
  "auto_selection": {
    "strategy": "longitude_latitude_grid",
    "delta_degrees": 45.0,
    "views_per_panorama": 32
  }
}
```

### Stage 2: Render Pinhole Images

Input:

```text
workspace/images/
selections.json
render parameters
```

Behavior:

- Render one pinhole image for each selection record.
- Use explicit render parameters such as width, height, horizontal FOV,
  vertical FOV, interpolation, and output extension.
- Do not require COLMAP or registered poses.

Output:

```text
renders/
render_report.json
```

The report should record:

- source image directory;
- selections path;
- render parameters;
- rendered image count;
- per-render output filenames;
- any skipped or failed selections.

Implemented command shape:

```bash
manual-pinhole-render-selected \
  --selections /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/selections.json \
  --output-dir /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/renders \
  --width 1280 \
  --height 1280 \
  --fov-x-deg 70 \
  --fov-y-deg 70
```

`--image-dir` is optional when `selections.json` contains `source.image_dir`.
`--output-dir` defaults to a `renders/` directory beside `selections.json`.
The command writes rendered images and `render_report.json` without loading
COLMAP, `pycolmap`, `sparse_best`, scene poses, or rig metadata.
Rendering shows a `tqdm` progress bar. Use `--workers N` to render source
panorama groups with multiple threads; the default is `--workers 1` for
predictable memory use.

### Stage 3: Attach Poses From Reconstruction

Input:

```text
selections.json
render_report.json
workspace/sparse_best/
```

Behavior:

- Read native equirectangular panorama poses from the COLMAP reconstruction.
- Match rendered pinhole selections back to their source panorama image names.
- Compute each pinhole camera pose from the panorama pose and selected
  longitude/latitude.
- Report selected panoramas that are missing from the registered COLMAP model.

Output:

```text
pose_report.json
sparse/0/cameras.txt
sparse/0/images.txt
sparse/0/points3D.txt
```

Pose attachment should report:

```text
selected view count
pose-attached view count
missing-pose view count
registered panorama count
```

For pose-aware exports, unresolved selections should either be skipped
explicitly or fail unless an option such as `--allow-missing-poses` is provided.

Implemented command shape:

```bash
manual-pinhole-attach-poses \
  --selections /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/selections.json \
  --render-report /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/renders/render_report.json \
  --sparse-best /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/sparse_best \
  --output-dir /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/pose-renders
```

This stage writes `pose_report.json` by matching flat rendered selections to
registered native equirectangular COLMAP image names. It also writes a COLMAP
text model under `sparse/0` for pose-attached views:

```text
sparse/0/cameras.txt
sparse/0/images.txt
sparse/0/points3D.txt
```

## Completed Implementation

1. Defined the flat `selections.json` schema and added a pure selection artifact
   builder for manual and auto-selected views.
2. Added a render stage that consumes `selections.json` and writes rendered
   images plus `render_report.json` without loading COLMAP.
3. Reworked the browser selector so it operates directly from an
   image directory and writes the flat selection schema.
4. Added native equirectangular pose attachment from `sparse_best`, using
   registered COLMAP image names instead of legacy `panorama_rig_config.json`.
5. Updated README examples after the behavior was implemented and verified.
6. Updated `colmap-visualize` so pose-attached pinhole exports can be viewed as
   pinhole camera frustums with the exported point cloud.
7. Removed stale legacy pinhole render command, metadata selector mode, and
   virtual-rig/metadata-sidecar code.

## Final Selector Behavior

The selector server supports only image-directory mode.

CLI shape:

```bash
manual-pinhole-select \
  --image-dir /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/images
```

`--output-dir` is optional and defaults to the parent directory of
`--image-dir`.

Behavior:

- scan `--image-dir` for equirectangular panorama images;
- launch the browser selector from those images;
- write flat `--output-dir/selections.json`;
- keep preview rendering available in the browser;
- store only direction selection fields in `selections.json`;
- do not require `metadata.json`, COLMAP, `pycolmap`, `sparse_best`, scene
  poses, or rig metadata.

The UI should treat render controls as preview-only in this stage. Width,
height, FOV, interpolation, and output extension belong to Stage 2 render
reports, not Stage 1 selections.

The API can continue grouping selections by panorama for display, but the file
on disk must remain the flat schema:

```text
selections[]
```

Implemented selector behavior:

- plain click previews without saving;
- `Cmd`/`Ctrl` click previews and saves;
- left/right arrow keys navigate panoramas;
- cursor-near-marker highlighting;
- `d` deletes the highlighted marker;
- ROI sampling writes the flat schema;
- auto-grid selection writes the flat schema.

## Remaining Follow-Up

`src/render_pinhole/tools/manual_pinhole_select_server.py` is still large. It is
no longer carrying legacy metadata branches, but a later refactor could split
server state/storage, HTTP handlers, and browser assets into smaller modules.
That refactor is intentionally separate from this completed pipeline change.

# Planned Pinhole COLMAP Reconstruction

## Purpose

Add a sibling COLMAP reconstruction pipeline under `src/colmap/pinhole/` that
uses rendered pinhole images as the reconstruction input.

This pipeline should be separate from the maintained native panorama pipeline
under `src/colmap/panorama/`. The panorama pipeline remains the direct
`EQUIRECTANGULAR` reconstruction path. The new pinhole pipeline should consume
the outputs of the completed pinhole rendering workflow:

```text
selections.json -> renders/render_report.json -> optional pose-renders/pose_report.json
```

The first implementation should target reconstruction from the rendered images
and intrinsics recorded in `render_report.json`.

## Agreed Baseline Behavior

Input:

```text
renders/render_report.json
```

Behavior:

- read rendered pinhole image records from `render_report.json`;
- link or copy those rendered images into a new COLMAP workspace;
- derive COLMAP `PINHOLE` camera intrinsics from each render record's width,
  height, `fov_x`, and `fov_y`;
- for the first implementation, require one shared camera setup across all
  rendered images;
- extract features with the same pycolmap feature options used by the panorama
  pipeline;
- match images using a normal overlap-based matcher, with sequential overlap as
  the default first target;
- run COLMAP incremental mapping without requiring panorama poses;
- write `sparse/`, `sparse_best/`, `cloud_colmap_raw.ply`, and a structured
  `pinhole_reconstruction_report.json`.

Initial command shape:

```bash
colmap-pinhole-reconstruct \
  --render-report /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/renders/render_report.json \
  --workspace /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/pinhole-reconstruction \
  --matcher sequential \
  --overlap 10 \
  --overwrite
```

The baseline result is a normal unscaled COLMAP sparse model reconstructed from
the rendered pinhole images. It is independent from the native panorama
reconstruction unless optional priors are explicitly enabled.

## Optional Panorama-Derived Pose Reuse

Optional input:

```text
pose-renders/pose_report.json
```

When a pose report is passed, the pinhole reconstruction can reuse
panorama-derived pinhole poses in two separate ways.

### Position Priors

This means:

```text
pose_report.views[].pinhole_camera_pose.center_world
```

is used as a prior camera center for the matching rendered pinhole image.

Initial command shape:

```bash
colmap-pinhole-reconstruct \
  --render-report /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/renders/render_report.json \
  --pose-report /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/pose-renders/pose_report.json \
  --use-position-priors \
  --workspace /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/pinhole-reconstruction-priors \
  --overlap 10 \
  --overwrite
```

Reports should make the prior behavior explicit:

```text
pose_prior_source: panorama_pose_report
pose_prior_type: camera_center
pose_prior_coordinate_frame: colmap_world_unscaled
pose_prior_count
missing_pose_prior_count
use_prior_position
```

Important limitation:

The panorama-derived centers are not external ground truth. They come from the
native equirectangular COLMAP reconstruction and therefore inherit that model's
unscaled coordinate frame, drift, and any pose errors. They are useful for
anchoring or regularizing the pinhole reconstruction, but they should not be
reported as metric or absolute camera locations.

### Full 6DoF Initial Poses

Full-pose initialization means:

```text
pose_report.views[].pinhole_camera_pose.center_world
pose_report.views[].pinhole_camera_pose.rotation_world_from_pinhole
```

are converted into COLMAP image poses and written as an initial text model under
the pinhole workspace. The mapper receives that model via pycolmap
`incremental_mapping(..., input_path=...)`.

This initializes the rendered pinhole images from the panorama-derived 6DoF
poses, but it does not fix them. COLMAP still extracts features, matches
rendered images, triangulates, and bundle-adjusts the reconstruction.

Initial command shape:

```bash
colmap-pinhole-reconstruct \
  --render-report /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/renders/render_report.json \
  --pose-report /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/pose-renders/pose_report.json \
  --initial-poses-from-pose-report \
  --workspace /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/pinhole-reconstruction-initial-poses \
  --overlap 10 \
  --overwrite
```

The mode remains experimental until representative runs show that it improves
registration, stability, or alignment compared with the baseline and
camera-center prior modes.

## Implementation Stages

### Stage 1: Render-Report Import And Intrinsics

Goal:

Build the pure input layer for `src/colmap/pinhole/`.

Tasks:

- create `src/colmap/pinhole/` with focused modules such as `paths.py`,
  `image_import.py`, `reports.py`, `pipeline.py`, and `cli.py`;
- parse `render_report.json`;
- validate rendered records and output files;
- require uniform width, height, `fov_x`, and `fov_y` for the first slice;
- derive a shared `PINHOLE` camera using `camera_from_fov`;
- link rendered images into `workspace/images/`;
- write a pinhole config/import report.

Verification:

- unit tests for render-report parsing;
- unit tests for camera derivation from FOV;
- unit tests for rejecting missing files and mixed intrinsics.

### Stage 2: Baseline Pinhole Reconstruction

Goal:

Run COLMAP reconstruction from rendered pinhole images without using panorama
poses.

Tasks:

- reuse shared feature and execution option dataclasses where practical;
- extract features into `database.db`;
- support sequential matching with configurable `--overlap`;
- run incremental mapping;
- select and write the best sparse model;
- export `cloud_colmap_raw.ply`;
- write `pinhole_reconstruction_report.json`.

Verification:

- tests for CLI argument validation and report construction;
- targeted tests around the pipeline boundaries using fakes where pycolmap is
  too heavy;
- one small real smoke run when data and pycolmap are available.

### Stage 3: Optional Position-Prior Parsing

Goal:

Read `pose_report.json` and match panorama-derived pinhole camera centers to
rendered image records, without changing reconstruction behavior yet.

Tasks:

- add `--pose-report`;
- match pose-report views to render records by `selection_id` and/or
  `output_name`;
- report matched, missing, and unusable priors;
- record prior coordinate frame and source;
- keep reconstruction identical unless `--use-position-priors` is enabled.

Verification:

- tests for pose-report matching;
- tests for missing priors and mismatched render records;
- report tests proving the mode is visible and auditable.

### Stage 4: Enable COLMAP Position Priors

Goal:

Use panorama-derived camera centers as optional COLMAP position priors.

Tasks:

- add `--use-position-priors`;
- write position priors into the COLMAP database or otherwise feed them through
  the supported pycolmap path;
- set mapper options such as `use_prior_position`;
- expose robust-loss and prior-loss-scale options only if needed after the
  first working implementation;
- report all prior-related mapper options.

Verification:

- unit tests for database prior writing if that path is used;
- a comparison run against the Stage 2 baseline:
  - registered image count;
  - sparse point count;
  - model count;
  - mean reprojection error;
  - elapsed time;
  - visual consistency with the panorama coordinate frame.

### Stage 5: Pair Planning Improvements

Goal:

Make matching more aware of pinhole render provenance after the baseline is
working.

Possible improvements:

- avoid or separately report same-source-panorama pairs, because multiple
  pinhole crops from one panorama share the same physical camera center and do
  not provide stereo baseline;
- add source-panorama-aware sequential pair planning;
- add optional vocabtree or keyframe-style pairing if baseline overlap matching
  is insufficient.

Verification:

- report pair counts by source;
- compare reconstruction quality and runtime against simple sequential overlap.

### Stage 6: Experimental Full-Pose Initialization

Goal:

Evaluate whether full panorama-derived pinhole poses can be used as an
initialization strategy without fixing camera poses.

Implemented first slice:

- generate a COLMAP input model from pose report data and pass it through
  `input_path`;
- keep mapper pose refinement enabled by leaving existing frames unfixed;
- report initialized, missing, and unusable pose counts.

Possible future implementation:

- use lower-level pycolmap reconstruction or mapper APIs if the text-model
  `input_path` seed proves insufficient.

Exit criteria:

- only keep this mode if it demonstrably improves registration, stability, or
  alignment compared with Stage 2 and Stage 4;
- label the mode experimental until it is verified on representative datasets.
