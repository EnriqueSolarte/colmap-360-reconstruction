Date: 2026-07-02
Project version: v0.1.0
Description: Redefines pinhole view rendering as selection, rendering, and pose attachment stages

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
- Support automatic grid selection, similar to the current `--auto-select`
  behavior.
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

## Implementation Plan

1. Define the flat `selections.json` schema and add a pure selection artifact
   builder for manual and auto-selected views.
2. Add a render stage that consumes `selections.json` and writes rendered
   images plus `render_report.json` without loading COLMAP.
3. Adapt or replace the browser selector so it can operate directly from an
   image directory and write the flat selection schema.
4. Add native equirectangular pose attachment from `sparse_best`, using
   registered COLMAP image names instead of legacy `panorama_rig_config.json`.
5. Update README examples after the behavior is implemented and verified.

## Selector Server Plan

The selector server should support a new image-directory mode while preserving
the existing metadata-based workflow until the new flow fully replaces it.

New CLI shape:

```bash
manual-pinhole-select \
  --image-dir /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/images \
  --output-dir /data/itri-3d-reconstruction/scene-reconstruction/2026-07-02-001/pinhole-selection
```

Behavior:

- scan `--image-dir` for equirectangular panorama images;
- launch the existing browser selector from those images;
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

### Selector Server Slices

1. Add image-directory server mode and flat `selections.json` read/write while
   preserving legacy metadata mode.
2. Make the existing `/api/state`, `/api/preview`, `/api/selections`,
   delete/reset/undo, and ROI endpoints dispatch to flat storage in image mode.
3. Simplify browser labels for image mode by hiding COLMAP scene/quality fields
   or replacing them with image-directory selection status.
4. Add automatic grid selection from the same flat schema, either as a CLI
   option or a browser command.

## First Slice

Implement the data-model slice first:

- create or extract focused helpers for the flat `selections.json` schema;
- support auto-grid generation into that schema;
- add tests that prove multiple selections per panorama and auto-selection use
  the same flat output shape;
- avoid changing the existing browser workflow until the schema is verified.
