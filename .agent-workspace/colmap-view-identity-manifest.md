Date: 2026-06-16
Project version: v0.1.0
Description: Define an upgrade that separates COLMAP view names from source panorama identity.

# COLMAP View Identity Manifest

## Goal

Upgrade the panorama reconstruction workflow so generated COLMAP virtual views
carry stable, explicit source identity metadata instead of relying on filenames
to encode every meaning.

This matters for:

- Continuing panorama reconstruction across multiple input batches.
- Mapping registered COLMAP virtual views back to original panorama files.
- Selecting manual pinhole views only from panoramas that actually participate
  in the final reconstructed scene.
- Computing pose and quality metadata per source panorama.
- Avoiding fragile reverse parsing of names such as
  `img-0002_shelf-1781171031546_00000051.png`.

## Current Behavior

The reconstruction pipeline currently names virtual views using:

```text
<virtual-camera-prefix>/<possibly-prefixed-source-filename>
```

Examples:

```text
pano_camera00/shelf-1781171031546_00000011.png
pano_camera01/shelf-1781171031546_00000011.png
pano_camera00/img-0002_shelf-1781171031546_00000051.png
```

Useful parts:

- `pano_cameraXX/` is useful because COLMAP `PER_FOLDER` camera mode can assign
  one sensor to each virtual camera direction.
- Keeping a recognizable source panorama filename is useful for debugging.

Problems:

- The filename is doing too many jobs at once:
  source panorama identity, virtual camera identity, continuation-batch identity,
  collision avoidance, and human-readable display name.
- First sessions and continuation sessions do not use the same naming rule. A
  first session may use `shelf-...png`, while a continuation session may use
  `img-0002_shelf-...png`.
- Downstream tools must reverse-engineer source identity from strings.
- The final scene may contain only a subset of all available source panoramas,
  but the source panorama directory can contain many more files.
- The generated metadata records render options and virtual camera definitions,
  but it does not provide a complete generated-view manifest grouped by source
  panorama.

## Design Principle

Separate these concepts:

```text
source panorama identity
input batch identity
COLMAP virtual image identity
virtual camera identity
registered reconstruction pose identity
human-readable filename
```

Filenames can remain readable, but they must not be the only source of truth.

## Recommended Identity Model

Introduce a stable per-panorama frame record:

```text
frame_key: stable identifier for one source panorama in one reconstruction
source_pano_path: original panorama file path
source_pano_filename: original panorama filename
source_pano_stem: original panorama stem
input_batch_id: optional batch/session label, such as img-0002
virtual_images: generated COLMAP images for this panorama frame
```

Each virtual image record should include:

```text
colmap_image_name: image name stored in COLMAP
virtual_camera_index: integer camera index
virtual_camera_prefix: pano_cameraXX/
yaw: virtual camera yaw in radians
pitch: virtual camera pitch in radians
cam_from_pano_rotation: fixed rig rotation
rendered_image_path: generated virtual image file
mask_path: generated ownership mask file
database_image_id: optional COLMAP database image id
registered_image_id: optional optimized COLMAP model image id
```

## Recommended Filename Policy

Keep the existing `pano_cameraXX/` folder structure.

Preferred COLMAP image-name shape:

```text
pano_camera00/<frame-key>.png
pano_camera01/<frame-key>.png
```

Example:

```text
pano_camera00/frame-000051.png
pano_camera01/frame-000051.png
```

Alternative human-readable shape:

```text
pano_camera00/shelf-1781171031546_00000051.png
pano_camera01/shelf-1781171031546_00000051.png
```

Rule:

- Use the same naming convention for first sessions and continuation sessions.
- Do not rely on prepending `img-0002_` to preserve uniqueness.
- If batch identity matters, store it in metadata as `input_batch_id`.
- If filename uniqueness is needed, use a stable `frame_key` rather than
  overloading the source filename.

## Proposed Manifest

Create a manifest such as:

```text
panorama_view_manifest.json
```

or add an equivalent `source_panorama_frames` section to
`panorama_rig_config.json`.

Recommended top-level shape:

```json
{
  "schema_version": 1,
  "input_image_path": "/path/to/current/input/batch",
  "workspace": "/path/to/colmap/workspace",
  "render_options": {},
  "camera": {},
  "views": [],
  "source_panorama_frames": []
}
```

Example frame record:

```json
{
  "frame_key": "frame-000051",
  "source_pano_filename": "shelf-1781171031546_00000051.png",
  "source_pano_stem": "shelf-1781171031546_00000051",
  "source_pano_path": "/data/.../shelf-1781171031546_00000051.png",
  "input_batch_id": "img-0002",
  "virtual_images": [
    {
      "virtual_camera_index": 0,
      "virtual_camera_prefix": "pano_camera00/",
      "colmap_image_name": "pano_camera00/frame-000051.png",
      "rendered_image_path": "/workspace/.../images/pano_camera00/frame-000051.png",
      "mask_path": "/workspace/.../masks/pano_camera00/frame-000051.png.png",
      "yaw": 0.0,
      "pitch": -0.6108652381980153,
      "cam_from_pano_rotation": []
    }
  ]
}
```

## Reconstruction Report Additions

After COLMAP mapping completes, enrich the manifest or report with optimized
registration data:

```text
registered: true/false
colmap_frame_id: optional frame id
registered_virtual_image_count
expected_virtual_image_count
active_virtual_camera_count
unique_point3D_count
triangulated_observation_count
mean_reprojection_error_px
median_reprojection_error_px
p90_reprojection_error_px
mean_track_length
connected_panorama_count
top_connected_panoramas
pano_camera_pose
```

Recommended `pano_camera_pose` fields:

```text
center_world
rotation_world_from_pano
transform_world_from_pano
pose_source_image_name
pose_source_image_id
coordinate_system: colmap_world_unscaled
```

The pose must be clearly labeled as COLMAP world coordinates and not metric
unless an external scale/alignment step has been applied.

## Impact On Manual Pinhole Selection

With this manifest, `manual-pinhole-render` should no longer scan every source
panorama and guess what the scene used.

Instead:

1. Read the scene workspace manifest or `sparse_best` model.
2. Use only source panorama frames that are registered in the final scene.
3. Resolve each registered frame back to the source panorama file.
4. Store pose and quality metadata in `pinhole-renders/metadata.json`.
5. Let the user manually select longitude/latitude directions only from those
   scene-registered panoramas.

This makes the manual workflow scene-aware and avoids showing panoramas that
were never part of the reconstruction.

## Compatibility Strategy

Do not break existing workspaces immediately.

Support two modes:

1. New manifest mode:
   read `panorama_view_manifest.json` or `source_panorama_frames` directly.
2. Legacy inference mode:
   infer frame identity from existing COLMAP image names such as
   `pano_camera00/img-0002_shelf-...png`.

Legacy inference should be explicit in metadata:

```text
identity_source: legacy_filename_inference
source_status: matched | missing | ambiguous
```

New workspaces should write:

```text
identity_source: explicit_manifest
source_status: matched
```

## Proposed Stages

### Stage 1: Add Manifest Writing

During virtual image rendering, write one manifest entry per source panorama
frame and per generated virtual camera image.

Acceptance criteria:

- The manifest records source path, source filename, frame key, input batch id,
  virtual camera index, COLMAP image name, rendered image path, and mask path.
- First sessions and continuation sessions use the same identity structure.
- Existing COLMAP reconstruction behavior still works.

### Stage 2: Add Registered Pose And Quality Enrichment

After sparse reconstruction, join the optimized COLMAP model back to the
manifest.

Acceptance criteria:

- The report records registered/unregistered status per source panorama frame.
- The report includes per-frame pose and quality summaries.
- The report distinguishes unscaled COLMAP coordinates from metric poses.

### Stage 3: Update Manual Selection Tool

Make `manual-pinhole-render` consume the explicit manifest when present.

Acceptance criteria:

- Only scene-registered source panoramas are listed.
- Source panorama resolution no longer depends on fragile string parsing.
- Legacy workspaces still work through an explicit inference fallback.

## Open Questions

- Should `frame_key` be sequential by reconstruction input order, source
  filename stem, or a stable hash of the source path?
- Should the manifest live as a separate `panorama_view_manifest.json`, or be
  embedded into `panorama_rig_config.json`?
- Should continuation runs preserve old frame keys exactly when copying an
  initial session?
- Should unregistered source panoramas appear in the final enriched manifest, or
  only in the raw render manifest?

## Recommendation

Add a separate `panorama_view_manifest.json` first. It keeps the upgrade focused
and avoids making `panorama_rig_config.json` too large. Later, the report can
reference the manifest path and include only summaries.

Use explicit identity fields as the source of truth. Keep filenames readable,
but treat them as paths, not as the database of meaning.
