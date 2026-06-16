Date: 2026-06-16
Project version: v0.1.0
Description: Define the discussion scope for manually selecting panorama view directions.

# Manual Selection Of Pinhole Views

## Goal

Assume the current repo can already reconstruct a scene well from panorama
images. The goal of this task is to add a small set of scripts that let a user
manually walk through all panorama views in an existing reconstruction, choose
one or more longitude/latitude directions per panorama, and render those
selected directions as pinhole images.

The selected pinhole images should be suitable as downstream input for Gaussian
Splatting reconstruction.

## Current Repo Context

The repo already has:

- Equirectangular-to-pinhole rendering through `render_pinhole.render_pinhole`.
- CLI rendering for one image and for a directory.
- Random and uniform-longitude view sampling.
- COLMAP panorama-rig reconstruction support that creates fixed virtual camera
  views from panoramas.
- Metadata output for generated pinhole images in the directory renderer.

The repo does not yet have:

- A manual review UI or command loop.
- A persistent selection file format for hand-picked longitude/latitude pairs.
- A render command that consumes a user-authored selection file.
- A clear mapping from reconstructed panorama records back to manually selected
  pinhole outputs.

## Working Assumptions

- The source of truth for the panorama list is an explicit panorama image
  directory passed by the user.
- The manual selection tool may also receive a reconstructed scene workspace so
  it can attach existing panorama camera poses to each selected view.
- A user may select multiple directions for one panorama.
- The same pinhole intrinsics should be used for all selected renders unless a
  later discussion explicitly changes this.
- The initial output should favor simple, inspectable files over a complex UI.
- Longitude and latitude values should remain in radians internally, matching
  the current renderer API.

## Confirmed Inputs

Initial target panorama directory:

```text
/data/itri-3d-reconstruction/itri-3d-reconstruction/shelf-1781171031546
```

Initial target reconstructed scene workspace:

```text
/data/itri-3d-reconstruction/scene-reconstruction/2026-06-15-0002/colmap/session-001-img-0002-img-0003-img-0004-img-0005-img-0006
```

## Discussion Decisions

- Panorama list source: pass a directory containing all panorama views.
- Reconstruction context: pass a reconstructed scene workspace when pose
  metadata is needed.
- Selection storage: JSON.
- First UI scope: show each panorama and preview pinhole crops. Sparse-model
  overlays are not required for the first useful version.
- Output naming: use the panorama stem plus a per-panorama selection index, with
  compact direction and render metadata for readability.
- Quality target: render the selected pinhole views at the best practical
  reconstruction quality, using stable intrinsics and avoiding crops that are
  visually weak or geometrically risky.

## Recommended Interface

Use a lightweight local browser tool, but do not build a full frontend app.

Recommended command shape:

```bash
manual-pinhole-select \
  --pano-dir /data/itri-3d-reconstruction/itri-3d-reconstruction/shelf-1781171031546 \
  --scene-workspace /data/itri-3d-reconstruction/scene-reconstruction/2026-06-15-0002/colmap/session-001-img-0002-img-0003-img-0004-img-0005-img-0006 \
  --output selections.json
```

Implementation direction:

- Python standard-library local HTTP server.
- One small vanilla HTML/JavaScript page.
- Browser displays the current panorama.
- Clicking the panorama proposes a longitude/latitude pair.
- Backend renders a preview crop using the existing pinhole renderer.
- User can save multiple selections per panorama.
- Keyboard shortcuts can handle next, previous, save, delete, and skip.

This is easier to use than a pure terminal workflow and much smaller than a
framework-based browser app. It also works naturally with remote development:
the user can open a forwarded local port in a browser.

## Selection JSON Contract

The JSON file should be explicit enough to reproduce the render and trace every
pinhole view back to the panorama and reconstruction pose.

Recommended top-level shape:

```json
{
  "schema_version": 1,
  "pano_dir": "/path/to/panoramas",
  "scene_workspace": "/path/to/reconstruction",
  "render_defaults": {
    "width": 1280,
    "height": 960,
    "fov_x": 1.221730476,
    "fov_y": 0.967145867,
    "interpolation": "linear"
  },
  "selections": []
}
```

Each selection should record:

- Source panorama path and panorama filename.
- Panorama index in the sorted input list.
- Selection index for that panorama.
- Longitude and latitude in radians.
- Longitude and latitude in degrees for readability.
- Render width, height, FOV, interpolation, and output extension.
- Panorama camera pose from the reconstruction, when available.
- Derived pinhole camera pose, when available.
- Output filename.
- Reviewer notes or tags.
- Creation time and tool version or repo version.

Pose fields should be optional in the first file contract if the exact mapping
from panorama image name to reconstruction image/frame ID is not yet confirmed.
The renderer can still work without poses, but Gaussian Splatting handoff is
stronger when poses are present.

## Rendered Filename Convention

Recommended pattern:

```text
<pano-stem>__sel-<idx>__lon-<deg>__lat-<deg>__w<width>_h<height>.png
```

Example:

```text
shelf-1781171031546_000123__sel-002__lon-p045.0__lat-m010.0__w1280_h960.png
```

Rules:

- Keep the original panorama stem first for easy traceability.
- Use zero-padded `sel-000`, `sel-001`, `sel-002` for multiple selections from
  the same panorama.
- Encode positive angles with `p` and negative angles with `m` so filenames do
  not contain ambiguous signs.
- Keep detailed camera pose and full precision angle values in JSON, not in the
  filename.
- Write all selected renders into one output directory unless downstream tools
  require grouping.

The JSON metadata remains the authoritative record. The filename is for human
navigation and quick debugging.

## Quality Checks Before Final Render

This does not mean judging aesthetics only. It means preventing bad selected
views from quietly becoming reconstruction input.

Recommended checks:

- Validate every referenced panorama file exists and can be opened.
- Validate every longitude is in `[-pi, pi]` after wrapping.
- Validate every latitude is in `[-pi / 2, pi / 2]`.
- Validate render width, height, FOV, and interpolation are consistent unless
  explicitly overridden.
- Warn if selected latitude is very close to the poles, because pinhole crops
  can become less stable there.
- Render preview thumbnails before the final batch so the user can catch blank,
  blurry, occluded, or uninformative views.
- Reject duplicate selections for the same panorama when longitude/latitude are
  nearly identical.
- Confirm panorama pose and pinhole pose are present if the output is intended
  to be used as pose-aware Gaussian Splatting input.
- Write a final `metadata.json` that records all selected outputs, camera
  intrinsics, source panorama paths, and pose availability.

For maximum rendered image quality, the first default should stay aligned with
the reconstruction defaults:

```text
width: 1280
height: 960
fov_x: 70 degrees / 1.221730476 radians
fov_y: 55.4 degrees / 0.967145867 radians
```

If compute and downstream memory allow it, add a final-render option for
`1600x1200` using the same FOV. Do not change FOV per image unless there is a
specific reconstruction reason, because variable intrinsics make downstream
camera metadata and debugging harder.

## Proposed Staged Direction

### Stage 1: Selection File Contract

Define a small, versioned selection file that records:

- Source panorama path or identifier.
- Selection index per panorama.
- Longitude and latitude in radians.
- Render dimensions, FOV, and interpolation.
- Panorama camera pose, when recoverable from the reconstruction workspace.
- Derived pinhole camera pose, when recoverable from the panorama pose and
  selected direction.
- Optional reviewer notes or tags.

This stage should include validation but should not build an interactive UI yet.

### Stage 2: Render From Selection File

Add a CLI that reads the selection file, renders the requested pinhole views,
and writes both images and metadata.

This proves the file contract is useful before investing in interaction design.

### Stage 3: Manual Selection Interface

Add the lightweight local browser interface for stepping through panoramas,
previewing candidate crops, and saving one or more selections per panorama.

### Stage 4: Reconstruction Handoff

Add output conventions and metadata needed by the Gaussian Splatting workflow.
This should include clear camera intrinsics and traceability back to the source
panoramas.

## Risks

- A manual workflow can become slow if it requires reviewing every panorama in a
  large sequence without shortcuts.
- If selection outputs are not traceable to source panoramas, downstream
  debugging will be painful.
- If FOV and intrinsics are changed per selected view, Gaussian Splatting and
  COLMAP handoff will need more careful camera metadata.
- A browser UI may be nicer but could be overkill for the first useful slice.

## Recommended First Discussion

Start by deciding the selection file contract and source-of-truth input. These
decisions determine the API shape and keep the first implementation slice small,
in line with `rule-of-implementation.md`.
