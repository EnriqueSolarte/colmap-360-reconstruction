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

- The input panoramas are already ordered or identifiable from the existing
  reconstruction workspace.
- A user may select multiple directions for one panorama.
- The same pinhole intrinsics should be used for all selected renders unless a
  later discussion explicitly changes this.
- The initial output should favor simple, inspectable files over a complex UI.
- Longitude and latitude values should remain in radians internally, matching
  the current renderer API.

## Discussion Questions

1. What is the source of truth for the panorama list: an image directory, a
   `colmap-panorama-reconstruct` workspace, or a COLMAP model plus metadata?
2. Should the first manual-selection tool be terminal-based, notebook-based, or
   browser-based?
3. Should selections be stored as JSON, JSONL, CSV, or a small SQLite database?
4. Do we need visual overlays from the reconstructed sparse model, or is showing
   each panorama and previewing pinhole crops enough for the first slice?
5. Should selected views be named by panorama filename plus selection index, or
   by reconstruction frame/image IDs?
6. What quality checks are required before rendering the final selected set?

## Proposed Staged Direction

### Stage 1: Selection File Contract

Define a small, versioned selection file that records:

- Source panorama path or identifier.
- Selection index per panorama.
- Longitude and latitude in radians.
- Render dimensions, FOV, and interpolation.
- Optional reviewer notes or tags.

This stage should include validation but should not build an interactive UI yet.

### Stage 2: Render From Selection File

Add a CLI that reads the selection file, renders the requested pinhole views,
and writes both images and metadata.

This proves the file contract is useful before investing in interaction design.

### Stage 3: Manual Selection Interface

Add the smallest practical interface for stepping through panoramas, previewing
candidate crops, and saving one or more selections per panorama.

The interface choice should be decided before implementation.

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
