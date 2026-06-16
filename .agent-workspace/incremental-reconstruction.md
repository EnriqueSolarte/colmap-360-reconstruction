Date: 2026-06-15
Project version: v0.1.0
Description: Record the accepted continuation design for panorama reconstruction.

# Incremental Reconstruction

## Goal

Add an option to `colmap-panorama-reconstruct` that can start from a previous
`colmap-panorama-reconstruct` session. That previous session provides the
initial optimized map and camera set for the next reconstruction run.

Example starting map:

- Previous `colmap-panorama-reconstruct` session:
  `/data/itri-3d-reconstruction/scene-reconstruction/2026-06-08-0001/colmap/sparse_best`

## Desired Workflow

The long-term workflow should support incremental scene growth:

1. Create a new COLMAP session, or start from a pre-optimized one.
2. Add more panorama views to that scene.
3. Optimize and save the new scene.
4. Later, reload that optimized scene, add more views, and optimize again.

## Design Decision

The input seed must come exclusively from a previous
`colmap-panorama-reconstruct` session, not from arbitrary COLMAP, VSLAM, or
third-party exports.

The previous session's optimized map should be treated as the initial camera
set for the next reconstruction run. Its existing cameras and intrinsics may be
refined during optimization, so the seed model is not considered fixed.

The output should be a new merged COLMAP model directory containing both the
initial map and the newly added views. This should be written as a new set of
COLMAP model files, such as `*.bin`, rather than treated as a separate model to
merge later.

## Implementation Rule

Follow `rule-of-implementation.md`: do not implement this in one pass. First
define the stages and tradeoffs, discuss the API and data assumptions, and only
then implement the agreed slice.

## Current Implementation

`colmap-panorama-reconstruct` now accepts `--initial-session`, constrained to a
previous workspace produced by the same command, or that workspace's
`sparse_best` directory. This constraint is intentional: the continuation logic
depends on the panorama rig metadata, copied database, copied virtual images,
masks, and compatible render settings from the prior session. Supporting
arbitrary COLMAP or VSLAM models would require different validation and model
normalization rules.

A seeded run creates a new workspace and copies the previous session state into
it. New panorama inputs are rendered into the same virtual camera folder layout,
but their filenames are prefixed with the new input folder name. This prevents
basename collisions while preserving the immediate `pano_cameraXX/` prefix that
COLMAP uses for camera grouping.

The seed model is preserved as the initial reconstruction, including its
existing rig and frame topology. After feature extraction, the temporary cameras,
singleton rigs, and singleton frames that COLMAP creates for the new images are
removed from the copied database. The new images are reassigned to the seed
camera IDs and inserted as one new multi-camera panorama frame in the existing
rig. This keeps the database, the input model, and COLMAP's rig verification in
agreement.

Mapping then runs with the copied `sparse_best` model as `input_path`, with a
single continuation model, fixed existing frames, fixed virtual intrinsics, and
bundle adjustment still allowed to refine poses and sparse points. The output is
a new merged reconstruction containing the prior registered frames plus the new
views; it is not a separate model to merge later.

The rationale is to treat the previous session as the authoritative camera and
rig scaffold while allowing COLMAP to extend and optimize the scene. The main
failure avoided by this design is an inconsistent frame graph: rebuilding or
globally reapplying rig configuration can leave frames referencing images that
the returned reconstruction no longer contains.

Remaining risks are narrow but real. The implementation assumes the previous
session database and `sparse_best` model are mutually consistent, and that the
current render settings match the previous rig. It also assumes COLMAP's
database schema for rigs, frames, and frame data remains compatible with the
pycolmap version in use. These assumptions are acceptable for the constrained
workflow, but they are the reason the input is deliberately limited to previous
`colmap-panorama-reconstruct` sessions.
