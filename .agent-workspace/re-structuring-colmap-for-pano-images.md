Date: 2026-06-05
Project version: v0.1.0
Description: Direction for restructuring the panorama reconstruction pipeline around COLMAP's recommended virtual-camera rig approach

# Restructuring COLMAP For Panorama Images

## Primary Direction

Follow COLMAP's officially recommended virtual-pinhole-rig approach for
spherical images. Use established COLMAP functionality and proven approaches
from other reliable sources. Do not invent a new reconstruction method when
the current COLMAP pipeline already provides an appropriate solution.

The improved pipeline should build on the successful behavior of
`legacy_pano_reconstruction.py`:

- Twelve fixed viewing directions per panorama.
- Known rigid rotations between virtual cameras.
- One shared camera center per panorama.
- Full-sphere visual coverage.
- Correct feature-ownership masks.
- Matching that understands panorama frames.
- Healthy temporal sampling.
- Loop detection.

The virtual cameras must be forced into a fixed rig structure so COLMAP does
not treat the twelve crops from one panorama as unrelated cameras.

## Implementation Priority

Do not spend time benchmarking how much the generic and rig-based
implementations differ numerically. The objective is to implement the strongest
practical panorama reconstruction pipeline, not to compare the two existing
implementations.

Start directly from Phase 1:

1. Make the maintained panorama reconstruction pipeline correct.
2. Use the legacy implementation as the behavioral reference.
3. Follow COLMAP's recommended rig-based reconstruction approach.
4. Validate that the maintained implementation preserves the important
   successful behavior of the legacy pipeline.

## Feature Strategy

Use SIFT features for the current implementation.

Structure the feature-extraction and matching code so that descriptors and
matchers can be replaced later. Future candidates may include:

- SIFT with LightGlue.
- ALIKED with LightGlue.
- Other improved feature descriptors and matchers supported by the selected
  reconstruction stack.

Do not implement or benchmark these alternatives yet.

## Deferred Work

The following work is not necessary for the current implementation:

- Panorama-specific artifact handling.
- Calibration and bundle-adjustment experiments beyond the fixed-rig pipeline.
- Metric scale and external pose integration.
- Dense reconstruction.

These topics may be revisited after the core sparse panorama-rig pipeline works
reliably.

## Confirmed Clarification

The original Note 2 ended with:

> We should highly consider force teh rig structure in teh asame manner as the
> legacy code that COLMAP

Confirmed meaning:

> Force the virtual-camera rig structure in the same manner as the legacy code
> and COLMAP's recommended panorama-rig approach.

This interpretation was confirmed when implementation of the plan was approved.
