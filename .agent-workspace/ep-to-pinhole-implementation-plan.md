Date: 2026-06-03
Project version: v0.0.0-unversioned
Description: Minimal implementation plan for EP-to-pinhole rendering

# Goal

Implement a small Python library that reads equirectangular projection images and renders pinhole views at requested orientations.

# Scope

This is a small utility for a larger reconstruction project. Keep the first version focused on correctness, simple APIs, and practical testing.

Inputs:
- EP image path or loaded image array.
- Output size `height`, `width`.
- Orientation `longitude`, `latitude` in radians.
- Pinhole field of view `fov_x`, `fov_y` in radians.
- Optional interpolation mode.

Output:
- Rendered pinhole image array.

# Decisions

- Use `numpy` for projection-map construction.
- Start with `cv2.remap` as the first rendering backend.
- Keep roll fixed at zero in v0.
- Use radians everywhere.
- Use camera frame: `+z` forward, `+x` right, `+y` downward.
- Longitude/yaw range is `[-pi, pi)`, with `0` at the horizontal center of the EP image.
- Latitude/pitch range is `[-pi/2, pi/2]`, positive upward.
- Add a Torch backend only after the OpenCV version is correct and benchmarks show a real need.

# Implementation Steps

1. Create a minimal Python package layout.
   - `src/ep_pinhole/`
   - `tests/`
   - basic package config

2. Implement image loading.
   - Read EP images from disk.
   - Preserve color channels clearly.
   - Avoid dataset-specific assumptions in the library.

3. Implement projection-map generation.
   - Build pinhole camera rays from `height`, `width`, `fov_x`, `fov_y`.
   - Rotate rays using `longitude` and `latitude`.
   - Convert rays to EP pixel coordinates.
   - Handle horizontal wrapping.

4. Implement rendering with OpenCV.
   - Use `cv2.remap`.
   - Support bilinear interpolation first.
   - Return a normal image array.

5. Add synthetic system tests.
   - Generate synthetic EP grid/checker images with known longitude/latitude content.
   - Verify center view, yaw offsets, pitch offsets, FOV behavior, and horizontal wrapping.

6. Add a small command-line smoke test.
   - Input EP image path.
   - Output pinhole image path.
   - Parameters: `height`, `width`, `longitude`, `latitude`, `fov_x`, `fov_y`.

7. Benchmark only after correctness tests pass.
   - Measure OpenCV render time on representative image sizes.
   - Decide later whether a Torch backend is justified.

# Acceptance Criteria

- A user can render one pinhole image from one EP image with explicit size, orientation, and FOV.
- Synthetic tests pass for orientation, FOV, and wrapping.
- The first implementation has no custom CUDA/OpenGL code.
- The API does not depend on the ITRI dataset layout.

# Not In Scope For v0

- Roll support.
- Camera calibration estimation.
- AprilTag detection.
- GPU/Torch backend.
- Real-time viewer.
- Dataset ingestion pipeline.
