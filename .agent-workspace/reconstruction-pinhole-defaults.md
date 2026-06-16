Date: 2026-06-03
Project version: v0.1.0
Description: Recommended pinhole render defaults for reconstruction workflows

# Reconstruction Pinhole Defaults

## Recommended Defaults

Use these as the first reconstruction-oriented smoke-test defaults:

```text
width: 1280
height: 960
fov_x: 1.221730476 radians
fov_y: 0.967145867 radians
```

Equivalent degrees:

```text
fov_x: 70.0 degrees
fov_y: 55.4 degrees
```

Example CLI call:

```bash
ep-pinhole-render \
  --input pano.png \
  --output pinhole.png \
  --height 960 \
  --width 1280 \
  --longitude 0.0 \
  --latitude 0.0 \
  --fov-x 1.221730476 \
  --fov-y 0.967145867
```

## Resolution Tiers

Use consistent intrinsics across all generated virtual pinhole views unless there is a specific reconstruction reason not to.

```text
fast smoke:
  width=1024
  height=768
  fov_x=70.0 degrees / 1.221730476 radians
  fov_y=55.4 degrees / 0.967145867 radians

default reconstruction:
  width=1280
  height=960
  fov_x=70.0 degrees / 1.221730476 radians
  fov_y=55.4 degrees / 0.967145867 radians

higher quality if compute allows:
  width=1600
  height=1200
  fov_x=70.0 degrees / 1.221730476 radians
  fov_y=55.4 degrees / 0.967145867 radians
```

## Rationale

The generated pinhole images are intended for COLMAP, GLOMAP, Depth Anything 3, and Gaussian Splatting style workflows.

The default should therefore favor:

- A normal perspective camera rather than an extreme wide-angle crop.
- Enough resolution for feature extraction and depth cues.
- Manageable memory and runtime.
- Stable shared intrinsics across generated views.
- Square-pixel camera geometry.

`1280x960` with `fov_x=70 degrees` is a practical default. It is large enough for useful local features but still moderate enough for repeated view generation and downstream processing.

The `fov_y` value is derived from the horizontal FOV and `4:3` aspect ratio so the resulting camera has square pixels:

```text
fov_y = 2 * atan(tan(fov_x / 2) * height / width)
```

For `width=1280`, `height=960`, and `fov_x=70 degrees`, this gives approximately `55.4 degrees`.

Avoid defaulting to `90+ degree` horizontal FOV for reconstruction crops unless there is a clear need for extra coverage. Wider FOV captures more scene context but increases edge stretching, which can hurt feature matching, monocular depth quality, and Gaussian Splatting consistency.

## Source Notes

COLMAP camera model guidance:

- URL: https://colmap.readthedocs.io/en/latest/cameras.html
- Relevant point: COLMAP recommends simple camera models such as `SIMPLE_PINHOLE` or `PINHOLE` when images are already undistorted. It warns that too-complex camera models can cause poor intrinsic estimates.

COLMAP feature extraction and large image guidance:

- URL: https://colmap.github.io/faq.html?highlight=sift
- Relevant point: large images can consume excessive RAM during feature extraction, especially on CPU, and users may need to reduce maximum image size.

Nerfstudio Splatfacto / Gaussian Splatting guidance:

- URL: https://docs.nerf.studio/nerfology/methods/splat.html
- Relevant point: Gaussian splatting works better when initialized from existing geometry such as COLMAP SfM points. Nerfstudio also notes that Gaussian rasterization assumes perspective cameras.

Depth Anything 3 API:

- URL: https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md
- Relevant point: Depth Anything 3 can default to input resolution when no processing resolution is provided, so the rendered pinhole resolution directly matters.

GLOMAP camera data structure reference:

- URL: https://deepwiki.com/colmap/glomap/5.2-camera-image-and-track
- Relevant point: GLOMAP uses camera structures with model type, focal length, principal point, distortion parameters, and image dimensions. This reinforces keeping virtual camera dimensions and intrinsics consistent.

## Practical Rule

For first reconstruction experiments, generate all pinhole views with:

```text
1280x960
70 degree horizontal FOV
55.4 degree vertical FOV
same intrinsics for all views
PINHOLE or SIMPLE_PINHOLE downstream camera model
```

Then increase to `1600x1200` only if feature quality or final reconstruction detail is inadequate and compute allows it.
