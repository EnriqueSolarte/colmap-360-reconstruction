# EP Pinhole

Render perspective pinhole images from equirectangular panoramas and run a raw
COLMAP sparse reconstruction on the generated pinholes.

The primary 360-image workflow is:

1. Render twelve fixed virtual pinhole views from each equirectangular image.
2. Assign every panorama direction to one virtual camera with feature masks.
3. Force the virtual cameras into a known, shared-center COLMAP rig.
4. Match panorama frames with rig verification and loop detection.
5. Reconstruct, summarize, or visualize the raw sparse model.

The generic `colmap-reconstruct` workflow remains available for ordinary
perspective-image folders and experiments with independently rendered views.

The COLMAP output is monocular and unscaled. Do not treat it as metric geometry
until it has been aligned with an external scale source.

## Install

From the repository root:

```bash
python -m pip install -e .
```

For development tests:

```bash
python -m pip install -e '.[test]'
pytest -q
```

For COLMAP reconstruction support:

```bash
python -m pip install -e '.[colmap]'
python -c "import pycolmap; print(pycolmap.has_cuda)"
```

The final command must print `True`. Installing the CUDA wheel alongside the
ordinary `pycolmap` wheel is not sufficient because both packages provide the
same Python module.

For Open3D visualization support:

```bash
python -m pip install -e '.[visualize]'
```

You can combine extras:

```bash
python -m pip install -e '.[test,colmap,visualize]'
```

## Defaults

The reconstruction-oriented pinhole defaults are shared by the renderer and
COLMAP helper:

```text
width: 1280
height: 960
fov_x: 1.221730476 radians  # 70.0 degrees
fov_y: 0.967145867 radians  # 55.4 degrees
```

For these defaults, COLMAP uses:

```text
camera model: PINHOLE
fx: 914.0147247
fy: 914.006024013
cx: 640.0
cy: 480.0
```

These values assume all generated pinhole images share the same resolution and
field of view.

## Render One Pinhole Image

```bash
ep-pinhole-render \
  --input pano.png \
  --output pinhole.png \
  --longitude 0.0 \
  --latitude 0.0
```

Angles and fields of view are in radians.

Useful options:

```bash
ep-pinhole-render \
  --input pano.png \
  --output pinhole.png \
  --height 960 \
  --width 1280 \
  --longitude 0.0 \
  --latitude 0.0 \
  --fov-x 1.221730476 \
  --fov-y 0.967145867 \
  --interpolation bilinear
```

## Render A Directory

Render multiple random pinhole views from each equirectangular image:

```bash
ep-pinhole-render-directory \
  --input-dir /path/to/equirectangular_images \
  --output-dir /path/to/pinhole_images \
  --num-views 5 \
  --seed 123
```

The command writes the pinhole images and a `metadata.json` file in the output
directory. The metadata records the render size, FOV, longitude, latitude, and
source image for each output.

To render one fixed latitude for all views:

```bash
ep-pinhole-render-directory \
  --input-dir /path/to/equirectangular_images \
  --output-dir /path/to/pinhole_images \
  --num-views 5 \
  --latitude 0.0
```

To constrain view sampling:

```bash
ep-pinhole-render-directory \
  --input-dir /path/to/equirectangular_images \
  --output-dir /path/to/pinhole_images \
  --num-views 3 \
  --longitude-min -0.5 \
  --longitude-max 0.5 \
  --latitude-min -0.2 \
  --latitude-max 0.2 \
  --longitude-jitter 0.05 \
  --latitude-jitter 0.03 \
  --seed 123
```

## Python API

```python
from ep_pinhole import EquirectangularToPinhole

renderer = EquirectangularToPinhole.from_file("pano.png")
image = renderer.render(
    height=960,
    width=1280,
    longitude=0.0,
    latitude=0.0,
    fov_x=1.221730476,
    fov_y=0.967145867,
)
```

`image` is a NumPy array.

## Reconstruct Directly From 360 Panoramas

Use the maintained panorama-rig pipeline for equirectangular 360 images:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-panorama-rig \
  --matcher sequential \
  --overlap 15 \
  --overwrite
```

The default overlapping layout follows COLMAP's panorama example: twelve
fixed `PINHOLE` cameras cover the sphere, share a camera center, and retain
known relative rotations. Feature-ownership masks prevent duplicate features
in overlapping crops. SIFT matching uses rig verification, expands sequential
pairs across complete panorama frames, and enables loop detection.

Virtual-image rendering precomputes the twelve spherical remapping grids once,
reuses them for every panorama, remaps complete RGB images with OpenCV, and
encodes images concurrently. The default output is lossless PNG with fast
compression:

```bash
--virtual-image-format png \
--png-compression 1 \
--image-save-workers 12
```

For substantially faster and smaller, but lossy, virtual images:

```bash
--virtual-image-format jpeg \
--jpeg-quality 95
```

COLMAP feature extraction, matching, and bundle adjustment use CUDA
automatically when the installed pycolmap build supports it:

```bash
--device auto \
--gpu-index -1
```

To require CUDA and fail instead of falling back to CPU:

```bash
--device cuda \
--gpu-index 0
```

Multiple GPU indices distribute feature extraction and matching across GPUs.
Bundle adjustment requests the first selected GPU:

```bash
--device cuda \
--gpu-index 0,1,2,3
```

Disable GPU bundle adjustment while retaining GPU SIFT with
`--no-gpu-bundle-adjustment`. Some CUDA pycolmap wheels are built with a
CPU-only Ceres solver. In that case, COLMAP logs a warning and automatically
falls back to CPU bundle adjustment.

Start with a small smoke reconstruction before a full run:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-panorama-smoke \
  --image-limit 12 \
  --no-loop-detection \
  --overwrite
```

Loop detection may download COLMAP's default vocabulary tree. To avoid that
download and make runs reproducible, pass a local tree with
`--vocab-tree-path /path/to/vocab_tree.bin`. The command automatically points
COLMAP/libcurl at `/etc/ssl/certs/ca-certificates.crt` when available; override
it with `--ssl-cert-file` when needed.

The workspace includes `panorama_rig_config.json` and
`panorama_rig_report.json`. The report records per-camera mask coverage,
feature health, matching settings, registered panorama frames, and sparse
model statistics.

## Inspect Pinhole Inputs For COLMAP

Before running COLMAP, inspect the pinhole folder and write a report:

```bash
colmap-reconstruct \
  --inspect-only \
  --image-dir /path/to/pinhole_images \
  --workspace /workspace/data/colmap
```

This writes:

```text
/workspace/data/colmap/colmap_input_report.json
```

The report records:

- image count;
- first and last image names;
- validated image resolution;
- FOV assumptions;
- computed COLMAP intrinsics.

Inspection does not require `pycolmap`.

## Run A COLMAP Smoke Reconstruction

Start with a small subset before attempting a full reconstruction:

```bash
colmap-reconstruct \
  --image-dir /path/to/pinhole_images \
  --workspace /workspace/data/colmap-smoke \
  --image-limit 100 \
  --matching exhaustive \
  --overwrite
```

Outputs:

```text
database.db
sparse/
sparse_best/
cloud_colmap_raw.ply
validation_report.json
colmap_input_report.json
```

The output is raw and unscaled. The validation report contains registration
count, sparse point count, matching database stats, and reprojection error.

To let COLMAP refine the shared virtual camera intrinsics during mapping, add:

```bash
--optimize-intrinsics
```

This refines focal length and principal point, but all images still share one
`PINHOLE` camera. Extra distortion parameters are not enabled.

When `--optimize-intrinsics` is used without explicit `--width`, `--height`,
`--fov-x`, or `--fov-y`, the COLMAP helper derives the camera resolution from
the input images and uses a generic initial pinhole guess. This is useful for
real pinhole image folders that are not rendered with the project defaults, for
example `1920x1080` frames.

If you pass `--width` or `--height` with `--optimize-intrinsics`, pass both.

### Matching Strategy

`--matching sequential` is the default and is scalable, but it may be too sparse
if neighboring frames do not overlap well.

`--matching exhaustive` is useful for small smoke tests. It registered much
more strongly on the first 100-image sample during development, but it is not
appropriate for a 12k-image full run.

For a large dataset, use smoke tests to choose a scalable strategy before
running the full sequence. A full exhaustive match over thousands of images is
usually too expensive.

## Visualize Or Summarize COLMAP Output

Summary mode works without Open3D:

```bash
colmap-visualize \
  --model /workspace/data/colmap-smoke/sparse_best \
  --summary-only
```

Interactive visualization requires Open3D and GUI/OpenGL support:

```bash
colmap-visualize \
  --model /workspace/data/colmap-smoke/sparse_best
```

To save a preview image:

```bash
colmap-visualize \
  --model /workspace/data/colmap-smoke/sparse_best \
  --save-image /workspace/data/colmap-smoke/map-preview.png \
  --summary-only
```

In headless containers this may fail if Open3D cannot create a rendering
window. Summary mode remains usable in that environment.

## Current Validated Example

The current development dataset:

```text
/data/itri-3d-reconstruction/itri-3d-reconstruction/pinhole_1780034936312
```

was inspected successfully:

```text
images: 12124
resolution: 1280x960
camera params: 914.0147247,914.006024013,640,480
```

A 100-image exhaustive smoke reconstruction produced:

```text
registered images: 95/100
sparse points: 14234
mean reprojection error: 0.72572 px
```

This is a good smoke result, not a final full-dataset strategy.

## Important Limits

- COLMAP output is raw monocular geometry in arbitrary scale and frame.
- Metric scale recovery is not implemented for the current dataset.
- The imported ArUco scale-recovery script is not part of the active workflow.
- The full 12k-image dataset needs a scalable matching plan; do not run full
  exhaustive matching naively.
- `data/` is ignored by git because COLMAP workspaces are generated artifacts.

## Run Tests

```bash
pytest -q
```
