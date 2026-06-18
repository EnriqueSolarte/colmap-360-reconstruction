# colmap-360-reconstruction

Run a raw COLMAP sparse reconstruction from equirectangular panoramas by
rendering fixed virtual pinhole views and registering them as a shared-center
rig.

The primary 360-image workflow is:

1. Render twelve fixed virtual pinhole views from each equirectangular image.
2. Assign every panorama direction to one virtual camera with feature masks.
3. Force the virtual cameras into a known, shared-center COLMAP rig.
4. Match panorama frames with rig verification and loop detection.
5. Reconstruct, summarize, or visualize the raw sparse model.

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

## Renderer Defaults

The standalone renderer helper constants are:

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

## Python API

```python
from render_pinhole import EquirectangularToPinhole

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

Continue from a previous `colmap-panorama-reconstruct` session by passing the
previous workspace, or its `sparse_best` model directory, as the initial map:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/new-panoramas \
  --workspace /workspace/data/colmap-panorama-continued \
  --initial-session /workspace/data/colmap-panorama-rig \
  --matcher sequential \
  --overlap 15
```

The initial session must be a prior panorama-rig workspace from this command,
not an arbitrary COLMAP or VSLAM export. The new input directory should contain
additional panoramas. During seeded runs, newly generated virtual images are
namespaced by the input folder name, such as `img-0001_shelf.png`, so repeated
camera filenames from different capture folders can coexist. The current
panorama-rig pipeline keeps virtual camera intrinsics fixed while COLMAP
continues optimizing poses and sparse points from the previous model.
Previous virtual images and masks are hardlinked into the new workspace by
default to avoid duplicating large files. If hardlinks are not supported between
the two workspace locations, the command falls back to normal copies. The
database is always copied because continuation runs mutate it.

Loop detection may download COLMAP's default vocabulary tree. To avoid that
download and make runs reproducible, pass a local tree with
`--vocab-tree-path /path/to/vocab_tree.bin`. The command automatically points
COLMAP/libcurl at `/etc/ssl/certs/ca-certificates.crt` when available; override
it with `--ssl-cert-file` when needed.

The workspace includes `panorama_rig_config.json` and
`panorama_rig_report.json`. The report records per-camera mask coverage,
feature health, matching settings, registered panorama frames, and sparse
model statistics.

## Manually Select Pinhole Render Directions

After a panorama-rig reconstruction exists, you can manually choose
longitude/latitude directions from the registered panoramas and render those
directions as pinhole images. This is useful when you want a curated set of
perspective views for Gaussian Splatting or other pose-aware training
pipelines.

First initialize a manual render workspace. The `--scene-workspace` decides
which panoramas are shown: only panorama frames registered in that COLMAP scene
are included. The `--pano-dir` is used to resolve those scene frame names back
to the original equirectangular image files.

```bash
manual-pinhole-render \
  --pano-dir /path/to/source-panoramas \
  --scene-workspace /path/to/panorama-rig-workspace \
  --output-dir /path/to/panorama-rig-workspace/pinhole-renders
```

To skip manual point-by-point selection and initialize a regular grid over
every registered panorama, add `--auto-select`. This samples a 45-degree
longitude/latitude grid inside the full equirectangular image, producing 32
pending pinhole views per panorama.

```bash
manual-pinhole-render \
  --pano-dir /path/to/source-panoramas \
  --scene-workspace /path/to/panorama-rig-workspace \
  --output-dir /path/to/panorama-rig-workspace/pinhole-renders \
  --auto-select
```

Then open the local browser selector:

```bash
manual-pinhole-select \
  --output-dir /path/to/panorama-rig-workspace/pinhole-renders \
  --host 127.0.0.1 \
  --port 8766
```

Open the printed URL, for example `http://127.0.0.1:8766`.

The panorama image is an equirectangular view:

- Horizontal position maps to longitude.
- The left edge is `-180` degrees, the center is `0` degrees, and the right
  edge is `+180` degrees.
- Vertical position maps to latitude.
- The top edge is `+90` degrees, the center is `0` degrees, and the bottom edge
  is `-90` degrees.

Basic selection behavior:

- Move the cursor over the panorama to inspect the full-width/full-height
  crosshair location.
- Click once to render only a low-resolution preview of that longitude/latitude
  direction.
- Press `Cmd` or `Ctrl` while clicking to save the direction as a selected
  pinhole view. If `Cmd-click save only` is disabled, the preview is rendered
  too.
- Use `Preview` to re-render the current numeric longitude/latitude fields.
- Use `Save` to save the current numeric longitude/latitude fields.
- Use `Check` beside a saved view to reload that saved direction and preview it.
- Use `Delete`, `Undo`, `Reset Pano`, and `Reset Collection` to manage saved
  selections.

For faster batch selection, enable `ROI sampling`, draw a box from the upper
left to the lower right of the region you want, and set `Delta deg`. The tool
samples a regular longitude/latitude grid inside that box using the chosen
angular spacing. Smaller deltas create more views; larger deltas create fewer
views.

Good manual selections usually follow these rules:

- Prefer textured, well-lit, non-blurry areas with useful geometry.
- Avoid crops dominated by blank walls, floor, ceiling, reflective surfaces, or
  heavy motion blur.
- Avoid latitudes very close to `+90` or `-90` degrees unless you explicitly
  need pole-facing views.
- Keep the FOV and resolution consistent unless you have a specific downstream
  reason to vary intrinsics.
- Use the per-panorama COLMAP quality fields as hints: panoramas with many
  triangulated observations and low reprojection error are usually safer
  pose anchors.

Saved selections are stored in:

```text
pinhole-renders/metadata.json
pinhole-renders/selections.json
```

When you are done selecting, render the final pinhole images:

```bash
manual-pinhole-render \
  --output-dir /path/to/panorama-rig-workspace/pinhole-renders \
  --render-selections
```

To also export a COLMAP text model for Gaussian Splatting training:

```bash
manual-pinhole-render \
  --output-dir /path/to/panorama-rig-workspace/pinhole-renders \
  --render-selections \
  --export-gaussian-splatting-colmap
```

The export writes:

```text
pinhole-renders/images/
pinhole-renders/sparse/0/cameras.txt
pinhole-renders/sparse/0/images.txt
pinhole-renders/sparse/0/points3D.txt
pinhole-renders/gaussian_splatting_export.json
```

## Visualize Or Summarize COLMAP Output

Summary mode works without Open3D:

```bash
colmap-visualize \
  --model /workspace/data/colmap-panorama-rig/sparse_best \
  --summary-only
```

Interactive visualization requires Open3D and GUI/OpenGL support:

```bash
colmap-visualize \
  --model /workspace/data/colmap-panorama-rig/sparse_best
```

For panorama-rig workspaces, the visualizer automatically draws each panorama
frame as a small surface sphere with a local RGB arrow coordinate frame instead
of drawing every virtual pinhole camera frustum. The sphere radius is one
quarter of the local frame size. The local frame is centered at the panorama
view center: +x points right, +y points down, and +z points through the
panorama center direction. You can override this if needed:

```bash
colmap-visualize \
  --model /workspace/data/colmap-panorama-rig/sparse_best \
  --camera-style sphere
```

To save a preview image:

```bash
colmap-visualize \
  --model /workspace/data/colmap-panorama-rig/sparse_best \
  --save-image /workspace/data/colmap-panorama-rig/map-preview.png \
  --summary-only
```

In headless containers this may fail if Open3D cannot create a rendering
window. Summary mode remains usable in that environment.

## Utility Scripts

Create a sampled directory of image symlinks:

```bash
tools-sampling \
  --source /path/to/images \
  --destination /path/to/sampled-images \
  --skip 4 \
  --idx 0
```

Extract video frames through the bundled shell wrapper:

```bash
tools-proc-video /path/to/video.mp4 5
```

## Important Limits

- COLMAP output is raw monocular geometry in arbitrary scale and frame.
- Metric scale recovery is not implemented for the current dataset.
- `data/` is ignored by git because COLMAP workspaces are generated artifacts.

## Run Tests

```bash
pytest -q
```
