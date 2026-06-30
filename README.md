# colmap-360-reconstruction

Run a raw COLMAP sparse reconstruction from equirectangular panoramas by
using COLMAP's native `EQUIRECTANGULAR` camera model directly.

The primary 360-image workflow is:

1. Hard-link or symlink source panoramas into the COLMAP workspace.
2. Extract SIFT features with one shared native equirectangular camera.
3. Match panorama images with sequential, exhaustive, or vocabulary-tree matching.
4. Reconstruct, summarize, or visualize the raw sparse model.

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

Use the maintained native equirectangular pipeline for 360 images:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-equirectangular \
  --matcher sequential \
  --overlap 15 \
  --overwrite
```

The command imports each source panorama into the COLMAP workspace with a hard
link when possible and a symbolic link fallback across filesystems. It then
uses COLMAP's native `EQUIRECTANGULAR` camera model with one shared camera,
`CameraMode.SINGLE`, and fixed equirectangular camera parameters derived from
the panorama resolution. No virtual pinhole views, masks, rig configuration,
or render-encoding arguments are used.

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
  --workspace /workspace/data/colmap-equirectangular-smoke \
  --image-limit 12 \
  --no-loop-detection \
  --overwrite
```

Loop detection may download COLMAP's default vocabulary tree. To avoid that
download and make runs reproducible, pass a local tree with
`--vocab-tree-path /path/to/vocab_tree.bin`. The command automatically points
COLMAP/libcurl at `/etc/ssl/certs/ca-certificates.crt` when available; override
it with `--ssl-cert-file` when needed.

For larger ordered panorama captures, keyframe matching can reduce the pair
list before COLMAP matching. The sequential-vocabtree selector clusters images
in capture order using vocabulary-tree candidate pairs, selects a keyframe per
cluster, and writes an imported pair list:

```bash
colmap-panorama-reconstruct \
  --input-image-path /path/to/panoramas \
  --workspace /workspace/data/colmap-equirectangular-keyframes \
  --matcher keyframes \
  --keyframe-selection sequential-vocabtree \
  --vocab-tree-path /path/to/vocab_tree.bin \
  --keyframe-backbone-policy gaps \
  --keyframe-backbone-overlap 1 \
  --overwrite
```

For deterministic testing without a vocabulary tree, use interval keyframes:

```bash
--matcher keyframes \
--keyframe-selection interval
```

To run only the keyframe clustering/pair-planning stage after image import and
feature extraction, use the standalone cluster command:

```bash
tools-download-colmap-vocab-tree \
  --variant 256k \
  --destination-dir /workspace/data/colmap-vocab-tree

colmap-panorama-clusters \
  --workspace /workspace/data/colmap-equirectangular-keyframes \
  --vocab-tree-path /workspace/data/colmap-vocab-tree/vocab_tree_flickr100K_words256K.bin \
  --keyframe-selection sequential-vocabtree
```

This writes `match_pairs.txt` for COLMAP matching and
`panorama_keyframe_cluster_report.json` with the selected keyframes, clusters,
and pair counts.

The workspace includes `panorama_equirectangular_config.json` and
`panorama_equirectangular_report.json`. The metadata marks `camera_setup` as
`native_equirectangular`, and the report records feature health, matching
settings, registered panorama images, and sparse model statistics.

## Manually Select Pinhole Render Directions

After an equirectangular panorama reconstruction exists, you can manually choose
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
  --scene-workspace /path/to/equirectangular-workspace \
  --output-dir /path/to/equirectangular-workspace/pinhole-renders
```

To skip manual point-by-point selection and initialize a regular grid over
every registered panorama, add `--auto-select`. This samples a 45-degree
longitude/latitude grid inside the full equirectangular image, producing 32
pending pinhole views per panorama.

```bash
manual-pinhole-render \
  --pano-dir /path/to/source-panoramas \
  --scene-workspace /path/to/equirectangular-workspace \
  --output-dir /path/to/equirectangular-workspace/pinhole-renders \
  --auto-select
```

Then open the local browser selector:

```bash
manual-pinhole-select \
  --output-dir /path/to/equirectangular-workspace/pinhole-renders \
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
- When `Cmd-click save only` is enabled, press `Shift` with `Cmd` or `Ctrl`
  while clicking to save and immediately advance to the next panorama.
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
  --output-dir /path/to/equirectangular-workspace/pinhole-renders \
  --render-selections
```

To also export a COLMAP text model for Gaussian Splatting training:

```bash
manual-pinhole-render \
  --output-dir /path/to/equirectangular-workspace/pinhole-renders \
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

### Example With An Existing Optimized Scene

Use this workflow when you already have:

- an optimized equirectangular panorama COLMAP model, for example:

```text
/data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/colmap/1781765016983-1781764929431-1781765089651-1781765215596-1781765283163-1781765391913-1781765475779-1781765557512/sparse_best
```

- the original panorama images, for example:

```text
/data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/pano-raw-views/all-merged
```

The COLMAP scene is the authority for which panorama frames are usable. The
raw panorama directory may contain extra images; those are only used to resolve
source files for the registered scene panoramas.

Choose an output directory for the selected pinhole workflow:

```text
/data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/renders-pinhole
```

Initialize the selection workspace:

```bash
manual-pinhole-render \
  --pano-dir /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/pano-raw-views/all-merged \
  --scene-workspace /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/colmap/1781765016983-1781764929431-1781765089651-1781765215596-1781765283163-1781765391913-1781765475779-1781765557512/sparse_best \
  --output-dir /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/renders-pinhole
```

Open the selector:

```bash
manual-pinhole-select \
  --output-dir /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/renders-pinhole \
  --host 127.0.0.1 \
  --port 8766
```

Select views in the browser, then stop the selector server before rendering the
final set. Avoid editing selections in the browser while the render/export
command is running.

Render the selected pinhole images and export a COLMAP text model for Gaussian
Splatting:

```bash
manual-pinhole-render \
  --output-dir /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/renders-pinhole \
  --render-selections \
  --scene-workspace /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/colmap/1781765016983-1781764929431-1781765089651-1781765215596-1781765283163-1781765391913-1781765475779-1781765557512/sparse_best \
  --export-gaussian-splatting-colmap
```

Passing `--scene-workspace` again during render/export is intentional and safe.
It ensures the exporter has the optimized panorama poses and the source COLMAP
`points3D`, even if the selection metadata was originally initialized without
scene pose information.

For a fully automatic baseline selection instead of browser selection, add
`--auto-select` during initialization:

```bash
manual-pinhole-render \
  --pano-dir /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/pano-raw-views/all-merged \
  --scene-workspace /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/colmap/1781765016983-1781764929431-1781765089651-1781765215596-1781765283163-1781765391913-1781765475779-1781765557512/sparse_best \
  --output-dir /data/itri-3d-reconstruction/scene-reconstruction/q200-lab-2026-06-18/renders-pinhole \
  --auto-select
```

`--auto-select` creates 32 pending views per registered panorama using a
45-degree longitude/latitude grid. It is useful for a baseline dataset, but a
manual or ROI-curated set is usually better when you want to avoid blank walls,
blurred views, ceiling/floor-heavy crops, or weakly textured areas.

## Visualize Or Summarize COLMAP Output

Summary mode works without Open3D:

```bash
colmap-visualize \
  --model /workspace/data/colmap-equirectangular/sparse_best \
  --summary-only
```

Interactive visualization requires Open3D and GUI/OpenGL support:

```bash
colmap-visualize \
  --model /workspace/data/colmap-equirectangular/sparse_best
```

For equirectangular workspaces, the visualizer automatically draws each
panorama as a small surface sphere with a local RGB arrow coordinate frame
instead of drawing a pinhole camera frustum. The sphere radius is one
quarter of the local frame size. The local frame is centered at the panorama
view center: +x points right, +y points down, and +z points through the
panorama center direction. You can override this if needed:

```bash
colmap-visualize \
  --model /workspace/data/colmap-equirectangular/sparse_best \
  --camera-style sphere
```

To save a preview image:

```bash
colmap-visualize \
  --model /workspace/data/colmap-equirectangular/sparse_best \
  --save-image /workspace/data/colmap-equirectangular/map-preview.png \
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
