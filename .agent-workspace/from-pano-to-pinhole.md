Date: 2026-06-04
Project version: v0.1.0
Description: Current EP-to-pinhole status and missing reconstruction work

# Current One-Minute Status

The EP-to-pinhole renderer is implemented and tested. The active backend is
`py360convert`, wrapped by the local `ep_pinhole` API so callers use radians,
project naming, and stable defaults.

The COLMAP input inspection stage is implemented. The current pinhole dataset
has been inspected successfully:

```text
images: 12124
resolution: 1280x960
camera model: PINHOLE
fx/fy/cx/cy: 914.0147247,914.006024013,640,480
```

A 100-image exhaustive COLMAP smoke reconstruction succeeded:

```text
registered images: 95/100
sparse points: 14234
mean reprojection error: 0.72572 px
```

This proves the basic pipeline can work, but it does not prove the full 12k
image reconstruction is ready.

# Current Missing Points

1. Full dataset reconstruction is not done.
   The project has not yet reconstructed all `12124` pinhole images.

2. Scalable matching is not decided.
   Exhaustive matching worked well for 100 images, but it is not a realistic
   full-dataset strategy. Sequential matching registered only `3/100` images
   in the stored smoke result, so the next work must tune or replace the
   matching strategy before a full run.

3. Render provenance is still an assumption.
   The COLMAP intrinsics assume all pinhole images were rendered with
   `1280x960`, `fov_x=1.221730476`, and `fov_y=0.967145867`. No source metadata
   has confirmed this yet.

4. Metric scale is not implemented.
   The current COLMAP model is raw monocular geometry in arbitrary scale and
   frame. Do not treat it as metric output until an external scale source is
   chosen.

5. Visualization is only partially validated.
   Summary mode is tested. Full Open3D visualization still depends on a working
   GUI/OpenGL environment.

# Next Step

Run a medium smoke reconstruction before attempting the full dataset:

```bash
colmap-reconstruct \
  --image-dir /data/itri-3d-reconstruction/itri-3d-reconstruction/pinhole_1780034936312 \
  --workspace /workspace/data/colmap-medium-sequential \
  --image-limit 300 \
  --matching sequential \
  --overlap 30 \
  --overwrite
```

If registration is still poor, do not increase to 12k images. Instead, test a
better scalable matching strategy or split the sequence into overlapping
chunks.

# Original Task Description

I’ll have access to a directory of images defined in equirectangular projection (EP).
I need to develop a small Python library that efficiently reads an EP image and enables me to render a pinhole image 
(with specified HxW dimensions) at a given orientation (latitude and longitude coordinates of the EP).

Let’s discuss the options for this project. Please be extremely critical and brutally honest in your suggestions. 
a given directory will be somthing like: /media/v1-data/parking-lot-reconstruction/itri-3d-reconstruction/1780034936312

# Original Discussion
-> Define missing camera parameters. HxW + lat/lon is not enough; you also need FOV or focal length, camera convention, and probably roll.
ANSWER: Define `fov_x` and `fov_y` as explicit hyperparameters of the pinhole render. Output resolution `HxW` is user-specified, while the EP image resolution only limits the maximum useful sampling detail. For v0, set roll angle to zero by default.

-> Define coordinate conventions explicitly: longitude/yaw direction, latitude/pitch sign, radians vs degrees, image origin, wrap/clamp behavior.
ANSWER: longitude/yaw direction defined from -pi to pi with 0 at the middle of the W EP. latitude/pitch sign: up positive and down negative. Camera frame +z axis principal point, to the right +x, downards +y.

-> Yes, let's use torch.grid_sample only if you need GPU batches or downstream PyTorch integration. We can process many images at once + I think is faster but we could start from cv2 version. 

-> Add validation data: synthetic equirectangular grid/checker images with known lat/lon labels. Real parking-lot images are not enough to catch projection bugs. 
ANSWER: Treat this as part of the implementation system test. First use synthetic EP validation images with exact known longitude/latitude content to verify the projection math, FOV behavior, wrapping, and orientation conventions. Then add real AprilTag/checkerboard captures as downstream real-data validation after the synthetic tests pass.
