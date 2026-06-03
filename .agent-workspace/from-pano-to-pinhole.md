Date: 2026-06-03
Project version: v0.0.0-unversioned
Description: Initial EP-to-pinhole library design decisions

# Task description

I’ll have access to a directory of images defined in equirectangular projection (EP).
I need to develop a small Python library that efficiently reads an EP image and enables me to render a pinhole image 
(with specified HxW dimensions) at a given orientation (latitude and longitude coordinates of the EP).

Let’s discuss the options for this project. Please be extremely critical and brutally honest in your suggestions. 
a given directory will be somthing like: /media/v1-data/parking-lot-reconstruction/itri-3d-reconstruction/1780034936312

# Discussion: 
-> Define missing camera parameters. HxW + lat/lon is not enough; you also need FOV or focal length, camera convention, and probably roll.
ANSWER: Define `fov_x` and `fov_y` as explicit hyperparameters of the pinhole render. Output resolution `HxW` is user-specified, while the EP image resolution only limits the maximum useful sampling detail. For v0, set roll angle to zero by default.

-> Define coordinate conventions explicitly: longitude/yaw direction, latitude/pitch sign, radians vs degrees, image origin, wrap/clamp behavior.
ANSWER: longitude/yaw direction defined from -pi to pi with 0 at the middle of the W EP. latitude/pitch sign: up positive and down negative. Camera frame +z axis principal point, to the right +x, downards +y.

-> Yes, let's use torch.grid_sample only if you need GPU batches or downstream PyTorch integration. We can process many images at once + I think is faster but we could start from cv2 version. 

-> Add validation data: synthetic equirectangular grid/checker images with known lat/lon labels. Real parking-lot images are not enough to catch projection bugs. 
ANSWER: Treat this as part of the implementation system test. First use synthetic EP validation images with exact known longitude/latitude content to verify the projection math, FOV behavior, wrapping, and orientation conventions. Then add real AprilTag/checkerboard captures as downstream real-data validation after the synthetic tests pass.