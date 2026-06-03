Date: 2026-06-03
Project version: v0.0.0-unversioned
Description: Implement EP-to-pinhole in agreed slices before code changes

# Rule Of Implementation

For the EP-to-pinhole library, do not implement the whole plan in one pass.

Before each implementation slice:

1. Identify the next narrow target from `ep-to-pinhole-implementation-plan.md`.
2. Propose the class, function, module, and test shape for that target.
3. Discuss tradeoffs with the user until the API and behavior are agreed.
4. Execute only the agreed slice.
5. Run the smallest useful verification for that slice.
6. Report what changed, what was verified, and what remains.

For this project, prefer:

- Simple, explicit Python APIs.
- Radians for all orientation and field-of-view parameters.
- `numpy` for projection maps.
- `cv2.remap` as the first rendering backend.
- No dataset-specific assumptions in library code.
- Tests built from synthetic equirectangular images before real dataset work.

The first implementation discussion should define the library package, the renderer class, and the main render function before creating source code.
