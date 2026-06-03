Date: 2026-06-03
Project version: v0.0.0-unversioned
Description: Use py360convert as the initial EP-to-pinhole backend

# EP-to-Pinhole Backend Decision

## Decision

Use `py360convert` as the initial backend for equirectangular-to-pinhole rendering.

The project should still expose its own small API so callers can use radians, project-specific naming, and stable behavior without depending directly on the upstream function signature.

## Primary Sources

- py360convert PyPI: https://pypi.org/project/py360convert/
- py360convert GitHub: https://github.com/sunset1995/py360convert

## Why py360convert

`py360convert` already implements the core operation needed for v0:

- Equirectangular image to perspective image conversion.
- NumPy array input and output.
- Configurable interpolation.
- Optional OpenCV acceleration when OpenCV is installed.
- MIT license.
- Maintained package with a recent PyPI release.

This avoids reimplementing projection math before the project has proven it needs custom behavior.

## Wrapper API Direction

Keep the local package API in radians:

```python
renderer.render(
    height=512,
    width=768,
    longitude=0.0,
    latitude=0.0,
    fov_x=1.2,
    fov_y=0.9,
)
```

Internally, convert to `py360convert.e2p(...)`, which expects degrees:

```python
py360convert.e2p(
    image,
    fov_deg=(degrees(fov_x), degrees(fov_y)),
    u_deg=degrees(longitude),
    v_deg=degrees(latitude),
    out_hw=(height, width),
    mode="bilinear",
)
```

## Implementation Impact

Update the v0 implementation order:

1. Create the minimal Python package layout.
2. Add `py360convert` as the rendering dependency.
3. Implement the renderer class and convenience function as a thin wrapper.
4. Add validation around image shape, output size, orientation, FOV, and interpolation mode.
5. Add synthetic tests to confirm our coordinate conventions match the project decisions.
6. Add the CLI smoke test.

## Alternatives Considered

OmniCV also supports equirectangular-to-perspective conversion, but it appears broader and less focused for this minimal v0 wrapper.

Torch/GPU camera libraries may become useful later if the project needs differentiable rendering, batching, or GPU acceleration. They are out of scope for v0.

## Open Checks

Before relying on `py360convert` for all tests, confirm:

- Its yaw and pitch sign conventions match our decisions, or document and compensate in the wrapper.
- Its FOV tuple ordering matches our `fov_x`, `fov_y` naming.
- Its interpolation mode names map cleanly from our public API.
