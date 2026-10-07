---
title: "axelera.runtime.op.img_preprocess"
---
# `axelera.runtime.op.img_preprocess`


Image preprocessing operators: Perspective and BarrelCorrect.

## Summary

| Name | Description |
|------|-------------|
| [Perspective](#perspective) | Apply perspective transformation to an image using a 3x3 homography matrix. |
| [BarrelCorrect](#barrelcorrect) | Correct barrel (radial + tangential) lens distortion using the Brown-Conrady model. |

---

### Perspective

**Alias:** `perspective`

Apply perspective transformation to an image using a 3x3 homography matrix.

Takes an image and applies a perspective warp using the provided transformation matrix.
The transformation uses backward mapping where the matrix maps output coordinates to
input coordinates, which is the standard convention in image processing libraries
like OpenCV's warpPerspective.

**Args:**

- **matrix**: 3x3 homography matrix as a list/tuple of 9 floats in row-major order: [m00, m01, m02, m10, m11, m12, m20, m21, m22] For backward mapping: input_coord = matrix @ output_coord
- **width**: Output width in pixels.
- **height**: Output height in pixels.
- **dst_format**: Output color format (default: None, keeps input format).

**Examples:**

```python
# Identity transformation (no change)
identity = [1.0, 0.0, 0.0,
           0.0, 1.0, 0.0,
           0.0, 0.0, 1.0]
op.perspective(matrix=identity, width=640, height=480)

# Translation: shift image by (10, 20) pixels
# Note: backward mapping, so use positive offsets to shift content left/up
translation = [1.0, 0.0, 10.0,
              0.0, 1.0, 20.0,
              0.0, 0.0, 1.0]
op.perspective(matrix=translation, width=640, height=480)

# Scale: zoom out by 2x
scale = [0.5, 0.0, 0.0,
        0.0, 0.5, 0.0,
        0.0, 0.0, 1.0]
op.perspective(matrix=scale, width=640, height=480)

# Pipeline with perspective correction
op.seq(
    op.perspective(matrix=homography, width=640, height=480),
    op.letterbox(640, 640),
    op.totensor(),
)
```

**Note:**

- The matrix uses backward mapping (output -> input coordinates)
- Bilinear interpolation is used for subpixel sampling
- Out-of-bounds pixels are filled with 0 (black)
- Uses OpenCL acceleration for RGB/BGR 3-channel images
- Falls back to CPU (OpenCV) for other formats or if OpenCL fails
- Both OpenCL and OpenCV paths produce consistent results (within interpolation precision)

**Constructor:**

```python
__init__(matrix: list[float] | tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), width: int = 640, height: int = 480, dst_format: ColorFormat | str | None = None)
```

---

### BarrelCorrect

**Alias:** `barrel_correct`

Correct barrel (radial + tangential) lens distortion using the Brown-Conrady model.

Takes an image and undistorts it using the provided camera intrinsics and distortion
coefficients, using backward mapping where for each output pixel the corresponding
source location in the distorted input is computed.

**Args:**

- **camera_props**: Original camera intrinsics as [fx, fy, cx, cy] in pixel space.
- **distort_coefs**: Distortion coefficients as [k1, k2, p1, p2, k3].
- **new_camera_props**: Intrinsics for the undistorted view as [fx, fy, cx, cy]. Defaults to camera_props (identity rectification).
- **width**: Output width in pixels (default: same as input).
- **height**: Output height in pixels (default: same as input).     dst_format: Output color format (default: None, keeps input format).

**Examples:**

```python

# Typical fisheye / wide-angle lens correction
camera_props = [800.0, 800.0, 640.0, 360.0]   # fx, fy, cx, cy
distort_coefs = [-0.3, 0.1, 0.0, 0.0, -0.02]  # k1, k2, p1, p2, k3
op.barrel_correct(
    camera_props=camera_props,
    distort_coefs=distort_coefs,
)

# With a different new camera matrix to change field of view
new_cam = [700.0, 700.0, 640.0, 360.0]
op.barrel_correct(
    camera_props=camera_props,
    distort_coefs=distort_coefs,
    new_camera_props=new_cam,
    width=1280,
    height=720,
            )
```

**Note:**

- Uses OpenCL acceleration for RGB/BGR 3-channel images.
- Falls back to CPU (cv2.remap + initUndistortRectifyMap) for other formats.
- camera_props and distort_coefs must match the format used during camera
  calibration (pixel-space intrinsics, not normalised).

**Constructor:**

```python
__init__(camera_props: list[float] | tuple[float, ...], distort_coefs: list[float] | tuple[float, ...], new_camera_props: list[float] | tuple[float, ...] | None = None, width: int = 0, height: int = 0, dst_format: ColorFormat | str | None = None)
```
