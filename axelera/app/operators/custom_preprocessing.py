# Copyright Axelera AI, 2023
# Custom pre-processing operators
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageOps
import cv2
import numpy as np

from axelera import types

from .. import config, gst_builder, logging_utils
from ..torch_utils import torch
from .base import PreprocessOperator, builtin
from .context import PipelineContext
from .utils import insert_color_convert, inspect_resize_status, add_alpha_channel

if not hasattr(Image, "Resampling"):  # if Pillow<9.0
    Image.Resampling = Image

LOG = logging_utils.getLogger(__name__)


@builtin
class PermuteChannels(PreprocessOperator):
    input_layout: types.TensorLayout
    output_layout: types.TensorLayout

    def _post_init(self):
        super()._post_init()
        self._enforce_member_type('input_layout')
        self._enforce_member_type('output_layout')
        if self.input_layout != self.output_layout:
            self._dimchg = _get_dimchg(self.input_layout, self.output_layout)

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        # GST pipe expects only NHWC
        if self.input_layout != types.TensorLayout.NHWC:
            raise ValueError(
                "PermuteChannels is only supported for NHWC layout in GStreamer, skipping"
            )

    def exec_torch(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(image)}")
        if image.ndim < 3:
            # all supported layouts are HW, so all we can actually do here is
            # add a channel dimension (should this be in ToTensor?)
            image = image.unsqueeze(0)
        elif self.input_layout != self.output_layout:
            _in, _out = self.input_layout.name, self.output_layout.name
            if image.ndim < 4:
                _in, _out = _in.replace('N', ''), _out.replace('N', '')
            axes = [_in.index(x) for x in _out]
            image = image.permute(*axes).contiguous()
        return image


_dimchgs = {
    "NCHW:NHWC": '2:0',
    "NHWC:NCHW": '0:2',
    "CHWN:NCHW": '0:3',
    "NCHW:CHWN": '3:0',
}


def _get_dimchg(input_layout: types.TensorLayout, output_layout: types.TensorLayout) -> str:
    try:
        return _dimchgs[f'{input_layout.name}:{output_layout.name}']
    except KeyError:
        raise ValueError(
            f"Unsupported input/output layouts: {input_layout.name}/{output_layout.name}"
        ) from None


@builtin
class Letterbox(PreprocessOperator):
    height: int
    width: int
    scaleup: bool = True
    half_pixel_centers: bool = False
    pad_val: int = 114
    # Never crop, always end up in scaleup==True if no image_width or height specified
    image_width: int = 1000000
    image_height: int = 1000000

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        # TODO: implement SQUISH and LETTERBOX_CONTAIN mode in torch and verify accuracy on YOLOs
        self.task_name = task_name
        inspect_resize_status(context)
        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if gst.getconfig().opencl:
            gst.axtransform(
                lib='libtransform_resize_cl.so',
                options=f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)}',
            )
        else:
            gst.axtransform(
                lib='libtransform_resize.so',
                options=f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)}',
            )

    def exec_torch(self, image: types.Image) -> types.Image:
        result = self._letterbox(
            image.aspil(),
            (self.height, self.width),
            scaleup=self.scaleup,
            color=(self.pad_val, self.pad_val, self.pad_val),
        )[0]
        return types.Image.frompil(result, image.color_format)

    def _letterbox(
        self,
        im: Image,
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int],
        rect: bool = False,
        scaleup: bool = True,
        stride: int = 32,
    ):
        """Resize and pad image while meeting stride-multiple constraints.
        new_shape: (height, width)
        """

        shape = im.size[::-1]  # shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))  # h, w
        dh, dw = (
            new_shape[0] - new_unpad[0],
            new_shape[1] - new_unpad[1],
        )  # hw padding

        # minimum rectangle
        # We don't support this mode due to the requirement of dynamic input dim
        # TODO: we should support detection with a fixed rectangle size which may increase performance
        if rect:
            dh, dw = np.mod(dh, stride), np.mod(dw, stride)  # wh padding

        dh /= 2  # divide padding into 2 sides
        dw /= 2

        if shape != new_unpad:  # resize
            if self.half_pixel_centers:
                im_resized = cv2.resize(
                    np.array(im), new_unpad[::-1], interpolation=cv2.INTER_LINEAR
                )
                im = Image.fromarray(im_resized)
            else:
                im = im.resize(new_unpad[::-1], Image.Resampling.BILINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        if im.mode == 'RGB':
            im = ImageOps.expand(im, border=(left, top, right, bottom), fill=color)
        elif im.mode == 'L':
            if isinstance(color, tuple):
                color = color[0]  # Use the first value of the tuple for grayscale
            im = ImageOps.expand(im, border=(left, top, right, bottom), fill=color)
        else:
            raise ValueError(f"Unsupported image mode: {im.mode}")

        return im, r, (dw, dh)


_supported_formats = [
    'RGB2GRAY',
    'GRAY2RGB',
    'RGB2BGR',
    'BGR2RGB',
    'BGR2GRAY',
    'GRAY2BGR',
]


@builtin
class ConvertColor(PreprocessOperator):
    """TODO: merge ConvertChannel and ConvertColor, format follow
    OpenCV cvtColor like RGB2BGR, YUV2RGB, this means developer must know
    exact input and output formats."""

    format: str

    def _post_init(self):
        super()._post_init()
        _, output_format = self.format.split('2')
        self.output_format = output_format.upper()
        if self.format.upper() not in _supported_formats:
            raise ValueError(f"Unsupported conversion: {self.format}")

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        self.task_name = task_name
        input_format, output_format = self.format.split('2')
        input_format = input_format.upper()
        output_format = output_format.upper()

        format_map = {
            'RGB': types.ColorFormat.RGB,
            'BGR': types.ColorFormat.BGR,
            'GRAY': types.ColorFormat.GRAY,
        }

        if context.color_format != format_map[input_format]:
            raise ValueError(
                f"Input color format mismatch. Expected {input_format}, but got {context.color_format}"
            )

        self.input_format = format_map[input_format]
        context.color_format = format_map[output_format]

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        vaapi = gst.getconfig() is not None and gst.getconfig().vaapi
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        format = f'{self.format.split("2")[1].lower()}'
        insert_color_convert(gst, format, vaapi, opencl)

    def exec_torch(
        self, image: Union[torch.Tensor, types.Image]
    ) -> Union[torch.Tensor, types.Image]:
        if isinstance(image, (types.Image, Image.Image)):
            result = image
        elif isinstance(image, np.ndarray):
            result = types.Image.fromarray(image, self.input_format)
        elif isinstance(image, torch.Tensor):
            return _convert_color_torch(image, self.format)
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        new_array = result.asarray(self.output_format)
        result.update(new_array, color_format=self.output_format)
        return result


@builtin
class ConvertColorInput(PreprocessOperator):
    """
    This is the color convert from the Input operator. It has been moved into
    a separate operator to allow for more flexibility in the pipeline.
    """

    format: types.ColorFormat = types.ColorFormat.RGB

    def _post_init(self):
        self._enforce_member_type('format')

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        vaapi = gst.getconfig() is not None and gst.getconfig().vaapi
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        insert_color_convert(gst, self.format, vaapi, opencl)

    def exec_torch(
        self, image: Union[torch.Tensor, types.Image]
    ) -> Union[torch.Tensor, types.Image]:
        return image


@builtin
class FaceAlign(PreprocessOperator):
    keypoints_key: Optional[str] = None  # Only used for torch pipeline
    width: int = 0
    height: int = 0
    padding: float = 0.0
    template_keypoints_x: List[float] = (
        []
    )  # Empty by default, will be populated based on detection
    template_keypoints_y: List[float] = (
        []
    )  # Empty by default, will be populated based on detection
    # Whether to use self-normalizing alignment when no template is provided
    use_self_normalizing: bool = False
    save_aligned_images: bool = True  # for debugging purposes

    # Default template values for reference (from OpenCV)
    _default_5pt_template_x = [
        30.2946 / 96,
        65.5318 / 96,
        48.0252 / 96,
        33.5493 / 96,
        62.7299 / 96,
    ]
    _default_5pt_template_y = [
        51.6963 / 96,
        51.5014 / 96,
        71.7366 / 96,
        92.3655 / 96,
        92.2041 / 96,
    ]

    def _post_init(self):
        if len(self.template_keypoints_x) != len(self.template_keypoints_y):
            raise ValueError("Number of template keypoints x and y must be equal")

        # Initialize cache for templates
        self._template_cache = {}

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        self._association = context.association or None

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        master_key = f'master_meta:{self._where};' if self._where else str()
        association_key = f'association_meta:{self._association};' if self._association else str()

        if gst.getconfig().opencl:
            lib = 'libtransform_facealign_cl.so'
        else:
            lib = 'libtransform_facealign.so'
        gst.axtransform(
            lib=lib,
            options=f'{master_key}'
            f'{association_key}'
            f'width:{self.width};'
            f'height:{self.height};'
            f'padding:{self.padding};'
            f'template_keypoints_x:{",".join(map(str, self.template_keypoints_x))};'
            f'template_keypoints_y:{",".join(map(str, self.template_keypoints_y))};'
            f'use_self_normalizing:{int(self.use_self_normalizing)};'
            f'save_aligned_images:{int(self.save_aligned_images)}',
        )

    def exec_torch(self, image, meta=None):
        """
        Align face images using facial keypoints from metadata.

        Args:
            image: The input image to align
            meta: Optional metadata containing facial keypoints

        Returns:
            Aligned face image
        """
        if meta is None or self.keypoints_key is None or self.keypoints_key not in meta:
            LOG.warning("FaceAlign requires metadata with keypoints")
            return image

        keypoints_meta = meta[self.keypoints_key]
        if not hasattr(keypoints_meta, 'keypoints') or keypoints_meta.keypoints is None:
            LOG.warning("No keypoints found in metadata")
            return image

        width = self.width if self.width > 0 else image.size[0]
        height = self.height if self.height > 0 else image.size[1]

        # Extract keypoints - the keypoints are stored as a numpy array
        try:
            img_array = image.asarray()  # Get image array at the beginning
            original_full_img = img_array.copy()  # Save a copy of the original full image
            landmarks = keypoints_meta.keypoints[0]  # Take the first set of keypoints
            LOG.debug(f"Keypoints meta type: {type(keypoints_meta).__name__}")

            # Process face bounding box if available
            if (
                hasattr(keypoints_meta, 'boxes')
                and keypoints_meta.boxes is not None
                and len(keypoints_meta.boxes) > 0
            ):
                box = keypoints_meta.boxes[0]
                LOG.debug(f"Bounding boxes available: {keypoints_meta.boxes}")

                # Ensure box coordinates are within image bounds
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, min(x1, img_array.shape[1] - 1))
                y1 = max(0, min(y1, img_array.shape[0] - 1))
                x2 = max(0, min(x2, img_array.shape[1] - 1))
                y2 = max(0, min(y2, img_array.shape[0] - 1))
                box = (x1, y1, x2, y2)

                # Check if landmarks are outside the box and expand if necessary
                if self._are_landmarks_outside_box(landmarks, box):
                    LOG.debug("Some landmarks are outside the bounding box - adjusting box")
                    box = self._expand_box_to_include_landmarks(box, landmarks, img_array.shape)

                # Crop the face and adjust landmarks
                img_array, landmarks, original_cropped_img = self._crop_face_and_adjust_landmarks(
                    img_array, landmarks, box
                )
            else:
                # No bounding box available - use the full image
                LOG.info("No bounding boxes available, using full image")
                original_cropped_img = img_array.copy()

            if hasattr(keypoints_meta, 'scores') and keypoints_meta.scores is not None:
                LOG.debug(f"Detection scores: {keypoints_meta.scores}")

            # Perform face alignment
            num_keypoints = landmarks.shape[0]
            LOG.debug(f"Found {num_keypoints} keypoints for face alignment")

            # Get template and calculate transformation
            template = self._get_template(num_keypoints, width, height)

            try:
                # Calculate transformation matrix
                M = self._get_transformation_matrix(
                    landmarks, template, num_keypoints, width, height
                )

                # Debug information
                LOG.debug(f"Source image shape: {img_array.shape}")
                LOG.debug(f"Target dimensions: {width}x{height}")

                # Check for issues that might cause alignment to fail
                if self._are_landmarks_outside_bounds(landmarks, img_array.shape):
                    LOG.debug(
                        "Some landmarks are outside the image bounds, which may cause incorrect alignment"
                    )

                if not np.all(np.isfinite(M)):
                    LOG.error(f"Invalid transformation matrix contains NaN or Inf values: {M}")
                    return image

                if img_array.shape[0] <= 0 or img_array.shape[1] <= 0:
                    LOG.error(f"Invalid image dimensions for warping: {img_array.shape}")
                    return image

                # Perform the actual alignment
                aligned_img = cv2.warpAffine(img_array, M[:2], (width, height))

                # Save debug images if requested
                if self.save_aligned_images:
                    self._save_debug_images(
                        original_full_img,
                        original_cropped_img,
                        aligned_img,
                        landmarks,
                        keypoints_meta,
                        num_keypoints,
                    )

                return types.Image.fromarray(aligned_img, image.color_format)
            except Exception as e:
                LOG.error(f"Error in alignment process: {str(e)}")
                return image

        except Exception as e:
            LOG.error(f"Face alignment failed: {str(e)}")
            # If available, log additional info about the shapes
            try:
                if 'landmarks' in locals() and 'template' in locals():
                    LOG.error(
                        f"Landmarks shape: {landmarks.shape}, Template shape: {template.shape}"
                    )
            except Exception:
                pass
            return image

    def _get_template(self, num_keypoints, width, height):
        """Get the appropriate template for face alignment based on the number of keypoints"""
        cache_key = f"{num_keypoints}_{width}_{height}_{self.padding}"

        if cache_key in self._template_cache:
            return self._template_cache[cache_key]

        template = None
        if self.template_keypoints_x and self.template_keypoints_y:
            template = np.array(list(zip(self.template_keypoints_x, self.template_keypoints_y)))
            template[:, 0] = ((template[:, 0] + self.padding) / (2 * self.padding + 1)) * width
            template[:, 1] = ((template[:, 1] + self.padding) / (2 * self.padding + 1)) * height
        else:
            if num_keypoints == 51 or num_keypoints == 68:
                # When using 51 or 68 points, get the standard template
                x, y = self._get_standard_51_point_template()
                x = ((np.array(x) + self.padding) / (2 * self.padding + 1)) * width
                y = ((np.array(y) + self.padding) / (2 * self.padding + 1)) * height
                template = np.array(list(zip(x, y)))

                # Update instance variables if not already set
                if not self.template_keypoints_x:
                    self.template_keypoints_x = x
                    self.template_keypoints_y = y
            else:
                template = self._get_position(
                    width, height, self.padding, num_points=min(num_keypoints, 51)
                )

                # Update instance variables if not already set
                if not self.template_keypoints_x and num_keypoints == 5:
                    self.template_keypoints_x = self._default_5pt_template_x.copy()
                    self.template_keypoints_y = self._default_5pt_template_y.copy()

        # Cache the template for future use
        self._template_cache[cache_key] = template
        return template

    def _transformation_from_points(self, points1, points2):
        """
        Calculate the transformation matrix for aligning points1 to points2.

        Args:
            points1: Source points
            points2: Target points

        Returns:
            Transformation matrix
        """
        points1 = np.array(points1, dtype=np.float64)
        points2 = np.array(points2, dtype=np.float64)

        # Check if points have valid shape
        if points1.shape[0] < 2 or points2.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 points for transformation. Got {points1.shape[0]} and {points2.shape[0]}"
            )

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)

        # Avoid division by zero
        if s1 < 1e-5:
            LOG.warning("Standard deviation of source points is too small, using minimum value")
            s1 = 1e-5

        if s2 < 1e-5:
            LOG.warning("Standard deviation of target points is too small, using minimum value")
            s2 = 1e-5

        points1 /= s1
        points2 /= s2

        points1_T = points1.T
        # points2_T = points2.T

        # Use SVD to find rotation matrix
        try:
            U, S, Vt = np.linalg.svd(np.dot(points1_T, points2))
            R = np.dot(U, Vt)

            scale = s2 / s1
            rotation = R
            translation = c2 - scale * np.dot(rotation, c1)

            M = np.zeros((3, 3), dtype=np.float64)
            M[0:2, 0:2] = scale * rotation
            M[0:2, 2] = translation
            M[2, 2] = 1.0

            # Check for invalid values
            if not np.all(np.isfinite(M)):
                LOG.error("Transformation matrix contains NaN or Inf values")
                # Fallback to identity transformation
                M = np.eye(3, dtype=np.float64)
                M[0:2, 2] = c2 - c1  # Simple translation only

            return M
        except Exception as e:
            LOG.error(f"SVD calculation failed: {str(e)}")
            # Fallback to identity transformation
            M = np.eye(3, dtype=np.float64)
            M[0:2, 2] = c2 - c1  # Simple translation only
            return M

    def _get_transformation_matrix(self, landmarks, template, num_keypoints, width, height):
        """Calculate the transformation matrix for face alignment"""
        if num_keypoints == 5 and self.use_self_normalizing:
            LOG.info("Using self-normalizing 5-point face alignment")
            left_eye = landmarks[0]
            right_eye = landmarks[1]

            # Calculate eye center and distance
            eye_center = (left_eye + right_eye) / 2
            eye_distance = np.linalg.norm(right_eye - left_eye)

            # Adjust desired eye position based on height
            desired_eye_y = height * 0.33
            desired_eye_distance = width * 0.42

            # Calculate scale to maintain proper eye distance
            scale = desired_eye_distance / max(eye_distance, 1.0)  # Avoid division by zero

            # Calculate angle of eyes
            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale)

            # Calculate translation to center eyes
            desired_eye_center_x = width / 2
            rotation_matrix[0, 2] += desired_eye_center_x - eye_center[0]
            rotation_matrix[1, 2] += desired_eye_y - eye_center[1]

            # Create full transformation matrix
            M = np.vstack([rotation_matrix, [0, 0, 1]])

            # Log the transformation details for debugging
            LOG.debug(f"Eye center: {eye_center}, Eye distance: {eye_distance}")
            LOG.debug(f"Scale: {scale}, Angle: {angle} degrees")
            LOG.debug(
                f"Translation: ({desired_eye_center_x - eye_center[0]}, {desired_eye_y - eye_center[1]})"
            )

            return M
        elif self.template_keypoints_x and self.template_keypoints_y:
            if num_keypoints == 5 and len(template) > 5:
                LOG.debug("Using 5-point to many-point face alignment")
                five_point_template = np.array(
                    list(zip(self.template_keypoints_x[:5], self.template_keypoints_y[:5]))
                )
                five_point_template[:, 0] *= width
                five_point_template[:, 1] *= height
                return self._transformation_from_points(landmarks, five_point_template)
            elif num_keypoints != len(template):
                LOG.debug(
                    f"Template has {len(template)} points but we have {num_keypoints}. Using only matching points."
                )
                min_points = min(num_keypoints, len(template))
                return self._transformation_from_points(
                    landmarks[:min_points], template[:min_points]
                )
            else:
                return self._transformation_from_points(landmarks, template)
        else:
            LOG.debug(
                "No template provided and self-normalizing not enabled. Using default template."
            )
            template = self._get_position(
                width, height, self.padding, num_points=min(num_keypoints, 51)
            )
            return self._transformation_from_points(landmarks, template[:num_keypoints])

    def _save_debug_images(
        self,
        original_full_img,
        original_cropped_img,
        aligned_img,
        landmarks,
        keypoints_meta,
        num_keypoints,
    ):
        """Save debug images for visualization and troubleshooting"""
        cv2.imwrite("aligned_face.jpg", aligned_img)
        cv2.imwrite("original_face_full.jpg", original_full_img)
        cv2.imwrite("original_face_cropped.jpg", original_cropped_img)

        debug_img = original_cropped_img.copy()
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(debug_img, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(
                debug_img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
            )

        # Draw connections between landmarks for better visualization (for 5-point landmarks)
        if num_keypoints == 5:
            # Connect eyes
            cv2.line(
                debug_img,
                (int(landmarks[0][0]), int(landmarks[0][1])),
                (int(landmarks[1][0]), int(landmarks[1][1])),
                (255, 255, 0),
                1,
            )
            # Connect nose to eyes
            cv2.line(
                debug_img,
                (int(landmarks[2][0]), int(landmarks[2][1])),
                (int(landmarks[0][0]), int(landmarks[0][1])),
                (255, 255, 0),
                1,
            )
            cv2.line(
                debug_img,
                (int(landmarks[2][0]), int(landmarks[2][1])),
                (int(landmarks[1][0]), int(landmarks[1][1])),
                (255, 255, 0),
                1,
            )
            # Connect mouth corners
            cv2.line(
                debug_img,
                (int(landmarks[3][0]), int(landmarks[3][1])),
                (int(landmarks[4][0]), int(landmarks[4][1])),
                (255, 255, 0),
                1,
            )
            # Connect nose to mouth
            cv2.line(
                debug_img,
                (int(landmarks[2][0]), int(landmarks[2][1])),
                (
                    (int(landmarks[3][0]) + int(landmarks[4][0])) // 2,
                    (int(landmarks[3][1]) + int(landmarks[4][1])) // 2,
                ),
                (255, 255, 0),
                1,
            )
        cv2.imwrite("landmarks_face.jpg", debug_img)

    def _get_standard_51_point_template(self):
        """
        Returns the standard 51-point template for face alignment.
        These coordinates are from standard facial landmark detection.
        """
        # Standard 51-point template coordinates
        x = [
            0.000213256,
            0.0752622,
            0.18113,
            0.29077,
            0.393397,
            0.586856,
            0.689483,
            0.799124,
            0.904991,
            0.98004,
            0.490127,
            0.490127,
            0.490127,
            0.490127,
            0.36688,
            0.426036,
            0.490127,
            0.554217,
            0.613373,
            0.121737,
            0.187122,
            0.265825,
            0.334606,
            0.260918,
            0.182743,
            0.645647,
            0.714428,
            0.793132,
            0.858516,
            0.79751,
            0.719335,
            0.254149,
            0.340985,
            0.428858,
            0.490127,
            0.551395,
            0.639268,
            0.726104,
            0.642159,
            0.556721,
            0.490127,
            0.423532,
            0.338094,
            0.290379,
            0.428096,
            0.490127,
            0.552157,
            0.689874,
            0.553364,
            0.490127,
            0.42689,
        ]

        y = [
            0.106454,
            0.038915,
            0.0187482,
            0.0344891,
            0.0773906,
            0.0773906,
            0.0344891,
            0.0187482,
            0.038915,
            0.106454,
            0.203352,
            0.307009,
            0.409805,
            0.515625,
            0.587326,
            0.609345,
            0.628106,
            0.609345,
            0.587326,
            0.216423,
            0.178758,
            0.179852,
            0.231733,
            0.245099,
            0.244077,
            0.231733,
            0.179852,
            0.178758,
            0.216423,
            0.244077,
            0.245099,
            0.780233,
            0.745405,
            0.727388,
            0.742578,
            0.727388,
            0.745405,
            0.780233,
            0.864805,
            0.902192,
            0.909281,
            0.902192,
            0.864805,
            0.784792,
            0.778746,
            0.785343,
            0.778746,
            0.784792,
            0.824182,
            0.831803,
            0.824182,
        ]

        return x, y

    def _get_position(self, size_x, size_y, padding=0.25, num_points=51):
        """
        Get the standardized facial landmark positions.

        Args:
            size_x: Width of the output image
            size_y: Height of the output image
            padding: Padding to add around the face
            num_points: Number of points to generate (5 or 51)

        Returns:
            Array of landmark positions
        """
        if num_points == 5:
            # Use default 5-point template
            x = self._default_5pt_template_x
            y = self._default_5pt_template_y
        else:
            # Use standard 51-point template or provided template based on available points
            if not self.template_keypoints_x or len(self.template_keypoints_x) < 51:
                # Get the standard 51-point template
                x, y = self._get_standard_51_point_template()
                # Only use the first num_points if we have too many
                if len(x) > num_points:
                    x = x[:num_points]
                    y = y[:num_points]
            else:
                x = self.template_keypoints_x
                y = self.template_keypoints_y
                # Only use the first num_points if we have too many
                if len(x) > num_points:
                    x = x[:num_points]
                    y = y[:num_points]

        x, y = np.array(x), np.array(y)

        # Adjust for padding
        x = (x + padding) / (2 * padding + 1)
        y = (y + padding) / (2 * padding + 1)

        # Scale to target size
        x = x * size_x
        y = y * size_y

        return np.array(list(zip(x, y)))

    def _is_point_outside_box(self, point, box, tolerance=1.0):
        """Check if a point is outside a bounding box with tolerance."""
        x1, y1, x2, y2 = box
        return (
            point[0] < x1 - tolerance
            or point[0] > x2 + tolerance
            or point[1] < y1 - tolerance
            or point[1] > y2 + tolerance
        )

    def _are_landmarks_outside_box(self, landmarks, box, tolerance=5.0):
        """Check if any landmarks are outside the bounding box."""
        x1, y1, x2, y2 = box
        return (
            np.any(landmarks[:, 0] < x1 - tolerance)
            or np.any(landmarks[:, 0] > x2 + tolerance)
            or np.any(landmarks[:, 1] < y1 - tolerance)
            or np.any(landmarks[:, 1] > y2 + tolerance)
        )

    def _are_landmarks_outside_bounds(self, landmarks, image_shape, tolerance=5.0):
        """Check if any landmarks are outside the image bounds."""
        return (
            np.any(landmarks[:, 0] < -tolerance)
            or np.any(landmarks[:, 0] >= image_shape[1] + tolerance)
            or np.any(landmarks[:, 1] < -tolerance)
            or np.any(landmarks[:, 1] >= image_shape[0] + tolerance)
        )

    def _expand_box_to_include_landmarks(self, box, landmarks, image_shape, margin=10):
        """Expand a box to include all landmarks with a margin."""
        x1, y1, x2, y2 = box

        # Get min/max landmark coordinates
        min_x = np.min(landmarks[:, 0])
        min_y = np.min(landmarks[:, 1])
        max_x = np.max(landmarks[:, 0])
        max_y = np.max(landmarks[:, 1])

        # Expand box to include all landmarks with margin
        new_x1 = max(0, min(x1, min_x - margin))
        new_y1 = max(0, min(y1, min_y - margin))
        new_x2 = min(image_shape[1] - 1, max(x2, max_x + margin))
        new_y2 = min(image_shape[0] - 1, max(y2, max_y + margin))

        # Make sure box has minimum dimensions (at least 1x1)
        if new_x2 <= new_x1:
            new_x2 = new_x1 + 1
        if new_y2 <= new_y1:
            new_y2 = new_y1 + 1

        # Ensure box is within image bounds
        new_x2 = min(new_x2, image_shape[1] - 1)
        new_y2 = min(new_y2, image_shape[0] - 1)

        # Only log if there's a meaningful difference
        if (
            abs(new_x1 - x1) >= 1
            or abs(new_y1 - y1) >= 1
            or abs(new_x2 - x2) >= 1
            or abs(new_y2 - y2) >= 1
        ):
            LOG.debug(f"Original box: [{x1}, {y1}, {x2}, {y2}]")
            LOG.debug(f"Expanded box: [{new_x1}, {new_y1}, {new_x2}, {new_y2}]")

        return new_x1, new_y1, new_x2, new_y2

    def _crop_face_and_adjust_landmarks(self, img_array, landmarks, box):
        """Crop the face using the box and adjust landmarks to the cropped image."""
        x1, y1, x2, y2 = box

        # Validate box dimensions
        if x2 <= x1 or y2 <= y1:
            LOG.debug(f"Invalid box dimensions: {box}. Using original image.")
            return img_array.copy(), landmarks.copy(), img_array.copy()

        # Ensure box is within image bounds
        img_height, img_width = img_array.shape[:2]
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))

        # Final validation - ensure we have valid dimensions
        if x2 <= x1 or y2 <= y1:
            LOG.debug(
                f"Box dimensions invalid after bounds check: [{x1}, {y1}, {x2}, {y2}]. Using original image."
            )
            return img_array.copy(), landmarks.copy(), img_array.copy()

        # Crop the image
        cropped_img = img_array[y1:y2, x1:x2].copy()

        # Extra validation - check cropped image dimensions
        if cropped_img.size == 0 or cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            LOG.debug(
                f"Cropped image has invalid dimensions: {cropped_img.shape}. Using original image."
            )
            return img_array.copy(), landmarks.copy(), img_array.copy()

        # Adjust landmarks to be relative to the cropped image
        adjusted_landmarks = landmarks.copy()
        adjusted_landmarks[:, 0] = adjusted_landmarks[:, 0] - x1
        adjusted_landmarks[:, 1] = adjusted_landmarks[:, 1] - y1

        LOG.debug(f"Original landmarks: {landmarks}")
        LOG.debug(f"Adjusted landmarks: {adjusted_landmarks}")
        LOG.debug(f"Cropped image shape: {cropped_img.shape}")

        return cropped_img, adjusted_landmarks, cropped_img.copy()


def get_output_format_spec(format: types.ColorFormat) -> str:
    return f';format:{add_alpha_channel(format)}' if format else ''


@builtin
class Polar(PreprocessOperator):
    width: int = None
    height: int = None
    size: int = None
    rotate180: bool = False
    center_x: float = 0.5
    center_y: float = 0.5
    max_radius: int = None
    inverse: bool = False
    linear_polar: bool = True
    format: types.ColorFormat = None

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        out = get_output_format_spec(self.format)

        oplib = 'libtransform_polar_cl.so' if opencl else 'libtransform_polar.so'
        sizestr = (
            f'size:{self.size};' if self.size else f'width:{self.width};height:{self.height};'
        )
        gst.axtransform(
            lib=oplib,
            options=f'center_x:{self.center_x};center_y:{self.center_y};'
            f'{sizestr}'
            f'inverse:{int(self.inverse)};'
            f'linear_polar:{int(self.linear_polar)};'
            f'max_radius:{self.max_radius}{out}',
        )

    def exec_torch(self, image):
        img = image.asarray()
        h, w = img.shape[:2]
        if self.size:
            self.width = self.size
            self.height = self.size
        if not self.width or not self.height:
            self.width = w
            self.height = h
        center = (int(self.center_y * h), int(self.center_x * w))
        max_radius = (
            self.max_radius
            if self.max_radius
            else min(center[1], center[0], w - center[0], h - center[1])
        )
        flags = cv2.WARP_FILL_OUTLIERS
        flags |= cv2.INTER_LINEAR
        if self.linear_polar:
            flags |= cv2.WARP_POLAR_LINEAR
        else:
            flags |= cv2.WARP_POLAR_LOG
        if self.inverse:
            flags |= cv2.WARP_INVERSE_MAP

        polar_img = cv2.warpPolar(
            img.transpose(1, 0, 2), (self.height, self.width), center, max_radius, flags
        )
        if self.rotate180:
            polar_img = cv2.rotate(polar_img, cv2.ROTATE_180)
        return types.Image.fromarray(polar_img.transpose(1, 0, 2), color_format=image.color_format)


@builtin
class Perspective(PreprocessOperator):
    camera_matrix: List[float]
    invert: bool = False
    format: types.ColorFormat = None

    def _post_init(self):
        if self.camera_matrix is None:
            raise ValueError("You must specify camera matrix")
        if isinstance(self.camera_matrix, str):
            if len(self.camera_matrix.split(",")) != 9:
                raise ValueError("Number of camera matrix values must be 9")
            self.camera_matrix = [float(item) for item in self.camera_matrix.split(',')]
        elif isinstance(self.camera_matrix, list):
            if len(self.camera_matrix) != 9:
                raise ValueError("Number of camera matrix values must be 9")
        else:
            raise ValueError(
                "Camera matrix must be either a comma-separated string or a list of floats"
            )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        matrix = np.array(self.camera_matrix).reshape(3, 3)
        if self.invert:
            matrix = np.linalg.inv(matrix)
        matrix = ','.join(f'{x:.6g}' for x in matrix.flatten())
        if opencl:
            out = get_output_format_spec(self.format)
            gst.axtransform(
                lib='libtransform_perspective_cl.so',
                options=f'matrix:{matrix}{out}',
            )
        else:
            vaapi = False  # Gst perspective element doesn't respect from offsets and strides in VAsurfaces (GstVideoMeta)
            if self.format:
                insert_color_convert(gst, self.format, vaapi, opencl)
            gst.perspective(matrix=matrix)

    def exec_torch(self, image):
        matrix = np.array(self.camera_matrix).reshape(3, 3)
        transformed_image = cv2.warpPerspective(image.asarray(), matrix, image.size)
        return types.Image.fromarray(transformed_image, color_format=image.color_format)


@builtin
class CameraUndistort(PreprocessOperator):
    fx: float
    fy: float
    cx: float
    cy: float
    distort_coefs: List[float]
    normalized: bool = True
    format: types.ColorFormat = None

    def _post_init(self):
        if self.distort_coefs is None:
            raise ValueError("You must specify camera matrix")
        if len(self.distort_coefs) != 5:
            raise ValueError("Number of distort coeficients must be 5")

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if gst.getconfig().opencl:
            out = get_output_format_spec(self.format)
            gst.axtransform(
                lib='libtransform_barrelcorrect_cl.so',
                options=f'camera_props:{self.fx},{self.fy},{self.cx},{self.cy};'
                f'normalized_properties:{int(self.normalized)};'
                f'distort_coefs:{",".join(str(coef) for coef in self.distort_coefs)}{out}',
            )
        else:
            # for non OpenCL path camera matrix need to be denormalized in yaml file
            if self.normalized:
                raise ValueError(
                    "CameraUndistort only supports non normalized camera matrix in non OpenCL path"
                )
            if self.format:
                insert_color_convert(gst, self.format, False, False)
            config = f'<?xml version=\"1.0\"?><opencv_storage><cameraMatrix type_id=\"opencv-matrix\"><rows>3</rows><cols>3</cols><dt>f</dt><data>{self.fx} 0. {self.cx} 0. {self.fy} {self.cy} 0. 0. 1.</data></cameraMatrix><distCoeffs type_id=\"opencv-matrix\"><rows>5</rows><cols>1</cols><dt>f</dt><data>{" ".join(str(coef) for coef in self.distort_coefs)}</data></distCoeffs></opencv_storage>'
            gst.cameraundistort(settings=config)

    def exec_torch(self, image):
        width, height = image.size
        if not self.normalized:
            width, height = 1, 1
        camera_matrix = np.array(
            [
                [self.fx * width, 0.0, self.cx * width],
                [0.0, self.fy * height, self.cy * height],
                [0.0, 0.0, 1.0],
            ]
        )
        camera_dist = np.array(self.distort_coefs, dtype=np.float64)
        new_camera, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, camera_dist, image.size, 0, image.size
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, camera_dist, None, new_camera, image.size, 5
        )
        new_image = cv2.remap(image.asarray(), mapx, mapy, cv2.INTER_LINEAR)

        return types.Image.fromarray(new_image, color_format=image.color_format)


def _convert_color_torch(img, format):
    format = getattr(cv2, f"COLOR_{format.upper()}")
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(img, format)
    if len(img.shape) == 2:  # Grayscale output has 2 dimensions (H,W)
        img = img[np.newaxis, :, :]
    else:
        img = img.transpose(2, 0, 1)
    return torch.from_numpy(img)


@builtin
class ContrastNormalize(PreprocessOperator):
    """Also known as contrast stretching"""

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.axtransform(lib='libtransform_contrastnormalize.so')

    def exec_torch(
        self, image: Union[torch.Tensor, types.Image]
    ) -> Union[torch.Tensor, types.Image]:
        import torchvision.transforms.functional as F

        if isinstance(image, types.Image):
            # If the input is a PIL image, convert it to a tensor and normalize it
            tensor_image = F.to_tensor(image.aspil())
            min_value = torch.min(tensor_image)
            max_value = torch.max(tensor_image)
            normalized_tensor = (tensor_image - min_value) / (max_value - min_value)
            # to_pil_image scale the values back to 0-255 automaticlly
            normalized_image = F.to_pil_image(normalized_tensor)
            normalized_image = types.Image.frompil(normalized_image, image.color_format)
        elif isinstance(image, torch.Tensor):
            # If the input is a tensor, normalize it directly
            min_value = torch.min(image)
            max_value = torch.max(image)
            normalized_image = (image - min_value) / (max_value - min_value)
        else:
            raise TypeError("Input should be a PIL image or a tensor.")
        return normalized_image


@builtin
class VideoFlip(PreprocessOperator):
    '''Perform simple video flip operations like rotation and flipping.'''

    method: config.VideoFlipMethod = config.VideoFlipMethod.clockwise

    def _post_init(self):
        self._enforce_member_type('method')
        self._first = True

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.videoflip(method=self.method.value)

    def exec_torch(
        self, image: Union[torch.Tensor, types.Image]
    ) -> Union[torch.Tensor, types.Image]:
        if self._first:
            LOG.warning("This rotation is not implemented for torch pipeline")
            self._first = True
        return image  # No-op for torch, as rotation is not implemented yet


@builtin
class _AddTiles(PreprocessOperator):
    """This is a n internal only class for inserting tiles into the pipeline.
    It is used to insert tiles into the pipeline at the specified position.
    """

    tiling: config.TilingConfig
    model_width: int = 0
    model_height: int = 0

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        options = (
            f'meta_key:axelera-tiles-internal;'
            f'tile_size:{self.tiling.size};'
            f'tile_overlap:{self.tiling.overlap};'
            f'tile_position:{self.tiling.position};'
            f'model_width:{self.model_width};'
            f'model_height:{self.model_height}'
            + (f';tile_json:{self.tiling.file}' if self.tiling.file else '')
        )
        gst.axinplace(lib="libinplace_addtiles.so", options=options)

    def exec_torch(
        self, image: Union[torch.Tensor, types.Image]
    ) -> Union[torch.Tensor, types.Image]:
        if self._first:
            LOG.warning("Tiling is not implemented for torch pipeline")
            self._first = False
        return image  # No-op for torch, as rotation is not implemented yet
