"""
YOLOv8/YOLO11 Model Utilities for Detection and Pose Estimation.

This module provides preprocessing and postprocessing for:
- YOLOv8/YOLO11 object detection models
- YOLOv8 pose estimation models

Note: YOLO11 dropped the 'v' prefix (e.g., YOLO11s not YOLOv11s).

These models share similar architectures:
- Split-head design with separate outputs
- DFL (Distribution Focal Loss) for box regression
- Same preprocessing (simple /255 normalization)

Supported models:
- Detection: YOLOv8s/m/l/x, YOLO11s/m/l/x (COCO)
- Pose: YOLOv8s/m/l/x-pose (COCO)
"""

from typing import List, Tuple
import numpy as np
import cv2
import logging
from axelera.runtime import TensorInfo

LOG = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

# COCO dataset class names (80 classes)
COCO_CLASSES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
]

# COCO keypoint names (17 keypoints)
COCO_KEYPOINTS = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]


# ============================================================================
# Object Detection - Preprocessing and Postprocessing
# ============================================================================


def preprocess_yolo_detection(
    image: np.ndarray, input_info: TensorInfo, quantize_fn, pad_fn
) -> np.ndarray:
    """
    Preprocess image for YOLO detection models.

    Pipeline:
    General Preprocessing:
      1. Resize to model dimensions
      2. Convert BGR → RGB
      3. Normalize to [0, 1] (YOLO: /255)
    Axelera-Required Processing:
      4. Quantize to int8
      5. Add hardware padding
      6. Handle batch dimension

    Args:
        image: Input image (BGR from OpenCV)
        input_info: TensorInfo from model.inputs()[0]
        quantize_fn: Function(normalized, scale, zero_point) -> quantized
        pad_fn: Function(unpadded, padding, zero_point) -> padded

    Returns:
        Preprocessed tensor ready for inference
    """
    batch, height, width, channels = input_info.unpadded_shape

    # Resize and convert color
    resized = cv2.resize(image, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # YOLO normalization: just /255
    normalized = rgb.astype(np.float32) / 255.0

    # Quantize and pad (Axelera-specific)
    quantized = quantize_fn(normalized, input_info.scale, input_info.zero_point)
    padded = pad_fn(quantized, input_info.padding[1:], input_info.zero_point)

    # Handle batch dimension
    if batch > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch, axis=0)
    else:
        padded = padded[np.newaxis, ...]

    return padded


def decode_yolo_detections(
    features: List[np.ndarray],
    num_classes: int = 80,
    conf_threshold: float = 0.25,
    input_size: int = 640,
) -> np.ndarray:
    """
    Decode YOLOv8/YOLO11 detection predictions.

    YOLOv8/YOLO11 split-head architecture:
    - First 3 outputs: Box regression (64 channels = 4 coords × 16 DFL bins)
    - Last 3 outputs: Classification (num_classes channels)

    Args:
        features: List of 6 feature maps in NCHW format
        num_classes: Number of classes (80 for COCO)
        conf_threshold: Confidence threshold
        input_size: Input image size

    Returns:
        Bounding boxes [x1, y1, x2, y2, confidence, class_id], shape (N, 6)
    """
    strides = [8, 16, 32]  # YOLOv8/YOLO11 strides

    reg_outputs = features[:3]  # Box regression
    cls_outputs = features[3:]  # Classification

    all_boxes = []

    for i, (reg_out, cls_out) in enumerate(zip(reg_outputs, cls_outputs)):
        _, reg_channels, h, _ = reg_out.shape
        stride = strides[i]

        # Generate anchor grid
        grid = np.arange(h, dtype=np.float32)
        yv, xv = np.meshgrid(grid, grid, indexing='ij')
        anchors = np.stack([xv, yv], axis=-1)
        anchors = (anchors + 0.5) * stride

        # Reshape outputs
        reg_out = reg_out.transpose(0, 2, 3, 1).reshape(-1, reg_channels)
        cls_out = cls_out.transpose(0, 2, 3, 1).reshape(-1, num_classes)

        # Apply sigmoid to classification
        cls_scores = 1.0 / (1.0 + np.exp(-cls_out))
        max_scores = np.max(cls_scores, axis=1)
        max_classes = np.argmax(cls_scores, axis=1)

        # Filter by confidence
        mask = max_scores > conf_threshold
        if not np.any(mask):
            continue

        filtered_reg = reg_out[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = max_classes[mask]
        filtered_anchors = anchors.reshape(-1, 2)[mask]

        # Decode boxes using DFL
        num_bins = reg_channels // 4
        filtered_reg = filtered_reg.reshape(-1, 4, num_bins)

        reg_exp = np.exp(filtered_reg)
        reg_dist = reg_exp / reg_exp.sum(axis=2, keepdims=True)

        proj = np.arange(num_bins, dtype=np.float32)
        decoded_bbox = np.sum(reg_dist * proj, axis=2) * stride

        # Convert ltrb to xyxy
        x1 = filtered_anchors[:, 0] - decoded_bbox[:, 0]
        y1 = filtered_anchors[:, 1] - decoded_bbox[:, 1]
        x2 = filtered_anchors[:, 0] + decoded_bbox[:, 2]
        y2 = filtered_anchors[:, 1] + decoded_bbox[:, 3]

        boxes = np.stack([x1, y1, x2, y2, filtered_scores, filtered_classes], axis=1)
        all_boxes.append(boxes)

    if all_boxes:
        return np.concatenate(all_boxes, axis=0)
    else:
        return np.zeros((0, 6))


def apply_nms(boxes: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """
    Non-Maximum Suppression for detections.

    Applies NMS per-class to avoid suppressing different object types.

    Args:
        boxes: Array [x1, y1, x2, y2, confidence, class_id], shape (N, 6)
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered boxes after NMS, shape (M, 6) where M <= N
    """
    if len(boxes) == 0:
        return np.zeros((0, 6))

    final_boxes = []
    unique_classes = np.unique(boxes[:, 5].astype(int))

    for cls_id in unique_classes:
        cls_mask = boxes[:, 5] == cls_id
        cls_boxes = boxes[cls_mask]

        # Sort by confidence
        indices = np.argsort(cls_boxes[:, 4])[::-1]
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Compute IoU
            current_box = cls_boxes[current]
            remaining_boxes = cls_boxes[indices[1:]]

            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

            box1_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            boxes_area = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
                remaining_boxes[:, 3] - remaining_boxes[:, 1]
            )
            union = box1_area + boxes_area - intersection

            iou = np.where(union > 0, intersection / union, 0)

            mask = iou < iou_threshold
            indices = indices[1:][mask]

        final_boxes.append(cls_boxes[keep])

    if final_boxes:
        return np.concatenate(final_boxes, axis=0)
    else:
        return np.zeros((0, 6))


def postprocess_yolo_detection(
    outputs: List[np.ndarray],
    output_infos: List[TensorInfo],
    depad_fn,
    dequantize_fn,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """
    Postprocess YOLO detection outputs.

    Pipeline:
    Axelera-Required Processing:
      1. Depad - Remove hardware channel alignment
      2. Dequantize - Convert int8 to float32
      3. Transpose - Convert NHWC to NCHW
      4. Decode predictions - Manual postamble implementation (DFL + sigmoid)
    General Postprocessing:
      5. NMS - Filter overlapping detections

    Args:
        outputs: List of output arrays from inference (int8, padded)
        output_infos: List of TensorInfo for each output
        depad_fn: Function(padded, padding) -> depadded
        dequantize_fn: Function(quantized, scale, zero_point) -> dequantized
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS

    Returns:
        Final detections [x1, y1, x2, y2, confidence, class_id], shape (N, 6)
    """
    # === Axelera-Required Processing ===
    # Depad and dequantize
    dequantized_outputs = []
    for output, info in zip(outputs, output_infos):
        depadded = depad_fn(output, info.padding)
        dequantized = dequantize_fn(depadded, info.scale, info.zero_point)
        dequantized_outputs.append(dequantized)

    # Transpose NHWC to NCHW
    features_nchw = [np.transpose(out, (0, 3, 1, 2)) for out in dequantized_outputs]

    # Decode detections (manual postamble: DFL + sigmoid + box decoding)
    input_size = output_infos[0].unpadded_shape[1]
    num_classes = dequantized_outputs[3].shape[-1]

    decoded_boxes = decode_yolo_detections(
        features_nchw,
        num_classes=num_classes,
        conf_threshold=conf_threshold,
        input_size=input_size,
    )

    # === General Postprocessing ===
    final_boxes = apply_nms(decoded_boxes, iou_threshold=iou_threshold)

    return final_boxes


# ============================================================================
# Pose Estimation - Preprocessing and Postprocessing
# ============================================================================


def preprocess_yolo_pose(
    roi: np.ndarray, input_info: TensorInfo, quantize_fn, pad_fn
) -> np.ndarray:
    """
    Preprocess ROI for YOLOv8-pose models.

    Same pipeline as detection preprocessing.

    Args:
        roi: Input ROI/image (BGR from OpenCV)
        input_info: TensorInfo from model.inputs()[0]
        quantize_fn: Function(normalized, scale, zero_point) -> quantized
        pad_fn: Function(unpadded, padding, zero_point) -> padded

    Returns:
        Preprocessed tensor ready for inference
    """
    # Same as detection preprocessing
    return preprocess_yolo_detection(roi, input_info, quantize_fn, pad_fn)


def decode_yolo_pose(
    features: List[np.ndarray],
    num_keypoints: int = 17,
    conf_threshold: float = 0.25,
    input_size: int = 640,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode YOLOv8-pose predictions.

    YOLOv8-pose outputs (9 feature maps):
    - Features 0-2: Box regression (64 channels, DFL format)
    - Features 3-5: Objectness/confidence (1 channel each)
    - Features 6-8: Keypoints (num_keypoints * 3 channels: x, y, visibility)

    Args:
        features: List of 9 feature maps in NCHW format
        num_keypoints: Number of keypoints (17 for COCO)
        conf_threshold: Confidence threshold
        input_size: Input image size

    Returns:
        boxes: [x1, y1, x2, y2, confidence], shape (N, 5)
        keypoints: [num_kpts, x, y, visibility], shape (N, num_keypoints, 3)
    """
    strides = [8, 16, 32]

    reg_outputs = features[:3]  # Box regression (DFL)
    conf_outputs = features[3:6]  # Objectness/confidence
    kpt_outputs = features[6:9]  # Keypoints

    all_boxes = []
    all_keypoints = []

    for i, (reg_out, conf_out, kpt_out) in enumerate(zip(reg_outputs, conf_outputs, kpt_outputs)):
        _, reg_channels, h, _ = reg_out.shape
        _, kpt_channels, _, _ = kpt_out.shape
        stride = strides[i]

        # Generate anchor grid
        grid = np.arange(h, dtype=np.float32)
        yv, xv = np.meshgrid(grid, grid, indexing='ij')
        anchors = np.stack([xv, yv], axis=-1)
        anchors = (anchors + 0.5) * stride

        # Reshape outputs
        reg_out = reg_out.transpose(0, 2, 3, 1).reshape(-1, reg_channels)
        conf_out = conf_out.transpose(0, 2, 3, 1).reshape(-1, 1)
        kpt_out = kpt_out.transpose(0, 2, 3, 1).reshape(-1, kpt_channels)

        # Extract confidence from objectness output and apply sigmoid
        box_conf = 1.0 / (1.0 + np.exp(-conf_out.squeeze()))

        # DEBUG: Log confidence statistics per scale
        LOG.debug(
            f"[Pose Decode Debug] Scale {i} (stride={stride}): "
            f"conf range=[{box_conf.min():.3f}, {box_conf.max():.3f}], "
            f"mean={box_conf.mean():.3f}, "
            f"count>{conf_threshold}={np.sum(box_conf > conf_threshold)}/{len(box_conf)}"
        )

        # Filter by confidence
        mask = box_conf > conf_threshold
        if not np.any(mask):
            continue

        filtered_reg = reg_out[mask]
        filtered_kpt = kpt_out[mask]
        filtered_conf = box_conf[mask]
        filtered_anchors = anchors.reshape(-1, 2)[mask]

        # Decode boxes using DFL
        num_bins = reg_channels // 4
        filtered_reg = filtered_reg.reshape(-1, 4, num_bins)

        reg_exp = np.exp(filtered_reg)
        reg_dist = reg_exp / reg_exp.sum(axis=2, keepdims=True)

        proj = np.arange(num_bins, dtype=np.float32)
        decoded_bbox = np.sum(reg_dist * proj, axis=2) * stride

        # Convert ltrb to xyxy
        x1 = filtered_anchors[:, 0] - decoded_bbox[:, 0]
        y1 = filtered_anchors[:, 1] - decoded_bbox[:, 1]
        x2 = filtered_anchors[:, 0] + decoded_bbox[:, 2]
        y2 = filtered_anchors[:, 1] + decoded_bbox[:, 3]

        boxes = np.stack([x1, y1, x2, y2, filtered_conf], axis=1)

        # Decode keypoints
        keypoints = filtered_kpt.reshape(-1, num_keypoints, 3)

        # Scale keypoints and apply sigmoid to visibility
        # Same logic as box decoding: offset * stride + anchor
        keypoints[:, :, 0] = (keypoints[:, :, 0] * 2.0 - 0.5) * stride + filtered_anchors[:, 0:1]
        keypoints[:, :, 1] = (keypoints[:, :, 1] * 2.0 - 0.5) * stride + filtered_anchors[:, 1:2]
        keypoints[:, :, 2] = 1.0 / (1.0 + np.exp(-keypoints[:, :, 2]))

        all_boxes.append(boxes)
        all_keypoints.append(keypoints)

    if all_boxes:
        return np.concatenate(all_boxes, axis=0), np.concatenate(all_keypoints, axis=0)
    else:
        return np.zeros((0, 5)), np.zeros((0, num_keypoints, 3))


def postprocess_yolo_pose(
    outputs: List[np.ndarray],
    output_infos: List[TensorInfo],
    depad_fn,
    dequantize_fn,
    conf_threshold: float = 0.25,
    num_keypoints: int = 17,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Postprocess YOLOv8-pose outputs.

    Model structure (9 outputs):
    - Outputs 0-2: Box regression (64 channels, DFL format)
    - Outputs 3-5: Objectness/confidence (1 channel each)
    - Outputs 6-8: Keypoints (51 channels = 17 keypoints * 3)

    Pipeline:
    Axelera-Required Processing:
      1. Depad - Remove hardware channel alignment
      2. Dequantize - Convert int8 to float32
      3. Transpose - Convert NHWC to NCHW
      4. Decode pose predictions - Manual postamble (DFL + objectness + keypoints)

    Args:
        outputs: List of 9 output arrays from inference (int8, padded)
        output_infos: List of TensorInfo for each output
        depad_fn: Function(padded, padding) -> depadded
        dequantize_fn: Function(quantized, scale, zero_point) -> dequantized
        conf_threshold: Confidence threshold
        num_keypoints: Number of keypoints (17 for COCO)

    Returns:
        boxes: [x1, y1, x2, y2, confidence], shape (N, 5)
        keypoints: [num_kpts, x, y, visibility], shape (N, num_keypoints, 3)
    """
    # === Axelera-Required Processing ===
    # Depad and dequantize
    dequantized_outputs = []
    for output, info in zip(outputs, output_infos):
        depadded = depad_fn(output, info.padding)
        dequantized = dequantize_fn(depadded, info.scale, info.zero_point)
        dequantized_outputs.append(dequantized)

    # Transpose NHWC to NCHW
    features_nchw = [np.transpose(out, (0, 3, 1, 2)) for out in dequantized_outputs]

    # DEBUG: Log model output shapes and value ranges
    LOG.debug(f"[Pose Postprocess Debug] Number of output feature maps: {len(features_nchw)}")
    for i, feat in enumerate(features_nchw):
        LOG.debug(
            f"[Pose Postprocess Debug] Feature {i}: shape={feat.shape}, "
            f"value range=[{feat.min():.3f}, {feat.max():.3f}], "
            f"mean={feat.mean():.3f}, std={feat.std():.3f}"
        )

    # Decode pose
    input_size = output_infos[0].unpadded_shape[1]

    boxes, keypoints = decode_yolo_pose(
        features_nchw,
        num_keypoints=num_keypoints,
        conf_threshold=conf_threshold,
        input_size=input_size,
    )

    # DEBUG: Log decode results
    LOG.debug(
        f"[Pose Postprocess Debug] Decoded {len(boxes)} boxes, {len(keypoints)} keypoint sets "
        f"(conf_threshold={conf_threshold})"
    )

    return boxes, keypoints
