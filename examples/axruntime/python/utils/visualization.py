"""
Visualization utilities for axelera.runtime examples.

This module provides functions to visualize model outputs:
- Object detection bounding boxes
- Pose estimation keypoints and skeleton
- Classification labels

All visualization functions use OpenCV for fast rendering.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


# ============================================================================
# COCO Pose Skeleton Definition
# ============================================================================

# COCO pose skeleton connections (pairs of keypoint indices)
COCO_POSE_SKELETON = [
    (0, 1),
    (0, 2),  # nose to eyes
    (1, 3),
    (2, 4),  # eyes to ears
    (0, 5),
    (0, 6),  # nose to shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (5, 6),  # shoulders
    (5, 11),
    (6, 12),  # shoulders to hips
    (11, 12),  # hips
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
]

# Colors for different body parts (BGR format)
SKELETON_COLORS = {
    'head': (255, 0, 0),  # Blue
    'torso': (0, 255, 0),  # Green
    'left_arm': (0, 0, 255),  # Red
    'right_arm': (255, 255, 0),  # Cyan
    'left_leg': (255, 0, 255),  # Magenta
    'right_leg': (0, 255, 255),  # Yellow
}


def get_skeleton_color(connection: Tuple[int, int]) -> Tuple[int, int, int]:
    """Get color for a skeleton connection based on body part."""
    i, j = connection

    # Head connections
    if (i, j) in [(0, 1), (0, 2), (1, 3), (2, 4)]:
        return SKELETON_COLORS['head']
    # Torso
    elif (i, j) in [(0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12)]:
        return SKELETON_COLORS['torso']
    # Left arm
    elif (i, j) in [(5, 7), (7, 9)]:
        return SKELETON_COLORS['left_arm']
    # Right arm
    elif (i, j) in [(6, 8), (8, 10)]:
        return SKELETON_COLORS['right_arm']
    # Left leg
    elif (i, j) in [(11, 13), (13, 15)]:
        return SKELETON_COLORS['left_leg']
    # Right leg
    elif (i, j) in [(12, 14), (14, 16)]:
        return SKELETON_COLORS['right_leg']
    else:
        return (255, 255, 255)  # White default


# ============================================================================
# Color Generation
# ============================================================================


def generate_colors(num_classes: int, seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for each class.

    Args:
        num_classes: Number of classes to generate colors for
        seed: Random seed for reproducibility

    Returns:
        List of (B, G, R) color tuples
    """
    np.random.seed(seed)
    colors = []

    for i in range(num_classes):
        # Use HSV color space for better distribution
        hue = int(180 * i / num_classes)
        saturation = 255
        value = 255

        # Convert HSV to BGR
        hsv = np.uint8([[[hue, saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))

    return colors


# ============================================================================
# Detection Visualization
# ============================================================================


def draw_detection_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    conf_threshold: float = 0.0,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes on image for object detection.

    Args:
        image: Input image (BGR format)
        boxes: Detection boxes [x1, y1, x2, y2, confidence, class_id], shape (N, 6)
        class_names: List of class names (optional)
        colors: List of colors for each class (optional, will be generated if None)
        conf_threshold: Minimum confidence to draw
        line_thickness: Thickness of bounding box lines

    Returns:
        Image with drawn bounding boxes
    """
    if len(boxes) == 0:
        return image.copy()

    result = image.copy()

    # Generate colors if not provided
    if colors is None:
        max_class_id = int(np.max(boxes[:, 5])) + 1
        colors = generate_colors(max_class_id)

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box

        if conf < conf_threshold:
            continue

        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get color for this class
        color = colors[cls_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, line_thickness)

        # Prepare label
        if class_names and cls_id < len(class_names):
            label = f'{class_names[cls_id]}: {conf:.2f}'
        else:
            label = f'class_{cls_id}: {conf:.2f}'

        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw label background
        label_y1 = max(y1 - label_h - baseline - 5, 0)
        label_y2 = y1
        cv2.rectangle(result, (x1, label_y1), (x1 + label_w + 5, label_y2), color, -1)

        # Draw label text
        cv2.putText(
            result,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return result


# ============================================================================
# Pose Visualization
# ============================================================================


def draw_pose_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    visibility_threshold: float = 0.5,
    keypoint_radius: int = 4,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw pose keypoints and skeleton on image.

    Args:
        image: Input image (BGR format)
        keypoints: Keypoints array, shape (N, num_keypoints, 3) where each keypoint is [x, y, visibility]
        boxes: Optional bounding boxes [x1, y1, x2, y2, confidence], shape (N, 5)
        visibility_threshold: Minimum visibility to draw keypoint
        keypoint_radius: Radius of keypoint circles
        line_thickness: Thickness of skeleton lines

    Returns:
        Image with drawn keypoints and skeleton
    """
    if len(keypoints) == 0:
        return image.copy()

    result = image.copy()

    # Draw each pose
    for pose_idx, kpts in enumerate(keypoints):
        # Draw skeleton connections first (so keypoints appear on top)
        for connection in COCO_POSE_SKELETON:
            pt1_idx, pt2_idx = connection

            if pt1_idx >= len(kpts) or pt2_idx >= len(kpts):
                continue

            x1, y1, vis1 = kpts[pt1_idx]
            x2, y2, vis2 = kpts[pt2_idx]

            # Draw line if both keypoints are visible
            if vis1 > visibility_threshold and vis2 > visibility_threshold:
                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                color = get_skeleton_color(connection)
                cv2.line(result, pt1, pt2, color, line_thickness, cv2.LINE_AA)

        # Draw keypoints
        for kpt_idx, (x, y, vis) in enumerate(kpts):
            if vis > visibility_threshold:
                center = (int(x), int(y))
                # Use different color based on keypoint position
                if kpt_idx < 5:  # Head keypoints
                    color = SKELETON_COLORS['head']
                elif kpt_idx < 11:  # Upper body
                    color = SKELETON_COLORS['torso']
                else:  # Lower body
                    color = SKELETON_COLORS['left_leg']

                cv2.circle(result, center, keypoint_radius, color, -1, cv2.LINE_AA)
                cv2.circle(result, center, keypoint_radius + 1, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw bounding box if provided
        if boxes is not None and pose_idx < len(boxes):
            x1, y1, x2, y2, conf = boxes[pose_idx]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw confidence
            label = f'pose: {conf:.2f}'
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return result


# ============================================================================
# Classification Visualization
# ============================================================================


def draw_classification_label(
    image: np.ndarray,
    class_id: int,
    confidence: float,
    class_names: Optional[List[str]] = None,
    position: str = 'top',
) -> np.ndarray:
    """
    Draw classification label on image.

    Args:
        image: Input image (BGR format)
        class_id: Predicted class ID
        confidence: Prediction confidence
        class_names: List of class names (optional)
        position: Position of label ('top', 'bottom', 'center')

    Returns:
        Image with drawn label
    """
    result = image.copy()

    # Prepare label text
    if class_names and class_id < len(class_names):
        label = f'{class_names[class_id]}'
        conf_text = f'Confidence: {confidence:.3f}'
    else:
        label = f'Class {class_id}'
        conf_text = f'Confidence: {confidence:.3f}'

    # Get text size for both lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    (conf_w, conf_h), _ = cv2.getTextSize(conf_text, font, font_scale * 0.7, thickness)

    # Calculate position
    h, w = image.shape[:2]
    max_w = max(label_w, conf_w)
    total_h = label_h + conf_h + 30

    if position == 'top':
        x = (w - max_w) // 2
        y = 50
    elif position == 'bottom':
        x = (w - max_w) // 2
        y = h - total_h - 20
    else:  # center
        x = (w - max_w) // 2
        y = (h - total_h) // 2

    # Draw background rectangle
    padding = 20
    cv2.rectangle(
        result,
        (x - padding, y - label_h - padding),
        (x + max_w + padding, y + conf_h + padding + 10),
        (0, 0, 0),
        -1,
    )

    # Draw border
    cv2.rectangle(
        result,
        (x - padding, y - label_h - padding),
        (x + max_w + padding, y + conf_h + padding + 10),
        (0, 255, 0),
        2,
    )

    # Draw label text
    cv2.putText(result, label, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Draw confidence text
    cv2.putText(
        result,
        conf_text,
        (x, y + label_h + 20),
        font,
        font_scale * 0.7,
        (200, 200, 200),
        thickness,
        cv2.LINE_AA,
    )

    return result


# ============================================================================
# File I/O
# ============================================================================


def save_visualization(image: np.ndarray, output_path: Path, create_dirs: bool = True) -> None:
    """
    Save visualization image to file.

    Args:
        image: Image to save
        output_path: Path to save image to
        create_dirs: Whether to create parent directories if they don't exist
    """
    if create_dirs:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), image)


def create_output_directory(base_name: str, base_path: Optional[Path] = None) -> Path:
    """
    Create output directory for visualizations.

    Args:
        base_name: Base name for output directory (e.g., 'yolo11')
        base_path: Base path for output directory (default: current directory)

    Returns:
        Path to created output directory
    """
    if base_path is None:
        base_path = Path.cwd()

    output_dir = base_path / f"outputs_{base_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir
