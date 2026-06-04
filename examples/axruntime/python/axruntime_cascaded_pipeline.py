#!/usr/bin/env python3
"""
Two-Stage Cascaded Pipeline Example with axelera.runtime

This example demonstrates:
1. Cascaded inference: Detection (Stage 1) → Pose Estimation (Stage 2)
2. Configurable batch size and AIPU core allocation per stage
3. Pipeline coordination: passing detection regions of interest to Stage 2
4. Resource sharing: 4 total AIPU cores (one Metis device) split between two models

Architecture:
    Stage 1: YOLO11s detection (finds person bounding boxes)
    Stage 2: YOLOv8s-pose estimation (estimates keypoints per person)

    Flow:
    1. Image → Stage 1 detection → Person boxes (in 640x640 space)
    2. Scale coordinates to original image dimensions
    3. Crop person ROIs from original image
    4. Stage 2 pose estimation per ROI

Usage:
    python axruntime_cascaded_pipeline.py \\
        --stage1-model build/yolo11s-coco-onnx/model.json \\
        --stage2-model build/yolov8spose-coco-onnx/model.json \\
        --stage1-cores 2 \\
        --stage2-cores 2 \\
        --images path/to/images/*.jpg
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from axelera.runtime import Context, Model, ModelInstance, Connection, TensorInfo

# Import YOLO utilities for preprocessing and postprocessing
from utils import yolo_utils
from utils import visualization

LOG = logging.getLogger(__name__)


# ============================================================================
# Axelera-Specific Helper Functions
# ============================================================================


def axelera_quantize(normalized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Axelera-specific: Quantize float32 tensor to int8."""
    quantized = np.round((normalized / scale) + zero_point)
    return quantized.clip(-128, 127).astype(np.int8)


def axelera_pad(unpadded: np.ndarray, padding: list, zero_point: int) -> np.ndarray:
    """Axelera-specific: Add hardware alignment padding."""
    return np.pad(unpadded, padding, mode='constant', constant_values=zero_point)


def axelera_depad(padded: np.ndarray, padding: list) -> np.ndarray:
    """Axelera-specific: Remove hardware padding."""
    slices = tuple(slice(start, -end if end else None) for start, end in padding)
    return padded[slices]


def axelera_dequantize(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Axelera-specific: Dequantize int8 tensor to float32."""
    return (quantized.astype(np.float32) - zero_point) * scale


# ============================================================================
# Worker Thread Pattern
# ============================================================================


class InferenceWorker(threading.Thread):
    """
    Worker thread that runs inference on a single ModelInstance.

    This pattern is essential for performance because axelera.runtime uses a
    synchronous API - the run() call blocks until inference completes. By using
    multiple worker threads with separate ModelInstances, we can overlap preprocessing
    (CPU) with inference (AIPU) and maximize utilization of the Metis AIPU cores.

    Each worker:
    1. Receives (frame_id, inputs, outputs) from its input queue
    2. Calls instance.run(inputs, outputs) - sends data to AIPU and waits for completion
    3. Returns (frame_id, outputs) to its own output queue
    """

    def __init__(self, instance: ModelInstance):
        """
        Initialize worker thread.

        Args:
            instance: ModelInstance for this worker to use
        """
        self.instance = instance
        self.inqueue = queue.Queue()  # Each worker has its own input queue
        self.outqueue = queue.Queue()  # Each worker has its own output queue
        super().__init__(daemon=True)
        self.start()

    def run(self):
        """Main worker loop - runs in separate thread."""
        while True:
            item = self.inqueue.get()
            if item is None:  # Shutdown signal
                break

            frame_id, inputs, outputs = item

            try:
                # This is a synchronous call - worker thread waits here until inference completes
                self.instance.run(inputs, outputs)
            except Exception as e:
                # Put exception in output queue so main thread can handle it
                self.outqueue.put(e)
                break
            else:
                # Put results in output queue
                self.outqueue.put((frame_id, outputs))

    def push(self, frame_id: int, inputs: List[np.ndarray], outputs: List[np.ndarray]):
        """Add work item to this worker's queue."""
        self.inqueue.put((frame_id, inputs, outputs))

    def pop(self):
        """
        Get next result from this worker's output queue.

        Returns:
            Tuple of (frame_id, outputs)

        Raises:
            Exception: If the worker encountered an error during inference
        """
        result = self.outqueue.get()
        if isinstance(result, Exception):
            raise result
        return result

    def shutdown(self):
        """Signal worker to shut down."""
        self.inqueue.put(None)


# ============================================================================
# Stage Pipeline Classes
# ============================================================================


class Stage1Detection:
    """
    Stage 1: Detection pipeline with configurable AIPU cores.

    Loads YOLO11s detection model and runs inference to detect objects.
    """

    def __init__(self, ctx: Context, model_path: Path, num_cores: int, batch_size: int = 1):
        self.ctx = ctx
        self.num_cores = num_cores
        self.batch_size = batch_size

        # Load model
        LOG.info(f"[Stage 1] Loading detection model from {model_path}")
        self.model = ctx.load_model(str(model_path))

        # Get tensor info
        self.input_info = self.model.inputs()[0]
        self.output_infos = self.model.outputs()

        # Calculate number of instances
        self.num_instances = num_cores // batch_size
        if num_cores % batch_size:
            LOG.warning(
                f"[Stage 1] Cores ({num_cores}) not divisible by batch size ({batch_size})"
            )

        LOG.info(
            f"[Stage 1] Creating {self.num_instances} instances "
            f"(batch_size={batch_size}, {num_cores} cores)"
        )

        # Create connections and instances
        self.connections = [
            ctx.device_connect(None, batch_size) for _ in range(self.num_instances)
        ]
        self.instances = [
            conn.load_model_instance(self.model, num_sub_devices=batch_size, aipu_cores=batch_size)
            for conn in self.connections
        ]

        # Create buffers
        self.inputs = [
            [np.zeros(t.shape, np.int8) for t in self.model.inputs()] for _ in self.instances
        ]
        self.outputs = [
            [np.zeros(t.shape, np.int8) for t in self.model.outputs()] for _ in self.instances
        ]

        # Create workers (each has its own input and output queues)
        self.workers = [InferenceWorker(inst) for inst in self.instances]

    def process_image(self, image: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Process a single image through Stage 1 detection.

        Returns: Array of detected bounding boxes [x1, y1, x2, y2, confidence, class_id]
        """
        # Preprocess using yolo_utils
        preprocessed = yolo_utils.preprocess_yolo_detection(
            image, self.input_info, axelera_quantize, axelera_pad
        )

        # Select worker
        worker_idx = frame_id % len(self.workers)

        # Copy to buffer and submit
        self.inputs[worker_idx][0][:] = preprocessed
        self.workers[worker_idx].push(frame_id, self.inputs[worker_idx], self.outputs[worker_idx])

        # Wait for result from the same worker
        result_frame_id, result_outputs = self.workers[worker_idx].pop()

        # Postprocess using yolo_utils
        boxes = yolo_utils.postprocess_yolo_detection(
            result_outputs, self.output_infos, axelera_depad, axelera_dequantize
        )

        return boxes

    def release(self):
        """Release all resources."""
        LOG.info("[Stage 1] Releasing resources")
        for worker in self.workers:
            worker.shutdown()
        for worker in self.workers:
            worker.join()


class Stage2Pose:
    """
    Stage 2: Pose estimation pipeline with configurable AIPU cores.

    Loads YOLOv8s-pose model and estimates keypoints for detected ROIs.
    """

    def __init__(self, ctx: Context, model_path: Path, num_cores: int, batch_size: int = 1):
        self.ctx = ctx
        self.num_cores = num_cores
        self.batch_size = batch_size

        # Load model
        LOG.info(f"[Stage 2] Loading pose model from {model_path}")
        self.model = ctx.load_model(str(model_path))

        # Get tensor info
        self.input_info = self.model.inputs()[0]
        self.output_infos = self.model.outputs()

        # Calculate number of instances
        self.num_instances = num_cores // batch_size
        if num_cores % batch_size:
            LOG.warning(
                f"[Stage 2] Cores ({num_cores}) not divisible by batch size ({batch_size})"
            )

        LOG.info(
            f"[Stage 2] Creating {self.num_instances} instances "
            f"(batch_size={batch_size}, {num_cores} cores)"
        )

        # Create connections and instances
        self.connections = [
            ctx.device_connect(None, batch_size) for _ in range(self.num_instances)
        ]
        self.instances = [
            conn.load_model_instance(self.model, num_sub_devices=batch_size, aipu_cores=batch_size)
            for conn in self.connections
        ]

        # Create buffers
        self.inputs = [
            [np.zeros(t.shape, np.int8) for t in self.model.inputs()] for _ in self.instances
        ]
        self.outputs = [
            [np.zeros(t.shape, np.int8) for t in self.model.outputs()] for _ in self.instances
        ]

        # Create workers (each has its own input and output queues)
        self.workers = [InferenceWorker(inst) for inst in self.instances]

    def process_roi(self, roi: np.ndarray, roi_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single ROI through Stage 2 pose estimation.

        Returns:
            boxes: Array of [x1, y1, x2, y2, confidence]
            keypoints: Array of [num_kpts, x, y, visibility]
        """
        # Get ROI dimensions (needed for coordinate rescaling later)
        roi_height, roi_width = roi.shape[:2]

        # Preprocess using yolo_utils
        preprocessed = yolo_utils.preprocess_yolo_pose(
            roi, self.input_info, axelera_quantize, axelera_pad
        )

        # Select worker
        worker_idx = roi_id % len(self.workers)

        # Copy to buffer and submit
        self.inputs[worker_idx][0][:] = preprocessed
        self.workers[worker_idx].push(roi_id, self.inputs[worker_idx], self.outputs[worker_idx])

        # Wait for result from the same worker
        result_roi_id, result_outputs = self.workers[worker_idx].pop()

        # Postprocess using yolo_utils
        boxes, keypoints = yolo_utils.postprocess_yolo_pose(
            result_outputs, self.output_infos, axelera_depad, axelera_dequantize
        )

        # Apply NMS to filter duplicate pose detections
        if len(boxes) > 0:
            # Sort by confidence
            sorted_indices = np.argsort(boxes[:, 4])[::-1]
            keep_indices = []

            while len(sorted_indices) > 0:
                # Keep the highest confidence detection
                current = sorted_indices[0]
                keep_indices.append(current)

                if len(sorted_indices) == 1:
                    break

                # Compute IoU with remaining boxes
                current_box = boxes[current]
                remaining_boxes = boxes[sorted_indices[1:]]

                x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
                y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
                x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
                y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

                intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

                box_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                remaining_area = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
                    remaining_boxes[:, 3] - remaining_boxes[:, 1]
                )
                union = box_area + remaining_area - intersection

                iou = np.where(union > 0, intersection / union, 0)

                # Keep boxes with IoU below threshold (0.45)
                mask = iou < 0.45
                sorted_indices = sorted_indices[1:][mask]

            # Filter both boxes and keypoints using NMS results
            keep_indices = np.array(keep_indices)
            original_count = len(boxes)
            boxes = boxes[keep_indices]
            keypoints = keypoints[keep_indices]

            if len(keep_indices) < original_count:
                LOG.debug(f"  ROI {roi_id}: NMS kept {len(keep_indices)}/{original_count} poses")

        # ====================================================================
        # CRITICAL: Coordinate Space Rescaling for Cascaded Pipelines
        # ====================================================================
        # This rescaling step is required for ANY two-stage cascaded application
        # where Stage 2 processes ROIs extracted from Stage 1 detections.
        #
        # Coordinate Flow:
        #   1. Stage 1 detects objects in original image (e.g., 1920x1080)
        #   2. We extract ROI from original image (e.g., 435x857 person crop)
        #   3. Preprocessing STRETCHES ROI to model input size (640x640)
        #   4. Model outputs coordinates in 640x640 space
        #   5. We must RESCALE coordinates back to ROI dimensions (435x857)
        #   6. Later, these ROI coordinates get transformed to full image space
        #
        # Without this rescaling, coordinates would be in wrong space and
        # misaligned with the actual ROI/image. This applies to any cascaded
        # application: detection→pose, detection→classification, etc.
        # ====================================================================
        if len(boxes) > 0 or len(keypoints) > 0:
            model_size = 640  # YOLOv8-pose model input size
            ratio_x = model_size / roi_width
            ratio_y = model_size / roi_height

            # Scale boxes from 640x640 back to ROI dimensions
            if len(boxes) > 0:
                boxes[:, [0, 2]] /= ratio_x  # x1, x2
                boxes[:, [1, 3]] /= ratio_y  # y1, y2

            # Scale keypoints from 640x640 back to ROI dimensions
            if len(keypoints) > 0:
                keypoints[:, :, 0] /= ratio_x  # x coordinates
                keypoints[:, :, 1] /= ratio_y  # y coordinates

        return boxes, keypoints

    def release(self):
        """Release all resources."""
        LOG.info("[Stage 2] Releasing resources")
        for worker in self.workers:
            worker.shutdown()
        for worker in self.workers:
            worker.join()


# ============================================================================
# Cascaded Pipeline Runner
# ============================================================================


def run_cascaded_pipeline(
    images: List[Path],
    stage1: Stage1Detection,
    stage2: Stage2Pose,
    detection_conf_threshold: float = 0.5,
    max_rois_per_image: int = 10,
    output_dir: Path = None,
):
    """
    Run cascaded inference pipeline:
        Image → Stage 1 (Detection) → ROIs → Stage 2 (Pose per ROI)

    Args:
        images: List of image paths
        stage1: Detection pipeline instance
        stage2: Pose estimation pipeline instance
        detection_conf_threshold: Confidence threshold for detections
        max_rois_per_image: Maximum ROIs to process per image
        output_dir: Optional directory to save visualizations
    """
    LOG.info(f"Processing {len(images)} images through cascaded pipeline")

    # Setup visualization directories and colors
    stage1_dir = None
    final_dir = None
    if output_dir is not None:
        stage1_dir = output_dir / "stage1_detection"
        final_dir = output_dir / "final_detection_pose"
        stage1_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

    colors = visualization.generate_colors(len(yolo_utils.COCO_CLASSES))

    total_detections = 0
    total_poses = 0

    for frame_id, image_path in enumerate(images):
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            LOG.warning(f"Failed to load image: {image_path}")
            continue

        orig_height, orig_width = image.shape[:2]

        # Stage 1: Detection
        boxes = stage1.process_image(image, frame_id)

        # Filter by confidence
        boxes = boxes[boxes[:, 4] > detection_conf_threshold]

        # Filter for "person" class only (class_id = 0 in COCO)
        PERSON_CLASS_ID = 0
        person_boxes = boxes[boxes[:, 5] == PERSON_CLASS_ID]

        # Limit ROIs
        person_boxes = person_boxes[:max_rois_per_image]

        num_persons = len(person_boxes)
        total_detections += num_persons

        LOG.info(
            f"Frame {frame_id}: Found {num_persons} person detection(s) "
            f"(threshold={detection_conf_threshold})"
        )

        # Scale boxes to original image coordinates for visualization
        scaled_person_boxes = person_boxes.copy()
        if len(scaled_person_boxes) > 0:
            scale_x = orig_width / 640.0
            scale_y = orig_height / 640.0
            scaled_person_boxes[:, 0] *= scale_x  # x1
            scaled_person_boxes[:, 1] *= scale_y  # y1
            scaled_person_boxes[:, 2] *= scale_x  # x2
            scaled_person_boxes[:, 3] *= scale_y  # y2

        # Save Stage 1 visualization (detection only)
        if output_dir is not None:
            stage1_vis = visualization.draw_detection_boxes(
                image, scaled_person_boxes, yolo_utils.COCO_CLASSES, colors
            )
            stage1_path = stage1_dir / image_path.name
            visualization.save_visualization(stage1_vis, stage1_path)
            LOG.debug(f"Saved stage1 visualization to {stage1_path}")

        if num_persons == 0:
            continue

        # Prepare for final visualization (will collect all keypoints)
        all_keypoints = []
        all_pose_boxes = []

        # Stage 2: Pose estimation for each person ROI
        for roi_idx, box in enumerate(person_boxes):
            x1, y1, x2, y2, conf, cls_id = box

            # Scale coordinates from model input size (640x640) to original image size
            scale_x = orig_width / 640.0
            scale_y = orig_height / 640.0

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Clip to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_width, x2), min(orig_height, y2)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                LOG.warning(
                    f"  Skipping ROI {roi_idx}: invalid box dimensions "
                    f"after clipping [{x1}, {y1}, {x2}, {y2}]"
                )
                continue

            # Extract ROI
            roi = image[y1:y2, x1:x2]

            LOG.debug(
                f"  Extracting ROI {roi_idx}: box=[{x1}, {y1}, {x2}, {y2}], "
                f"size={x2-x1}x{y2-y1}, conf={conf:.3f}"
            )

            # Run pose estimation
            pose_boxes, keypoints = stage2.process_roi(roi, roi_idx)

            if len(keypoints) > 0:
                total_poses += len(keypoints)
                visible_kpts = np.sum(keypoints[0, :, 2] > 0.5)
                LOG.info(
                    f"  ROI {roi_idx}: Detected {len(keypoints)} pose(s), "
                    f"{visible_kpts}/17 keypoints visible"
                )

                # Transform keypoints from ROI coordinates to original image coordinates
                for kpts in keypoints:
                    transformed_kpts = kpts.copy()
                    transformed_kpts[:, 0] += x1  # x offset
                    transformed_kpts[:, 1] += y1  # y offset
                    all_keypoints.append(transformed_kpts)

                # Transform pose boxes from ROI coordinates to original image coordinates
                for pose_box in pose_boxes:
                    transformed_box = pose_box.copy()
                    transformed_box[0] += x1  # x1
                    transformed_box[1] += y1  # y1
                    transformed_box[2] += x1  # x2
                    transformed_box[3] += y1  # y2
                    all_pose_boxes.append(transformed_box)
            else:
                LOG.debug(f"  ROI {roi_idx}: No poses detected")

        # Save final visualization (detection + pose)
        if output_dir is not None and len(all_keypoints) > 0:
            final_vis = image.copy()
            # First draw detection boxes
            final_vis = visualization.draw_detection_boxes(
                final_vis, scaled_person_boxes, yolo_utils.COCO_CLASSES, colors
            )
            # Then draw pose keypoints on top
            keypoints_array = np.array(all_keypoints)
            final_vis = visualization.draw_pose_keypoints(final_vis, keypoints_array)
            final_path = final_dir / image_path.name
            visualization.save_visualization(final_vis, final_path)
            LOG.debug(f"Saved final visualization to {final_path}")

    LOG.info(
        f"Pipeline complete: {total_detections} person(s) detected, "
        f"{total_poses} pose(s) estimated"
    )

    # Warn if no poses were detected despite person detections
    if total_detections > 0 and total_poses == 0:
        LOG.warning(
            f"Note: {total_detections} person(s) detected but 0 poses estimated. "
            f"Check confidence thresholds or ROI quality."
        )

    if output_dir is not None:
        LOG.info(f"Visualizations saved to {output_dir}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model paths
    parser.add_argument(
        '--stage1-model',
        type=Path,
        required=True,
        help='Path to Stage 1 detection model (e.g., build/yolo11s-coco-onnx/model.json)',
    )
    parser.add_argument(
        '--stage2-model',
        type=Path,
        required=True,
        help='Path to Stage 2 pose model (e.g., build/yolov8spose-coco-onnx/model.json)',
    )

    # Core allocation
    parser.add_argument(
        '--stage1-cores',
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help='Number of AIPU cores for Stage 1 (default: 2)',
    )
    parser.add_argument(
        '--stage2-cores',
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help='Number of AIPU cores for Stage 2 (default: 2)',
    )

    # Batch sizes
    parser.add_argument(
        '--stage1-batch', type=int, default=1, help='Batch size for Stage 1 (default: 1)'
    )
    parser.add_argument(
        '--stage2-batch', type=int, default=1, help='Batch size for Stage 2 (default: 1)'
    )

    # Input images
    parser.add_argument('--images', type=Path, nargs='+', required=True, help='Input image paths')

    # Thresholds
    parser.add_argument(
        '--detection-threshold',
        type=float,
        default=0.5,
        help='Detection confidence threshold (default: 0.5)',
    )
    parser.add_argument(
        '--max-rois', type=int, default=10, help='Maximum ROIs to process per image (default: 10)'
    )

    # Logging
    parser.add_argument(
        '-v', '--verbose', action='count', default=0, help='Increase verbosity (use -vv for debug)'
    )

    parser.add_argument(
        '--save-visualizations',
        action='store_true',
        help='Save visualization outputs to outputs_cascaded/ directory',
    )

    args = parser.parse_args()

    # Configure logging
    log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    logging.basicConfig(
        level=log_levels.get(args.verbose, logging.DEBUG),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    # Validate core allocation
    total_cores = args.stage1_cores + args.stage2_cores
    if total_cores > 4:
        LOG.error(
            f"Total cores ({total_cores}) exceeds available cores (4). "
            f"Adjust --stage1-cores and --stage2-cores."
        )
        return 1

    LOG.info(
        f"Core allocation: Stage 1 = {args.stage1_cores}, "
        f"Stage 2 = {args.stage2_cores}, Total = {total_cores}/4"
    )

    try:
        # Create output directory if visualization is enabled
        output_dir = None
        if args.save_visualizations:
            output_dir = visualization.create_output_directory('cascaded')
            LOG.info(f"Saving visualizations to {output_dir}")

        # Create context
        with Context() as ctx:
            # Initialize Stage 1: Detection
            stage1 = Stage1Detection(ctx, args.stage1_model, args.stage1_cores, args.stage1_batch)

            # Initialize Stage 2: Pose
            stage2 = Stage2Pose(ctx, args.stage2_model, args.stage2_cores, args.stage2_batch)

            # Run cascaded pipeline
            run_cascaded_pipeline(
                args.images, stage1, stage2, args.detection_threshold, args.max_rois, output_dir
            )

            # Cleanup
            stage1.release()
            stage2.release()

        LOG.info("Pipeline completed successfully")
        return 0

    except Exception as e:
        LOG.error(f"Pipeline failed: {e}", exc_info=args.verbose >= 2)
        return 1


if __name__ == '__main__':
    exit(main())
