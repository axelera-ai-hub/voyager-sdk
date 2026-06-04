#!/usr/bin/env python3
"""
YOLO11 Detection Example with ONNX Runtime Postamble Processing

This example demonstrates using axelera.runtime with ONNX Runtime to handle
postamble graph processing. It shows:
- Loading a compiled model and checking for postamble graph
- Using ONNX Runtime to execute compiler-extracted operations
- Simplified postprocessing (only NMS needed after postamble)
- Comparison with manual postamble implementation approach

Model: yolo11s-coco-onnx
Cores: 4 (full utilization of single Metis device)

The postamble graph for YOLO11s contains the detection head operations:
- DFL (Distribution Focal Loss) for box regression
- Sigmoid activation for classification
- Box coordinate transformations
- Output format: [1, 84, 8400] where 84 = 4 box coords + 80 classes

See axruntime_yolo11.py for the manual implementation approach that doesn't
use the postamble graph (implements these operations directly in numpy).
"""

import argparse
import collections.abc
import json
import logging
import queue
import threading
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from axelera.runtime import Context, ModelInstance, TensorInfo

# Import YOLO utilities for preprocessing and postprocessing
from utils import yolo_utils
from utils import visualization

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
LOG = logging.getLogger(__name__)


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
                # While this worker is blocked, OS scheduler will switch to other threads
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
# Axelera-Specific Helper Functions
#
# These functions handle the hardware-specific operations required by the
# Metis accelerator: quantization and padding (input tensors) and their inverses
# (on output tensors).
# ============================================================================


def axelera_quantize(normalized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """
    Axelera-specific: Quantize float32 tensor to int8.

    Formula: quantized = round((float / scale) + zero_point).clip(-128, 127)
    """
    quantized = np.round((normalized / scale) + zero_point)
    return quantized.clip(-128, 127).astype(np.int8)


def axelera_pad(unpadded: np.ndarray, padding: list, zero_point: int) -> np.ndarray:
    """
    Axelera-specific: Add hardware alignment padding.

    IMPORTANT: Pad with zero_point, not zero! This ensures padded regions
    represent the correct value after dequantization.
    """
    return np.pad(unpadded, padding, mode='constant', constant_values=zero_point)


def axelera_depad(padded: np.ndarray, padding: list) -> np.ndarray:
    """
    Axelera-specific: Remove hardware padding.
    """
    slices = tuple(slice(start, -end if end else None) for start, end in padding)
    return padded[slices]


def axelera_dequantize(quantized: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """
    Axelera-specific: Dequantize int8 tensor to float32.

    Formula: float = (int8 - zero_point) * scale
    """
    return (quantized.astype(np.float32) - zero_point) * scale


# ============================================================================
# ONNX Runtime Postamble Handling
# ============================================================================


def initialize_postamble_session(
    model, model_path: Path
) -> Optional[Tuple[ort.InferenceSession, List[str], List[str]]]:
    """
    Initialize ONNX Runtime session for postamble graph if it exists.

    Args:
        model: Loaded axelera.runtime Model object
        model_path: Path to model.json (to locate manifest.json and postamble graph)

    Returns:
        Tuple of (session, input_names, output_names) if postamble exists, None otherwise
    """
    # Read manifest.json to check for postamble graph
    model_dir = model_path.parent
    manifest_path = model_dir / "manifest.json"

    LOG.info(f"Checking for postamble graph in: {model_dir}")

    postamble_path = None
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                # Try both possible field names (postamble_graph and postprocess_graph)
                postamble_filename = manifest.get('postamble_graph') or manifest.get(
                    'postprocess_graph'
                )
                if postamble_filename:
                    postamble_path = model_dir / postamble_filename
                    LOG.info(f"Found postamble reference in manifest: {postamble_filename}")
                else:
                    LOG.info(
                        "No postamble_graph or postprocess_graph field found in manifest.json"
                    )
        except Exception as e:
            LOG.warning(f"Failed to read manifest.json: {e}")
    else:
        LOG.warning(f"Manifest file not found: {manifest_path}")

    if not postamble_path or not postamble_path.exists():
        if postamble_path:
            LOG.warning(f"Postamble graph file not found at: {postamble_path}")
        LOG.info("No postamble graph available - using manual postprocessing")
        return None

    LOG.info(f"Loading postamble graph: {postamble_path}")

    # Configure ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Thread configuration: use 1 for single-threaded applications
    # Can increase for multi-threaded applications if profiling shows benefit
    sess_options.intra_op_num_threads = 1

    # Create session
    session = ort.InferenceSession(
        str(postamble_path), sess_options, providers=['CPUExecutionProvider']
    )

    # Get input/output metadata
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    LOG.info(f"Postamble ONNX: {len(input_names)} inputs, {len(output_names)} outputs")

    return session, input_names, output_names


def prepare_postamble_inputs(
    raw_outputs: List[np.ndarray], output_infos: List[TensorInfo], input_names: List[str]
) -> dict:
    """
    Transform raw AIPU outputs to postamble ONNX inputs.

    Performs the three required transformations:
    1. Depadding (remove channel padding)
    2. Transpose NHWC → NCHW
    3. Dequantize int8 → float32

    Args:
        raw_outputs: List of raw int8 arrays from AIPU (NHWC format, padded)
        output_infos: List of TensorInfo objects with quantization params
        input_names: List of postamble input names (from ONNX session)

    Returns:
        Dictionary mapping input names to numpy arrays (NCHW format, float32)
    """
    postamble_inputs = {}

    for idx, (raw_data, info) in enumerate(zip(raw_outputs, output_infos)):
        # Get dimensions (AIPU outputs are NHWC format)
        N, H, W, C = info.shape

        # Get channel padding info
        c_pad_left, c_pad_right = info.padding[3]
        actual_channels = C - c_pad_left - c_pad_right

        # Step 1: Reshape and remove padding
        tensor_nhwc = raw_data.reshape(N, H, W, C)
        tensor_nhwc = tensor_nhwc[:, :, :, c_pad_left : c_pad_left + actual_channels]

        # Step 2: Transpose to NCHW (ONNX standard format)
        tensor_nchw = np.transpose(tensor_nhwc, (0, 3, 1, 2))

        # Step 3: Dequantize: float_value = (int8_value - zero_point) * scale
        tensor_float = (tensor_nchw.astype(np.float32) - info.zero_point) * info.scale

        # Map to postamble input name
        postamble_inputs[input_names[idx]] = tensor_float

    return postamble_inputs


# ============================================================================
# Preprocessing and Postprocessing
# ============================================================================


def preprocess(image: np.ndarray, input_info: TensorInfo) -> np.ndarray:
    """
    Preprocess image for YOLO detection.

    Delegates to yolo_utils.preprocess_yolo_detection() which handles:
    - Resize, BGR→RGB, normalize
    - Quantization and padding
    - Batch dimension handling
    """
    return yolo_utils.preprocess_yolo_detection(image, input_info, axelera_quantize, axelera_pad)


def postprocess_with_postamble(
    postamble_output: np.ndarray,
    frame_id: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    input_size: int = 640,
) -> np.ndarray:
    """
    Postprocess YOLO detection output from postamble graph.

    The postamble graph outputs [1, 84, 8400] where:
    - 84 = 4 box coordinates + 80 class scores
    - 8400 = total anchor points across all scales (80*80 + 40*40 + 20*20)

    We only need to apply NMS since the postamble already:
    - Decoded boxes using DFL
    - Applied sigmoid to class scores
    - Transformed box coordinates

    Args:
        postamble_output: Output from postamble graph [1, 84, 8400]
        frame_id: Frame number for logging
        conf_threshold: Confidence threshold for detection
        iou_threshold: IoU threshold for NMS
        input_size: Model input size for coordinate scaling

    Returns:
        Final detections [x1, y1, x2, y2, confidence, class_id], shape (N, 6)
    """
    # postamble_output shape: [1, 84, 8400]
    # Reshape to [8400, 84] for easier processing
    predictions = postamble_output[0].T  # [8400, 84]

    # Split into boxes and scores
    boxes = predictions[:, :4]  # [8400, 4] - x_center, y_center, width, height
    scores = predictions[:, 4:]  # [8400, 80] - class scores

    # Get max score and class for each detection
    max_scores = np.max(scores, axis=1)
    max_classes = np.argmax(scores, axis=1)

    # Filter by confidence threshold
    mask = max_scores > conf_threshold
    if not np.any(mask):
        LOG.info(f"Frame {frame_id}: Found 0 detection(s)")
        return np.zeros((0, 6))

    filtered_boxes = boxes[mask]
    filtered_scores = max_scores[mask]
    filtered_classes = max_classes[mask]

    # Convert from center format to corner format
    x_center, y_center, width, height = (
        filtered_boxes[:, 0],
        filtered_boxes[:, 1],
        filtered_boxes[:, 2],
        filtered_boxes[:, 3],
    )
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Stack into [N, 6] format
    detections = np.stack([x1, y1, x2, y2, filtered_scores, filtered_classes], axis=1)

    # Apply NMS
    final_boxes = yolo_utils.apply_nms(detections, iou_threshold=iou_threshold)

    # Log results
    LOG.info(f"Frame {frame_id}: Found {len(final_boxes)} detection(s)")

    for box in final_boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_id = int(cls_id)
        class_name = (
            yolo_utils.COCO_CLASSES[cls_id]
            if cls_id < len(yolo_utils.COCO_CLASSES)
            else f"class_{cls_id}"
        )
        LOG.info(f"  {class_name}: {conf:.3f} at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    return final_boxes


# ============================================================================
# Frame Source Generator
# ============================================================================


def frame_generator(
    video_path: Path, max_frames: int = None
) -> collections.abc.Generator[tuple, None, None]:
    """
    Yield frames one at a time from video file, image directory, or single image.

    This generator pattern processes frames as they're read, matching how frames
    arrive from cameras or network streams.

    Args:
        video_path: Path to video file, image directory, or single image
        max_frames: Maximum number of frames to yield (None = all)

    Yields:
        Tuple of (frame, filename) where frame is numpy array (BGR format) and filename is str
    """
    frame_count = 0

    if video_path.is_dir():
        LOG.info(f"Processing images from directory: {video_path}")
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_paths.extend(sorted(video_path.glob(ext)))

        for img_path in image_paths:
            if max_frames and frame_count >= max_frames:
                break
            frame = cv2.imread(str(img_path))
            if frame is not None:
                LOG.debug(f"Loaded {img_path.name}")
                yield frame, img_path.name
                frame_count += 1

    elif video_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        LOG.info(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_count >= max_frames:
                    break

                LOG.debug(f"Loaded frame {frame_count}")
                filename = f"frame_{frame_count:06d}.jpg"
                yield frame, filename
                frame_count += 1
        finally:
            cap.release()

    else:
        LOG.info(f"Processing single image: {video_path}")
        frame = cv2.imread(str(video_path))
        if frame is None:
            raise ValueError(f"Failed to load image: {video_path}")
        yield frame, video_path.name
        frame_count += 1

    LOG.info(f"Processed {frame_count} frame(s)")


def initialize_model(model_path: Path, aipu_cores: int = 4):
    """
    Initialize the model and set up inference resources.

    This version includes postamble graph handling using ONNX Runtime.

    Args:
        model_path: Path to model.json (compiled model)
        aipu_cores: Number of AIPU cores to use (default 4 = full Metis device)

    Returns:
        Tuple of (context, model, input_infos, output_infos, workers, inputs, outputs, postamble_info)
    """
    LOG.info("Initializing axelera.runtime...")
    ctx = Context()
    try:
        LOG.info(f"Loading model from: {model_path}")
        model = ctx.load_model(str(model_path))

        # Get input and output tensor information
        input_infos = model.inputs()
        output_infos = model.outputs()
        LOG.info(f"Model has {len(input_infos)} input(s) and {len(output_infos)} output(s)")

        # Initialize postamble session if postamble graph exists
        postamble_info = initialize_postamble_session(model, model_path)

        input_info = input_infos[0]
        LOG.info(f"Input shape: {input_info.unpadded_shape}")
        LOG.info(
            f"Input quantization: scale={input_info.scale:.6f}, zero_point={input_info.zero_point}"
        )

        # Create model instances
        batch_size = input_info.shape[0]
        LOG.info(f"Model batch size: {batch_size}")

        num_instances = aipu_cores // batch_size
        LOG.info(f"Creating {num_instances} model instance(s)")

        # Create connections
        connections = []
        for i in range(num_instances):
            conn = ctx.device_connect(device=None, num_sub_devices=batch_size)
            connections.append(conn)

        # Load model instances
        instances = []
        for i, conn in enumerate(connections):
            instance = conn.load_model_instance(
                model, num_sub_devices=batch_size, aipu_cores=batch_size
            )
            instances.append(instance)

        # Prepare buffers
        inputs = []
        outputs = []
        for i in range(num_instances):
            instance_inputs = [np.zeros(info.shape, info.dtype) for info in input_infos]
            inputs.append(instance_inputs)

            instance_outputs = [np.zeros(info.shape, info.dtype) for info in output_infos]
            outputs.append(instance_outputs)

        # Create worker threads
        workers = [InferenceWorker(instance) for instance in instances]
        LOG.info(f"Created {len(workers)} worker thread(s)")

        LOG.info("Initialization complete")
        return ctx, model, input_infos, output_infos, workers, inputs, outputs, postamble_info

    except Exception:
        ctx.release()
        raise


def run_realtime_inference(
    frame_source: collections.abc.Generator[tuple, None, None],
    input_infos: List[TensorInfo],
    output_infos: List[TensorInfo],
    workers: List[InferenceWorker],
    inputs: List[List[np.ndarray]],
    outputs: List[List[np.ndarray]],
    postamble_info: Optional[Tuple[ort.InferenceSession, List[str], List[str]]],
    output_dir: Path = None,
):
    """
    Run inference on frames with postamble processing.

    Args:
        frame_source: Generator yielding (frame, filename) tuples
        input_infos: Input tensor metadata
        output_infos: Output tensor metadata
        workers: Worker threads
        inputs: Pre-allocated input buffers
        outputs: Pre-allocated output buffers
        postamble_info: Tuple of (session, input_names, output_names) or None
        output_dir: Optional directory to save visualizations
    """
    input_info = input_infos[0]

    # Setup visualization colors
    colors = visualization.generate_colors(len(yolo_utils.COCO_CLASSES))

    # Store frames with their metadata for visualization later
    frame_data = {}

    try:
        prefill = len(workers)
        out_frame_id = 0

        # Process frames
        for in_frame_id, (frame, filename) in enumerate(frame_source):
            # Store frame data for later visualization
            frame_data[in_frame_id] = (frame.copy(), filename)

            # Preprocess
            LOG.debug(f"Preprocessing frame {in_frame_id}")
            preprocessed = preprocess(frame, input_info)

            # Submit to worker
            worker_idx = in_frame_id % len(workers)
            inputs[worker_idx][0][:] = preprocessed
            workers[worker_idx].push(in_frame_id, inputs[worker_idx], outputs[worker_idx])

            # Collect results after prefill
            if in_frame_id >= prefill:
                worker_to_collect = out_frame_id % len(workers)
                result_frame_id, result_outputs = workers[worker_to_collect].pop()

                LOG.debug(f"Postprocessing frame {result_frame_id}")

                # Run postamble processing if available
                if postamble_info is not None:
                    session, input_names, output_names = postamble_info

                    # Transform AIPU outputs for postamble
                    postamble_inputs = prepare_postamble_inputs(
                        result_outputs, output_infos, input_names
                    )

                    # Run postamble ONNX graph
                    postamble_outputs = session.run(output_names, postamble_inputs)

                    # Postprocess: only NMS needed
                    boxes = postprocess_with_postamble(
                        postamble_outputs[0],
                        result_frame_id,
                        conf_threshold=0.25,
                        iou_threshold=0.45,
                    )
                else:
                    # No postamble - would need manual implementation
                    # (See axruntime_yolo11.py for manual approach)
                    raise RuntimeError(
                        "No postamble graph found - use axruntime_yolo11.py for manual approach"
                    )

                # Visualize and save if output_dir is provided
                if output_dir is not None:
                    result_frame, result_filename = frame_data[result_frame_id]

                    # Scale boxes from model space (640x640) to original image coordinates
                    scaled_boxes = boxes.copy()
                    if len(scaled_boxes) > 0:
                        orig_height, orig_width = result_frame.shape[:2]
                        scale_x = orig_width / 640.0
                        scale_y = orig_height / 640.0
                        scaled_boxes[:, 0] *= scale_x  # x1
                        scaled_boxes[:, 1] *= scale_y  # y1
                        scaled_boxes[:, 2] *= scale_x  # x2
                        scaled_boxes[:, 3] *= scale_y  # y2

                    vis_image = visualization.draw_detection_boxes(
                        result_frame, scaled_boxes, yolo_utils.COCO_CLASSES, colors
                    )
                    output_path = output_dir / result_filename
                    visualization.save_visualization(vis_image, output_path)
                    LOG.debug(f"Saved visualization to {output_path}")

                    # Clean up frame data to save memory
                    del frame_data[result_frame_id]

                out_frame_id += 1

        # Drain remaining results
        for _ in range(prefill):
            worker_to_collect = out_frame_id % len(workers)
            result_frame_id, result_outputs = workers[worker_to_collect].pop()

            LOG.debug(f"Postprocessing frame {result_frame_id}")

            if postamble_info is not None:
                session, input_names, output_names = postamble_info
                postamble_inputs = prepare_postamble_inputs(
                    result_outputs, output_infos, input_names
                )
                postamble_outputs = session.run(output_names, postamble_inputs)
                boxes = postprocess_with_postamble(postamble_outputs[0], result_frame_id)
            else:
                raise RuntimeError(
                    "No postamble graph found - use axruntime_yolo11.py for manual approach"
                )

            # Visualize and save if output_dir is provided
            if output_dir is not None:
                result_frame, result_filename = frame_data[result_frame_id]

                # Scale boxes from model space (640x640) to original image coordinates
                scaled_boxes = boxes.copy()
                if len(scaled_boxes) > 0:
                    orig_height, orig_width = result_frame.shape[:2]
                    scale_x = orig_width / 640.0
                    scale_y = orig_height / 640.0
                    scaled_boxes[:, 0] *= scale_x  # x1
                    scaled_boxes[:, 1] *= scale_y  # y1
                    scaled_boxes[:, 2] *= scale_x  # x2
                    scaled_boxes[:, 3] *= scale_y  # y2

                vis_image = visualization.draw_detection_boxes(
                    result_frame, scaled_boxes, yolo_utils.COCO_CLASSES, colors
                )
                output_path = output_dir / result_filename
                visualization.save_visualization(vis_image, output_path)
                LOG.debug(f"Saved visualization to {output_path}")

                # Clean up frame data
                del frame_data[result_frame_id]

            out_frame_id += 1

        LOG.info(f"Successfully processed {out_frame_id} frame(s)")
        if output_dir is not None:
            LOG.info(f"Visualizations saved to {output_dir}")

    except Exception as e:
        LOG.error(f"Inference failed: {e}", exc_info=True)
        raise


def cleanup(ctx: Context, workers: List[InferenceWorker]):
    """
    Clean up resources when shutting down.

    Args:
        ctx: Context to release
        workers: Worker threads to shut down
    """
    LOG.info("Shutting down...")

    for worker in workers:
        worker.shutdown()

    for worker in workers:
        worker.join(timeout=5.0)

    ctx.release()

    LOG.info("Shutdown complete")


def main():
    """Parse arguments and run inference."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'model_path',
        type=Path,
        help='Path to model.json (e.g., build/yolo11s-coco-onnx/model.json)',
    )

    parser.add_argument(
        'video_path', type=Path, help='Path to video file, image directory, or single image'
    )

    parser.add_argument(
        '--aipu-cores', type=int, default=4, help='Number of AIPU cores to use (default: 4)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (default: all)',
    )

    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    parser.add_argument(
        '--save-visualizations',
        action='store_true',
        help='Save visualization outputs to outputs_onnxruntime/ directory',
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    if not args.model_path.exists():
        print(f"Error: Model path not found: {args.model_path}")
        return 1

    if not args.video_path.exists():
        print(f"Error: Video path not found: {args.video_path}")
        return 1

    try:
        # Create output directory if visualization is enabled
        output_dir = None
        if args.save_visualizations:
            output_dir = visualization.create_output_directory('onnxruntime')
            LOG.info(f"Saving visualizations to {output_dir}")

        # Initialize with postamble support
        ctx, model, input_infos, output_infos, workers, inputs, outputs, postamble_info = (
            initialize_model(args.model_path, args.aipu_cores)
        )

        # Create frame generator
        frames = frame_generator(args.video_path, args.max_frames)

        # Run inference
        run_realtime_inference(
            frames, input_infos, output_infos, workers, inputs, outputs, postamble_info, output_dir
        )

        # Clean up
        cleanup(ctx, workers)

        return 0
    except Exception as e:
        LOG.error(f"Failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    exit(main())
