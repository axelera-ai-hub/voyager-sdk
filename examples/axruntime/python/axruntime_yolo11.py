#!/usr/bin/env python3
"""
YOLO11 Detection Example for axelera.runtime Low-Level API

This example demonstrates using axelera.runtime to run inference on a YOLO11s
detection model with configurable batch sizes. It shows:
- Loading a compiled model and understanding tensor metadata
- Connecting to an Axelera Metis device and configuring resources
- Setting up worker threads for parallel inference
- Configurable batch sizes and core allocation
- Example preprocessing and postprocessing pipeline
- Best-practice resource management

Model: yolo11s-coco-onnx
Cores: 4 (full utilization of single Metis device)
Batch sizes: Supports models compiled with batch_size=1, 2, or 4
"""

import argparse
import collections.abc
import logging
import queue
import threading
from pathlib import Path
from typing import List
import cv2
import numpy as np
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
# Note that these operations can be optimized in some cases by fusing them with other
# preprocessing steps. In this example, they are separated for simplicity.
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
# Preprocessing and Postprocessing Wrappers
#
# These wrap the YOLO11-specific pre- and post-processing functions found in
# yolo_utils.py, in addition to the Axelera helper functions
# and logging functionality.
# ============================================================================


def preprocess(image: np.ndarray, input_info: TensorInfo) -> np.ndarray:
    """
    Preprocess image for YOLO detection.

    Delegates to yolo_utils.preprocess_yolo_detection() which handles:
    - Resize, BGR→RGB, normalize
    - Quantization and padding (using our helpers above)
    - Batch dimension handling
    """
    return yolo_utils.preprocess_yolo_detection(image, input_info, axelera_quantize, axelera_pad)


def postprocess(
    outputs: List[np.ndarray],
    output_infos: List[TensorInfo],
    frame_id: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """
    Postprocess YOLO detection outputs.

    Delegates to yolo_utils.postprocess_yolo_detection() which handles:
    - Depadding and dequantization (using our helpers above)
    - YOLO prediction decoding (DFL + classification)
    - Non-Maximum Suppression (NMS)

    Then logs the results for this example.
    """
    final_boxes = yolo_utils.postprocess_yolo_detection(
        outputs,
        output_infos,
        axelera_depad,
        axelera_dequantize,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

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
    arrive from cameras or network streams. Only the current frame is in memory
    at any given time.

    Args:
        video_path: Path to video file, image directory, or single image
        max_frames: Maximum number of frames to yield (None = all)

    Yields:
        Tuple of (frame, filename) where frame is numpy array (BGR format) and filename is str
    """
    frame_count = 0

    if video_path.is_dir():
        # Image directory: yield each image as it's read
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
        # Video file: yield each frame as it's read
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
        # Single image file
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

    Call this ONCE at application startup, then use the returned objects
    for all inferences over the application lifetime.

    In this example, only one Model is used (YOLO11s), and four Connections
    and ModelInstances are created to utilize all four cores.

    For using multiple Models at once, see axruntime_cascaded_pipeline.py.

    Steps:
    1. Create Context (root object for resource management)
    2. Load Model and inspect tensor metadata
    3. Understand batch size and model instance relationship
    4. Create Connections to device
    5. Load ModelInstances for parallel execution
    6. Pre-allocate input/output buffers
    7. Create worker thread pool

    Args:
        model_path: Path to model.json (compiled model)
        aipu_cores: Number of AIPU cores to use (default 4 = full Metis device)

    Returns:
        Tuple of (context, model, input_infos, output_infos, workers, inputs, outputs)
    """
    # ==========================
    # Step 1: Create Context
    # ==========================
    # Context is the root object that manages all runtime resources.
    # Keep this Context alive for the application lifetime.

    LOG.info("Initializing axelera.runtime...")
    ctx = Context()
    try:
        LOG.info(f"Loading model from: {model_path}")

        # ==========================
        # Step 2: Load Model
        # ==========================
        # Model contains the compiled model weights and metadata.
        # This steps loads the model into host memory, not onto the Axelera AIPU.
        # This function expects the path to the model.json file in the compiled model directory.
        # Load this once and reuse for multiple model instances.
        model = ctx.load_model(str(model_path))

        # ==========================
        # Step 3: Inspect Tensor Metadata
        # ==========================
        # Get input and output tensor information
        # This metadata tells us how to prepare data for inference and what
        # buffer sizes to allocate for inputs and outputs
        input_infos = model.inputs()
        output_infos = model.outputs()

        LOG.info(f"Model has {len(input_infos)} input(s) and {len(output_infos)} output(s)")

        # For this example, we'll work with the first input
        input_info = input_infos[0]

        LOG.info(f"Input shape (with padding): {input_info.shape}")
        LOG.info(f"Input shape (without padding): {input_info.unpadded_shape}")
        LOG.info(f"Input padding: {input_info.padding}")
        LOG.info(
            f"Input quantization: scale={input_info.scale:.6f}, zero_point={input_info.zero_point}"
        )
        LOG.info(f"Input dtype: {input_info.dtype}")

        # ==========================
        # Step 4: Understand Batch Size and Model Instances
        # ==========================
        # Batch size is determined at compile time and fixed in the model.
        # The batch size determines how many images are processed in a single inference call.
        #
        # Examples with 4 AIPU cores on Metis:
        # - batch_size=1: Create 4 ModelInstances (4 cores / 1 = 4 instances)
        #                 Process 4 images in parallel, each on 1 core
        # - batch_size=2: Create 2 ModelInstances (4 cores / 2 = 2 instances)
        #                 Process 2 batches in parallel, each batch has 2 images on 2 cores
        # - batch_size=4: Create 1 ModelInstance  (4 cores / 4 = 1 instance)
        #                 Process 1 batch with 4 images on all 4 cores
        #
        # Formula: num_instances = aipu_cores / batch_size

        batch_size = input_info.shape[0]
        LOG.info(f"Model batch size: {batch_size}")

        # Calculate number of instances
        if aipu_cores % batch_size != 0:
            LOG.warning(
                f"AIPU cores ({aipu_cores}) not evenly divisible by batch size ({batch_size}). "
                f"This may lead to underutilization."
            )

        num_instances = aipu_cores // batch_size
        LOG.info(f"Creating {num_instances} model instance(s) for {aipu_cores} AIPU cores")
        LOG.info(
            f"Formula: {num_instances} instances = {aipu_cores} cores / {batch_size} batch_size"
        )

        # ==========================
        # Step 5: Create Connections
        # ==========================
        # Connection represents reserved hardware resources (subdevices/cores).
        # We create one Connection per ModelInstance (1:1 pattern).
        #
        # device_connect parameters:
        # - device: Which Axelera device to connect to. None means automatically select available device
        # - num_sub_devices: Number of cores to reserve (should match batch_size)

        connections = []
        for i in range(num_instances):
            conn = ctx.device_connect(device=None, num_sub_devices=batch_size)
            connections.append(conn)
            LOG.info(f"Created connection {i+1}/{num_instances}")

        # ==========================
        # Step 6: Load ModelInstances
        # ==========================
        # Loads an instance of the specified Model onto the Axelera device.
        # Each ModelInstance:
        # - Is tied to a specific Connection (allocated AIPU cores)
        # - Can run inference independently
        # - Should be used by only one thread at a time
        #
        # load_model_instance parameters:
        # - model: The Model object to instantiate
        # - num_sub_devices: Number of AIPU cores for this instance (1 for batch_size=1)
        # - aipu_cores: L2 memory allocation (1 core = 25% of device's L2 memory)

        instances = []
        for i, conn in enumerate(connections):
            instance = conn.load_model_instance(
                model, num_sub_devices=batch_size, aipu_cores=batch_size
            )
            instances.append(instance)
            LOG.info(f"Loaded model instance {i+1}/{num_instances}")

        # ==========================
        # Step 7: Prepare Buffers
        # ==========================
        # Pre-allocate input and output buffers for each instance.
        # Reusing buffers is important for performance - don't allocate new ones each frame!
        #
        # Each instance needs its own buffers because multiple instances
        # run in parallel in different threads.

        inputs = []
        outputs = []
        for i in range(num_instances):
            # Allocate input buffers
            instance_inputs = [np.zeros(info.shape, info.dtype) for info in input_infos]
            inputs.append(instance_inputs)

            # Allocate output buffers
            instance_outputs = [np.zeros(info.shape, info.dtype) for info in output_infos]
            outputs.append(instance_outputs)

        LOG.info(f"Allocated buffers for {num_instances} instance(s)")

        # ==========================
        # Step 8: Create Worker Thread Pool
        # ==========================
        # Create one worker thread per ModelInstance.
        # Each worker has its own input and output queues.

        workers = [InferenceWorker(instance) for instance in instances]
        LOG.info(f"Created {len(workers)} worker thread(s)")

        LOG.info("Initialization complete")
        return ctx, model, input_infos, output_infos, workers, inputs, outputs

    except Exception:
        # If initialization fails, clean up and re-raise
        ctx.release()
        raise


def run_realtime_inference(
    frame_source: collections.abc.Generator[tuple, None, None],
    input_infos: List[TensorInfo],
    output_infos: List[TensorInfo],
    workers: List[InferenceWorker],
    inputs: List[List[np.ndarray]],
    outputs: List[List[np.ndarray]],
    output_dir: Path = None,
):
    """
    Run inference on frames as they arrive from a stream.

    This function processes frames one-at-a-time as they're yielded from the
    frame_source generator, demonstrating the pattern used when frames arrive
    continuously from cameras, video files, or network streams.

    The initialize_model() function is called ONCE at startup, then this function
    processes frames continuously using the pre-allocated resources.

    Args:
        frame_source: Generator yielding (frame, filename) tuples
        input_infos: Input tensor metadata from model
        output_infos: Output tensor metadata from model
        workers: List of inference worker threads
        inputs: Pre-allocated input buffers
        outputs: Pre-allocated output buffers
        output_dir: Optional directory to save visualizations
    """
    input_info = input_infos[0]  # Use first input

    # Setup visualization colors
    colors = visualization.generate_colors(len(yolo_utils.COCO_CLASSES))

    # ==========================
    # Run Inference Pipeline
    # ==========================
    # PIPELINE PATTERN
    # This pattern processes frames as they arrive, keeping all workers busy:
    #
    # 1. Submit first N frames (prefill) to start all workers
    # 2. For each subsequent frame: collect one result, then submit the new frame
    # 3. After all frames submitted, drain remaining N results
    #
    # This hides inference latency by overlapping preprocessing (CPU) with inference (AIPU).

    # Store frames with their metadata for visualization later
    frame_data = {}

    try:
        prefill = len(workers)
        out_frame_id = 0

        # Process frames as they arrive from the generator
        for in_frame_id, (frame, filename) in enumerate(frame_source):
            # Store frame data for later visualization
            frame_data[in_frame_id] = (frame.copy(), filename)

            # Preprocess the incoming frame
            LOG.debug(f"Preprocessing frame {in_frame_id}")
            preprocessed = preprocess(frame, input_info)

            # Round-robin assignment to workers
            worker_idx = in_frame_id % len(workers)

            # Copy preprocessed data into this worker's input buffer
            inputs[worker_idx][0][:] = preprocessed

            # Submit to worker for inference
            workers[worker_idx].push(in_frame_id, inputs[worker_idx], outputs[worker_idx])
            LOG.debug(f"Submitted frame {in_frame_id} to worker {worker_idx}")

            # After prefill phase, collect one result for each new frame submitted
            if in_frame_id >= prefill:
                # Collect from workers in round-robin order (preserves submission order)
                worker_to_collect = out_frame_id % len(workers)
                result_frame_id, result_outputs = workers[worker_to_collect].pop()
                LOG.debug(f"Postprocessing frame {result_frame_id}")
                boxes = postprocess(result_outputs, output_infos, result_frame_id)

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

        # Drain phase: Collect remaining results from the last N frames
        for _ in range(prefill):
            # Collect from workers in round-robin order (preserves submission order)
            worker_to_collect = out_frame_id % len(workers)
            result_frame_id, result_outputs = workers[worker_to_collect].pop()
            LOG.debug(f"Postprocessing frame {result_frame_id}")
            boxes = postprocess(result_outputs, output_infos, result_frame_id)

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

    Call this when your application exits to properly release all resources.

    Args:
        ctx: Context to release
        workers: Worker threads to shut down
    """
    LOG.info("Shutting down...")

    # Shut down all workers gracefully
    for worker in workers:
        worker.shutdown()

    for worker in workers:
        worker.join(timeout=5.0)

    # Release all runtime resources
    # This releases ModelInstances, Connections, Model, and Context
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
        help='Save visualization outputs to outputs_yolo11/ directory',
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
            output_dir = visualization.create_output_directory('yolo11')
            LOG.info(f"Saving visualizations to {output_dir}")

        # Initialize once at startup
        ctx, model, input_infos, output_infos, workers, inputs, outputs = initialize_model(
            args.model_path, args.aipu_cores
        )

        # Create frame generator (yields frames one at a time)
        frames = frame_generator(args.video_path, args.max_frames)

        # Run inference on streaming frames
        run_realtime_inference(
            frames, input_infos, output_infos, workers, inputs, outputs, output_dir
        )

        # Clean up on shutdown
        cleanup(ctx, workers)

        return 0
    except Exception as e:
        LOG.error(f"Failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    exit(main())
