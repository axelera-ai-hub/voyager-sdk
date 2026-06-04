#!/usr/bin/env python3
"""
Quick Start Implementation: Minimal Axelera Integration for ImageNet Classification

This example demonstrates the MINIMAL code needed to integrate Axelera hardware
into an existing inference pipeline. It runs end-to-end with ImageNet ResNet models.

========================================
WHAT YOU NEED TO CHANGE FOR YOUR MODEL:
========================================
1. preprocess_imagenet() → your_preprocess()
   - Change resize, normalization, etc. for your model
2. postprocess_imagenet() → your_postprocess()
   - Change output decoding for your model type
3. Everything else stays the same!

Usage:
    python axruntime_quickstart.py <model_path> <image_path> [--cores 4] [--labels labels.txt]

Example:
    python axruntime_quickstart.py build/resnet50/model.json test.jpg --labels imagenet_labels.txt
"""

import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from axelera.runtime import Context, TensorInfo
from threading import Thread
from queue import Queue

# Import visualization utilities
from utils import visualization

# ============================================================================
# Global State (initialized once at startup)
# ============================================================================
axelera_ctx = None
axelera_workers = []
axelera_inputs = []
axelera_outputs = []
axelera_input_info = None
axelera_output_infos = []


# ============================================================================
# Worker Thread (enables parallel inference across AIPU cores)
# ============================================================================
class InferenceWorker(Thread):
    """Runs inference on dedicated thread."""

    def __init__(self, instance):
        super().__init__(daemon=True)
        self.instance = instance
        self.inqueue = Queue()
        self.outqueue = Queue()
        self.start()

    def run(self):
        while True:
            item = self.inqueue.get()
            if item is None:  # Shutdown signal
                break

            frame_id, inputs, outputs = item
            try:
                # Blocking call - inference on Axelera hardware
                self.instance.run(inputs, outputs)
                self.outqueue.put((frame_id, outputs))
            except Exception as e:
                self.outqueue.put(e)
                break


# ============================================================================
# STEP 1: Initialize Once at Startup
# ============================================================================
def initialize_axelera(model_path: str, num_cores: int = 4):
    """
    Initialize Axelera hardware. Call this ONCE at application startup.

    Args:
        model_path: Path to compiled model.json file
        num_cores: Number of AIPU cores to use (4 = full Metis device)

    Returns:
        Tuple of (input_info, output_infos) for quantization parameters
    """
    global axelera_ctx, axelera_workers, axelera_inputs, axelera_outputs
    global axelera_input_info, axelera_output_infos

    # 1. Create context (root object for all resources)
    axelera_ctx = Context()

    # 2. Load compiled model into host memory
    model = axelera_ctx.load_model(model_path)

    # 3. Get tensor metadata (scale, zero_point, padding, shape)
    axelera_input_info = model.inputs()[0]
    axelera_output_infos = model.outputs()

    # 4. Calculate number of parallel instances
    # Formula: num_instances = num_cores / batch_size
    batch_size = axelera_input_info.shape[0]
    num_instances = num_cores // batch_size

    # 5. Create connections (reserve hardware cores) and load model instances
    connections = []
    instances = []
    for _ in range(num_instances):
        conn = axelera_ctx.device_connect(None, num_sub_devices=batch_size)
        connections.append(conn)

        instance = conn.load_model_instance(
            model, num_sub_devices=batch_size, aipu_cores=batch_size
        )
        instances.append(instance)

    # 6. Pre-allocate buffers (one set per instance for parallel execution)
    for _ in range(num_instances):
        inp = [np.zeros(info.shape, np.int8) for info in model.inputs()]
        axelera_inputs.append(inp)

        out = [np.zeros(info.shape, np.int8) for info in model.outputs()]
        axelera_outputs.append(out)

    # 7. Create worker threads (one per instance)
    axelera_workers = [InferenceWorker(inst) for inst in instances]

    return axelera_input_info, axelera_output_infos


# ============================================================================
# STEP 2: Run Inference in Your Loop
# ============================================================================
def run_inference(preprocessed_int8: np.ndarray) -> list:
    """
    Run inference on Axelera hardware. Call this in your processing loop.

    This replaces your existing model.run() call.

    Args:
        preprocessed_int8: Your preprocessed int8 array (already quantized & padded)

    Returns:
        List of int8 output arrays (you must dequantize in postprocessing)
    """
    global axelera_workers, axelera_inputs, axelera_outputs

    # Select worker (round-robin)
    frame_id = run_inference.counter
    worker_idx = frame_id % len(axelera_workers)
    worker = axelera_workers[worker_idx]
    run_inference.counter += 1

    # Copy into worker's pre-allocated buffer
    axelera_inputs[worker_idx][0][:] = preprocessed_int8

    # Submit to worker (async)
    worker.inqueue.put((frame_id, axelera_inputs[worker_idx], axelera_outputs[worker_idx]))

    # Collect result (blocks until inference completes)
    result = worker.outqueue.get()
    if isinstance(result, Exception):
        raise result

    result_frame_id, int8_outputs = result

    return int8_outputs


# Initialize frame counter
run_inference.counter = 0


# ============================================================================
# STEP 3: Cleanup at Shutdown
# ============================================================================
def cleanup_axelera():
    """Release Axelera hardware resources. Call this ONCE at shutdown."""
    global axelera_ctx, axelera_workers

    # Shutdown worker threads
    for worker in axelera_workers:
        worker.inqueue.put(None)

    for worker in axelera_workers:
        worker.join(timeout=5.0)

    # Release all hardware resources
    axelera_ctx.release()


# ============================================================================
# EXAMPLE: ImageNet Classification Pre/Post Processing
# (Replace these with your own model-specific functions)
# ============================================================================

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STDDEV = [0.229, 0.224, 0.225]


def preprocess_imagenet(image_path: Path, input_info: TensorInfo) -> np.ndarray:
    """
    EXAMPLE preprocessing for ImageNet ResNet models.
    REPLACE THIS with your model's preprocessing!

    Args:
        image_path: Path to input image
        input_info: TensorInfo from initialize_axelera()

    Returns:
        int8 array ready for inference (quantized + padded)
    """
    # Get target shape (without batch dimension)
    batch, height, width, _ = input_info.unpadded_shape

    # Read and resize image
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize using ImageNet statistics
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STDDEV)

    # Quantize: float32 → int8
    quantized = np.round(image / input_info.scale + input_info.zero_point)
    quantized = quantized.clip(-128, 127).astype(np.int8)

    # Pad for hardware alignment (skip first dimension - batch)
    padded = np.pad(
        quantized, input_info.padding[1:], mode='constant', constant_values=input_info.zero_point
    )

    # Handle batch size > 1
    if batch > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch, axis=0)

    return padded


def postprocess_imagenet(
    int8_output: np.ndarray, output_info: TensorInfo, labels: list, image_path: Path
) -> dict:
    """
    EXAMPLE postprocessing for ImageNet classification.
    REPLACE THIS with your model's postprocessing!

    Args:
        int8_output: Raw int8 output from inference
        output_info: TensorInfo for this output
        labels: List of class labels
        image_path: Path to input image (for logging)

    Returns:
        dict with 'class_id', 'label', 'score'
    """
    # Depad and dequantize
    depadded = int8_output[tuple(slice(b, -e if e else None) for b, e in output_info.padding)]
    depadded = depadded.squeeze()
    float_output = (depadded.astype(np.float32) - output_info.zero_point) * output_info.scale

    # Top-1 classification
    class_id = np.argmax(float_output)
    score = float_output[class_id]
    label = labels[class_id] if class_id < len(labels) else "(no label)"

    print(f"{image_path.name}: class={class_id} label='{label}' score={score:.3f}")

    return {'class_id': int(class_id), 'label': label, 'score': float(score)}


# ============================================================================
# Main Example
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'model_path', type=Path, help='Path to model.json (or directory containing it)'
    )
    parser.add_argument(
        'image_paths', type=Path, nargs='+', help='Path(s) to image(s) or directory'
    )
    parser.add_argument('--cores', type=int, default=4, help='Number of AIPU cores (default: 4)')

    # Default labels path
    default_labels = os.path.expandvars(
        "$AXELERA_FRAMEWORK/ax_datasets/labels/imagenet1000_clsidx_to_labels.txt"
    )
    parser.add_argument(
        '--labels',
        type=Path,
        default=default_labels,
        help='Path to labels file (default: ImageNet 1000)',
    )
    parser.add_argument(
        '--save-visualizations',
        action='store_true',
        help='Save visualization outputs to outputs_classification/ directory',
    )
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model_path
    if model_path.is_dir():
        model_path = model_path / "model.json"
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        return 1

    # Load labels
    labels = args.labels.read_text().splitlines() if args.labels.exists() else []

    # Collect image paths
    image_paths = []
    for path in args.image_paths:
        if path.is_dir():
            image_paths.extend(path.glob("*.jpg"))
            image_paths.extend(path.glob("*.png"))
        else:
            image_paths.append(path)

    if not image_paths:
        print("Error: No images found")
        return 1

    print(f"=== Fast Path Integration Example ===")
    print(f"Model: {model_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Cores: {args.cores}\n")

    # Create output directory if visualization is enabled
    output_dir = None
    if args.save_visualizations:
        output_dir = visualization.create_output_directory('classification')
        print(f"Saving visualizations to {output_dir}\n")

    # STEP 1: Initialize once at startup
    input_info, output_infos = initialize_axelera(str(model_path), args.cores)
    print(f"✓ Initialized with {len(axelera_workers)} worker(s)\n")

    try:
        # STEP 2: Process images
        results = []
        for image_path in image_paths:
            # Your preprocessing (CHANGE THIS for your model)
            preprocessed = preprocess_imagenet(image_path, input_info)

            # Inference on Axelera (THIS STAYS THE SAME)
            outputs = run_inference(preprocessed)

            # Your postprocessing (CHANGE THIS for your model)
            result = postprocess_imagenet(outputs[0], output_infos[0], labels, image_path)
            results.append(result)

            # Visualize and save if enabled
            if output_dir is not None:
                # Load original image for visualization
                image = cv2.imread(str(image_path))
                if image is not None:
                    vis_image = visualization.draw_classification_label(
                        image, result['class_id'], result['score'], labels
                    )
                    output_path = output_dir / image_path.name
                    visualization.save_visualization(vis_image, output_path)

        print(f"\n✓ Processed {len(results)} image(s)")
        if output_dir is not None:
            print(f"✓ Visualizations saved to {output_dir}")

    finally:
        # STEP 3: Cleanup at shutdown
        cleanup_axelera()
        print("✓ Cleaned up")

    return 0


if __name__ == '__main__':
    exit(main())
