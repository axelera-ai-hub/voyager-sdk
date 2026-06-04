# Cascaded Inference Pipelines

## Overview

This guide explains how to run multiple models in sequence using the low-level `axelera.runtime` Python API. The example demonstrates a two-stage pipeline: YOLO11s object detection followed by YOLOv8s-pose pose estimation. If you haven't completed the [Quick Start](axruntime-python-quickstart.md) yet, start there to understand the basic API usage and core concepts.

Cascaded pipelines require careful AIPU core allocation, coordinate space management, and understanding of the low-level resource management APIs.

**Prerequisites:**
- Completed [Quick Start](axruntime-python-quickstart.md) and [Basic Usage Tutorial](basic-usage-tutorial.md)
- Understand Context, Model, Connection, ModelInstance, and resource allocation
- SDK virtual environment activated: `source venv/bin/activate`
- Compiled models:
  - Detection model: `voyager-sdk/build/yolo11s-coco-onnx/1/model.json`
  - Pose model: `voyager-sdk/build/yolov8spose-coco-onnx/1/model.json`
  - You can download both precompiled models with: `axdownloadmodel yolo11s-coco-onnx yolov8spose-coco-onnx`


**Complete example:** [axruntime_cascaded_pipeline.py](../../../examples/axruntime/python/axruntime_cascaded_pipeline.py)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Input Image (original dimensions, e.g., 1920×1080) │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
    ┌───────────────────────────┐
    │   Stage 1: Detection      │
    │   Model: YOLO11s          │
    │   Cores: 2 (configurable) │
    │   Batch: 1 (configurable) │
    └────────────┬──────────────┘
                 │
                 │ Person boxes (in 640×640 model space)
                 ▼
    ┌───────────────────────────┐
    │  Scale coordinates to     │
    │  original image space     │
    │  (640×640 → 1920×1080)    │
    └────────────┬──────────────┘
                 │
                 │ Scaled boxes in image space
                 ▼
    ┌───────────────────────────┐
    │  Crop person ROIs from    │
    │  original image           │
    │  (e.g., 435×857 crops)    │
    └────────────┬──────────────┘
                 │
                 │ Person ROI images (variable sizes)
                 ▼
    ┌───────────────────────────┐
    │   Stage 2: Pose           │
    │   Model: YOLOv8s-pose     │
    │   Cores: 2 (configurable) │
    │   Preprocess: Stretch     │
    │   ROI → 640×640           │
    └────────────┬──────────────┘
                 │
                 │ Keypoints in 640×640 model space ⚠️
                 ▼
    ┌───────────────────────────┐
    │ ⚠️ CRITICAL STEP:         │
    │  Rescale coordinates      │
    │  640×640 → ROI size       │
    │  (e.g., 640×640 → 435×857)│
    └────────────┬──────────────┘
                 │
                 │ Keypoints in ROI space
                 ▼
    ┌───────────────────────────┐
    │  Apply NMS (filter        │
    │  duplicate detections)    │
    └────────────┬──────────────┘
                 │
                 │ Filtered keypoints
                 ▼
    ┌───────────────────────────┐
    │  Transform to image space │
    │  (ROI coords → full image)│
    └────────────┬──────────────┘
                 │
                 ▼
    ┌───────────────────────────┐
    │   Final Results           │
    │   Person boxes + Poses    │
    │   (all in image space)    │
    └───────────────────────────┘
```

---

## Using Multiple Models

The basic API components remain the same - Context, Model, Connection, ModelInstance. Here's how to use them for multiple models:

- **Context**: Only one Context for all models
- **Model**: One Model object per model (2 in this example: YOLO11s and YOLOv8s-pose)
- **Connection and ModelInstance**: Specific to each model instance (see [Batch Size and AIPU Cores](#batch-size-and-aipu-cores-deep-dive))

### Implementation Pattern

In the example, there is a Python class for each pipeline stage (Stage1Detection and Stage2Pose). These classes own the Connection and ModelInstance objects for each stage loaded onto the AIPU. While this example shows a cascaded pipeline, the API usage is the same for independent models.

---

## Batch Size and AIPU Cores Deep Dive

For the basics of batch size and the formula `num_instances = aipu_cores / batch_size`, see [Basic Usage Tutorial - Batch Size](basic-usage-tutorial.md#batch-size-aipu-cores-and-num_instances).

### Core Allocation Principle

When using **multiple models** (cascaded pipelines), you must allocate AIPU cores between models. Each Metis device has 4 cores total.

**Key principle:**

**Total cores used = Σ(batch_size × num_instances) for each model ≤ 4**

The number of InferenceWorker threads should match the number of ModelInstances in use at once.

### Example Allocations

Core allocation should be based on performance testing. In general, heavier models should have more cores allocated.

**Example allocations for 4 cores:**

| Configuration | Stage 1 Cores | Stage 2 Cores | Use Case |
|--------------|---------------|---------------|----------|
| Balanced | 2 | 2 | Similar computation time per stage |
| Detection-heavy | 3 | 1 | Stage 1 processes full images, Stage 2 processes small ROIs |
| Pose-heavy | 1 | 3 | Few detections per frame, expensive pose detection |

### Multi-Model Examples

| Model | Batch Size | AIPU Cores Allocated | Num Instances | Notes |
|-------|-----------|---------------------|---------------|-------|
| Detection | 1 | 2 | 2 | Stage 1: 2 cores |
| Pose | 1 | 2 | 2 | Stage 2: 2 cores, total = 4 |
| Detection | 2 | 2 | 1 | Stage 1: 2 cores |
| Pose | 2 | 2 | 1 | Stage 2: 2 cores, total = 4 |
| Detection | 1 | 3 | 3 | Stage 1: 3 cores |
| Classifier | 1 | 1 | 1 | Stage 2: 1 core, total = 4 |

### Code Pattern

```python
# Model was compiled with batch_size (check model.json)
# Assumed to be batch_size=1 here
batch_size = model.inputs()[0].shape[0]

# Determine how many cores to use for this model
aipu_cores = 2  # For cascaded pipeline, sharing 4 cores between 2 models

# Calculate number of instances
num_instances = aipu_cores // batch_size

# Create connections - one per instance
connections = [
    ctx.device_connect(None, batch_size)
    for _ in range(num_instances)
]

# Load model instances
instances = [
    conn.load_model_instance(
        model,
        num_sub_devices=batch_size,
        aipu_cores=batch_size  # Cores per instance
    )
    for conn in connections
]
```

**Important notes:**
1. `num_sub_devices` = batch_size (reserves correct number of sub-devices; a sub-device is another name for a core)
2. `aipu_cores` = must equal batch_size (allocates L2 resources correctly)
3. Total cores used = `num_instances × batch_size`
4. For each model, follow the same pattern shown in [axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py#L356-L384)

---

## Critical: Coordinate Space Rescaling

**This rescaling step is required for ANY two-stage cascaded application where Stage 2 processes ROIs extracted from Stage 1 detections.**

### The Problem

When Stage 2 processes cropped ROIs, the model outputs are in the **model's input space** (e.g., 640×640), not the ROI's original dimensions. Without rescaling, coordinates will be misaligned with the actual image.

### Coordinate Flow

```
1. Stage 1 detects objects in original image (e.g., 1920×1080)
   ↓
2. Extract ROI from original image (e.g., 435×857 person crop)
   ↓
3. Preprocessing STRETCHES ROI to model input size (640×640)
   ↓
4. Model outputs coordinates in 640×640 space ❌ WRONG SPACE!
   ↓
5. MUST RESCALE coordinates back to ROI dimensions (435×857) ✅
   ↓
6. Transform ROI coordinates to full image space
```

### Implementation

See [axruntime_cascaded_pipeline.py:415-446](../../../examples/axruntime/python/axruntime_cascaded_pipeline.py#L415-L446) for full implementation:

```python
# After Stage 2 postprocessing, but before using coordinates
roi_height, roi_width = roi.shape[:2]  # Original ROI dimensions
model_size = 640  # Model input size

# Calculate scaling ratios
ratio_x = model_size / roi_width
ratio_y = model_size / roi_height

# Scale boxes from 640×640 back to ROI dimensions
if len(boxes) > 0:
    boxes[:, [0, 2]] /= ratio_x  # x1, x2
    boxes[:, [1, 3]] /= ratio_y  # y1, y2

# Scale keypoints from 640×640 back to ROI dimensions
if len(keypoints) > 0:
    keypoints[:, :, 0] /= ratio_x  # x coordinates
    keypoints[:, :, 1] /= ratio_y  # y coordinates
```

### Applicable to All Cascaded Applications

This rescaling applies to:
- Detection → Pose estimation
- Detection → Classification (if classifier returns spatial info)
- Detection → Segmentation (masks in ROI space)
- Detection → Re-identification (feature maps with spatial coordinates)

Without this rescaling, your Stage 2 outputs will be in the wrong coordinate system and unusable for visualization or downstream processing.

---

## NMS for Duplicate Detection Filtering

When Stage 2 processes ROIs, you may get duplicate detections for the same object. Apply NMS (Non-Maximum Suppression) to filter overlapping detections based on IoU (Intersection over Union).

See [axruntime_cascaded_pipeline.py:368-413](../../../examples/axruntime/python/axruntime_cascaded_pipeline.py#L368-L413) for full NMS implementation:

```python
# After postprocessing, before coordinate rescaling
if len(boxes) > 0:
    # Sort by confidence
    sorted_indices = np.argsort(boxes[:, 4])[::-1]
    keep_indices = []

    while len(sorted_indices) > 0:
        current = sorted_indices[0]
        keep_indices.append(current)

        if len(sorted_indices) == 1:
            break

        # Compute IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[sorted_indices[1:]]

        # ... IoU calculation ...

        # Keep boxes with IoU below threshold (0.45)
        mask = iou < 0.45
        sorted_indices = sorted_indices[1:][mask]

    # Filter both boxes and keypoints
    boxes = boxes[keep_indices]
    keypoints = keypoints[keep_indices]
```

**When to use NMS:**
- Multiple overlapping detections on same object
- High confidence threshold produces many candidates
- Stage 2 model outputs multiple predictions per ROI

---

## Best Practices

### Cascaded Pipelines

1. **CRITICAL: Always rescale Stage 2 outputs** from model space to ROI space
2. **Apply NMS** when Stage 2 produces multiple overlapping detections
3. **Validate total cores ≤ 4** before loading models
4. **Start with 2+2 core split** and adjust based on profiling
5. **Use batch_size=1** unless you have specific batching requirements
6. **Profile each stage** to identify bottlenecks (CPU preprocessing vs inference time)
7. **Consider ROI count** - many small ROIs may benefit from more Stage 2 cores
8. **Clip ROI coordinates** to image bounds to avoid crashes

### Batch Size Configuration

1. **Check compiled batch size** from `model.inputs()[0].shape[0]`
2. **Calculate num_instances correctly**: `aipu_cores // batch_size`
3. **Set num_sub_devices = batch_size** when calling `device_connect()`
4. **Set aipu_cores = batch_size** when calling `load_model_instance()`
5. **Handle batch_size > 1** in preprocessing (repeat or stack images)

---

## Related Documentation

- [Basic Usage Tutorial](basic-usage-tutorial.md) - Core API concepts and batch size fundamentals
- [Multiple Devices](multiple-devices.md) - Scaling cascaded pipelines across devices
- [API Reference](../../reference/apis/axruntime-py.md) - Complete API documentation

---

**Last Updated**: 2026-02-11
