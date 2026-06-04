
# Basic Usage Tutorial: axelera.runtime Low-Level API

## Introduction

This tutorial explains the building blocks of `axelera.runtime` - the objects you'll work with, how to structure your application for performance, and why things work the way they do. After completing this tutorial, you'll understand how to build robust, high-performance applications using the low-level API.

**New to the API?** Start with the [Quick Start guide](axruntime-python-quickstart.md) first, then return here to understand the "why" behind the patterns.

**What you'll learn:**
- The API object hierarchy and how objects relate to each other
- Why worker threads are essential for performance
- How to allocate AIPU cores efficiently (batch size, num_instances)
- The required Axelera-specific transformations (padding, quantization, depadding, dequantization)
- Resource management patterns to avoid leaks

**Examples used in this tutorial:**
- **[axruntime_quickstart.py](../../../examples/axruntime/python/axruntime_quickstart.py)** - Minimal ImageNet classification showing core pattern
- **[axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py)** - Production-ready YOLO detection with prefill/drain pattern

---

## Contents

**Foundation Concepts** - Understanding the API structure
- [Core API Concepts](#core-api-concepts) - Object hierarchy and resource ownership
- [Understanding Padding and Quantization](#understanding-padding-and-quantization) - Required transformations

**Step-by-Step Walkthrough** - Building an application
- [Walkthrough: Basic Usage Example](#walkthrough-basic-usage-example) - 7 steps from initialization to cleanup

**Performance & Best Practices**
- [Batch Size and Core Allocation](#batch-size-aipu-cores-and-num_instances) - Maximizing hardware utilization
- [Resource Management Best Practices](#resource-management-best-practices) - Avoiding leaks and crashes

**Next Steps**
- [Next Steps](#next-steps) - Advanced guides (double buffering, cascaded pipelines, etc.)

---


## Prerequisites

- Completed the [Quick Start guide](axruntime-python-quickstart.md)
- SDK installed and virtual environment activated (`source venv/bin/activate`)
- Compiled model available (e.g., `build/resnet50-imagenet-onnx/model.json`)
  - You can download pre-compiled model with (`axdownloadmodel resnet50-imagenet-onnx`)

---

## Core API Concepts

### Object Hierarchy

The axelera.runtime API follows a clear hierarchy:

```
Context (root object, manages all resources)
├── Model (loaded model metadata and weights)
└── Connection (reserved hardware resources)
    └── ModelInstance (executable model on hardware)
```

**Key points:**
- **Context** is the entry point - create it first
- **Model** contains the compiled model - can be reused for multiple instances
- **Connection** reserves AIPU cores on an Axelera device - one per ModelInstance
- **ModelInstance** is ready for inference - one per thread

### Resource Ownership

The Axelera resource hierarchy follows a parent-child model:

1. **Parent releases children**: Releasing a parent automatically releases all child resources
2. **Context controls lifecycle**: Context manages when all Axelera resources are freed
3. **Explicit control**: You decide when to create and release resources

**Good pattern (context manager):**
```python
with Context() as ctx:
    model = ctx.load_model("model.json")
    # ... use model ...
# Context and all children automatically released here
```

**Also valid (explicit release):**
```python
ctx = Context()
model = ctx.load_model("model.json")
# ... use model ...
ctx.release()  # Releases Context, Model, and all other children
```

---

## Foundational Concepts

Before diving into the walkthrough, understand these two critical concepts that affect how you structure your application.

### Understanding Padding and Quantization

The Axelera Metis AIPU performs inference using INT8 quantized models with hardware-aligned tensors. When using the low-level API, you must handle these transformations manually (higher-level APIs do this automatically).

#### The Pipeline

```
Float32 Input → Pre-processing → Quantize → INT8 → Pad → INT8 Padded Input
                                                                ↓
                                                         Inference (AIPU)
                                                                ↓
Float32 Output ← Post-processing ← Dequantize ← Depad ← INT8 Padded Output
```

Here, pre-processing and post-processing include the normal pre- or post-processing you would do for this model on any platform, in addition to any operations contained in the [pre- or post-amble ONNX graph](postamble-processing.md). 

**Axelera-Required Processing:**
1. **Input**: Quantize (FP32 → INT8) then Pad (add channel alignment)
2. **Output**: Depad (remove channel alignment) then Dequantize (INT8 → FP32)

#### Formulas

**Input preprocessing:**
```python
# 1. Quantize: float32 → int8
quantized = np.round(normalized / input_info.scale + input_info.zero_point)
quantized = quantized.clip(-128, 127).astype(np.int8)
# Note: Normalization (e.g., /255.0) and quantization can often be fused into a single operation for efficiency

# 2. Pad: Add hardware alignment (skip batch dimension)
padded = np.pad(quantized, input_info.padding[1:],
                mode='constant', constant_values=input_info.zero_point)
```

**Output postprocessing:**
```python
# 1. Depad: Remove alignment padding
depadded = output[tuple(slice(b, -e if e else None)
                        for b, e in output_info.padding)]

# 2. Dequantize: int8 → float32
result = (depadded.astype(np.float32) - output_info.zero_point) * output_info.scale
```

**Helper functions:** See [`axelera_quantize()`, `axelera_pad()`, `axelera_depad()`, `axelera_dequantize()` in yolo_utils.py](../../../examples/axruntime/python/utils/yolo_utils.py).

### Batch Size, AIPU Cores, and num_instances

Understanding how batch size relates to hardware utilization is crucial for performance.

**Key facts:**
- One Metis device has **4 AIPU cores**
- Batch size is **fixed at compile time** (check `model.inputs()[0].shape[0]`)
- A model with batch_size=N uses **N cores per inference**

**The formula:** `num_instances = 4 cores // batch_size`

**Examples:**
| Batch Size | Num Instances | Cores Used | Behavior |
|------------|---------------|------------|----------|
| 1 | 4 | 4 | Process images individually across 4 parallel instances |
| 2 | 2 | 4 | Process 2 images at a time across 2 parallel instances |
| 4 | 1 | 4 | Process 4 images together in a single batched inference |

**Why this matters:**
- Each ModelInstance needs its own worker thread
- More instances = more parallelism = better CPU utilization while waiting for AIPU
- See [Worker Thread Pattern](#step-6-running-inference-with-worker-threads) for details

**Example code:** See [`axruntime_yolo11.py:356-384`](../../../examples/axruntime/python/axruntime_yolo11.py#L356-L384) for the calculation pattern.

---

## Walkthrough: Basic Usage Example

Let's walk through the core API concepts using [axruntime_quickstart.py](../../../examples/axruntime/python/axruntime_quickstart.py) and [axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py) as references. The Quick Start example shows the simplest integration pattern, while the YOLO11 example demonstrates production-ready patterns like prefill/drain pipelining.

### Step 1: Creating Context and Loading Model

The first step is always creating a Context and loading your model:

```python
with Context() as ctx:
    LOG.info(f"Loading model from: {model_path}")
    model = ctx.load_model(str(model_path))
```

**What's happening:**
- `Context()` initializes the runtime and prepares to access Axelera hardware
- `ctx.load_model()` loads a compiled model from a `model.json` file
- The `with` creates a Python context manager to ensure automatic cleanup when done, a best practice for resource management

**Model file structure:**
Your model directory typically contains:
- `model.json` - Model metadata and configuration. When loading a model, this is the file path you need to specify.
- `manifest.json` - Tensor quantization parameters
- Model weights and compiled binary

For a complete initialization function with detailed step-by-step comments, see [`initialize_model()` in axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py#L286-L467).

### Step 2: Understanding TensorInfo

After loading the model, inspect its input and output tensors:

```python
input_infos = model.inputs()
output_infos = model.outputs()

input_info = input_infos[0]  # First input tensor

LOG.info(f"Input shape (with padding): {input_info.shape}")
LOG.info(f"Input shape (without padding): {input_info.unpadded_shape}")
LOG.info(f"Input padding: {input_info.padding}")
LOG.info(f"Input quantization: scale={input_info.scale}, zero_point={input_info.zero_point}")
```

**TensorInfo contains:**
| Property | Description | Example |
|----------|-------------|---------|
| `shape` | Full tensor shape **including** padding | `(1, 224, 224, 4)` |
| `unpadded_shape` | Logical shape **without** padding | `(1, 224, 224, 3)` |
| `padding` | Padding per dimension `[(start, end), ...]` | `[(0, 0), (0, 0), (0, 0), (0, 1)]` |
| `scale` | Quantization scale factor | `0.003921` |
| `zero_point` | Quantization zero point | `0` |
| `dtype` | Data type (always `np.int8` for quantized) | `int8` |
| `size` | Total size in bytes | `150528` |

**Why two shapes?**
- `unpadded_shape`: The actual image dimensions (what you resize to)
- `shape`: The buffer size you allocate (includes hardware alignment padding)

The information in TensorInfo allows you to easily perform the Axelera-specific pre- and post-processing steps described in [Understanding Padding and Quantization](#understanding-padding-and-quantization).

### Step 3: Connecting to Device

Next, look at the model batch size and create Connections to the hardware:

```python
batch_size = input_info.shape[0]  # From model, fixed at model compile time
num_instances = aipu_cores // batch_size  # Calculate instances

# Create one connection per model instance
connections = []
for i in range(num_instances):
    conn = ctx.device_connect(device=None, num_sub_devices=batch_size)
    connections.append(conn)
```

**Critical concept: Batch size and model instances**

The batch size (read from the model info) determines how many ModelInstances we create using the formula `num_instances = aipu_cores / batch_size`. With batch_size=1 and 4 cores, we create 4 ModelInstances. See [Batch Size, AIPU Cores, and num_instances](#batch-size-aipu-cores-and-num_instances) for details on different batch sizes.

**Parameters explained:**
- `device=None`: Auto-select an available device
- `num_sub_devices`: Number of cores to reserve (matches batch_size)

### Step 4: Loading ModelInstance

Create a ModelInstance for each Connection:

```python
instances = []
for i, conn in enumerate(connections):
    instance = conn.load_model_instance(
        model,
        num_sub_devices=batch_size,
        aipu_cores=batch_size
    )
    instances.append(instance)
```

**Parameters explained:**
- `model`: The Model object to instantiate
- `num_sub_devices`: Number of AIPU cores for this instance (matches batch_size)
- `aipu_cores`: L2 memory allocation (1 core = 25% of device's L2 memory). Should match num_sub_devices.

**Key points:**

Each ModelInstance is independent and can run inference in parallel, and should only be used by one thread. Each ModelInstance is tied to the Connection that created it.

### Step 5: Preparing Buffers and Preprocessing

**Buffer allocation:**

Pre-allocate buffers for inputs and outputs to avoid allocating memory during inference. Each model instance has its own buffers:

```python
inputs = []
outputs = []
for i in range(num_instances):
    instance_inputs = [np.zeros(info.shape, info.dtype) for info in input_infos]
    inputs.append(instance_inputs)

    instance_outputs = [np.zeros(info.shape, info.dtype) for info in output_infos]
    outputs.append(instance_outputs)
```

**Why separate buffers per instance?** Multiple workers run in parallel in different threads, so each needs its own buffers to avoid data races.

**Preprocessing:**

Preprocessing splits into two parts:

**General Preprocessing** (model-specific):
1. Resize, color conversion, normalization

**Axelera-Required Processing** (needed for axelera.runtime API):
2. **Quantization**: Convert float32 → int8
3. **Padding**: Add hardware channel alignment
4. **Preamble graph** (if your model has one): Some models have compiler-extracted preprocessing operations

See [Quick Start - Preprocessing](axruntime-python-quickstart.md#1-preprocessing) for the complete pattern, or [`preprocess_yolo_detection()` in yolo_utils.py](../../../examples/axruntime/python/utils/yolo_utils.py#L58-L103) for a production implementation with YOLO models. For preamble/postamble graphs, see [ONNX Postamble Processing](postamble-processing.md).

### Step 6: Running Inference with Worker Threads

The examples use a worker thread pattern for maximum performance. The call to `instance.run(inputs, outputs)` is a blocking call - it will wait until the AIPU finishes inference on the input and returns it over the PCIe bus. If used in a simple single-threaded design, only one ModelInstance (and thus one AIPU core) will ever be used at once - this will significantly bottleneck performance. Therefore, we recommend a multi-threaded design to make sure that the OS scheduler is able to execute other tasks (including inference on other worker threads) while waiting for results.

**Worker thread implementation:**

The [`InferenceWorker`](../../../examples/axruntime/python/axruntime_quickstart.py#L47-L70) class creates a dedicated thread for each ModelInstance:
- Each worker has its own input and output queues
- The worker's `run()` method loops continuously, calling `instance.run()` (the blocking inference call)
- While one worker waits for inference, the OS scheduler can run other workers

See [Quick Start - Run Inference](axruntime-python-quickstart.md#2-run-inference-in-your-loop) for the complete `run_inference()` implementation.

**Pipeline execution patterns:**

**Simple round-robin** (used in Quick Start): Each frame is submitted to the next available worker and results are collected immediately. Simple but not maximum throughput.

**Prefill/drain pattern** (used in YOLO11 example): For maximum throughput:
1. **Prefill**: Submit N frames to all workers before collecting any results
2. **Steady state**: For each new frame, collect one result then submit the new frame
3. **Drain**: After all frames submitted, collect the remaining N results

This overlaps preprocessing (CPU) with inference (AIPU) for 2-3x throughput improvement. See [`run_realtime_inference()` in axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py#L470-L552) for the complete implementation with detailed comments.

**Why round-robin collection preserves order:** Each worker only processes its assigned frames (Worker 0: frames 0,4,8... Worker 1: frames 1,5,9...), so polling workers in order (0→1→2→3→0...) reconstructs the original submission order.

### Step 7: Postprocessing and Cleanup

**Postprocessing:**

Postprocessing splits into two parts:

**Axelera-Required Processing** (needed for axelera.runtime API):
1. **Depad**: Remove hardware channel alignment padding
2. **Transpose**: Convert NHWC → NCHW format
3. **Dequantize**: Convert int8 → float32
4. **Postamble graph** (if your model has one): Run extracted operations via ONNX Runtime or manual implementation

To check if your model has a postamble graph, look for `postprocess_graph.onnx` in your model directory or check the `postamble_graph`/`postprocess_graph` field in `manifest.json`. We provide code examples for handling all of these Axelera-specific steps in this documentation.

**General Postprocessing** (same as any AI framework):
5. **Model-specific operations**: NMS for detection, argmax for classification, etc.

See [Quick Start - Postprocessing](axruntime-python-quickstart.md#2-postprocessing) for the complete pattern, or [`postprocess_yolo_detection()` in yolo_utils.py](../../../examples/axruntime/python/utils/yolo_utils.py#L252-L303) for a production implementation. Our examples provide helper functions for all Axelera-required processing steps.

**Note on YOLO11 Postamble Graph:** The YOLO11 model has a postamble graph containing the detection head operations (DFL for box regression, sigmoid activation for classification, and box coordinate transformations). The [`axruntime_yolo11.py`](../../../examples/axruntime/python/axruntime_yolo11.py) example implements these postamble operations manually in [`yolo_utils.decode_yolo_detections()`](../../../examples/axruntime/python/utils/yolo_utils.py#L111-L192), while [`axruntime_yolo11_onnxruntime.py`](../../../examples/axruntime/python/axruntime_yolo11_onnxruntime.py) uses ONNX Runtime to execute the postamble graph automatically. Both approaches produce identical results - the manual approach gives more control, while ONNX Runtime is simpler. See the [Post-Processing with ONNX Postamble Graphs](postamble-processing.md) guide for details.

**Cleanup:**

```python
finally:
    # Shut down workers
    for worker in workers:
        worker.shutdown()
    for worker in workers:
        worker.join(timeout=5.0)

    # Context manager automatically releases:
    # - ModelInstances
    # - Connections
    # - Model
    # - Context
```

Using a Python context manager ensures all resources are freed, even if an exception occurs. See [Quick Start - Cleanup](axruntime-python-quickstart.md#3-cleanup-at-shutdown) for the complete implementation.

---

## Resource Management Best Practices

### Context Manager Pattern

**Recommended:** Always use the `with` statement for Context. This has the following advantages of automatic cleanup, even if exceptions occur.

```python
with Context() as ctx:
    model = ctx.load_model("model.json")
    conn = ctx.device_connect()
    instance = conn.load_model_instance(model)
    # ... use instance ...
# Everything automatically released here
```
### Parent-Child Ownership

Remember the hierarchy:

```
Context
├── Model ─┐
└── Connection ─┐
    └── ModelInstance
```

**Rules:**
1. Releasing Context releases everything
2. Releasing Connection releases ModelInstance
3. You can release children independently of the parent

**Example:**
```python
with Context() as ctx:
    model1 = ctx.load_model("model1.json")
    conn1 = ctx.device_connect()
    instance1 = conn1.load_model_instance(model1)

    # Use instance1...

    # Release instance1 and conn1, but keep Context and model1
    instance1.release()
    conn1.release()

    # Load a different model
    model2 = ctx.load_model("model2.json")
    conn2 = ctx.device_connect()
    instance2 = conn2.load_model_instance(model2)

    # Use instance2...
# Context releases model2, instance2, conn2 automatically
```

### Explicit Release

If not using context manager, always call `release()`:

```python
ctx = Context()
try:
    model = ctx.load_model("model.json")
    # ... use model ...
finally:
    ctx.release()  # Ensures cleanup even if exception occurs
```

## Next Steps

Now that you understand the basics, explore the task-based guides:

**Performance Optimization:**
- **[Double Buffering](double-buffering.md)** - DMA pipelining for maximum throughput

**Multi-Model Applications:**
- **[Cascaded Pipelines](cascaded-pipelines.md)** - Running multiple models in sequence
- **[Multiple Devices](multiple-devices.md)** - Scaling across multiple Metis devices

**Advanced Topics:**
- **[ONNX Postamble Processing](postamble-processing.md)** - Using compiler-extracted post-processing graphs
- **[API Reference](../../reference/apis/axruntime-py.md)** - Complete API documentation

## Questions?

For issues or feedback, visit the [Axelera GitHub repository](https://github.com/axelera-ai-hub/voyager-sdk/issues).
