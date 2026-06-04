# Quick Start: Integrating Axelera Inference

## Introduction

This quick start guide gets you running inference on Axelera hardware in 10 minutes using the low-level `axelera.runtime` Python API. You'll learn the minimal integration pattern using three simple functions: `initialize_axelera()`, `run_inference()`, and `cleanup_axelera()`.

This low-level API provides direct control over model loading, device management, and inference execution. It requires working with threading, queue management, and AIPU resource management. For more advanced, production-ready applications, we recommend starting here and then going through the other tutorials to ensure you follow best practices.

**Key requirement:** Using axelera.runtime requires additional processing steps (depadding, transposing, dequantizing, and handling postamble graphs if your model has one) beyond standard AI inference. All our examples include helper functions for these Axelera-required operations.

**Note:** Most developers should consider using the higher-level InferenceStream (Python) or AxInferenceNet (C++) APIs, which handle threading, resource management, and these transformations automatically. This low-level API is for users who need fine-grained control or have specific requirements not covered by the higher-level APIs. For API overview and when to use this API, see the [main documentation](../../reference/apis/axruntime-py.md).

## What You Get

- **Working example**: [axruntime_quickstart.py](../../../examples/axruntime/python/axruntime_quickstart.py) - ImageNet classification (350 lines)
- **Three functions**: `initialize_axelera()`, `run_inference()`, `cleanup_axelera()`
- **Pattern to follow**: Shows exactly what to change for your model

## Prerequisites

- Voyager SDK installed and activated (`source venv/bin/activate`)
- Compiled model (`build/your-model/model.json`)
    - In this example, use pre-compiled `ResNet-50` model: `axdownloadmodel resnet50-imagenet-coco`
    - To use your own model, compile it first (see [Compiler CLI Guide](../../reference/compiler/compiler-cli.md)) and use the path `build/<pipeline-name>/<model-name>/<core_count>/model.json`

---

## Run the Example

```bash
python examples/axruntime/python/axruntime_quickstart.py \
    build/resnet50/model.json \
    test_image.jpg \
    --labels imagenet_labels.txt
```

## Complete Example Flow

### With Axelera Integration
```python
# STEP 1: Initialize once at startup
input_info, output_infos = initialize_axelera("build/model.json", num_cores=4)

try:
    # STEP 2: Process frames
    for frame in frames:
        preprocessed = your_preprocess(frame, input_info)
        outputs = run_inference(preprocessed)      # Axelera inference
        results = your_postprocess(outputs[0], output_infos[0])
        print(results)
finally:
    # STEP 3: Cleanup at shutdown
    cleanup_axelera()
```

---

## Integration Pattern

### 1. Initialize (Once at Startup)

```python
def initialize_axelera(model_path: str, num_cores: int = 4):
    """Initialize Axelera hardware. Call this ONCE at application startup."""
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
    # Formula: num_instances = num_cores // batch_size
    batch_size = axelera_input_info.shape[0]
    num_instances = num_cores // batch_size

    # 5. Create connections and load model instances
    connections = []
    instances = []
    for _ in range(num_instances):
        conn = axelera_ctx.device_connect(None, num_sub_devices=batch_size)
        connections.append(conn)
        instance = conn.load_model_instance(model,
                                           num_sub_devices=batch_size,
                                           aipu_cores=batch_size)
        instances.append(instance)

    # 6. Pre-allocate buffers (one set per instance)
    for _ in range(num_instances):
        inp = [np.zeros(info.shape, np.int8) for info in model.inputs()]
        axelera_inputs.append(inp)
        out = [np.zeros(info.shape, np.int8) for info in model.outputs()]
        axelera_outputs.append(out)

    # 7. Create worker threads (one per instance)
    # Worker threads are needed because instance.run() is a blocking call.
    # Multi-threading lets the OS run other workers while one waits for the AIPU.
    axelera_workers = [InferenceWorker(inst) for inst in instances]

    return axelera_input_info, axelera_output_infos
```

**Call this once at startup.** Returns `input_info` and `output_infos` containing quantization parameters you'll need for preprocessing/postprocessing.

### 2. Run Inference (In Your Loop)

```python
def run_inference(preprocessed_int8: np.ndarray) -> list:
    """
    Run inference on Axelera hardware. Call this in your processing loop.
    Replaces your existing model.run() call.
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
```

### 3. Cleanup (At Shutdown)

```python
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
```

---

## Adapt for Your Model

Open [axruntime_quickstart.py](../../../examples/axruntime/python/axruntime_quickstart.py) and modify two functions:

**If you have your own pre- and postprocessing code already:** 
- Preprocessing and postprocessing need quantization/depadding steps added for compatibility with the Axelera hardware- see how to do these steps in this example
- You need to load your compiled model instead of the default model in this example
  - The Axelera compiler created a folder for your model- point the model loading code in this example to the `model.json` file in that folder.


### 1. Preprocessing

```python
def preprocess_imagenet(image_path: Path, input_info: TensorInfo) -> np.ndarray:
    """REPLACE THIS with your model's preprocessing!"""
    # Get target shape (without batch dimension)
    batch, height, width, _ = input_info.unpadded_shape

    # Read and resize image
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize using ImageNet statistics
    image = image.astype(np.float32) / 255.0
    image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STDDEV)

    # Quantize: float32 → int8 (KEEP THIS PATTERN)
    quantized = np.round(image / input_info.scale + input_info.zero_point)
    quantized = quantized.clip(-128, 127).astype(np.int8)

    # Pad for hardware alignment (KEEP THIS PATTERN)
    padded = np.pad(quantized, input_info.padding[1:],
                    mode='constant', constant_values=input_info.zero_point)

    # Handle batch size > 1
    if batch > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch, axis=0)

    return padded
```

**Change**: Resize dimensions, normalization (ImageNet mean/stddev), color format
**Keep**: Quantization and padding formulas - always use `input_info.scale`, `input_info.zero_point`, and `input_info.padding`

### 2. Postprocessing

```python
def postprocess_imagenet(int8_output: np.ndarray, output_info: TensorInfo,
                        labels: list, image_path: Path) -> dict:
    """REPLACE THIS with your model's postprocessing!"""
    # Depad and dequantize (KEEP THIS PATTERN)
    depadded = int8_output[tuple(slice(b, -e if e else None)
                                 for b, e in output_info.padding)]
    depadded = depadded.squeeze()
    float_output = (depadded.astype(np.float32) - output_info.zero_point) * output_info.scale

    # Top-1 classification (CHANGE THIS for your model)
    class_id = np.argmax(float_output)
    score = float_output[class_id]
    label = labels[class_id] if class_id < len(labels) else "(no label)"

    return {'class_id': int(class_id), 'label': label, 'score': float(score)}
```

**Keep**: Depadding and dequantization formulas - always use `output_info.padding`, `output_info.scale`, and `output_info.zero_point`
**Change**: Output decoding logic (classification → detection/segmentation/etc.)

> **Note**: If your output doesn't match what you expect (wrong shape, unexpected values), your compiled model may have a separate post-processing graph that is needed to get the same output. See [ONNX Postamble Processing](postamble-processing.md) for details on how to handle this.

---

## Key Requirements

### Quantization (Input)

**Always required** - convert float32 → int8 before inference:

```python
quantized = np.round(normalized / input_info.scale + input_info.zero_point)
quantized = quantized.clip(-128, 127).astype(np.int8)
```

Use `input_info.scale` and `input_info.zero_point` from the model. These are calculated when the model is compiled.

### Padding (Input)

**Always required** - for hardware alignment:

```python
padded = np.pad(quantized, input_info.padding[1:],
                mode='constant', constant_values=input_info.zero_point)
```

- Use `input_info.padding[1:]` (skip batch dimension)
- Pad with `input_info.zero_point`, not `0`

### Depadding (Output)

**Always required**:

```python
depadded = int8_output[tuple(slice(b, -e if e else None)
                             for b, e in output_info.padding)]
```

Use `output_info.padding` from the model.

### Dequantization (Output)

**Always required** - convert int8 → float32 after inference:

```python
float_output = (depadded.astype(np.float32) - output_info.zero_point) * output_info.scale
```

Use `output_info.scale` and `output_info.zero_point` from the model.

---

## Performance Notes

This fast path implementation uses a simple round-robin worker pattern:
- ✅ Multi-threaded: Uses all AIPU cores in parallel
- ✅ Buffer reuse: Pre-allocates buffers for performance
- ✅ Simple integration: Minimal code changes

For **maximum throughput**, use the prefill/drain pipeline pattern shown in [axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py).

---


## Next Steps

1. **Get it working**: Run the example with your model
2. **Validate accuracy**: Compare outputs with your existing inference engine
3. **Optimize if needed**: Add pipeline pattern from [axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py)
4. **Learn more**: Read [Basic Usage Tutorial](basic-usage-tutorial.md) for API details and explanations

---

## Related Documentation

- [Basic Usage Tutorial](basic-usage-tutorial.md) - Full API walkthrough with explanations
- [axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py) - Complete example with prefill/drain pattern
- [Double Buffering](double-buffering.md) - Performance optimization
- [Cascaded Pipelines](cascaded-pipelines.md) - Multi-model applications
- [ONNX Postamble Processing](postamble-processing.md) - Using post-processing graphs
