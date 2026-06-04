# Post-Processing with ONNX Postamble Graphs

## Overview

This guide explains how to handle postamble graphs with the low-level `axelera.runtime` Python API. When models are compiled for Axelera's Metis AIPU, the compiler may split the model graph to extract operations that are not efficient to run on the AIPU (like certain pooling, regression, or normalization operations). These extracted operations typically appear at the beginning or end of the model, and are available as preamble and postamble ONNX graphs along with other compiled model artifacts. If you do not handle the processing of the operations in these graphs, you will get incorrect output from your model. This guide explains how to incorporate the post-processing graphs correctly with your application so you get the results you expect.

When using the low-level API, you must manually handle postamble processing that higher-level APIs (InferenceStream, AxInferenceNet) manage automatically. If you haven't completed the [Quick Start](axruntime-python-quickstart.md) yet, start there to understand basic inference execution and tensor transformations.

For most models, the pre-processing "preamble" ONNX graph is empty, meaning all operations at the beginning of the model are executed on the AIPU in the core graph. This guide focuses on the post-processing "postamble" graph, but the same techniques apply to any preamble graph.

**Voyager SDK API-level differences:**
- **Python InferenceStream and C++ AxInferenceNet**: Handle pre/postamble graphs automatically
- **Low-level `axelera.runtime` API**: You must handle postamble processing manually

**Prerequisites:**
- Completed [Quick Start](axruntime-python-quickstart.md) and [Basic Usage Tutorial](basic-usage-tutorial.md)
- Understand TensorInfo, quantization, padding, and tensor transformations
- ONNX Runtime installed (included with SDK)
- SDK activated: `source venv/bin/activate`

---

## When Do You Need This?

### Check if your model has a postamble graph

You can check your compiled model for a postamble graph in two ways:

**1. Check manifest.json (programmatic):**
```python
import json
from pathlib import Path

model_dir = Path("build/yolo11s-coco-onnx")
manifest_path = model_dir / "manifest.json"

if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
        postamble = manifest.get('postamble_graph') or manifest.get('postprocess_graph')
        if postamble:
            print(f"Postamble graph: {postamble}")
```

**2. Check for postprocess_graph.onnx file:**
- Path: `voyager-sdk/build/<pipeline_name>/<model_name>/<aipu_cores>/postprocess_graph.onnx`
- Example: `voyager-sdk/build/yolo11s-coco-onnx/yolo11s-coco-onnx/1/postprocess_graph.onnx`

If you use the compiler CLI, look in the folder you specified with the -o option.

### Viewing Postamble Graphs

To inspect the postamble ONNX graph and understand what operations it contains:

```bash
# Install Netron (if not already installed)
pip install netron

# View the postamble graph
netron /path/to/your_model_postamble.onnx
```

This helps you understand:
- What operations were extracted from your model
- Input/output shapes and types
- Whether manual implementation would be feasible

---

## Model Pipeline Architecture

When a postamble graph exists, your pipeline becomes:

```
1. Preprocess (CPU)
   ↓
2. Core Model (AIPU) → raw int8 NHWC outputs with padding
   ↓
3. Axelera-Required Processing:
   a. Depad outputs
   b. Transpose NHWC → NCHW
   c. Dequantize int8 → float32
   d. Run postamble ONNX graph
   ↓
4. General Postprocessing (NMS, etc.)
```

**Note that you must depad, transpose, and dequantize the raw output from the Axelera AIPU before you run the postamble graph.**

**Performance note:** Postamble operations are typically much lighter on computation compared to AIPU inference. The compute-intensive layers in the middle of the network are executed on the AIPU.

---

## Implementation: Two Approaches

There are two ways to handle ONNX postamble graphs:

### Approach 1: ONNX Runtime (Recommended)

**Example:** [axruntime_yolo11_onnxruntime.py](../../../examples/axruntime/python/axruntime_yolo11_onnxruntime.py)

- Easier to implement - just load and run the ONNX file
- Guaranteed to match original model behavior
- Good performance on most platforms

### Approach 2: Manual Implementation

**Example:** [axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py) + [yolo_utils.py](../../../examples/axruntime/python/utils/yolo_utils.py)

- Requires understanding the postamble graph operations (view with Netron)
- More control over implementation and optimization
- Can be faster for simple operations

Both approaches are valid and produce identical results. Choose based on your needs:
- Use ONNX Runtime for complex postamble graphs or when development speed matters
- Use manual implementation when you need fine-grained control or have simple operations

---

## ONNX Runtime Approach

Complete example: [axruntime_yolo11_onnxruntime.py](../../../examples/axruntime/python/axruntime_yolo11_onnxruntime.py)

### Step 1: Initialize ONNX Runtime Session

During model initialization, check for and load the postamble graph:

```python
import json
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Tuple, List

def initialize_postamble_session(model, model_path: Path) -> Optional[Tuple[ort.InferenceSession, List[str], List[str]]]:
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

    postamble_path = None
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
                # Try both possible field names (postamble_graph and postprocess_graph)
                postamble_filename = manifest.get('postamble_graph') or manifest.get('postprocess_graph')
                if postamble_filename:
                    postamble_path = model_dir / postamble_filename
        except Exception as e:
            print(f"Warning: Failed to read manifest.json: {e}")

    if not postamble_path or not postamble_path.exists():
        print("No postamble graph available - using manual postprocessing")
        return None

    print(f"Loading postamble graph: {postamble_path}")

    # Configure ONNX Runtime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Thread configuration: use 1 for single-threaded applications
    # Can increase for multi-threaded applications if profiling shows benefit
    sess_options.intra_op_num_threads = 1

    # Create session
    session = ort.InferenceSession(
        str(postamble_path),
        sess_options,
        providers=['CPUExecutionProvider']
    )

    # Get input/output metadata for later use
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    print(f"Postamble ONNX: {len(input_names)} inputs, {len(output_names)} outputs")
    return session, input_names, output_names
```

**Usage in your initialization:**
```python
from axelera.runtime import Context
from pathlib import Path

# Model path (passed from command line)
model_path = Path("build/yolo11s-coco-onnx/model.json")

# Existing initialization
ctx = Context()
model = ctx.load_model(str(model_path))
input_infos = model.inputs()
output_infos = model.outputs()
# ... create connections and instances ...

# Add postamble initialization
postamble_info = initialize_postamble_session(model, model_path)
if postamble_info:
    postamble_session, postamble_input_names, postamble_output_names = postamble_info

    # Verify postamble inputs match AIPU outputs
    assert len(postamble_input_names) == len(output_infos), \
        f"Postamble expects {len(postamble_input_names)} inputs but model has {len(output_infos)} outputs"
```

### Step 2: Transform AIPU Outputs for Postamble

The raw AIPU outputs must be depadded, transposed, and dequantized before inputting to the ONNX postamble graph.

**Key transformations** (always in this order):
1. **Depad**: Remove hardware channel padding using `info.padding[3]`
2. **Transpose**: Convert NHWC → NCHW format
3. **Dequantize**: Convert int8 → float32 using `(value - zero_point) * scale`

```python
def prepare_postamble_inputs(raw_outputs, output_infos, input_names):
    """
    Transform raw AIPU outputs to postamble ONNX inputs.

    Performs the three required transformations:
    1. Depadding (remove channel padding)
    2. Transpose NHWC → NCHW
    3. Dequantize int8 → float32
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
        tensor_nhwc = tensor_nhwc[:, :, :, c_pad_left:c_pad_left + actual_channels]

        # Step 2: Transpose to NCHW (ONNX standard format)
        tensor_nchw = np.transpose(tensor_nhwc, (0, 3, 1, 2))

        # Step 3: Dequantize: float_value = (int8_value - zero_point) * scale
        tensor_float = (tensor_nchw.astype(np.float32) - info.zero_point) * info.scale

        # Map to postamble input name
        postamble_inputs[input_names[idx]] = tensor_float

    return postamble_inputs
```

See [axruntime_yolo11_onnxruntime.py](../../../examples/axruntime/python/axruntime_yolo11_onnxruntime.py) for complete implementation.

### Step 3: Use in Inference Loop

After AIPU inference, run postamble processing before your application postprocessing:

```python
# After AIPU inference
result_outputs = workers[worker_idx].pop()  # Get AIPU outputs

# Run postamble if available
if postamble_info is not None:
    session, input_names, output_names = postamble_info
    postamble_inputs = prepare_postamble_inputs(result_outputs, output_infos, input_names)
    processed_outputs = session.run(output_names, postamble_inputs)
else:
    processed_outputs = result_outputs  # Use directly (need manual processing)

# Your application postprocessing
detections = postprocess(processed_outputs[0], ...)
```

---

## Manual Implementation Approach

Complete example: [axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py) + [yolo_utils.py](../../../examples/axruntime/python/utils/yolo_utils.py)

Instead of using ONNX Runtime, you can manually implement the postamble operations in numpy. This approach:
1. Requires viewing the postamble graph with Netron to understand operations
2. Implements those operations yourself (DFL, sigmoid, box transforms, etc.)
3. Gives you full control over implementation and optimization

**For YOLO11s**, the postamble contains the detection head operations. The manual implementation in [yolo_utils.py](../../../examples/axruntime/python/utils/yolo_utils.py) shows:

```python
# Stage 1: Axelera transformations (lines 280-288)
depadded = axelera_depad(output, info.padding)
dequantized = axelera_dequantize(depadded, info.scale, info.zero_point)
features_nchw = np.transpose(dequantized, (0, 3, 1, 2))

# Stage 2: Manual postamble operations (lines 106-187)
# Implements DFL, sigmoid, anchor generation, box decoding
decoded_boxes = decode_yolo_detections(features_nchw, ...)

# Stage 3: Application postprocessing
final_boxes = apply_nms(decoded_boxes, ...)
```

The key is that **stages 1 and 2 together replace what the ONNX postamble graph does**. You must implement the specific operations from your postamble graph.

**When to use manual implementation:**
- Postamble contains operations you want to optimize or customize
- You need minimal latency (can fuse operations)
- You prefer not to depend on ONNX Runtime
- The postamble operations are simple enough to implement reliably

---

## Key Points

### Required Transformations

Always perform these transformations in this order for postamble inputs:

```python
# 1. Depad: Remove hardware alignment padding
tensor = tensor[:, :, :, c_pad_left:c_pad_left + actual_channels]

# 2. Transpose: NHWC → NCHW (ONNX standard)
tensor = np.transpose(tensor, (0, 3, 1, 2))

# 3. Dequantize: int8 → float32
tensor = (tensor.astype(np.float32) - zero_point) * scale
```

### Common Pitfalls

1. **Forgetting NHWC → NCHW transpose**: AIPU outputs are NHWC, ONNX expects NCHW
2. **Ignoring padding**: Always depad using `info.padding[3]` (channel dimension)
3. **Wrong quantization params**: Use exact values from `info.scale` and `info.zero_point`
4. **Order of operations**: Always depad → transpose → dequantize

### When to Use ONNX Runtime vs Manual Implementation

**Use ONNX Runtime when:**
- Postamble contains complex operations (convolutions, normalization layers)
- You want guaranteed parity with original model behavior
- Development speed is more important than minimal latency
- You're targeting x86 platforms where ONNX Runtime performance is excellent

**Use manual implementation when:**
- Postamble contains only simple operations (reshapes, basic math)
- You need absolute minimum latency (especially important on ARM platforms)
- You want to customize or optimize specific operations
- You want to fuse postamble operations with your application post-processing

---

## Performance and Debugging

**ONNX Runtime Threading**: Use `sess_options.intra_op_num_threads = 1` for single-threaded applications. Increase only if profiling shows benefits.

**Performance Impact**: Postamble processing adds minimal overhead. AIPU inference dominates total latency for most models.

**Debugging**: Print tensor shapes and statistics at each stage (raw AIPU → postamble input → postamble output) to verify transformations are correct.

---

## Complete Examples: YOLO11s Detection

Two complete examples are provided showing both approaches:

**ONNX Runtime Approach**: [axruntime_yolo11_onnxruntime.py](../../../examples/axruntime/python/axruntime_yolo11_onnxruntime.py)
- Uses ONNX Runtime to execute the postamble graph
- Simpler postprocessing (only NMS needed after postamble)
- Postamble graph outputs [1, 84, 8400] with decoded boxes and class scores

**Manual Implementation Approach**: [axruntime_yolo11.py](../../../examples/axruntime/python/axruntime_yolo11.py)
- Manually implements postamble operations in numpy (see [yolo_utils.py](../../../examples/axruntime/python/utils/yolo_utils.py))
- Depad/transpose/dequantize in `yolo_utils.postprocess_yolo_detection()` (lines 280-288)
- Manual DFL + sigmoid + box decoding in `yolo_utils.decode_yolo_detections()` (lines 106-187)
- NMS in `yolo_utils.apply_nms()` (lines 189-249)

Both examples produce identical detection results. The ONNX Runtime version is simpler to implement, while the manual version provides more control over the implementation.

---

## Summary

Some models don't require postamble handling, but when needed:

1. **Check** if your model has a postamble graph by reading `manifest.json`
2. **Initialize** ONNX Runtime session for the postamble graph
3. **Transform** AIPU outputs: depad → transpose → dequantize
4. **Run** postamble ONNX inference
5. **Continue** with your normal application post-processing

This approach ensures exact output matching your original ONNX model while leveraging Axelera hardware acceleration for the core computation.

---

## Related Documentation

- [Basic Usage Tutorial](basic-usage-tutorial.md) - Understanding TensorInfo and quantization
- [API Reference](../../reference/apis/axruntime-py.md) - Complete API documentation

---

**Last Updated**: 2026-02-11
