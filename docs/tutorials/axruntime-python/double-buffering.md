# Double Buffering for Performance (Advanced)

## Overview

This guide covers double buffering, an **advanced performance optimization** for the low-level `axelera.runtime` Python API. Double buffering is a hardware feature that requires careful buffer management and result tracking. If you haven't completed the [Quick Start](axruntime-python-quickstart.md) yet, start there to understand the basic API usage.

Double buffering is a feature in the Metis runtime that enables pipelined inference execution. When enabled, it overlaps data transfer time with computation time for higher throughput.

**Prerequisites:**
- Completed [Quick Start](axruntime-python-quickstart.md) and [Basic Usage Tutorial](basic-usage-tutorial.md)
- Understand worker threads and inference patterns
- Knowledge of threading, queue management, and buffer tracking
- SDK virtual environment activated:: `source venv/bin/activate`
- Compiled model available (e.g., `build/resnet50-imagenet-onnx/model.json`)
  - You can download pre-compiled model with (`axdownloadmodel resnet50-imagenet-onnx`)

**Download example model:**
```bash
axdownloadmodel resnet50-imagenet-onnx
```

---

## What is Double Buffering?

When double buffering is enabled:

1. While the AIPU processes inference N, the host can transfer data for inference N+1
2. This overlaps data transfer time with computation time
3. Results are delayed by 2 inference cycles when using round-robin workers

**Trade-off**: Increased latency (2-frame delay) for higher throughput.

### Pipeline Comparison

**Without double buffering:**
```
run(input0) → result0
run(input1) → result1
run(input2) → result2
```

**With double buffering (single worker):**
```
run(input0) → dummy0
run(input1) → dummy1
run(input2) → result0  # Result delayed by 2 cycles
run(input3) → result1
run(input4) → result2
```

**With double buffering (N workers, round-robin):**
```
# Worker 0: run(input0) → dummy0
# Worker 1: run(input1) → dummy1
# Worker 0: run(input2) → dummy2
# Worker 1: run(input3) → dummy3
# Worker 0: run(input4) → result0  # Delay = 2 × N cycles
# Worker 1: run(input5) → result1
```

**Critical insight**: With N workers, you must:
1. **Drop the first 2×N results** (dummy outputs)
2. **Push 2×N extra dummy frames** at the end to flush the pipeline
3. **Buffer the input→output association** to match results with correct inputs

---

## When to Use Double Buffering

**Use double buffering when:**
- Throughput is more important than latency
- Processing batch workloads (e.g., video files, image directories)
- DMA transfer time is significant compared to inference time
- You can tolerate 2-frame result delay

**Don't use double buffering when:**
- Latency is more important than throughput
- Processing single images or small batches
- Avoiding application complexity is more important than maximizing throughput

---

## Implementation

**Complete example:** [axruntime_dblbuff.py](../../../examples/axruntime/python/axruntime_dblbuff.py)

**Run the example:**
```bash
# With double buffering (default)
python examples/axruntime/python/axruntime_dblbuff.py \
    build/resnet50-imagenet-onnx/model.json images/ --repeat 10

# Without double buffering (for comparison)
python examples/axruntime/python/axruntime_dblbuff.py \
    build/resnet50-imagenet-onnx/model.json images/ --repeat 10 --no-double-buffer
```

### Step 1: Enable double buffering when loading model instance

```python
instances = [
    conn.load_model_instance(
        model,
        num_sub_devices=batch_size,
        aipu_cores=batch_size,
        double_buffer=True  # Enable double buffering
    )
    for conn in connections
]
```

### Step 2: Calculate number of dummy results to drop

```python
# With N workers and double buffering, first 2×N results are dummy
num_to_drop = len(workers) * 2 if double_buffer else 0

# Add dummy frames at end to flush pipeline
input_paths += [None] * num_to_drop
```

### Step 3: Buffer the input→output association

```python
awaiting_result = collections.deque()

for in_frameno, image_path in enumerate(input_paths):
    # Preprocess and submit (or submit dummy frame)
    if image_path is None:
        inputs[next_available][0][:] = np.zeros_like(inputs[next_available][0])
    else:
        input = preprocess(image_path, input_infos[0])
        inputs[next_available][0][:] = input

    workers[next_available].push(image_path, inputs[next_available], outputs[next_available])

    if in_frameno >= prefill:
        # Retrieve result
        out_path, outs = workers[next_ready].pop()

        # Buffer the association
        awaiting_result.append(out_path)

        if out_frameno >= num_to_drop:
            # Now we have valid results - match with correct input
            out_path = awaiting_result.popleft()
            postprocess(out_frameno - num_to_drop, count, out_path, outs[0], labels, output_infos[0])

        out_frameno += 1
```

**Key points:**
1. **Track input paths separately** from output results using `awaiting_result` deque
2. **Skip first `num_to_drop` results** - these are dummy frames from double buffer warm-up
3. **Push dummy frames** at the end to drain the pipeline

### Step 4: Drain remaining results

```python
# Drain the remaining workers to extract the last N frames
for _ in range(prefill):
    next_ready = out_frameno % len(workers)
    out_path, outs = workers[next_ready].pop()
    awaiting_result.append(out_path)

    if out_frameno >= num_to_drop:
        out_path = awaiting_result.popleft()
        postprocess(out_frameno - num_to_drop, count, out_path, outs[0], labels, output_infos[0])

    out_frameno += 1
```

---

## Best Practices

1. **Always drop the first 2×N results** where N = number of workers
2. **Push 2×N dummy frames at the end** to flush the pipeline
3. **Use a deque or similar structure** to track input→output association
4. **Measure performance** - not all workloads benefit equally
5. **Consider latency requirements** - 2-frame delay may be unacceptable for real-time apps

---

## Performance Considerations

**Throughput gains**: Typically 10-30% improvement for workloads with significant DMA transfer overhead.

**Latency penalty**: Results delayed by 2×N frames (where N = number of workers).

**Memory overhead**: Requires buffering input associations (minimal).

**When it helps most:**
- Large input/output tensors (significant transfer time)
- Models with balanced compute vs. transfer time
- Batch processing workloads

**When it helps least:**
- Very fast models (compute time << transfer time)
- Small tensors (minimal transfer overhead)
- Single-image latency-critical applications

---

## Related Documentation

- [Basic Usage Tutorial](basic-usage-tutorial.md) - Understanding worker threads and inference patterns
- [API Reference](../../reference/apis/axruntime-py.md) - ModelInstance.load_model_instance() parameters

---

**Last Updated**: 2026-02-11
