---
title: "axelera.runtime.op.combinators"
---
# `axelera.runtime.op.combinators`


Pipeline combinators for composing operators.

Building blocks for constructing inference pipelines:

- Seq: Sequential composition - execute operators in order, piping output to next input
- Par: Parallel execution - run multiple operators on same input, return tuple of results
- ForEach: Collection iteration - apply operators to each element in a list
- Pack: Collect positional arguments into a single tuple
- Unpack: Mark a tuple for argument unpacking to the next operator
- Itemgetter: Extract an element from a tuple by index (like operator.itemgetter)
- identity: Pass input through unchanged
- constant: Always return a fixed value regardless of input

## Summary

| Name | Description |
|------|-------------|
| [Seq](#seq) | Sequential operator that executes operators in order, piping output to next input. |
| [Par](#par) | Parallel operator that executes multiple operators with same input, returns tuple. |
| [ForEach](#foreach) | Apply operators to each element in a collection, preserving the original collection. |
| [Pack](#pack) | Collect positional arguments into a plain tuple. |
| [Unpack](#unpack) | Mark a tuple or list for argument unpacking to the next operator. |
| [Itemgetter](#itemgetter) | Extract element by index from a tuple value (like `operator.itemgetter`). |
| [identity](#identity) | Return the input unchanged. |
| [constant](#constant) | Return the given constant value. |

---

### Seq

**Alias:** `seq`

Sequential operator that executes operators in order, piping output to next input.

**Input:** Any - accepts whatever the first operator in the sequence accepts.

**Output:** Any - returns whatever the last operator in the sequence returns.

**Examples:**

```python
# Image preprocessing pipeline
op.seq(
    op.letterbox(640, 640),
    op.totensor(),
    op.normalize(mean=[...], std=[...]),
)
# Input: np.ndarray (H, W, C) -> Output: np.ndarray (C, H, W) normalized

# Full detection pipeline
op.seq(
    op.load('yolov8n-coco'),
    op.decode_detections(...),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)
# Input: np.ndarray (preprocessed image) -> Output: list[DetectedObject]
```

**Methods:**

#### copy

```python
copy() -> Seq
```

Return an independent copy of this pipeline with its own DMA buffer pools.

Every operator is copied so per-run buffer wiring (which sets each op's
`output_pool`) never collides between copies. Stateless operators
(color_convert, letterbox, decode_detections, and so on) and containers
are shallow-copied, which preserves their concrete type and read-only
config verbatim: a `LoadedSeq` keeps its file/model_width/model_height/
output_shapes_original, a Par/ForEach keeps its tuple builder, and any
applied preprocess fusion is carried over unchanged. Only the inference
operator (`AxRuntimeModel`, the sole owner of DmaBufPool/allocator state)
is rebuilt, so each copy gets its own buffer pools and can run concurrently
without shared state. The model itself is cached by path, so the AIPU cores
are still shared between copies.

Use this when running the same pipeline topology on multiple sources in
parallel via `scheduler.current().as_available(streamables)` or
`scheduler.current().round_robin(streamables)`.

Example::

    pipeline = op.seq(op.letterbox(640, 640), op.load('model.axm'), ...)
    pipelines = [pipeline.copy() for _ in sources]

#### optimized

```python
optimized(policy: str | None = 'inherit') -> Seq
```

Return a new Seq with the registered fusions applied.

The pipeline is lowered to flat blocks first, so the result does not depend
on how it was grouped. Children are deliberately not optimized ahead of the
parent: that would settle a preprocess before the parent had seen the model
it feeds, and the driver reaches the same result on a later pass anyway. This
Seq is never modified; with fusion off, it is itself the result, since nothing
was rewritten.
`policy` selects the preprocess fusion kernel: a mode ('auto' picks per
target capability, 'none' returns the raw op sequence) or a kernel name
to force it -- see `optimizer.policies()` for the live vocabulary.
'inherit' (default) keeps the policy already in effect (default 'auto');
None is v1.8's deprecated spelling of it.

#### save_axe

```python
save_axe(path: str, *, name: str = '', description: str = '', task: str = '') -> None
```

Save this pipeline as an .axe file.

**Args:**

- **path**: Output path (must end in .axe)
- **name**: Human-readable pipeline name
- **description**: Pipeline description
- **task**: Task type ('detection', 'classification', etc.). Taken from the pipeline's own `ax_<task>` result operator when not given.

**Raises:**

- **ValueError**: If path doesn't end in .axe
- **ValueError**: If the pipeline has been optimized, or holds two models whose file names collide
- **FileNotFoundError**: If a referenced model file doesn't exist

**Example:**

pipeline = op.seq(
    op.letterbox(640, 640),
    op.totensor(),
    op.load('yolov8n.axm'),
    op.decode_detections(algo='yolov8', name='decode'),
    op.nms(iou_threshold=0.45, name='nms'),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)
pipeline.save_axe('yolov8n-coco.axe', name='YOLOv8n COCO', task='detection')

#### batch

```python
batch(*inputs: tuple[list[Tensor] | Tensor, ...]) -> list[Any]
```

Process a batch of inputs through the sequential operators.

**Args:**

- ***inputs**: Multiple input items to process as a batch.

**Returns:** `list[Any]` -- A list of results corresponding to each input item.

#### stream

```python
stream(source, *, frame_callback=None, split_at: str | None = None, max_in_flight: int | None = None, queue_depths: list[int] | None = None)
```

Run this pipeline over the frames of `source`, yielding `(input, result)`.

Forwards to the current scheduler (see `scheduler.current`); frames are
pipelined across the available AIPU cores. `max_in_flight` bounds how many
frames may be in flight at once (`None` -> four per AIPU core, deeper than
the core count); results are yielded in source order.

**Args:**

- **source**: a path/URL string, or a `cv` reader yielding frames.
- **split_at**: optional `name` of the operator at which to split this pipeline into a preprocessing prefix (that operator and everything before it) and a main tail (everything after it). Name the split operator explicitly, e.g. `op.perspective(..., name='undistort')` then `stream(..., split_at='undistort')`. The prefix runs in the thread driving the stream and its output feeds the main tail, so the preprocess runs exactly once per frame. The split point must be before the first `op.load` (inference) operator. `ValueError` is raised if no operator has that name, if it is the last operator (empty main tail), or if it is at or after the first `op.load`.
- **frame_callback**: optional `frame_callback(frame)` callback fired after each frame has been run through the preprocessing prefix (when `split_at` is given) and before inference is submitted.  `frame` is the preprocessed, display-ready image (or the raw decoded frame when `split_at` is `None`).  Useful for forking a low-latency display path without waiting for inference results.  The callback runs synchronously on the thread driving the stream, in the frame's critical path: no further frame is preprocessed or submitted to inference until it returns, so a slow callback throttles the whole pipeline.  Keep it light and fast: a handful of cheap operations such as handing the frame to a display, or pushing it onto a queue.  Do not do heavy work (large copies, encoding, resizing) or block (locks held under contention, synchronous disk/network I/O, `time.sleep`); offload any such work to your own thread and return immediately.  Treat `frame` as read-only and short-lived: on the threaded path it is the same object handed to inference, and a raw decoded frame may be a view into a recycled decoder buffer.  Copy it if you need to keep it beyond the call.
- **max_in_flight**: in-flight window; `None` uses four per AIPU core, `1` runs inline in the calling thread (handy for debugging).
- **queue_depths**: if given, appended with the source's decode queue depth right after each pull, when the source exposes `queue_depth()`.

**Returns:** A generator of `(input, result)` pairs.

---

### Par

**Alias:** `par`

Parallel operator that executes multiple operators with same input, returns tuple.

**Input:** Any - the same input is passed to all parallel operators.

**Output:**

tuple or NamedTuple - if all operators have names, returns NamedTuple
with named fields; otherwise returns regular tuple.

**Examples:**

```python
# Run two classifiers on same image
op.par(
    op.seq(op.load('age-model', name='age'), op.ax_classification(...), op.top_k(k=1)),
    op.seq(
        op.load('gender-model', name='gender'),
        op.ax_classification(...), op.top_k(k=1)
    )
)
# Input: np.ndarray
#   -> Output: NamedTuple(age=list[Classification], gender=list[Classification])

# Parallel processing in cascade
op.for_each(
    'results',
    op.crop_roi(property='bbox'),
    op.par(
        op.seq(..., name='classifier1'),
        op.seq(..., name='classifier2'),
    ),
)
```

---

### ForEach

**Alias:** `for_each`

Apply operators to each element in a collection, preserving the original collection.

**Data-flow:**

1. Receives a collection (list) from the previous operator
2. Applies the contained operators to EACH element
3. Returns NamedTuple(original_collection, processed_results)

The original collection is passed through unchanged, allowing you to align
results with their inputs (e.g., match classifications back to detections).

**Args:**

- ***operators**: First positional string (optional) names the OUTPUT collection.        Remaining items are the operators to apply to each element.
- **iter**: Name for the INPUT collection in the result tuple.   Default: 'input'   Auto-inference: When used in op.seq(), automatically inherits the name                 from the previous operator if it has one.
- **name**: Alternative way to specify output name (keyword-only).
- **save**: Optional path to save intermediate results.

**Returns:**

NamedTuple with two fields:
- Field 1 (input): The original collection (named by iter parameter)
- Field 2 (output): List of processed results (named by first positional arg or name)

**Examples:**

```python
# Basic usage - input collection named 'input' by default
op.seq(
    op.ax_detection(...),  # Outputs list of DetectedObject
    op.for_each(
        'classifications',  # Output field name
        op.crop_roi(...),
        op.classify(...),
    ),
)
# Result: NamedTuple(input=[DetectedObject, ...], classifications=[Classification, ...])

# Custom input field name - explicitly specify iter parameter
op.seq(
    op.filter(class_ids=[op.CocoClasses.person]),  # Outputs filtered persons
    op.for_each(
        'ages',             # Output field name
        op.crop_roi(...),
        op.classify(...),
        iter='persons',     # Input field name (must be explicit!)
    ),
)
# Result: NamedTuple(persons=[DetectedObject, ...], ages=[Classification, ...])

# Access results
result = pipeline(img)
for detection, classification in zip(result.input, result.classifications):
    print(f"{detection} classified as {classification}")
```

**Constructor:**

```python
__init__(iter_name: str = iter)
```

---

### Pack

**Alias:** `pack`

Collect positional arguments into a plain tuple.

Opposite of `unpack()`. Takes `*args` and returns them as a regular tuple
(not marked for unpacking), so the next operator receives one single value.

**Examples:**

```python
# An unnamed Par's results are unpacked; pack collects them back into a tuple
op.seq(
    op.par(op.process_a(), op.process_b()),  # -> result_a, result_b (unpacked)
    op.pack(),                                # -> (result_a, result_b) plain tuple
    op.final_combiner(),                      # receives one tuple
)

# Common pattern: pack + itemgetter to extract from unpacked args
op.seq(op.pack(), op.itemgetter(0))   # accepts arbitrary number of args,
                                      # returns the first one
```

---

### Unpack

**Alias:** `unpack`

Mark a tuple or list for argument unpacking to the next operator.

Takes a single value. If it is a tuple or list, marks it so that the
pipeline passes its elements as separate positional arguments
to the next operator. Non-sequence values pass through unchanged.

This is a pure type-cast (like `std::move` in C++): it does not call
or wrap another operator.

**Examples:**

```python
# decode_segmentation returns (detections, protos) as a plain tuple.
# unpack() marks it so the next operator receives two separate args.
op.seq(
    op.decode_segmentation(algo='yolov8', num_classes=80),
    op.unpack(),
    op.par(
        op.seq(op.pack(), op.itemgetter(0), op.nms()),
        op.seq(op.pack(), op.itemgetter(1)),
    ),
)
```

**Note:**

Only needed when an operator returns a plain tuple that should be
unpacked. An unnamed `Par` already unpacks its results automatically.

---

### Itemgetter

**Alias:** `itemgetter`

Extract element by index from a tuple value (like `operator.itemgetter`).

**Examples:**

```python
op.itemgetter(0)                                    # (a, b) -> a
op.seq(op.itemgetter(0), op.nms())                  # (det, protos) -> nms(det)
op.par(op.itemgetter(1), op.itemgetter(0))          # (a, b) -> (b, a)

# To extract from positional args, compose with pack:
op.seq(op.pack(), op.itemgetter(0))                 # a, b -> (a, b) -> a
```

**Constructor:**

```python
__init__(index: int | slice)
```

---

### identity

```python
identity(x)
```

Return the input unchanged.

---

### constant

```python
constant(*args, value) -> Operator
```

Return the given constant value.
