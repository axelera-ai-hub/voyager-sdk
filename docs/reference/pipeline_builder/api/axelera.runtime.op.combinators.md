# `axelera.runtime.op.combinators`

Pipeline combinators for composing operators.

Building blocks for constructing inference pipelines:

- Seq: Sequential composition - execute operators in order, piping output to next input
- Par: Parallel execution - run multiple operators on same input, return tuple of results
- ForEach: Collection iteration - apply operators to each element in a list
- Pack: Collect positional arguments into a single tuple
- Unpack: Mark a tuple for argument unpacking to the next operator
- ItemGetter: Extract an element from a tuple by index (like operator.itemgetter)
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
| [ItemGetter](#itemgetter) | Extract element by index from a tuple value (like `operator.itemgetter`). |
| [identity](#identity) | Return the input unchanged. |
| [constant](#constant) | Return the given constant value. |

---

### Seq

Sequential operator that executes operators in order, piping output to next input.

Input: Any -- accepts whatever the first operator in the sequence accepts.

Output: Any -- returns whatever the last operator in the sequence returns.

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
    op.axdetection(class_id_type=op.CocoClasses),
)
# Input: np.ndarray (preprocessed image) -> Output: list[DetectedObject]
```

---

### Par

Parallel operator that executes multiple operators with same input, returns tuple.

Input: Any -- the same input is passed to all parallel operators.

Output: tuple or NamedTuple -- if all operators have names, returns NamedTuple
with named fields; otherwise returns regular tuple.

**Examples:**

```python
# Run two classifiers on same image
op.par(
    op.seq(op.load('age-model', name='age'), op.axclassification(...), op.topk(k=1)),
    op.seq(op.load('gender-model', name='gender'), op.axclassification(...), op.topk(k=1)),
)
# Input: np.ndarray
#   -> Output: NamedTuple(age=list[Classification], gender=list[Classification])

# Parallel processing in cascade
op.foreach(
    'results',
    op.croproi(property='bbox'),
    op.par(
        op.seq(..., name='classifier1'),
        op.seq(..., name='classifier2'),
    ),
)
```

---

### ForEach

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
    op.axdetection(...),  # Outputs list of DetectedObject
    op.foreach(
        'classifications',  # Output field name
        op.croproi(...),
        op.classify(...),
    ),
)
# Result: NamedTuple(input=[DetectedObject, ...], classifications=[Classification, ...])

# Custom input field name - explicitly specify iter parameter
op.seq(
    op.filter(class_ids=[op.CocoClasses.person]),  # Outputs filtered persons
    op.foreach(
        'ages',             # Output field name
        op.croproi(...),
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

Collect positional arguments into a plain tuple.

Opposite of `unpack()`. Takes `*args` and returns them as a regular tuple
(not `_Unpacked`), so the next operator receives one single value.

**Examples:**

```python
# After unnamed Par produces _Unpacked, pack collects args back into a tuple
op.seq(
    op.par(op.process_a(), op.process_b()),  # -> _Unpacked(result_a, result_b)
    op.pack(),                                # -> (result_a, result_b) plain tuple
    op.final_combiner(),                      # receives one tuple
)

# Common pattern: pack + itemgetter to extract from unpacked args
op.seq(op.pack(), op.itemgetter(0))   # accepts arbitrary number of args,
                                      # returns the first one
```

---

### Unpack

Mark a tuple or list for argument unpacking to the next operator.

Takes a single value. If it is a tuple or list, wraps it in `_Unpacked`
so that `_reduce_ops` passes its elements as separate positional arguments
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
unpacked. Unnamed `Par` already produces `_Unpacked` automatically.

---

### ItemGetter

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
constant(args=(), value) -> Operator
```

Return the given constant value.
