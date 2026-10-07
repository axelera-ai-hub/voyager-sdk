---
title: "axelera.runtime.scheduler"
---
# `axelera.runtime.scheduler`


## Summary

| Name | Description |
|------|-------------|
| [Scheduler](#scheduler) | Owns the runtime connections and model instances used to run pipelines. |
| [create](#create) | Create a scheduler and make it the current one for the enclosing block. |
| [current](#current) | Return the scheduler in scope, or the lazily-created process default. |

---

### Scheduler

Owns the runtime connections and model instances used to run pipelines.

A scheduler reserves AIPU cores on the devices it is given and keeps each
model resident on its allocated core(s), spreading work across them (see
`op.load`'s `core_allocation` and `batch`). You rarely construct one
directly:

- **Implicit (default).** Running a pipeline with no scheduler in scope --
  `detections = pipeline(image)` -- lazily creates a single process-wide
  default that is reused for subsequent calls and closed at interpreter exit.
  `scheduler.current().close()` tears it down early (a later call recreates it).

- **Explicit (scoped).** Use `scheduler.create` to choose devices or give a
  thread its own scheduler; it binds the scheduler as current for the block
  and closes it on exit. While one is in scope the default is never created::

      with scheduler.create(devices='metis-0:1:0'):
          for img in source:
              detections = pipeline(img)

Constructing a `Scheduler` directly owns resources but does not bind itself
as current, so pipelines won't find it via `scheduler.current`; you must
`close` it yourself when done. `scheduler.create` does both for you -- it
binds the scheduler as current and closes it on exit.

Threading: each scheduler has its own worker pool, and every worker resolves
`scheduler.current` back to the scheduler that owns it, so distinct threads
can each run under their own `with scheduler.create(...)` block.

Device ownership is exclusive: a scheduler reserves the sub-devices (cores) it
connects to, so two live schedulers cannot both claim the same cores. With no
`devices` argument a scheduler grabs *all* detected cores, so a second
concurrent scheduler then fails to connect (`num_sub_devices > available`).
To run multiple schedulers at once -- e.g. one per thread -- give each a
disjoint set of devices via `devices=[...]`.

**Args:**

- **devices**: Device selector string -- a comma-separated list of zero-based indices and/or names, e.g. `'0,1'` or `'metis-0:1:0'` (the same syntax as `axrunmodel`/`axdevice`'s `-d/--devices`, parsed by `axelera.runtime.select_devices`). `None` or `''` uses all devices.
- **num_workers**: Size of the worker thread pool used for parallel `batch` execution and for the frames `stream` runs concurrently. `None` (default) is two per AIPU core. Clamped to at least 1. This does not set how many frames `stream` keeps in flight -- that is four per core either way; pass `max_in_flight` to change it.
- **device_config**: Board settings to write to the cores this scheduler runs models on (see `device_config.DeviceConfig`). Overrides what a model declared on `op.load`; `AXELERA_CONFIGURE_BOARD` overrides both. Resolved per board model, so a board this does not name is left to the model's own config.

**Constructor:**

```python
__init__(devices: str | None = None, num_workers: int | None = None, device_config: DeviceConfig | None = None)
```

**Properties:**

- **`is_connected`** (`bool`) — Whether this scheduler has connected to real hardware yet.
- **`opencl_context`** — The shared OpenCL context for this scheduler, created on first access.

**Methods:**

#### close

```python
close() -> None
```

Release all resources owned by this scheduler.

Shuts down the worker pool, then releases the runtime context, which
cascades to every device connection and model instance created from it
(see `axelera.runtime.Context.release`). Idempotent. Models held by the
shared, process-wide model cache are not affected.

If this scheduler is the lazily-created process default, the default slot is
cleared too, so a later `scheduler.current()` with nothing in scope creates a
fresh one. This is how you tear the default down early --
`scheduler.current().close()` -- without a dedicated free function.

#### exec_model

```python
exec_model(model: str, inputs: Sequence[np.ndarray | int], outputs: Sequence[np.ndarray | int])
```

Execute the model with the given inputs on a scheduler-selected core.

**Args:**

- **model**: The path to the model to execute.
- **inputs**: The input data (ndarrays, or dmabuf fds as ints).
- **outputs**: The output buffers to fill (ndarrays, or dmabuf fds as ints).

#### batch

```python
batch(seq: op.Seq, inputs: tuple[list[_core.Tensor] | _core.Tensor, ...]) -> list[Any]
```

Process a batch of known inputs through the pipeline, in parallel.

Unlike streaming, the whole input set is known up front, so the scheduler
spreads the work over as many cores as is worthwhile -- evicting cores from
other models only when the batch is large enough to amortise the switch cost.
Results are returned in input order.

It is not safe to call concurrently from two threads on the same scheduler
when the batches share a model: the results stay correct, but the batches can
fall back to slower, non-evicting scheduling.

#### stream

```python
stream(source: str | cv.Reader, pipeline: op.Operator, *, frame_callback: typing.Callable | None = None, split_at: str | None = None, max_in_flight: int | None = None, queue_depths: list[int] | None = None) -> typing.Generator
```

Run `pipeline` over the frames of `source`, yielding `(input, result)`.

Usually reached via `pipeline.stream(source)`, which forwards to the
current scheduler. Frames are pipelined across the available AIPU cores by
default; `max_in_flight` bounds how many frames may be in flight (submitted
but not yet yielded), and `None` resolves to four frames per AIPU core --
deeper than both the core count and the worker pool, so one frame's host
work overlaps another's device call and a finishing worker finds the next
frame already queued. Results are always yielded in source order.

When max_in_flight is 1, then the operators are called in the current thread,
this aids debugging. But if max_in_flight is greater than 1, then the operators
are called in a worker pool to utilise multiple cores.

`split_at`, if provided, is the `name` of the operator at which to split
the pipeline into a preprocessing prefix (that operator and everything before
it) and a main tail (everything after it). Name operators explicitly to mark
the split point, e.g.::

    pipeline = op.seq(
        op.perspective(matrix=homography, width=640, height=480, name='undistort'),
        op.letterbox(640, 640),
        op.totensor(),
        op.load('yolov8n-coco'),
        ...
    )
    pipeline.stream(source, frame_callback=display, split_at='undistort')

The prefix runs in the thread driving the stream and its output is fed to the
main tail, which is submitted to the worker pool; the preprocessing therefore
happens exactly once per frame. `ValueError` is raised if the pipeline has no
operator with that name.

`frame_callback`, if provided, is called as `frame_callback(frame)` immediately
after each frame has been run through the preprocessing prefix (when
`split_at` is given) and before inference is submitted. `frame` is
therefore the preprocessed image (e.g. geometrically corrected and ready for
display), not the raw decoded frame. When `split_at` is `None` there is no
preprocessing prefix, so `frame` is the raw decoded frame.

The callback runs synchronously on the thread driving the stream (the calling
thread here; the per-stream thread for `as_available`), on the frame's
critical path: the next frame is not preprocessed or submitted to inference
until it returns.  It MUST therefore be light and fast: a few cheap
operations such as handing the frame to a display or pushing it onto a queue.
Do not run heavy work (large copies, encoding, resizing) or block (contended
locks, synchronous disk/network I/O, `time.sleep`): a slow callback stalls
the decode queue and throttles the entire pipeline.  Offload any such work to
your own thread and return immediately.

Treat `frame` as read-only and short-lived.  On the threaded path it is the
same object handed to inference, and a raw decoded frame may be a view into a
recycled decoder buffer, so mutating it corrupts inference and retaining it
past the callback (e.g. queuing it for an async display) can show recycled
pixels.  Copy the frame if you need to keep it beyond the call.

An `OperatorProfiler` enabled on the calling thread records correctly either
way: the scheduler puts it in scope for each pool task.
Every frame pulled from `source` is timed and, when a profiler is active,
recorded as a `record_stage('source_pull', ...)`: decode never becomes an
Operator, so `record()`, which only sees operator calls, cannot
attribute it. The pull runs on this thread either way, so it reads the active
profiler straight from context.

`queue_depths`, if given, is appended with the source's decode buffer
occupancy right after each pull (via `source.queue_depth()`, skipped when
the source has none). Queue depth is an occupancy, not a duration, so unlike
the pull's own cost it cannot become a profiler stage.

#### multistream

```python
multistream(streams: typing.Iterable[typing.Iterable], *, order: str = 'as_available', on_frame: typing.Callable | None = None)
```

Run several pre-built streams together under one entry point.

The three dedicated methods vary along two independent axes; this exposes
both as keywords (the methods remain as convenience shortcuts):

**Order:**

'as_available' runs each stream on its own thread and surfaces frames
in arrival order; 'round_robin' pulls one frame per stream in turn from
the calling thread (serial, strict interleave, no overlap).

on_frame:
    None -> generator mode: yields `(stream_id, input, result)` and the
    caller's loop is the single consumer. A callable -> callback mode:
    `on_frame(stream_id, input, result)` is invoked per frame and the
    call blocks until every stream finishes. With `order='as_available'`
    the callback runs on each stream's own thread (so consumer work can
    parallelise); with `order='round_robin'` it runs in the calling
    thread.

The four combinations map to::

    order='as_available', on_frame=None  -> as_available   (generator)
    order='round_robin',  on_frame=None  -> round_robin    (generator)
    order='as_available', on_frame=cb    -> with_callback  (per-stream threads)
    order='round_robin',  on_frame=cb    -> serial round-robin callback

Completion `(stream_id, None, exc_or_None)` and per-order error contracts
are exactly those of the method each case dispatches to; see those methods.

**Returns:**

A generator of `(stream_id, input, result)` in generator mode, or
``None`` in callback mode (which blocks until all streams finish).

#### with_callback

```python
with_callback(streams: typing.Iterable[typing.Iterable], callback)
```

Run multiple pre-built streams concurrently, invoking *callback* for each frame.

Each stream is iterated in its own thread so they run independently and
deliver frames as they become available, not round-robin. Use a separate
pipeline copy per stream (via `pipeline.copy()`) to avoid shared operator
state.

The callback runs on each stream's own worker thread, so it is entered
concurrently (one live invocation per stream) and must be thread-safe; any
shared state it touches needs its own synchronization. That concurrency is
the point: unlike `as_available`, which funnels every frame through the
single consumer thread, `with_callback` is the one variant whose
consumer-side work parallelises across streams, which is what makes it
worthwhile on a host-bound pipeline.

This blocks until every stream finishes, and a stream is only checked for
teardown between frames; a source that never yields another frame (e.g. a
wedged camera) will therefore block the caller. Add your own stop signal
(a flag the callback raises on, or a bounded source) if you need to bail.

Error contract: the first error (from a stream, the callback, or the final
sentinel callback) stops the other streams and is re-raised once they have
all finished. When several fail, the one that surfaces is the first thread
to *record* an error, not the first error in wall-clock time, so with
concurrent failures which exception propagates is not deterministic.

**Args:**

- **streams**: iterable of stream generators, each yielding `(input, result)` pairs      (e.g. from `pipeline.copy().stream(source)`).
- **callback**: called as `callback(stream_id, img, result)` for each frame, and       `callback(stream_id, None, exc)` when a stream finishes (`exc` is       an exception if the stream failed, otherwise `None`).

#### round_robin

```python
round_robin(streams: typing.Iterable[typing.Iterable]) -> typing.Generator
```

Yield `(stream_id, input, result)` by cycling through streams in order.

When a stream is exhausted, yields `(stream_id, None, None)` once and
removes it. Stops when all streams are exhausted.

Frames are pulled one stream at a time via `next()` in the calling
thread, so the streams never overlap: wall-clock time is the sum of every
stream's per-frame cost, not the max across streams, and this does not
improve even when the per-frame work releases the GIL. It bites hardest
when the per-stream *source* is the slow part (e.g. several live cameras,
the case this is aimed at), where a slow stream stalls all the others.
Use `as_available` or `with_callback` when you want them to run
concurrently; reach for `round_robin` only when deterministic, strictly
interleaved ordering matters more than throughput.

Error contract: a failing stream yields `(stream_id, None, exc)` and is
dropped, but the exception is not re-raised until every remaining stream
has drained, so healthy streams always deliver all of their frames first.
(`as_available` instead raises as soon as the failing stream surfaces,
dropping other streams' undelivered frames.)

#### as_available

```python
as_available(streams: typing.Iterable[typing.Iterable]) -> typing.Generator
```

Run multiple pre-built streams concurrently, yielding `(stream_id, input, result)`.

Each stream is iterated in its own thread so they run independently and
yield frames as they become available, not round-robin. Use a separate
pipeline copy per stream (via `pipeline.copy()`) to avoid shared operator
state.

Aggregate throughput only improves when the per-frame work releases the
GIL (native or numpy-heavy post-processing). For a host/Python-bound
pipeline these threads buy latency, fairness and interleaving rather than
throughput.

Error contract: the exception is raised as soon as the failing stream's
sentinel reaches the front of the queue, so frames from other streams that
have not yet been dequeued are dropped (typically only a handful arrive
before the raise). Use `round_robin` instead if you need every other
stream drained to completion before the error surfaces.

Note: producer threads only check for teardown between frames, so a stream
parked in a blocking source decoder (e.g. a wedged camera) is not force-
stopped. The generator's cleanup waits at most 5 seconds for it
rather than blocking the caller forever; the thread is a daemon and is
reaped at interpreter exit.

**Args:**

- **streams**: iterable of stream generators, each yielding `(input, result)` pairs      (e.g. from `pipeline.copy().stream(source)`).

**Returns:** `typing.Generator` -- A generator of `(stream_id, input, result)` triples.

---

### create

```python
create(devices: str | None = None, num_workers: int | None = None, device_config: DeviceConfig | None = None) -> typing.Generator[Scheduler, None, None]
```

Create a scheduler and make it the current one for the enclosing block.

The scheduler owns the runtime connections and model instances for pipelines
run inside the block. Use this to target specific devices, or to give a thread
its own scheduler -- each `with scheduler.create(...)` binds only the calling
context::

    with scheduler.create(devices='metis-0:1:0'):
        for img in source:
            detections = pipeline(img)

`devices` is a selector string (see `Scheduler`): a comma-separated list of
zero-based indices and/or names, e.g. `'0,1'`; `None` uses all devices.

`device_config` writes board settings to the cores this scheduler uses,
overriding what any model declared on `op.load`::

    limits = DeviceConfig({'110m': BoardConfig(mvm_limitation=90)})
    with scheduler.create(devices='metis-0:1:0', device_config=limits):
        ...

While a scheduler is in scope the process default is never created. On exit the
scheduler is closed and its connections and instances released.

**Raises:**

- **RuntimeError**: if the process default scheduler already owns the devices -- it would be created the first time a pipeline runs with no scheduler in scope (e.g. `pipeline(img)`), leaving no devices for this one. Create the scheduler before running any pipeline, or release the default with `scheduler.current().close()`.

---

### current

```python
current() -> Scheduler
```

Return the scheduler in scope, or the lazily-created process default.

Resolves, in order: a scheduler set by `scheduler.create` in this context; the
scheduler that owns the current worker thread;
otherwise the process-wide default, created on first use. This is what lets
`pipeline(image)` just work with no explicit scheduler.
