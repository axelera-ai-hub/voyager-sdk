---
title: "axelera.runtime.profiler"
---
# `axelera.runtime.profiler`


Lightweight operator profiler for Pipeline Builder pipelines.

This module provides embedded profiling for Pipeline Builder operators with minimal
overhead.
Profiling is embedded directly in the operator execution path to preserve
buffer reuse and operator fusion optimizations. Work that never becomes an Operator
(e.g. `Scheduler.stream()`'s source pull) records through `record_stage()` instead,
landing in the same `records` list and the same `summary()` table under `kind='stage'`.

Key Features:
    - Minimal overhead: ~77ns per operator when disabled (0.0023% CPU)
    - Thread-safe: one profiler instance can record from many worker threads
    - Plugin design: No hard dependencies, optional import
    - Preserves optimizations: No wrapper layers that break buffer reuse

Usage:
    Basic profiling with context manager. `op` and `profiler` are attributes of
    the package, so import them off it: `from axelera.runtime.profiler import
    OperatorProfiler` raises ModuleNotFoundError.

        >>> from axelera.runtime import op, profiler
        >>>
        >>> pipeline = op.seq(op.letterbox(640, 640), op.load('yolov8n-coco'))
        >>> prof = profiler.OperatorProfiler()
        >>> with prof:
        ...     result = pipeline(image)
        >>> stats = prof.summary()
        >>> print(stats)

    Integration with benchmarking:
        >>> prof = profiler.OperatorProfiler()
        >>> with prof:
        ...     for _ in range(100):
        ...         result = pipeline(image)
        >>> stats = prof.summary()
        >>> for name, metrics in stats.items():
        ...     print(f"{name}: {metrics['mean_ms']:.3f} ms")

Thread Safety:
    A single OperatorProfiler instance can be shared across the scheduler's worker
    pool: the scheduler puts the submitting thread's profiler in scope for
    each task (pool threads do not inherit the caller's context vars), and appends
    are lock-protected. This is what lets a benchmark profile a pipeline running
    with `max_in_flight > 1` (multiple AIPU cores in flight) instead of falling back
    to a serial run to get a profile at all. The records of frames in flight together
    interleave and overlap in wall clock (see `summary()`).

See Also:
    The Pipeline Builder Profiling guide (profiling.md): how to read the report these
    fields produce, plus the py-spy recipe for GIL-level profiling questions
    this module does not answer.

## Summary

| Name | Description |
|------|-------------|
| [OperatorProfiler](#operatorprofiler) | Lightweight operator profiler for Pipeline Builder pipelines. |
| [ProfileRecord](#profilerecord) | Single timing record for an operator execution or a named pipeline stage. |
| [ResourceMonitor](#resourcemonitor) | Per-process CPU% and peak-RSS sampler exposed as a context manager. |
| [achieved_in_flight_depth](#achieved_in_flight_depth) | Average number of frames actually in flight, by Little's Law (L = lambda * W). |
| [blocked_label](#blocked_label) | '', 'BLOCKED', or 'STRONGLY BLOCKED' from how far wall time exceeds cpu time. |
| [debug_logging_warning](#debug_logging_warning) | Warning string if `logger_name` would emit DEBUG records right now, else None. |
| [device_occupancy_pct](#device_occupancy_pct) | Device occupancy: percentage of the DEVICE's core-time actually busy. |

---

### OperatorProfiler

Lightweight operator profiler for Pipeline Builder pipelines.

Captures per-operator execution timing with minimal overhead. When enabled,
the profiler hooks into the operator execution path to time each operator.

Thread-safe: one instance can be shared across the scheduler's worker pool
(see module docstring); record() is lock-protected against concurrent
appends from multiple worker threads.

**Usage:**

Context manager (recommended):
    >>> profiler = OperatorProfiler()
    >>> with profiler:
    ...     result = pipeline(image)
    >>> stats = profiler.summary()

Manual control:
    >>> profiler = OperatorProfiler()
    >>> profiler.enable()
    >>> result = pipeline(image)
    >>> profiler.disable()
    >>> stats = profiler.summary()

**Performance:**

- Disabled overhead: ~77ns per operator (negligible)
- Enabled overhead: ~2-5μs per operator (acceptable for benchmarking)
- For 10-operator pipeline @ 30fps: 0.0023% CPU overhead when disabled

**Attributes:**

- **records** (`list[ProfileRecord]`): List of all ProfileRecord instances captured

**Constructor:**

```python
__init__()
```

**Methods:**

#### record

```python
record(operator: Any, elapsed_seconds: float, cpu_seconds: float = 0.0, scope: tuple[str, ...] = (), *, minor_faults: int = 0) -> None
```

Record timing for one operator execution.

The pipeline calls this automatically when profiling is enabled.
Not typically called by user code. Safe to call from multiple worker threads
at once (e.g. a `Scheduler.stream()` run with `max_in_flight > 1`): appends are
lock-protected, so concurrent frames on different cores never lose a record.

**Args:**

- **operator**: Operator instance (must have name attribute or __class__.__name__)
- **elapsed_seconds**: Wall-clock execution time in seconds
- **cpu_seconds**: CPU time the executing thread spent (time.thread_time() delta)
- **scope**: Names of the containers the op ran inside, outermost first
- **minor_faults**: Minor page faults the executing thread took during the call

**Example:**

>>> # Called internally by the pipeline:
>>> start = time.perf_counter()
>>> cpu_start = time.thread_time()
>>> result = operator(data)
>>> elapsed = time.perf_counter() - start
>>> cpu_elapsed = time.thread_time() - cpu_start
>>> profiler.record(operator, elapsed, cpu_elapsed, ('yolov8n-coco',),
...                 minor_faults=550)

#### record_stage

```python
record_stage(name: str, elapsed_seconds: float, cpu_seconds: float = 0.0, *, minor_faults: int = 0) -> None
```

Record timing for one named pipeline stage that is not an Operator.

Some pipeline work never becomes an `Operator` and so cannot reach
`record()` at all: the leading example is `Scheduler.stream()`'s
source pull: a `cv.Reader`/`VideoSource` is not, and should not become,
an Operator just so its decode cost can be timed. `record_stage()` is
the entry point for that case: same lock-protected append as `record()`
so it is exactly as thread-safe, but keyed by a plain
name instead of an operator instance.

The resulting record carries `kind='stage'`, and `summary()` folds it
into the same ranked table as operators (by `total_ms`) rather than
leaving it structurally invisible to it, while still shipping `kind`
through so a renderer can mark it as not-an-operator.

Deliberately no `scope`, which is the one way this is not parallel with
`record()`. A scope names the containers an operator ran inside, and a
stage is by definition work that is not in the operator graph: the source
pull feeds the pipeline rather than sitting in it, so there is no model
or named stage that ran it and every stage groups under `pipeline`. Add
the parameter when a caller has a stage that genuinely runs inside a
model, not before: an always-empty argument would read like an
attribution that was lost rather than one that never existed.

**Args:**

- **name**: Display name for the stage (e.g. 'source_pull').
- **elapsed_seconds**: Wall-clock time in seconds.
- **cpu_seconds**: CPU time the executing thread spent (time.thread_time() delta). Pass this when available: the `blocked` classification needs it exactly as much for a stage as for an operator.
- **minor_faults**: Minor page faults the executing thread took. For `source_pull` this is the CONSUMING thread, so it counts what the pull itself takes, not what decode takes: a `cv.VideoSource` decodes on its own thread into a bounded queue and the consumer only pops from it. Expect 0 there even on a source that allocates every frame. Decode's cost shows up as this row's wall-minus-cpu (waiting on an empty queue) and in the queue-depth metric beside it, not as faults.

#### enable

```python
enable() -> None
```

Enable profiling for operator execution.

This activates the profiling path in operator execution by:
- Enabling the global profiling fast-path switch
- Setting the current profiler in a context-local variable

After calling enable(), all operator executions in Seq and ForEach
will be timed and recorded.

**Note:**

Prefer using the context manager (with profiler:) over manual enable/disable.
Context manager ensures disable() is called even if exceptions occur.

**Raises:**

- **ImportError**: If the axelera.runtime.op module cannot be imported

#### disable

```python
disable() -> None
```

Disable profiling for operator execution.

This deactivates the profiling path in operator execution and clears the
current profiler reference.

After calling disable(), operator executions return to full-speed
with minimal overhead (~77ns per operator).

**Note:** Safe to call multiple times. If profiler was not enabled, this is a no-op.

#### summary

```python
summary() -> dict[str, dict[str, Any]]
```

Aggregate timing statistics by qualified operator or stage name.

Groups all recorded timings (operators via `record()`, named stages via
`record_stage()`) by `ProfileRecord.qualified_name` and computes summary
statistics. Results are sorted by total_ms (descending), so the most
expensive rows appear first regardless of kind: one ranked table, not two.

**Returns:**

`dict[str, dict[str, Any]]` -- Dictionary mapping qualified name to statistics:
`dict[str, dict[str, Any]]` -- { 'model/operator_name': {     'count': int,        # Number of executions     'total_ms': float,   # Total wall time in milliseconds     'mean_ms': float,    # Average wall time per execution     'min_ms': float,     # Minimum wall time     'max_ms': float,     # Maximum wall time     'cpu_total_ms': float,  # Total CPU time (time.thread_time())     'cpu_mean_ms': float,   # Average CPU time per execution     'cpu_min_ms': float,    # Minimum CPU time     'cpu_max_ms': float,    # Maximum CPU time     'faults_total': int,    # Minor page faults over every call     'faults_mean': float,   # Minor page faults per call     'kind': str,         # 'operator' or 'stage'     'blocked': str,      # '', 'BLOCKED', or 'STRONGLY BLOCKED'     'scope': tuple,      # containers that ran it, outermost first     'operator_class': str,  # class behind a row named for its role     'order': int,        # position the row first ran in }, ...
`dict[str, dict[str, Any]]` -- }

**Example:**

>>> stats = profiler.summary()
>>> for name, metrics in stats.items():
...     print(f"{name:30s} {metrics['mean_ms']:8.3f} ms "
...           f"({metrics['count']} calls)")

Output:
    yolov8n-coco/AxRuntimeModel     42.156 ms (100 calls)
    letterbox                        5.234 ms (100 calls)
    yolov8n-coco/postamble           2.445 ms (100 calls)
    totensor                         0.892 ms (100 calls)

**Notes:**

- Returns empty dict if no records were captured
- All times are in milliseconds (ms) for readability
- `order` is the position the operator was first recorded in, which a report
  uses to present a pipeline in execution order rather than by cost. It is
  the order records arrived, so read it, like the times below, off a serial
  run: with frames in flight together nothing holds two frames' operators
  apart in the record list
- `faults_mean` is what a row's memory allocation costs, which wall and cpu
  time cannot separate from arithmetic. A row doing arithmetic on a buffer
  it already owns reads 0; a row allocating a large output every call reads
  in the hundreds, because the kernel supplies and zeroes those pages inside
  the timed interval. Faults rather than bytes: the probe is ~650 ns, where
  counting bytes needs a numpy allocator handler or tracemalloc, neither of
  which belongs on a per-frame path. Always 0 where the platform has no
  `resource` module (Windows)
- A 0 means the call did not have to take pages from the kernel, which is
  not the same as "did not allocate", and is the reading you want either
  way: an allocation the allocator serves from a cached arena costs nothing
  to serve. Whether a repeated allocation keeps costing depends on the
  allocator's arena behaviour, which the measured figures in the
  Pipeline Builder Profiling guide set out
- Over a concurrent run (`stream` or `batch` on more than one core) the
  operators of frames in flight together overlap in wall clock, so these
  times sum to more than the elapsed time and a mean is the cost of one
  call, not a latency. `bench.benchmark_pipeline` profiles at the
  declared concurrency for that reason, and reports frame latency from
  a separate serial pass.

#### reset

```python
reset() -> None
```

Clear all recorded timing data.

Use when you want to reuse the same profiler instance for multiple
benchmark runs without accumulated history.

**Example:**

>>> profiler = OperatorProfiler()
>>>
>>> # First benchmark run
>>> with profiler:
...     for _ in range(100):
...         pipeline(image1)
>>> stats1 = profiler.summary()
>>>
>>> # Clear data
>>> profiler.reset()
>>>
>>> # Second benchmark run (fresh statistics)
>>> with profiler:
...     for _ in range(100):
...         pipeline(image2)
>>> stats2 = profiler.summary()

**Note:**

Does not affect enabled/disabled state. If profiler was enabled,
it remains enabled after reset().

---

### ProfileRecord

Single timing record for an operator execution or a named pipeline stage.

**Attributes:**

- **operator_name** (`str`): Display name (operator.name or class name; the stage name for a `record_stage()` record)
- **operator_class** (`str`): Class name of the operator; the stage name again for a `record_stage()` record, since a stage has no class
- **elapsed_seconds** (`float`): Wall-clock execution time in seconds
- **cpu_seconds** (`float`): CPU time the executing thread spent, from time.thread_time(); wall minus cpu is time that thread was off-CPU (GIL, a lock, waiting on the device), not computing.
- **timestamp** (`float`): Time when record was created (time.time())
- **scope** (`tuple[str, ...]`): The named containers this op ran inside, outermost first (see `qualified_name`); empty for a stage
- **kind** (`str`): 'operator' (via `record()`) or 'stage' (via `record_stage()`). Carried into `summary()` so a renderer can tell the two apart without importing anything beyond the returned dict.
- **minor_faults** (`int`): Minor page faults the executing thread took during the call: the kernel supplying and zeroing fresh pages. This is what an allocation costs, and neither wall nor cpu time separates it from arithmetic. A constant 0 wherever the probe cannot run, where it means "not measured".

**Constructor:**

```python
__init__(operator_name: str, operator_class: str, elapsed_seconds: float, cpu_seconds: float, timestamp: float, scope: tuple[str, ...] = (), kind: str = 'operator', minor_faults: int = 0)
```

**Properties:**

- **`qualified_name`** (`str`) — `scope/.../operator_name`: the operator, and the part of the pipeline that ran it.

---

### ResourceMonitor

Per-process CPU% and peak-RSS sampler exposed as a context manager.

While the `with` block runs, a daemon background thread samples the
CURRENT process (`psutil.Process()`, current pid) every `interval`
seconds, so other processes on the box do not contaminate the reading.
`__exit__` stops and joins the thread.

`result()` returns::

    {'cpu_pct': <avg cpu%>, 'cpu_cores': <avg cpu% / 100>,
     'cpu_count': <logical cores available>, 'peak_mem_mb': <peak rss MB>}

`cpu_pct` is psutil's per-core-summed value (can exceed 100 on a
multi-core box); `cpu_cores` is that figure divided by 100; `cpu_count`
is the host's logical core count, so a caller can report cores used
against cores available instead of a percentage that means nothing
without knowing the box size.

Adapted from `axelera.app.utils.SystemResourceMonitor` and kept here in
runtime (rather than imported from axelera.app) to preserve the zoo/app
decouple: axzoo depends on runtime, never on axelera.app.

**Usage:**

>>> with ResourceMonitor() as m:
...     do_work()
>>> m.result()
{'cpu_pct': 142.0, 'cpu_cores': 1.42, 'peak_mem_mb': 812.3}

**Constructor:**

```python
__init__(interval: float = 0.1)
```

---

### achieved_in_flight_depth

```python
achieved_in_flight_depth(throughput_fps: float, latency_ms: float) -> float
```

Average number of frames actually in flight, by Little's Law (L = lambda * W).

`throughput_fps` is the arrival/departure rate (lambda) and `latency_ms` the
average time a frame spends in the pipeline (W); their product is the average
number of frames concurrently in the system. Compare against the requested
`--in-flight` window: well below it means something other than the declared
concurrency window is limiting throughput.

`latency_ms` (W) MUST be a residency measured at the same concurrency as
`throughput_fps`, e.g. a streamed per-frame residency at that `cores`, not a
serial (single-stream) latency. Mixing regimes makes the result algebraically
`fps_concurrent / fps_serial`, a speedup ratio wearing a depth's units, and
since contention can only make W grow, a serial W turns this into a lower
bound on the depth rather than the achieved value.

---

### blocked_label

```python
blocked_label(wall_ms: float, cpu_ms: float) -> str
```

'', 'BLOCKED', or 'STRONGLY BLOCKED' from how far wall time exceeds cpu time.

A thread that is genuinely computing has wall ~= cpu; wall >> cpu means the
thread spent the rest of the interval off-CPU (GIL contention, a lock, or
waiting on the device) while nominally "running" concurrently with other
frames, which elapsed_seconds alone cannot tell apart from more compute.
`cpu_ms <= 0` with nonzero wall is the extreme of that: no CPU time at all
was attributed to this thread while it held the wall clock.

---

### debug_logging_warning

```python
debug_logging_warning(logger_name: str = 'axelera.runtime2') -> str | None
```

Warning string if `logger_name` would emit DEBUG records right now, else None.

The device execution path (`scheduler.Scheduler.exec_model`) logs one DEBUG
line per frame; enabling it (directly, or via a CLI verbosity flag that raises
the root logger to DEBUG, which this logger then inherits) measurably slows
the pipeline and silently distorts any timing captured in the same run. A
timed benchmark should check this before trusting its own numbers.

---

### device_occupancy_pct

```python
device_occupancy_pct(throughput_fps: float, device_ms: float, device_cores: int) -> float
```

Device occupancy: percentage of the DEVICE's core-time actually busy.

`throughput_fps * device_ms / 1000` is device-busy-seconds consumed per
wall-second; dividing by `device_cores`, the device-seconds the hardware has
available per wall-second, gives occupancy as a fraction. One number then says
whether the device or the host is the bottleneck, without hand-deriving it.

`device_cores` is the chip's core count, NOT the in-flight
window the caller asked for. Those are different numbers and only the first one
is a capacity: the window is how many frames may be outstanding, which a caller
picks freely and `axzoo benchmark --in-flight` documents as "a window, NOT a core
count". Dividing by it makes the metric scale with a knob instead of with the
hardware, and it does so in the wrong direction. Measured on a 48-core host, same
model and clip, only the window changed:

    window 4:  171.3 fps x 6.63 ms -> 1.14 core-equivalents busy
    window 8:  267.2 fps x 6.92 ms -> 1.85 core-equivalents busy

The device did 63% more work at the wider window. Over the window those read
29.0% and 23.2%, so occupancy FELL as utilisation rose; over the 48 real cores
they read 2.4% and 3.9%, which tracks the work.

Both measured arguments must come from the SAME pass, not merely the same
window: profiling overhead shifts `device_ms` and `throughput_fps` by different
amounts, so pairing one pass's throughput with another pass's device_ms
multiplies two numbers that never coexisted in one run.
