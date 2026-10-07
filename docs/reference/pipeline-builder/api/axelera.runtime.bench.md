---
title: "axelera.runtime.bench"
---
# `axelera.runtime.bench`


Public benchmark helper: latency, throughput, and device throughput.

`benchmark_pipeline(runnable, video)` measures an already-built runtime
pipeline and returns a `BenchmarkResult`; `format_benchmark(result)` renders it
as the human report. Model recipes (`--benchmark`) and `axzoo benchmark` share
this one measurement path, so their numbers cannot diverge.

Everything here operates on runtime's own objects (`Operator.stream`,
`scheduler.current()`, `OperatorProfiler`, `ResourceMonitor`), which is why it
lives beside them rather than in a model catalog: benchmarking a pipeline needs
no catalog, and a recipe must not acquire one to measure what it just built.

Dependency-light: no argparse, no cv2. The heavier runtime submodules are
imported lazily inside the measurement functions so importing this module stays
cheap.

The quantities:

  - **Latency** (ms): the end-to-end per-frame distribution over N serial runs.
    Always single-stream; `cores` never touches it. This is a percentile
    distribution, not a throughput figure.
  - **Per-operator/stage + host/device split** (ms): a *second* streamed pass,
    profiler enabled, over the same `video`/`frames`/`cores` window as the
    throughput measurement below: the table is produced at the declared
    concurrency and explains the throughput number, instead of a serial guess
    at it. The source pull is one row in this same table: it never
    becomes an Operator, so `record()` cannot attribute it, but it is timed on
    the calling thread of this same pass and recorded via `record_stage()`, so
    `summary()` ranks it by total_ms alongside every operator rather than
    reporting it separately (`kind` marks which rows are stages, not
    operators). That row is the PULL, not decode itself: a `cv.VideoSource`
    decodes on its own thread into a bounded queue, so the row's wall time is
    mostly waiting on that queue and its cpu and faults are the pop's, near
    zero. Read decode's own health from this row's wall-minus-cpu together with
    the queue depth below, never from its faults. Each row also carries mean
    minor page faults, which is what its memory allocation costs: neither wall
    nor cpu time separates a row waiting for the kernel to supply and zero fresh
    pages from one doing arithmetic.
    `device_ms` is a per-frame latency shape measured at that concurrency, not a
    throughput.
  - **Throughput** (fps): a streamed loop over the video
    (`runnable.stream(video, max_in_flight=cores)`), profiler disabled so this
    number stays clean of profiling overhead. `cores=None` (the default) takes
    the window `stream` itself picks, so the figure is what an application gets;
    `cores == 1` is the serial floor, which leaves most of the chip idle.
  - **Queue depth**: the source's decode buffer occupancy, sampled on that same
    throughput pass (mostly-empty: source-starved; mostly-full: back-pressured
    onto decode; neither, with cores idle: the submission loop itself). Unlike
    the pull's own timing, a queue depth has no wall/cpu shape, so it cannot
    become a profiler stage and is reported on its own.
  - **Occupancy**: device occupancy (% of the `cores` window actually busy, from
    the profiled pass's own throughput and device_ms, so both terms come from the
    same run), host cores used vs. available, and achieved in-flight depth
    (Little's Law: throughput times the streamed per-frame residency
    `device_ms + host_ms` at that same concurrency, not the serial latency mean,
    since contention only grows residency and a serial W would bias the depth
    low), tying the above together instead of leaving the reader to hand-derive
    them.
  - **Device throughput** (fps): the same streamed model isolated to the on-chip
    inference op (preprocess/postprocess + host packing excluded), at the same
    `cores`. This is the fps counterpart to `device_ms`; occupancy is what
    relates the two above `cores == 1`.

A run with per-frame DEBUG logging enabled refuses to measure at all (see
`axelera.runtime.profiler.debug_logging_warning`), raising `BenchmarkRefused`:
that logging measurably slows the device path and would silently invalidate
every number above. `axzoo benchmark` checks this itself before opening the
device, so a CLI user never reaches the raise; it stays in `benchmark_pipeline`
as a safety net for a direct API caller (e.g. a generated recipe's
`--benchmark`).

## Summary

| Name | Description |
|------|-------------|
| [BenchmarkRefused](#benchmarkrefused) | Raised when a benchmark refuses to measure at all. |
| [BenchmarkResult](#benchmarkresult) | The measured quantities, one dict per section of the human report. |
| [benchmark_pipeline](#benchmark_pipeline) | Measure an already-built `runnable` over `video` (or a single image). |
| [format_benchmark](#format_benchmark) | Render a `BenchmarkResult` as the human report string. |
| [latency_stats](#latency_stats) | Distribution summary of per-iteration end-to-end latencies (ms). |
| [refuse_if_debug_logging](#refuse_if_debug_logging) | Raise `BenchmarkRefused` if per-frame DEBUG logging is active right now. |

---

### BenchmarkRefused

Raised when a benchmark refuses to measure at all.

A dedicated type (not a bare `ValueError`), so a caller (the CLI, or a
generated recipe's `--benchmark`) can catch exactly this refusal and
print an actionable message, without swallowing an unrelated `ValueError`
(e.g. "no frames decoded" from an empty video).

---

### BenchmarkResult

The measured quantities, one dict per section of the human report.

Each field is a plain dict so the result serializes with `dataclasses.asdict`.

UNITS CHANGE for anyone tracking these across versions (`axzoo benchmark --json`
feeds a tracked perf table). `host_device`'s `device_ms`/`host_ms` used to be the
two halves of ONE frame's single-stream latency, and are now per-frame averages of
wall time summed across concurrently running workers, measured at the declared
concurrency. They no longer reconcile with `latency` and they grow with the
in-flight window: on one host and model, `host_ms` went 27.34 (single-stream) to
37.28 at a window of 4 and 54.19 at 16, i.e. from 109% of the latency mean to 207%.
A row from before this change is not comparable with a row after it, and the step
is a units change, not a regression.

**Constructor:**

```python
__init__(latency: dict, per_operator: dict, host_device: dict, throughput: dict, source: dict, device_throughput: dict, resources: dict, occupancy: dict)
```

---

### benchmark_pipeline

```python
benchmark_pipeline(runnable, video: str, *, warmup: int = 10, iterations: int = 1000, cores: int | None = None, frames: int | None = None, seconds: float | None = None, device_frames: int | None = None, device_seconds: float | None = None) -> BenchmarkResult
```

Measure an already-built `runnable` over `video` (or a single image).

`runnable` is a recipe `build_pipeline(axm)` result (optimized), owning the
model-appropriate input adaptation (its preprocess leads with the color
convert). Both `axzoo benchmark` and the standalone recipe obtain it from
that same call, so they measure the same pipeline. Runs `warmup` untimed
frames, then a serial latency distribution over `iterations` runs of the
first frame, a streamed throughput over `video` (`cores` in flight; `None`
resolves to the scheduler's own `stream` default), a
profiled pass at that same concurrency for the per-operator/host-device
breakdown, and an isolated device-op throughput.

`frames` and `seconds` bound the e2e stream, whichever is reached first, and
the profiled pass takes the same pair so the two cover the same window. A
frame count alone cannot serve models of different speeds: the same count is
a minute on one model and two seconds on another, and what settles a
throughput reading is how long it ran. The device-op measurement takes its own
budget (`device_frames`, `device_seconds`, or both, whichever is reached
first) because the two want different units: a frame count that gives the
e2e run a sensible length can be a fraction of a second on the isolated chip
op, which is where that measurement stops meaning anything. Neither given
runs it for 10 seconds.

Refuses to measure at all when per-frame DEBUG logging is active (see
`axelera.runtime.profiler.debug_logging_warning`), raising `BenchmarkRefused`:
that logging measurably slows the device path, so a report produced under it
would be silently wrong rather than merely incomplete. `axzoo benchmark`
checks this itself before opening the device, so this is a safety net for a
caller that skips that check (e.g. a generated recipe's `--benchmark`).

---

### format_benchmark

```python
format_benchmark(result: BenchmarkResult) -> str
```

Render a `BenchmarkResult` as the human report string.

---

### latency_stats

```python
latency_stats(samples_ms: list[float]) -> dict[str, float]
```

Distribution summary of per-iteration end-to-end latencies (ms).

`jitter` is peak-to-peak (max - min); `stddev` is the sample standard
deviation. Both are reported because they answer different questions:
stddev for typical spread, jitter for the worst-case swing.

---

### refuse_if_debug_logging

```python
refuse_if_debug_logging() -> None
```

Raise `BenchmarkRefused` if per-frame DEBUG logging is active right now.

Shared by `axzoo benchmark` (which calls this before opening the device or
loading the `.axm`, so a CLI user gets a fast, correctly-exit-coded refusal)
and `benchmark_pipeline` (a safety net for a direct caller, e.g. a generated
recipe's `--benchmark`, that skips the CLI's own check). See
`axelera.runtime.profiler.debug_logging_warning` for why this matters: that
logging measurably slows the device path, so a report produced under it
would be silently wrong rather than merely incomplete.
