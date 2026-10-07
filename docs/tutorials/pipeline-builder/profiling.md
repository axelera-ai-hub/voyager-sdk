---
title: "Profiling"
---
# Profiling Pipeline Builder pipelines

Two tools, for two different questions. Start with the built-in one.

## Quick start

The whole report, for a model you already have. Either front door prints the
same thing, because both call the same measurement path:

```bash
axzoo benchmark yolov8n-coco --video clip.mp4 --in-flight 4
python yolov8n_coco_axrelay.py --benchmark clip.mp4   # a generated model recipe
```

To profile a pipeline you built yourself, wrap the run in an `OperatorProfiler`
and read `summary()`:

```python
from axelera.runtime import op, profiler

pipeline = op.seq(op.letterbox(640, 640), op.to_tensor(), op.load('yolov8n-coco.axm'))

prof = profiler.OperatorProfiler()
with prof:
    for image, result in pipeline.stream('clip.mp4', max_in_flight=4):
        pass

for name, m in prof.summary().items():
    print(f"{name:40} {m['mean_ms']:7.2f} ms  {m['faults_mean']:6.0f} faults  "
          f"{m['count']:5} calls")
```

Import the submodules off `axelera.runtime` as above. `axelera.runtime.profiler`
is the name a page or a traceback shows, but it is installed as an attribute of
the package rather than as an importable path, so
`from axelera.runtime.profiler import OperatorProfiler` raises
`ModuleNotFoundError`. The same is true of `bench`.

`summary()` returns one row per operator, keyed by `model/operator` (a stage such
as `source_pull` has no model, so it keys as a bare name), sorted most expensive
first. Nothing is recorded outside the `with` block. A run with no profiler active
pays one flag test per operator, about 77 ns; none of the clocks, the fault probe
or the record append run.

For what each column of `summary()` means field by field, see the
[Profiler reference](../../reference/pipeline-builder/api/axelera.runtime.profiler.md); this page is about
reading the report those fields produce.

To render the same rollup the benchmark prints rather than your own loop, use
`bench.benchmark_pipeline(pipeline, 'clip.mp4')` and pass its `BenchmarkResult`
to `bench.format_benchmark`. That path raises `bench.BenchmarkRefused` if DEBUG
logging would invalidate the numbers, so catch it and print the reason.

## The report

One real run, `yolov8n-coco` over a 200-frame clip on a host with 128 CPU cores
and 48 AIPU cores:

```
Latency e2e (ms, N=30, serial):
  min 24.13  mean 24.60  p50 24.30  p95 25.52  p99 28.63  p99.9 n/a(<1000)  max 29.80  stddev 1.04  jitter 5.67
Per-operator/stage time (ms, streamed at in_flight=4), execution order within each model.
  Mean/CPU/Faults are per CALL: a cascade body runs once per detection, so read Count beside them.
    Name                                   Mean      CPU   Faults      Min      Max   Count  Flag
  yolov8n-coco                        3542.85 ms total (81.8% of profiled)
    AxRuntimeModel                         6.70     0.29        0     6.09     7.96     200  DEVICE WAIT
    Depad                                  0.03     0.03        0     0.02     0.23     200
    Dequant                                2.02     1.79      290     1.05     4.66     200
    _AlignToModelOutput                    0.02     0.02        0     0.02     0.04     200
    _ListUnpack                            0.00     0.00        0     0.00     0.01     200
    postamble (OnnxModel)                  8.94     8.83        3     7.50    14.07     200
  pipeline                             790.40 ms total (18.2% of profiled)
    source_pull                            0.08     0.03        0     0.01     6.96     203  STAGE BLOCKED
    ResizeClOperator                       2.18     1.78       30     1.31     4.32     200
    DecodeDetections                       1.53     1.36      122     0.96     3.29     200
    Nms                                    0.04     0.04        0     0.03     0.43     200
    ToImageSpace                           0.07     0.07        0     0.05     0.15     200
    AxDetection                            0.04     0.04        0     0.03     0.07     200
Host/device wall-time share (streamed at in_flight=4; device = compute + data transfer; per-frame averages, not one frame's latency decomposition):
  device (inference)  6.70 ms/frame  (30.9% of summed wall)
  host   (pre/post)   14.96 ms/frame  (69.1% of summed wall)
  device_ms is a per-frame average, not a throughput; see occupancy below.
Throughput (wall-clock, streamed pipeline):
  200 frames in 1.22 s  ->  164.6 fps  (in_flight=4)
  host cpu 6663.0% (=66.63 cores)  peak mem 413.2 MB
Source queue depth (decode buffer occupancy):
  mean 28.1  min 0  max 29
Occupancy (at the declared concurrency; answers where the throughput above goes):
  device occupancy   2.4%  (of 48 device cores)
  host cores used    66.63 of 128 available
  in-flight depth    3.7 achieved  (in-flight window 4)
Device throughput (wall-clock, streamed chip op only; excludes pre/post + pack):
  200 frames in 0.36 s  ->  559.2 fps  (in_flight=4)
  this is the fps counterpart to device_ms above; the two are latency vs throughput, not a contradiction.
```

Four rows to read together. `AxRuntimeModel` is flagged `DEVICE WAIT`: wall 6.70
against cpu 0.29, because it waits on the chip, which is what it is for.
`postamble` is the opposite, wall 8.94 against cpu 8.83, so it is the largest
genuine compute cost in the pipeline. `Dequant` costs 2.02 ms with 290 faults,
which says a good part of it is the kernel handing over fresh pages rather than
arithmetic. And `source_pull` reads wall 0.08 against cpu 0.03 with 0 faults: it
is the queue pop, not decode (see below), and its `BLOCKED` flag is the consumer
waiting on the queue.

The occupancy block cross-checks against the rows above it. It takes its fps from
the profiled pass, which ran at about 172 fps here (the report does not print it;
the 164.6 under `Throughput` is a separate streamed pass): 2.4% of 48 cores is
about 1.15 cores busy, which is `fps x device_ms` = 172 x 6.70 ms; and the depth
is `fps x (device_ms + host_ms)` = 172 x 21.66 ms = 3.7 of the 4 requested. The
device is nowhere near the constraint here; the host postamble is.

## What a refused run looks like

Per-frame DEBUG logging on the device path measurably slows a run (see
`profiler.debug_logging_warning()`), so a timed run refuses to report while it is active
rather than printing a number that is quietly about 33% low:

```
Error: DEBUG logging is enabled (axelera.runtime2); per-frame DEBUG logging costs
roughly 33% of throughput on the device path (measured 220.7 fps with it vs 331.8
fps without, same command otherwise), which would invalidate a timed report.
Rerun without -v/DEBUG logging.
```

`axzoo benchmark` checks this before it opens the device or loads the `.axm`, and
a standalone recipe checks it before the build, so the refusal costs you no
compile. Rerun without `-v`.

## How to read the report

Work through these in order. Each answers a narrower question than the last,
and an earlier "no" makes the later ones moot.

1. **Is the run trustworthy?** If it refused to report because per-frame
   DEBUG logging was active, fix that and rerun; nothing below is worth
   reading from a run that never happened.

2. **Where is the wall time?** The `Per-operator/stage time` table is one
   block per model, most expensive model first, so a cascade does not read as
   one flat list in which two models' rows interleave. Within a model the rows
   are in execution order, which is what makes a fusion that did not fire
   legible: its `NchwToNhwc`, `Quant` and `Pad` stay adjacent where the
   optimized pipeline shows a single `QuantizeTransposePad`. Rows the pipeline
   runs outside any model, decode included, group under `pipeline`.
   `source_pull` is the frame pull timed on the streaming pass, not an
   `Operator`; it and any other row flagged `STAGE` are not operators. It times
   the PULL, not decode: a `VideoSource` decodes on its own thread into a
   bounded queue, so this row's cpu and faults are the queue pop's and read
   near zero however hard decode is working. Decode shows up instead as this
   row's wall-minus-cpu (waiting on an empty queue) and in `Source queue
   depth`. Rank models by the
   group header, never by scanning the `Mean` column: those are per call, and
   a cascade breaks the two apart (see Cascades below).

3. **For an expensive row, is it computing or waiting?** Compare `Mean`
   (wall) against `CPU`. When they track together, the row is doing real
   compute and the fix is a faster kernel. When wall pulls away from a flat
   `CPU`, the thread spent that time off-CPU (a lock, the GIL, or a device
   wait), and the fix is concurrency or contention, not a faster kernel. The
   device op is flagged `DEVICE WAIT` rather than `BLOCKED`/`STRONGLY
   BLOCKED`: it waits on the chip by design, so that classification would
   otherwise fire on every run and tell you nothing.

4. **Is an expensive row allocating?** `Faults` is minor page faults per
   call: the kernel supplying and zeroing fresh pages, which is what memory
   allocation costs and what neither time column can separate from
   arithmetic. A row that allocates a large output every frame reads in the
   hundreds; one that reuses a buffer it already owns reads 0. So a large
   `Mean` with a `CPU` that tracks it and a fault count in the hundreds is an
   allocation cost, and the fix is to reuse the output buffer rather than to
   speed up the kernel. The column is a constant 0 wherever the probe cannot
   run: Windows has no `resource` module, and macOS and the BSDs have one
   without `RUSAGE_THREAD`. On those a 0 means "not measured", not "took no
   pages", so read the column only on Linux.

   A 0 means the call did not have to take pages from the kernel, which is not
   quite "did not allocate": an allocation served from a cached arena is free,
   and reads as free. glibc raises its mmap threshold when it sees repeated
   large frees, up to a 32 MiB ceiling, so a repeated allocation below that
   size can fault on its first few calls and then settle at 0, while one above
   it is a fresh mapping every call and keeps costing. Measured on a repeated
   16 MiB allocation the counts go 520, 507, 0, 0, 0; at 32 MiB and above they
   stay at 528 every round. Either way the column is reporting what the call
   actually paid.

5. **Which resource is the constraint?** `Occupancy` reports device
   occupancy (percentage of the DEVICE's own cores actually kept busy, never
   of the `--in-flight` window, which is a knob and not a capacity) next to
   host cores used against available. High device occupancy
   with host cores to spare points at the device; the reverse points at the
   host; both near their ceiling means neither alone explains the throughput.

6. **If neither is saturated, who is starving whom?** Still in `Occupancy`,
   compare the achieved in-flight depth against the requested `--in-flight`;
   well below it means something other than the declared concurrency window
   is limiting throughput. Then check `Source queue depth`: mostly empty
   means the pipeline is source-starved (decode cannot keep up); mostly full
   means it is back-pressured onto decode; low and cores sitting idle points
   at the submission loop itself rather than decode or the device.

7. **What you must not compare across sections.** `device_ms` (in
   `Per-operator/stage time` and `Host/device wall-time share`) is a latency
   in ms per frame at the declared concurrency; device throughput (the
   `Device throughput` section) is an fps. The two only reciprocate
   (`1000/device_ms == fps`) at `--in-flight 1`; at higher concurrency, read
   `Occupancy` instead of dividing one into the other. `Host/device
   wall-time share` is itself a per-frame average of wall time summed across
   concurrently running workers, not a decomposition of one frame's latency,
   so it will not add up to (or otherwise reconcile with) the single-stream
   `Latency e2e` block above it.

## Cascades: read Count, not just Mean

A cascade ([`op.for_each` over a detector's output](cascade-tutorial.md)) runs its body once per
detection, so within one report some rows have a Count several times the frame
count while the models around them have one. Reading down the Mean column then
gets the answer backwards. Five frames, twenty detections each:

```
    Name                                   Mean      CPU   Faults      Min      Max   Count  Flag
  cascade/ages/age_gender               55.39 ms total (77.5% of profiled)
    crop                                   0.05     0.05        0     0.05     0.12     100
    cls_infer                              0.50     0.50        0     0.50     0.50     100
  cascade/yolov8n_coco                  16.04 ms total (22.5% of profiled)
    det_preproc                            1.20     1.20        0     1.20     1.21       5
    det_infer                              2.00     2.00        0     2.00     2.00       5
```

`det_infer` has the largest Mean in the report, 2.00 ms, four times `cls_infer`'s
0.50 ms. It is also the cheaper of the two: it runs once per frame where
`cls_infer` runs twenty times, so it costs 10 ms of the run against the
classifier's 50 ms. Optimising the row with the biggest Mean would be work spent
on 22.5% of the frame.

The group totals get this right (77.5% against 22.5%) because a total is over the
whole run and so already carries the call count. So rank models by the group
header, and use the per-row columns only to ask why one row inside a model is
expensive. For a row's per-frame cost, divide its total by the frame count the
`Throughput` section prints.

Scope comes from the containers that ran the operator, so a cascade body is
attributed to its own model rather than merged into its parent
(`cascade/ages/age_gender/cls_infer`). Name your models: an unnamed `op.load(...)`
falls back to the stem of its `.axm`, and two unnamed models sharing a basename
merge into one row.

## External: py-spy

The built-in profiler answers *where* time goes and *whether* the device or
host is the bottleneck, but not *which Python frame* is holding the GIL when
a row shows up `BLOCKED`. For that, reach for
[py-spy](https://github.com/benfred/py-spy) instead of adding a profiling
dependency to the runtime. It is not vendored here and should not become one:
it is used occasionally, and a sampling profiler with a flame-graph output
needs a human to read it, not an API to call.

Two gotchas that will otherwise cost you the whole session:

1. **Attaching to a running process fails under the default `ptrace_scope=1`.**
   Spawn the target as py-spy's own child instead of attaching after the fact:

   ```bash
   py-spy record -o profile.svg -- python3 my_pipeline.py video.mp4
   ```

   Attaching to an already-running pid (`py-spy record -o profile.svg --pid <pid>`)
   needs either root or a relaxed `ptrace_scope`; spawning as a child sidesteps
   the permission check entirely.

2. **Pin the interpreter by absolute path, or py-spy profiles the wrong
   install.** `py-spy record -- python3 ...` resolves `python3` from `PATH` at
   spawn time, which is whichever environment happens to be active in that
   shell. If you have more than one, that is not necessarily the one holding
   the pipeline you meant to profile. Name the interpreter outright:

   ```bash
   py-spy record -o profile.svg -- /path/to/venv/bin/python my_pipeline.py
   ```

   `python -c "import sys; print(sys.executable)"` in the environment you want
   prints the path to use. Confirm it before reading the flame graph: a
   silently-wrong `python3` profiles a different install and reports no error
   saying so.
