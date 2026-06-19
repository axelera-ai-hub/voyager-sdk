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
- **num_workers**: Size of the worker thread pool used for parallel `batch` execution. Clamped to at least 1.

**Methods:**

#### close

```python
close() -> None
```

Release all resources owned by this scheduler.

Shuts down the worker pool, then releases the runtime context, which
cascades to every device connection and model instance created from it
(see `axelera.runtime.Context.release`). Idempotent. Models held by the
shared, process-wide `_model_context`/`_load_model` cache are not affected.

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
other models only when the batch is large enough to amortise the switch cost
(see `_plan_batch`/`_batch_core_target`). Results are returned in input order.

TODO: not safe to call concurrently from two threads on the same scheduler
when the batches share a model. Core *selection* stays correct (it is all
under `self._cond`), but `_batch_budget` is keyed by model path with no
per-call scoping or refcounting: the second `_plan_batch` clobbers the
first's budget, and the first batch to finish pops the key in `_end_batch`,
pulling the eviction-growth budget out from under the other still-running
batch -- which then silently degrades to non-evicting scheduling (slower,
not wrong). To support concurrent batches, scope the budget per call
(e.g. refcount per model, or a budget token threaded through the run)
rather than a single shared `{model: budget}` dict.

#### stream

```python
stream(source: str | cv.Reader, pipeline: op.Operator, *, max_in_flight: int | None = None) -> typing.Generator
```

Run `pipeline` over the frames of `source`, yielding `(input, result)`.

Usually reached via `pipeline.stream(source)`, which forwards to the
current scheduler. Frames are pipelined across the available AIPU cores by
default; `max_in_flight` bounds how many frames may be in flight (submitted
but not yet yielded), and `None` resolves to the scheduler's core count.
Results are always yielded in source order.

When max_in_flight is 1, then the operators are called in the current thread,
this aids debugging. But if max_in_flight is greater than 1, then the operators
are called in a worker pool to utilise multiple cores.

---

### create

```python
create(devices: str | None = None, num_workers: int = 4)
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
scheduler that owns the current worker thread (see `_bind_worker_scheduler`);
otherwise the process-wide default, created on first use. This is what lets
`pipeline(image)` just work with no explicit scheduler.
