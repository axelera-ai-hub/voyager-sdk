#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""A tiny dependency-driven job scheduler.

Runs a DAG of jobs, dispatching each one as soon as its dependencies finish, so the
independent stages of an eval overlap (device codegen ∥ calibration/quantization ∥
bundle export). It deliberately assumes overlapping is a win -- the user opts in via
``--parallel`` -- so there is no resource accounting: every ready job runs
immediately. Thread jobs run in a ThreadPoolExecutor (they hold un-picklable Python
state -- torch models, the runtime); ``in_process`` jobs run in a spawn ProcessPoolExecutor
(TVM codegen is not concurrency-safe in-process and its args/result are picklable).

Each job's ``fn`` is called with a single dict argument mapping each dependency's name to
its result. A process job's fn must be picklable (a top-level function or a functools.partial
of one) and the deps it declares must be picklable too.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
import logging
import multiprocessing as mp
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Job:
    """One node in the job graph.

    fn(inputs) is invoked with ``inputs = {dep_name: dep_result}`` once every dep has
    completed. in_process runs it in the spawn process pool.
    """

    name: str
    deps: tuple[str, ...]
    fn: Callable[[dict], Any]
    in_process: bool = False


def run_jobs(
    jobs: list[Job], *, max_processes: int = 4, timings: dict | None = None
) -> dict[str, Any]:
    """Run a job DAG to completion, overlapping everything that can overlap.

    Returns {job_name: result}. Validates that deps exist; raises if the graph deadlocks
    (a cycle leaves jobs unrun) and propagates the first job exception (the pools shut
    down on the way out of the ``with`` block). If ``timings`` is given, it is filled with
    {job_name: wall_clock_seconds} (submit→completion, in this process — captures thread and
    spawned-process jobs alike; note overlapping jobs' times can sum to more than the wall total).
    """
    by_name = {j.name: j for j in jobs}
    if len(by_name) != len(jobs):
        raise ValueError("duplicate job names in graph")
    for j in jobs:
        for d in j.deps:
            if d not in by_name:
                raise ValueError(f"job {j.name!r} depends on unknown job {d!r}")

    results: dict[str, Any] = {}
    submitted: set[str] = set()
    started: dict[str, float] = {}
    n_proc_jobs = sum(1 for j in jobs if j.in_process)
    proc_cm = (
        ProcessPoolExecutor(
            max_workers=max(1, min(max_processes, n_proc_jobs)),
            mp_context=mp.get_context("spawn"),
        )
        if n_proc_jobs
        else nullcontext(None)
    )

    with ThreadPoolExecutor(thread_name_prefix="eval-job") as threads, proc_cm as procs:
        futs: dict = {}

        def submit_ready() -> None:
            for name, job in by_name.items():
                if name in submitted or not all(d in results for d in job.deps):
                    continue
                inputs = {d: results[d] for d in job.deps}
                ex = procs if job.in_process else threads
                started[name] = time.perf_counter()
                futs[ex.submit(job.fn, inputs)] = name
                submitted.add(name)

        submit_ready()
        while futs:
            done, _ = wait(set(futs), return_when=FIRST_COMPLETED)
            for fut in done:
                name = futs.pop(fut)
                results[name] = fut.result()  # re-raises any job exception here
                if timings is not None:
                    timings[name] = time.perf_counter() - started[name]
            submit_ready()

    unrun = set(by_name) - results.keys()
    if unrun:
        raise RuntimeError(f"jobs never ran (dependency cycle?): {sorted(unrun)}")
    return results
