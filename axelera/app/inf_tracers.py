#!/usr/bin/env python
# Copyright Axelera AI, 2024
from __future__ import annotations

import abc
import argparse
import collections
import dataclasses
import fcntl
import itertools
import logging
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

from . import logging_utils

if TYPE_CHECKING:
    from . import config, display
    from .pipe import network

LOG = logging_utils.getLogger(__name__)
MAX_LOG_SIZE = 1e6

supported_tracers = []


def fps_from_latency(us: float):
    return 1e6 / us if us > 0 else 0


def fps_from_latency_s(s: float):
    return 1 / s if s > 0 else 0


@dataclasses.dataclass
class TraceMetric:
    key: str
    '''This is a unique key per metric type to identify the metric when used as meta data.'''

    title: str
    '''A title for this metric, e.g. 'Core Temp'.'''

    value: float
    '''The current value of the metric.'''

    max: float = 1.0
    '''The upper scale to use in a speedometer.

    This does not usually represent the maximum value yet measured.
    '''

    unit: str = ''
    '''Display unit, for example '°C' or 'fps'.'''

    def draw(self, draw: display.Draw, **kwargs):
        draw.draw_speedometer(self)

    def visit(self, visitor, *args, **kwargs):
        visitor(self, *args, **kwargs)


def _add_to_environ(environ, key, value):
    existing = environ.get(key, '')
    values = existing.split(',') if existing else []
    values.append(value)
    environ[key] = ','.join(values)


class Tracer(abc.ABC):
    key = '__dummy_tracer__'
    title = 'Dummy'

    def start_monitoring(self):
        pass

    def stop_monitoring(self):
        pass

    @abc.abstractmethod
    def get_metrics(self) -> Sequence[TraceMetric]:
        return []

    def update(self, frame_result):
        pass

    def initialize_models(
        self, model_infos: network.ModelInfos, metis: config.Metis, core_clocks: dict[int, int]
    ) -> None:
        del model_infos
        del metis
        del core_clocks


def _run(args: List[str]):
    LOG.trace("$ %s", ' '.join(args))
    p = subprocess.run(args, capture_output=True, universal_newlines=True)
    LOG.trace("> %s retcode=%d stdout=%s stderr=%s", args[0], p.returncode, p.stdout, p.stderr)
    return p


def _first_and_only_model_info(model_infos, tracer):
    return model_infos.singular_model(f"Only one model is supported with {tracer} tracer")


def _determine_fps_multiplier(requested_cores, model_infos, metis):
    # If the model batch is > 1 then it must support batching, and we will ignore the
    # user requested cores.
    model_info = _first_and_only_model_info(model_infos, 'aipu')
    ncores = model_infos.determine_execution_cores(model_info.name, requested_cores, metis)
    return ncores


class _ThreadedTracerAdapter(Tracer):
    '''Adapts a tracer to run in a separate thread on a period basis.

    The metrics returned are cached, and returned on request.
    '''

    def __init__(self, tracer: Tracer, period=1.0):
        self._condition = threading.Condition()
        self._running = False
        self._period = period
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._last_metrics = []
        self._tracer = tracer

    @property
    def key(self):
        return self._tracer.key

    @property
    def title(self):
        return self._tracer.title

    def initialize_models(
        self, model_infos: network.ModelInfos, metis: config.Metis, core_clocks: dict[int, int]
    ) -> None:
        self._tracer.initialize_models(model_infos, metis, core_clocks)

    def start_monitoring(self):
        if not self._running:
            self._tracer.start_monitoring()
            self._running = True
            self._thread.start()

    def stop_monitoring(self):
        with self._condition:
            self._running = False
            self._condition.notify()
        self._thread.join()
        self._tracer.stop_monitoring()
        self._last_metrics = self._tracer.get_metrics()

    def _monitor(self):
        while 1:
            new_metrics = self._tracer.get_metrics()
            with self._condition:
                self._last_metrics = new_metrics or self._last_metrics
                if self._condition.wait_for(lambda: not self._running, self._period):
                    break

    def get_metrics(self):
        with self._condition:
            return self._last_metrics


def _drop_old_chunks(chunks, now, max_age=5):
    oldest = now - max_age
    if len(chunks) > 1:
        chunks[:] = list(itertools.dropwhile(lambda x: x[0] < oldest, chunks))


class _TritonTrace:
    def __init__(
        self,
        reset_args: list[str],
        initial_args: list[str],
        continuous: list[str],
        restore_args: list[str],
    ):
        if reset_args:
            _run(["triton_trace"] + reset_args)
        _run(["triton_trace"] + initial_args)
        cmd = ["stdbuf", "-oL", "triton_trace", "--clear-buffer"] + continuous
        LOG.trace("Running %s as subprocess to collect log", ' '.join(cmd))
        self._p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._restore = lambda: _run(["triton_trace"] + restore_args)
        stdout_fd = self._p.stdout.fileno()
        flags = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
        fcntl.fcntl(stdout_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def read(self) -> str:
        log = []
        while text_bin := self._p.stdout.read():
            log.append(text_bin.decode('utf-8'))
        return ''.join(log)

    def prepare_to_stop(self):
        LOG.trace("SIGTERM triton_trace process")
        self._p.terminate()

    def stop(self) -> str:
        try:
            LOG.trace(f"Waiting for triton_trace process")
            try:
                out, _ = self._p.communicate(timeout := 2)

            except subprocess.TimeoutExpired as e:
                LOG.warning(f"Failed to stop triton_trace process after {timeout}s, killing it")
                self._p.kill()
                out = e.stdout
            return out.decode('utf-8') if isinstance(out, bytes) else (out or '')
        finally:
            self._restore()
            self._p = None


class AipuTracer(Tracer):
    '''A class to monitor AIPU inference server using triton_trace.'''

    key = '__device_fps__'
    title = 'Metis'

    _PATTERN = re.compile(r"Megakernel  ?took (\d+) cycles")

    def __init__(self, aipu_cores: int):
        super().__init__()
        self._tritons: dict[int, _TritonTrace] = {}
        self._requested_cores = list(range(aipu_cores))
        from .config import DEFAULT_CORE_CLOCK

        self._clock_durations_in_us = {n: 1 / DEFAULT_CORE_CLOCK for n in self._requested_cores}
        self._fps_multiplier = 1
        self._all_chunks = collections.defaultdict(list)
        self._max = 0.0
        self._max_age = 5
        self._next_drop = time.time() + 5.0

    def initialize_models(
        self, model_infos: network.ModelInfos, metis: config.Metis, core_clocks: dict[int, int]
    ) -> None:
        self._clock_durations_in_us = {n: 1.0 / c for n, c in core_clocks.items()}
        self._fps_multiplier = _determine_fps_multiplier(
            len(self._requested_cores), model_infos, metis
        )
        LOG.debug(f"Device FPS multiplier: {self._fps_multiplier}")

    def start_monitoring(self):
        for i in self._requested_cores:
            self._tritons[i] = _TritonTrace(
                [], ["--alog-level", f'{i}:wrn'], ["--alog", f"{i}"], ["--alog-level", f'{i}:err']
            )

    def _read_trace_output(self) -> dict[str, str]:
        return {coren: p.read() for coren, p in self._tritons.items()}

    def _process_trace_output(self, outputs: dict[str, str]):
        now = time.time()
        for coren, log in outputs.items():
            if matches := self._PATTERN.findall(log):
                cycles = np.array([int(m) for m in matches], dtype=np.int32)
                self._all_chunks[coren].append((now, cycles))
        if now > self._next_drop:
            for chunks in self._all_chunks.values():
                _drop_old_chunks(chunks, now, self._max_age)
            self._next_drop = now + 1

    def _make_metric(self, fps, coren=None):
        key_s = f'-c{coren}' if coren is not None else ''
        title_s = f' Core {coren}' if coren is not None else ''
        return TraceMetric(
            f'{self.key}{key_s}', f'{self.title}{title_s}', fps, max(self._max * 1.05, 1.0), 'fps'
        )

    def get_metrics(self):
        '''Return FPS of metis.'''
        if self._tritons:
            self._process_trace_output(self._read_trace_output())
        metrics = []
        if LOG.isEnabledFor(logging.DEBUG):
            # show per-core. TODO should do per-model, not per-core
            for coren, now_chunks in self._all_chunks.items():
                if now_chunks:
                    nows, chunks = zip(*now_chunks)
                    cycles = np.concatenate(chunks)
                    latency = self._clock_durations_in_us[coren]
                    median = np.median(cycles) * latency
                    fps = fps_from_latency(median)
                    if LOG.isEnabledFor(logging_utils.TRACE):
                        minl = np.min(cycles) * latency
                        mean = np.mean(cycles) * latency
                        maxl = np.max(cycles) * latency
                        LOG.trace(
                            f"device({coren}) latency(ms): {median:1f} nsamples={len(cycles)} min={minl:.1f} max={maxl:.1f} mean_l={mean:.1f}",
                        )
                    metrics.append(self._make_metric(fps, coren))

        flattened = []
        for coren, now_chunks in self._all_chunks.items():
            if now_chunks:
                nows, chunks = zip(*now_chunks)
                cycles = np.concatenate(chunks)
                cycles = cycles * self._clock_durations_in_us[coren]
                flattened.append(cycles)
        if flattened:
            flattened = np.concatenate(flattened)
            median = np.median(cycles)
            fps = fps_from_latency(median) * self._fps_multiplier
            self._max = max(self._max, fps)
            metrics.append(self._make_metric(fps))
        return metrics

    def stop_monitoring(self):
        for p in self._tritons.values():
            p.prepare_to_stop()
        outputs = {coren: p.stop() for coren, p in self._tritons.items()}
        # don't forget to get final samples
        self._process_trace_output(outputs)
        self._tritons.clear()


class CoreTempTracer(Tracer):
    '''A class to monitor AIPU inference server using triton_trace.'''

    key = '__core_temp__'
    title = 'Core Temp'

    _PATTERN = re.compile(r"^\[[^\]]+\] .*core_temps=\[([\d,]+)\]\s*$", flags=re.M)

    def __init__(self):
        super().__init__()
        self._triton: _TritonTrace | None = None
        self._last = [0.0] * 5
        self._max = max(self._last)

    def start_monitoring(self):
        self._triton = _TritonTrace(
            ["--slog-level", 'err'],
            ["--slog-level", f'inf:collector'],
            ["--slog"],
            ["--slog-level", 'err'],
        )

    def _process_trace_output(self, log: str):
        if matches := self._PATTERN.findall(log):
            self._last = [int(x) for x in matches[-1].split(',')]

    def get_metrics(self):
        '''Return FPS of metis.'''
        if self._triton:
            self._process_trace_output(self._triton.read())
        value = float(max(self._last))
        self._max = max(value, self._max)
        return [TraceMetric(self.key, self.title, value, self._max, '°C')]

    def stop_monitoring(self):
        LOG.trace("SIGTERM triton_trace slog process")
        self._triton.prepare_to_stop()
        output = self._triton.stop()
        self._triton = None
        self._process_trace_output(output)


_axr_mappings = {
    'inp': 'Memcpy-in',
    'krn': 'Kernel',
    'out': 'Memcpy-out',
    'tot': 'Host',
    'pat': 'Patch Parameters',
}
_axr_entry = re.compile(r"\b(\w+):([\da-f]+)\b")


def parse_axr_stats(logs: str) -> Dict[str, float]:
    stats = collections.defaultdict(list)
    for line in logs.splitlines():
        for key, value in _axr_entry.findall(line):
            if key not in ('idx', 'beg'):
                stats[key].append(int(value, 16) / 1e9)
    return {_axr_mappings.get(k, k): np.array(v) for k, v in stats.items()}


class HostTracer(Tracer):
    '''A class to monitor host timing using timings at level zero execute level.'''

    key = '__host_fps__'
    title = 'Host'

    def __init__(self, aipu_cores: int):
        super().__init__()
        self._temp_file = tempfile.NamedTemporaryFile(delete=True)
        self._file_path = Path(self._temp_file.name)
        self._all_chunks = collections.defaultdict(list)
        self._last_read_position = 0
        self._requested_cores = aipu_cores
        self._fps_multiplier = 1
        self._max_stats = {'Host': 0.0}
        self._running = False
        self._max_age = 5
        self._next_drop = time.time() + 5.0
        LOG.debug(f"HostTracer will write to {self._file_path}")

    def initialize_models(
        self, model_infos: network.ModelInfos, metis: config.Metis, core_clocks: dict[int, int]
    ) -> None:
        del core_clocks
        os.environ['AXE_LZ_PERF_LOG_PATH'] = str(self._file_path)
        _add_to_environ(os.environ, 'AXE_PROFILING_CONFIG', 'HOST_TEMPLATE')
        self._fps_multiplier = _determine_fps_multiplier(self._requested_cores, model_infos, metis)
        LOG.debug(f"Host FPS multiplier: {self._fps_multiplier}")

    def start_monitoring(self):
        self._running = True

    def stop_monitoring(self):
        self._running = False

    def _read_logs(self):
        if not self._file_path.exists():
            LOG.debug(f"{self._file_path} not found")
            return ''
        with self._file_path.open('r+') as f:
            f.seek(self._last_read_position)
            logs = f.read()
            self._last_read_position = f.tell()

            if f.tell() > MAX_LOG_SIZE:
                f.seek(0)
                f.truncate(0)  # Empty the original log file
                self._last_read_position = 0  # Reset reading position
        return logs

    def _process_logs(self, logs):
        # TODO we ought to know if using tvm or axruntime but for now try both
        now = time.time()
        if stats := parse_axr_stats(logs):
            for s, v in stats.items():
                self._all_chunks[s].append((now, v))
            if LOG.isEnabledFor(logging_utils.TRACE):
                stats = ' '.join(
                    f'{k}:{np.median(v)*1000000.0:.1f}us ({v.shape})' for k, v in stats.items()
                )
                LOG.trace(f"Host latencies: {stats}")
        if now > self._next_drop:
            for chunks in self._all_chunks.values():
                _drop_old_chunks(chunks, now, self._max_age)
            self._next_drop = now + 1

    def get_metrics(self):
        if self._running:
            self._process_logs(self._read_logs())
        metrics = []
        for k, sample_chunks in self._all_chunks.items():
            if (k == 'Host' or LOG.isEnabledFor(logging.DEBUG)) and sample_chunks:
                cycles = np.concatenate([chunk for _, chunk in sample_chunks])
                v = np.median(cycles)
                LOG.trace(f"Host {k} latency: {v*1e6:.1f}us ({len(sample_chunks[-1])} samples)")
                if k == 'Host':
                    m = fps_from_latency_s(v) * self._fps_multiplier
                else:
                    m = v * 1.0e6  # show as microseconds latency, not fps
                maxm = self._max_stats[k] = max(self._max_stats.get(k, m), m)
                maxm = max(maxm * 1.05, 1.0)  # the 1.05 keeps the needle to the far right.
                unit = 'fps' if k == 'Host' else 'us'
                metrics.append(TraceMetric(k.lower(), k, m, maxm, unit))
        return metrics


class CpuTracer(Tracer):
    '''A class to monitor CPU usage.'''

    key = '__cpu_usage__'
    title = 'CPU %'

    def __init__(self):
        super().__init__()
        self._rolling = 0.0
        self._smoothing = 0.3

    def get_metrics(self) -> list[TraceMetric]:
        pid = os.getpid()
        ncores = max(multiprocessing.cpu_count(), 1)
        expr = re.compile(r"^\s*\d+(?:\.\d+)?$", flags=re.M)
        p = subprocess.run(f"ps -p {pid} -o %cpu", shell=True, capture_output=True, text=True)
        if m := expr.search(p.stdout):
            total = float(m.group(0))
            percore = float(total) / ncores
            LOG.trace(f"CPU Usage is {total:.1f} on {ncores} cores == {percore:.1f}%")
            if self._rolling == 0.0:
                self._rolling = percore
            else:
                self._rolling = self._smoothing * percore + (1 - self._smoothing) * self._rolling
            return [TraceMetric(self.key, self.title, self._rolling, 100.0, '%')]
        else:
            LOG.debug(f"CPU Usage monitoring failed to parse : {p.stdout!r}")
            return []


class End2EndTracer(Tracer):
    key = '__end_to_end_fps__'
    title = 'End-to-end'

    def __init__(self):
        self._last = 0.0
        self._last_chunk = 0.0
        self._this_chunk = []
        self._duration_in_us = 1e6
        self._interval = 1.0
        MAX_AGE = 5
        self._chunks = collections.deque(maxlen=int(MAX_AGE / self._interval))
        self._fps = 0.0
        self._max = 30.0

    def update(self, frame_result):
        if self._last == 0.0:
            self._last_chunk = self._last = time.time()
            return
        now = time.time()
        self._this_chunk.append(now - self._last)
        self._last = now
        if (now - self._last_chunk) >= self._interval:
            self._chunks.append(np.array(self._this_chunk, dtype=float))
            self._this_chunk.clear()
            self._last_chunk = now
            samples = np.concatenate(self._chunks)
            mean = np.mean(samples) * self._duration_in_us
            self._fps = fps_from_latency(mean)
            if LOG.isEnabledFor(logging_utils.TRACE):
                median = np.median(samples) * self._duration_in_us
                minl = np.min(samples) * self._duration_in_us
                maxl = np.max(samples) * self._duration_in_us
                LOG.trace(
                    f"e2e latency(ms): {median:1f} nsamples={len(samples)} min={minl:.1f} max={maxl:.1f} mean_l={mean:.1f}",
                )
            self._max = max(self._max, self._fps * 1.05)

    def get_metrics(self):
        return [TraceMetric(self.key, self.title, self._fps, self._max, 'fps')]


class Statistics:
    def __init__(self, key, title, max_bin_value, num_bins=200):
        self.key = key
        self.title = title
        self.max_bin_value = max_bin_value
        self.num_bins = num_bins
        self.sample_count = 0
        self.min, self.max, self.mean, self._sumvar = float('inf'), float('-inf'), 0.0, 0.0
        self.bins = np.zeros((self.num_bins,), np.int32)

    @property
    def bin_size(self):
        return self.max_bin_value / self.num_bins

    @property
    def stddev(self):
        return math.sqrt(self._sumvar / self.sample_count) if self.sample_count else 0.0

    def bin_for_value(self, value):
        if value < 0:
            return 0
        return max(0, min(int(value / self.bin_size), self.num_bins - 1))

    def update(self, value):
        self.sample_count += 1
        self.bins[self.bin_for_value(value)] += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        # Welford's method to calculate mean/std dev in a single pass
        delta = value - self.mean
        self.mean += delta / self.sample_count
        self._sumvar += (value - self.mean) * delta

    def draw(self, draw: display.Draw):
        draw.draw_statistics(self)


class StreamStatistics:
    def __init__(self):
        self._last_time = time.time()
        self.latency = Statistics('__stream_latency__', 'Latency', 10000.0)
        self.interval = Statistics('__stream_intervals__', 'Intervals', 1000.0)

    def update(self, latency):
        self.latency.update(latency * 1000)
        now = time.time()
        self.interval.update((now - self._last_time) * 1000)
        self._last_time = now

    def __repr__(self):
        # This is not directly used anywhere, but it's useful for debugging
        cells = []
        for prefix, s in [('l', self.latency), ('i', self.interval)]:
            cells.append(f'{prefix}-min:{int(s.min):<5d}')
            cells.append(f'{prefix}-mean:{int(s.mean):<5d}')
            cells.append(f'{prefix}-max:{int(s.max):<5d}')
            cells.append(f'{prefix}-stddev:{int(s.stddev):<5d}')
        return f'frames:{self.latency.sample_count:<5d} ' + ' '.join(cells)


class StreamTiming(Tracer):
    key = '__stream_timing__'
    title = '__stream_statistics__'

    def __init__(self):
        self.timing = collections.defaultdict(StreamStatistics)
        self.total = 0

    def update(self, frame_result):
        stream_id = frame_result.stream_id
        self.total += 1
        t = self.timing[stream_id]
        latency = frame_result.sink_timestamp - frame_result.src_timestamp
        t.update(latency)

    def get_metrics(self):
        metrics = []
        for sid, t in self.timing.items():
            i = t.interval
            l = t.latency
            d = f'#{sid} ' if len(self.timing) > 1 else ''
            metrics.append(TraceMetric(i.key, f'{d}Jitter', i.stddev, i.max, 'ms'))
            metrics.append(TraceMetric(l.key, f'{d}{l.title}', l.mean, l.max, 'ms'))
        if len(self.timing) > 1:
            mean_latency = sum(t.latency.mean for t in self.timing.values()) / len(self.timing)
            mean_jitter = sum(t.interval.stddev for t in self.timing.values()) / len(self.timing)
            metrics.append(TraceMetric(i.key, 'Jitter', mean_jitter, i.max, 'ms'))
            metrics.append(TraceMetric(l.key, l.title, mean_latency, l.max, 'ms'))
        return metrics


_key = lambda x: x.strip('_')

supported_tracers = {
    _key(AipuTracer.key): lambda aipu_cores: _ThreadedTracerAdapter(
        AipuTracer(aipu_cores), period=0.3
    ),
    _key(CoreTempTracer.key): lambda aipu_cores: _ThreadedTracerAdapter(
        CoreTempTracer(), period=1.0
    ),
    _key(HostTracer.key): lambda aipu_cores: _ThreadedTracerAdapter(
        HostTracer(aipu_cores), period=0.3
    ),
    _key(CpuTracer.key): lambda aipu_cores: _ThreadedTracerAdapter(CpuTracer(), period=3.0),
    _key(End2EndTracer.key): lambda aipu_cores: End2EndTracer(),
    _key(StreamTiming.key): lambda aipu_cores: StreamTiming(),
}


def create_tracers(*requested: tuple[str, ...], _aipu_cores: int = 4) -> List[Tracer]:
    tracers = []
    # disable tracing by default, tracers below will enable it if necessary
    os.environ.setdefault('AXE_PROFILING_CONFIG', '')
    for tracer in requested:
        try:
            ctor = supported_tracers[tracer]
        except KeyError:
            LOG.warning(
                f"Unsupported tracer: {tracer}, valid tracers are: {' '.join(supported_tracers.keys())}"
            )
        else:
            tracers.append(ctor(_aipu_cores))
    return tracers


def create_tracers_from_args(args: argparse.Namespace) -> List[Tracer]:
    requested = []
    # disable tracing by default, tracers below will enable it if necessary
    if args.show_temp:
        requested.append(_key(CoreTempTracer.key))
    if args.show_stats or args.show_device_fps:
        requested.append(_key(AipuTracer.key))
    if args.show_stats or args.show_host_fps:
        requested.append(_key(HostTracer.key))
    if args.show_cpu_usage:
        requested.append(_key(CpuTracer.key))
    if args.show_system_fps:
        requested.append(_key(End2EndTracer.key))
    if args.show_stream_timing:
        requested.append(_key(StreamTiming.key))
    return create_tracers(*requested, _aipu_cores=args.aipu_cores)
