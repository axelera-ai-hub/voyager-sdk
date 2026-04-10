#!/usr/bin/env python
# Copyright Axelera AI, 2025
from __future__ import annotations

import abc
import argparse
import collections
import dataclasses
import fcntl
import itertools
import logging
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

from . import config, device_manager, logging_utils, utils

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

    max_scale_value: float = 1.0
    '''The upper scale to use in a speedometer.

    This does not usually represent the maximum value yet measured.
    '''

    unit: str = ''
    '''Display unit, for example '°C' or 'fps'.'''

    def draw(self, draw: display.Draw):
        draw.draw_speedometer(self)

    def visit(self, visitor, *args, **kwargs):
        visitor(self, *args, **kwargs)

    def text_report(self) -> str:
        '''Return a string representation of the metric for text reports.'''
        return f"{self.value:.1f}{self.unit}"


@dataclasses.dataclass
class TraceMetricWithStats(TraceMetric):
    '''A metric that has min/max/mean and stddev statistics.

    Note that the `value` property returns the median value, which is the primary value
    for compatibility with the TraceMetric class.
    '''

    mean: float = 0.0
    '''The mean value of the samples collected.'''

    min: float = 0.0
    max: float = 0.0
    '''The minimum and maximum values of the samples collected.'''

    std: float = 0.0
    '''The standard deviation of the samples collected.'''

    @property
    def median(self) -> float:
        '''Return the median value of the samples collected.

        This is an alias for `value` and is provided for clarity.
        '''
        return self.value

    # def draw(self, draw: display.Draw):
    #  TODO render other stats in a speedometer somehow
    #     stats_text = f"{self.median:.1f} (min:{self.min_val:.1f} max:{self.max_val:.1f} σ:{self.stddev:.1f}){self.unit}"
    #     draw.text((10, 10), f"{self.title}: {stats_text}", (255, 255, 255, 255))

    def text_report(self) -> str:
        '''Return a string representation of the metric for text reports.'''
        return f"{self.median:.1f}{self.unit} (min:{self.min:.1f} max:{self.max:.1f} σ:{self.std:.1f} x̄:{self.mean:.1f}){self.unit}"


def _add_to_environ(environ, key, value):
    existing = environ.get(key, '')
    values = existing.split(',') if existing else []
    values.append(value)
    environ[key] = ','.join(values)


class Tracer(abc.ABC):
    '''Abstract base class for all tracers.

    Tracers can be used to monitor the inference pipeline, and collect metrics
    about how it is performing.

    For example the following tracer collects metrics about the number of detections. Adding
    this to the list of tracers passed to create_inference_stream will result in a speedometer
    showing the number of detections per frame.

      class DetectionsTracer(inf_tracers.Tracer):
          key = '__detections__'
          title = 'Detections'
          def __init__(self):
              self._v, self._max = 0, 0
          def update(self, frame_result):
              self._v = len(frame_result.detections)
              self._max = max(self._max, self._v)
          def get_metrics(self):
              return [inf_tracers.TraceMetric(self.key, self.title, self._v, self._max, ' dets')]
    '''

    key = '__dummy_tracer__'
    '''A unique key for this tracer, used to identify it in the application.

    It should begin and end with double underline (_), and should not contain
    spaces or special characters.
    '''

    title = 'Dummy'
    '''A title for this tracer, used to display it in the application.'''

    def start_monitoring(self):
        '''This is called just before inference starts.

        This is used in some tracers to start a subprocess, thread, or similar.
        '''
        pass

    def stop_monitoring(self):
        '''This is called just after inference ends.

        This is used to close any resources, threads, or subprocesses that were
        started in start_monitoring.
        '''
        pass

    @abc.abstractmethod
    def get_metrics(self) -> Sequence[TraceMetric]:
        '''Return a list of TraceMetric objects to be shown in speedometers.

        This is not called on every inference, but rather on a periodic basis when the metrics
        should be drawn, so it is important not to do any time based calculations in this function.
        '''
        return []

    def update(self, frame_result):
        '''Called whenever an inference result is returned from the inference stream.

        frame_result is an instance of FrameResult.
        '''
        pass

    def initialize_models(
        self,
        model_infos: network.ModelInfos,
        metis: config.Metis,
        core_clocks: dict[int, int],
        devices: list[str],
    ) -> None:
        '''Called before inference with information about the loaded models, and connected devices.

        This might be useful to determine what sort of metrics to collect.
        '''
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


def _determine_fps_multiplier(requested_cores, num_devices, model_infos, metis):
    # If the model batch is > 1 then it must support batching, and we will ignore the
    # user requested cores.
    model_info = _first_and_only_model_info(model_infos, 'aipu')
    ncores = model_infos.determine_execution_cores(model_info.name, requested_cores, metis)
    return ncores * num_devices


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
        self,
        model_infos: network.ModelInfos,
        metis: config.Metis,
        core_clocks: dict[int, int],
        devices: list[str],
    ) -> None:
        self._tracer.initialize_models(model_infos, metis, core_clocks, devices)

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
        device: str,
        reset_args: list[str],
        initial_args: list[str],
        continuous: list[str],
        restore_args: list[str],
    ):
        tt = ['axtrace'] + (['--device', device] if device else [])
        if reset_args:
            _run(tt + reset_args)
        _run(tt + initial_args)
        cmd = ["stdbuf", "-oL"] + tt + ["--clear-buffer"] + continuous
        LOG.trace("Running %s as subprocess to collect log", ' '.join(cmd))
        self._p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self._restore = lambda: _run(tt + restore_args)
        stdout_fd = self._p.stdout.fileno()
        flags = fcntl.fcntl(stdout_fd, fcntl.F_GETFL)
        fcntl.fcntl(stdout_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def read(self) -> str:
        log = []
        while text_bin := self._p.stdout.read():
            log.append(text_bin.decode('utf-8'))
        return ''.join(log)

    def prepare_to_stop(self):
        LOG.trace("SIGTERM axtrace process")
        self._p.terminate()

    def stop(self) -> str:
        try:
            LOG.trace("Waiting for axtrace process")
            try:
                out, _ = self._p.communicate(timeout := 2)

            except subprocess.TimeoutExpired as e:
                LOG.warning(f"Failed to stop axtrace process after {timeout}s, killing it")
                self._p.kill()
                out = e.stdout
            return out.decode('utf-8') if isinstance(out, bytes) else (out or '')
        finally:
            self._restore()
            self._p = None


class AipuTracer(Tracer):
    '''A class to monitor AIPU inference server using axtrace.'''

    key = '__device_fps__'
    title = 'Metis'

    _PATTERN = re.compile(r"Megakernel  ?took (\d+) cycles")

    def __init__(self, aipu_cores: int):
        super().__init__()
        self._tritons: dict[tuple[int, int], _TritonTrace] = {}
        self._requested_cores = list(range(aipu_cores))
        from .config import DEFAULT_CORE_CLOCK

        self._clock_durations_in_us = {n: 1 / DEFAULT_CORE_CLOCK for n in self._requested_cores}
        self._fps_multiplier = 1
        self._all_chunks = collections.defaultdict(list)
        self._max = 0.0
        self._max_age = 5
        self._next_drop = time.time() + 5.0

    def initialize_models(
        self,
        model_infos: network.ModelInfos,
        metis: config.Metis,
        core_clocks: dict[int, int],
        devices: list[str],
    ) -> None:
        self._devices = devices
        self._clock_durations_in_us = {n: 1.0 / c for n, c in core_clocks.items()}
        self._fps_multiplier = _determine_fps_multiplier(
            len(self._requested_cores), len(self._devices), model_infos, metis
        )
        LOG.debug(f"Device FPS multiplier: {self._fps_multiplier}")

    def start_monitoring(self):
        for dn, d in enumerate(self._devices):
            for i in self._requested_cores:
                self._tritons[(dn, i)] = _TritonTrace(
                    d,
                    [],
                    ["--alog-level", f'{i}:wrn'],
                    ["--alog", f"{i}"],
                    ["--alog-level", f'{i}:err'],
                )

    def _read_trace_output(self) -> dict[str, str]:
        return {devn_coren: p.read() for devn_coren, p in self._tritons.items()}

    def _process_trace_output(self, outputs: dict[tuple[int, int], str]):
        now = time.time()
        for (devn, coren), log in outputs.items():
            if matches := self._PATTERN.findall(log):
                cycles = np.array([int(m) for m in matches], dtype=np.int32)
                self._all_chunks[(devn, coren)].append((now, cycles))
        if now > self._next_drop:
            for chunks in self._all_chunks.values():
                _drop_old_chunks(chunks, now, self._max_age)
            self._next_drop = now + 1

    def _make_metric(self, fps, devn=None, coren=None):
        key_s = f'__device_d{devn}_{coren}_fps__' if coren is not None else self.key
        title_s = f' Core d{devn}.{coren}' if coren is not None else ''
        return TraceMetric(key_s, f'{self.title}{title_s}', fps, max(self._max * 1.05, 1.0), 'fps')

    def get_metrics(self):
        '''Return FPS of metis.'''
        if self._tritons:
            self._process_trace_output(self._read_trace_output())
        metrics = []
        if LOG.isEnabledFor(logging.DEBUG):
            # show per-core. TODO should do per-model, not per-core
            for (devn, coren), now_chunks in self._all_chunks.items():
                if now_chunks:
                    nows, chunks = zip(*now_chunks)
                    cycles = np.concatenate(chunks)
                    latency = self._clock_durations_in_us[coren]
                    median = np.median(cycles) * latency
                    fps = fps_from_latency(median)
                    metrics.append(self._make_metric(fps, devn, coren))

        flattened = []
        for (devn, coren), now_chunks in self._all_chunks.items():
            if now_chunks:
                nows, chunks = zip(*now_chunks)
                cycles = np.concatenate(chunks)
                cycles = cycles * self._clock_durations_in_us[coren]
                flattened.append(cycles)
        if flattened:
            flattened = np.concatenate(flattened)
            median = np.median(flattened)
            fps = fps_from_latency(median) * self._fps_multiplier
            self._max = max(self._max, fps)
            metrics.append(self._make_metric(fps))
        return metrics

    def stop_monitoring(self):
        for p in self._tritons.values():
            p.prepare_to_stop()
        outputs = {devn_coren: p.stop() for devn_coren, p in self._tritons.items()}
        # don't forget to get final samples
        self._process_trace_output(outputs)
        self._tritons.clear()


class CoreTempTracer(Tracer):
    '''A class to monitor AIPU inference server using axtrace.'''

    key = '__core_temp__'
    title = 'Core Temp'

    _PATTERN = re.compile(r"^\[[^\]]+\] .*core_temps=\[([\d,]+)\]\s*$", flags=re.M)

    def __init__(self):
        super().__init__()
        self._tritons: list[_TritonTrace] = []
        self._last: dict[int, list[float]] = {}
        self._max = 200.0

    def initialize_models(
        self,
        model_infos: network.ModelInfos,
        metis: config.Metis,
        core_clocks: dict[int, int],
        devices: list[str],
    ) -> None:
        del model_infos
        del metis
        del core_clocks
        self._devices = devices

    def start_monitoring(self):
        self._tritons = [
            _TritonTrace(
                d,
                ["--slog-level", 'err'],
                ["--slog-level", 'inf:collector'],
                ["--slog"],
                ["--slog-level", 'err'],
            )
            for d in self._devices
        ]

    def _process_trace_output(self, devicen, log: str):
        if matches := self._PATTERN.findall(log):
            self._last[devicen] = [int(x) for x in matches[-1].split(',')]

    def get_metrics(self):
        '''Return FPS of metis.'''
        for n, t in enumerate(self._tritons):
            self._process_trace_output(n, t.read())
        if not self._last:
            max_value = 0.0
        else:
            max_value = float(max(max(x) for x in self._last.values()))
            self._max = max(max_value, self._max)
        m = [TraceMetric(self.key, self.title, max_value, self._max, '°C')]
        if LOG.isEnabledFor(logging_utils.DEBUG):
            for d, coresvals in self._last.items():
                for c, v in enumerate(coresvals):
                    key = f"__core_d{d}_{c}_temp__"
                    title = f"Core d{d}.{c} Temp"
                    m.append(TraceMetric(key, title, v, self._max, '°C'))
        return m

    def stop_monitoring(self):
        LOG.trace("SIGTERM axtrace slog process")
        for t in self._tritons:
            t.prepare_to_stop()
        outputs = [t.stop() for t in self._tritons]
        self._tritons = []
        # don't forget to get final samples
        for n, o in enumerate(outputs):
            self._process_trace_output(n, o)


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
        self,
        model_infos: network.ModelInfos,
        metis: config.Metis,
        core_clocks: dict[int, int],
        devices: list[str],
    ) -> None:
        del core_clocks
        os.environ['AXE_LZ_PERF_LOG_PATH'] = str(self._file_path)
        _add_to_environ(os.environ, 'AXE_PROFILING_CONFIG', 'HOST_TEMPLATE')
        self._fps_multiplier = _determine_fps_multiplier(
            self._requested_cores, len(devices), model_infos, metis
        )
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


class End2EndInfRateTracer(End2EndTracer):
    key = '__end_to_end_infs__'
    title = 'Inf. Rate'

    def __init__(self):
        super().__init__()
        self._infs = self._fps
        self._max_infs = self._max

    def update(self, frame_result):
        super().update(frame_result)
        self._infs = self._fps * frame_result.inferences
        self._max_infs = max(self._max_infs, self._infs * 1.05)

    def get_metrics(self):
        return [TraceMetric(self.key, self.title, self._infs, self._max_infs, 'inf/s')]


class Statistics:
    def __init__(self, max_samples: int = 10000, dtype=np.float32):
        self.sample_count = 0
        self._max_samples = max_samples
        self._dtype = dtype
        self._samples = np.zeros((self._max_samples,), self._dtype)
        self._verbose = LOG.isEnabledFor(logging_utils.DEBUG)

    @staticmethod
    def _npproperty(npfunc, default=0):
        def getter(self) -> float:
            if self.sample_count == 0:
                return default
            return float(npfunc(self._samples[0 : min(self.sample_count, self._max_samples)]))

        return property(getter)

    min = _npproperty(np.min, default=float('inf'))
    max = _npproperty(np.max, default=float('-inf'))
    mean = _npproperty(np.mean, default=0.0)
    median = _npproperty(np.median, default=0.0)
    std = _npproperty(np.std, default=0.0)

    def update(self, value):
        self._samples[self.sample_count % self._max_samples] = self._dtype(value)
        self.sample_count += 1

    def as_metric(
        self,
        key: str,
        title: str,
        unit: str,
        scale: float,
    ) -> TraceMetricWithStats:
        # TODO really we ought to pass the unscaled values and format with x1000 when we render
        median = self.median * scale
        mean = self.mean * scale
        min = self.min * scale
        max = self.max * scale
        std = self.std * scale
        max_scale_value = max * 1.05 if max != float('-inf') else 500.0
        return TraceMetricWithStats(key, title, median, max_scale_value, unit, mean, min, max, std)


class Latency(Tracer):
    key = '__latency__'
    title = 'Latency'

    def __init__(self):
        self._stats = collections.defaultdict(Statistics)
        self._all_streams = Statistics()
        self._show_per_stream: bool | None = None
        self._frame_no = 0

    def update(self, frame_result):
        if self._show_per_stream is None:
            # read this lazily because logging has not been configured at startup
            self._show_per_stream = LOG.isEnabledFor(logging_utils.DEBUG)
        if self._frame_no > 200:  # the first 200 frames latency are ignored
            latency = frame_result.sink_timestamp - frame_result.src_timestamp
            self._all_streams.update(latency)
            if self._show_per_stream:
                self._stats[frame_result.stream_id].update(latency)
        self._frame_no += 1

    def get_metrics(self) -> list[TraceMetric]:
        m = []
        m.append(self._all_streams.as_metric('__latency__', 'Latency', 'ms', 1000))
        if self._show_per_stream and len(self._stats) > 1:
            for s, t in sorted(self._stats.items(), key=lambda x: x[0]):
                m.append(t.as_metric(f'__latency_{s}__', f'Latency (stream {s})', 'ms', 1000))
        return m


_key = lambda x: x.strip('_')

supported_tracers = {
    _key(CpuTracer.key): lambda aipu_cores: _ThreadedTracerAdapter(CpuTracer(), period=3.0),
    _key(End2EndTracer.key): lambda aipu_cores: End2EndTracer(),
    _key(End2EndInfRateTracer.key): lambda aipu_cores: End2EndInfRateTracer(),
    _key(Latency.key): lambda aipu_cores: Latency(),
}

# AIPU-specific tracers will be conditionally added at creation time
_aipu_tracers = {
    _key(AipuTracer.key): lambda aipu_cores: _ThreadedTracerAdapter(
        AipuTracer(aipu_cores), period=0.3
    ),
    _key(CoreTempTracer.key): lambda aipu_cores: _ThreadedTracerAdapter(
        CoreTempTracer(), period=1.0
    ),
    _key(HostTracer.key): lambda aipu_cores: _ThreadedTracerAdapter(
        HostTracer(aipu_cores), period=0.3
    ),
}


def create_tracers(
    *requested: tuple[str, ...], _aipu_cores: int = 4, pipe_type: str = None
) -> List[Tracer]:
    '''Create a list of tracers based on the requested tracer keys.

    Tracers provide real-time metrics that help you better understand device resource utilization,
    performance and thermal characteristics.

    This function takes a list of metrics as its arguments and returns an object to provide the
    `tracers` argument in the function `create_inference_stream`.  The available tracers are:

    * `end_to_end_fps` - collects end-to-end fps metric
    * `core_temp` - collects the temperature of the metis core
    * `cpu_usage` - collects information about the host CPU usage
    * `latency`  - collects metrics about stream latency

    Tracers passed to `create_inference_stream` will be started automatically when the stream is
    started, and periodically queried for their current value which will be shown in the
    application UI as a speedometer.

    Alternatively the application can also access these tracers through the stream object ::

      for frame_result in stream:
          metrics = stream.get_all_metrics()
          core_temp = metrics['core_temp']
          print(f"Core temperature: {core_temp.value} {core_temp.unit}")

    Args:
        *requested: The tracer keys to create.
        _aipu_cores: The number of AIPU cores to use.
        pipe_type: The pipe type being used (e.g. 'torch', 'gst', 'torch-aipu').
    '''
    tracers = []
    unsupported = []
    # disable tracing by default, tracers below will enable it if necessary
    os.environ.setdefault('AXE_PROFILING_CONFIG', '')

    available_tracers = supported_tracers.copy()
    has_aipu = False
    if pipe_type in ['gst', 'torch-aipu'] and not config.env.inference_mock:
        try:
            saved = utils.LOG.level
            utils.LOG.setLevel(logging_utils.logging.ERROR)
            has_aipu = device_manager.detect_metis_type() != config.Metis.none
        finally:
            utils.LOG.setLevel(saved)

        if has_aipu:
            available_tracers.update(_aipu_tracers)

    for tracer in requested:
        try:
            ctor = available_tracers[tracer]
        except KeyError:
            unsupported.append(tracer)
        else:
            tracers.append(ctor(_aipu_cores))

    if unsupported:
        s = 's' if len(unsupported) > 1 else ''
        LOG.warning(
            f"Unsupported tracer{s}: {', '.join(unsupported)}: valid tracers are: {', '.join(available_tracers.keys())}"
        )
    return tracers


def create_tracers_from_args(args: argparse.Namespace) -> List[Tracer]:
    '''Create a list of tracers based on the parsed command line arguments.

    This is a convenience function to create tracers based on the command line arguments in a
    similar way to how inference.py does so.

    If you are using `config.create_inference_argparser` to create the command line parser,
    then this function provides a simple way to create the tracers based on the
    command line arguments ::

        from axelera.app import config, inf_tracers
        parser = config.create_inference_argparser(description='My app description')
        args = parser.parse_args()
        tracers = inf_tracers.create_tracers_from_args(args)
        stream = create_inference_stream(
            network="yolov5m-v7-coco-tracker",
            sources=args.sources,
            pipe_type=args.pipe,
            tracers=tracers,
        )

    '''
    requested = []
    # disable tracing by default, tracers below will enable it if necessary
    if args.show_temp:
        requested.append(_key(CoreTempTracer.key))
    if args.show_device_fps:
        requested.append(_key(AipuTracer.key))
    if args.show_host_fps:
        requested.append(_key(HostTracer.key))
    if args.show_cpu_usage:
        requested.append(_key(CpuTracer.key))
    if args.show_system_fps:
        requested.append(_key(End2EndTracer.key))
    if args.show_inference_rate:
        requested.append(_key(End2EndInfRateTracer.key))
    if args.show_latency:
        requested.append(_key(Latency.key))
    pipe_type = getattr(args, 'pipe', 'gst')
    return create_tracers(*requested, _aipu_cores=args.aipu_cores, pipe_type=pipe_type)


def display_tracers(tracers: List[Tracer], file=sys.stdout):
    '''Display the tracers to the console.

    This is called by the application to display the tracers in the UI.
    '''
    all_metrics = list(itertools.chain.from_iterable(t.get_metrics() for t in tracers))
    widest = max(len(m.title) for m in all_metrics) if all_metrics else 0
    for m in all_metrics:
        print(f"{m.title.ljust(widest)} : {m.text_report()}", file=file)
