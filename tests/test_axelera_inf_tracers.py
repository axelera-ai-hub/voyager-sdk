# Copyright Axelera AI, 2023
import functools
import itertools
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from axelera import types
from axelera.app import config, inf_tracers, network


def make_aipu_tracer(frequency=600e6, log_content=''):
    mis = network.ModelInfos()
    mi = types.ModelInfo('squeezenet', 'Classification', [3, 224, 224])
    mi.manifest = Mock()
    mi.manifest.input_shapes = [[1, 3, 224, 224]]
    mis.add_model(mi, '/path/to/manifest.json')
    tracer = inf_tracers.AipuTracer(1)
    tracer.initialize_models(mis, config.Metis.pcie, {0: frequency})
    tracer.start_monitoring = Mock(return_value=None)
    mock_triton = Mock()
    reads = [log_content]

    def read(*args, **kwargs):
        try:
            return reads.pop(0)
        except IndexError:
            return b''

    mock_triton.read = read
    tracer._tritons = {0: mock_triton}
    tracer._running = True
    return tracer


@pytest.mark.parametrize(
    "frequency,log_content,expected_fps",
    [
        (
            600,
            "Megakernel  took 100000 cycles\n"
            "Megakernel  took 100000 cycles\n"
            "Megakernel  took 100000 cycles\n",
            6000,
        ),
        (
            800,
            "Megakernel  took 200000 cycles\n"
            "Megakernel  took 200000 cycles\n"
            "Megakernel  took 200000 cycles\n",
            4000,
        ),
        (
            300,
            "Megakernel  took 600000 cycles\n"
            "Megakernel  took 600000 cycles\n"
            "Megakernel  took 600000 cycles\n",
            500.0,
        ),
        (
            600,
            "Megakernel  took 10000000 cycles\n"  # huge outlier
            "Megakernel  took 500000 cycles\n"
            "Megakernel  took 600000 cycles\n",  # median
            1000.0,
        ),
        (
            600,
            "[M] __tvm_main__ 300123 cycles\n[B] Megakernel executed (0)\n[B] Megakernel took 300000 cycles\n[B] Enter supervisor mode\n",
            2000,
        ),
    ],
)
def test_get_metric_fps(frequency, log_content, expected_fps):
    aipu_tracer = make_aipu_tracer(frequency, log_content=log_content)
    metrics = aipu_tracer.get_metrics()[0].value
    assert round(metrics) == round(expected_fps), f"{frequency=}"


def test_empty_statistics():
    s = inf_tracers.Statistics('k', 't', 400, 4)
    assert s.min == float('inf')
    assert s.max == float('-inf')
    assert s.mean == 0.0
    assert s.stddev == 0.0
    assert s.sample_count == 0
    np.testing.assert_equal(s.bins, np.zeros((4,), dtype=np.int32))


def test_statistics():
    s = inf_tracers.Statistics('k', 't', 400, 4)
    for v in [100, 200, 100, 200, 100, 200]:
        s.update(v)
    assert s.max == 200
    assert s.min == 100
    assert s.mean == 150
    assert s.stddev == 50
    assert s.sample_count == 6
    np.testing.assert_equal(s.bins, np.array([0, 3, 3, 0], dtype=np.int32))


def test_stream_statistics():
    times = [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 1.0]
    with patch.object(time, 'time', iter(times).__next__):
        s = inf_tracers.StreamStatistics()
        for v in [0.1, 0.2, 0.1, 0.2, 0.1, 0.2]:
            s.update(v)
    assert repr(s) == (
        'frames:6     l-min:100   l-mean:150   l-max:200   l-stddev:50   '
        + ' i-min:99    i-mean:150   i-max:200   i-stddev:49   '
    )


def _generate_fps_time(fps):
    return functools.partial(next, itertools.count(42.0, 1 / fps))


def test_e2e_tracer_no_update():
    with patch.object(time, 'time', _generate_fps_time(10)):
        fps = inf_tracers.End2EndTracer()
        assert 0.0 == round(fps.get_metrics()[0].value, 1)


def test_e2e_tracer_one_update():
    with patch.object(time, 'time', _generate_fps_time(10)):
        fps = inf_tracers.End2EndTracer()
        fps.update(None)
        assert 0.0 == round(fps.get_metrics()[0].value, 1)


def test_e2e_tracer_one_second_of_updates():
    with patch.object(time, 'time', _generate_fps_time(10)):
        fps = inf_tracers.End2EndTracer()
        for _ in range(100):
            fps.update(None)
        assert 10.0 == round(fps.get_metrics()[0].value, 1)


def test_e2e_tracer_five_seconds_of_updates_fps_99():
    with patch.object(time, 'time', _generate_fps_time(99)):
        fps = inf_tracers.End2EndTracer()
        for n in range(99 * 5):
            fps.update(None)
        assert 99.0 == round(fps.get_metrics()[0].value, 1)


def test_e2t_tracer_ten_updates_fps_100_1():
    with patch.object(time, 'time', _generate_fps_time(100.1)):
        fps = inf_tracers.End2EndTracer()
        for n in range(1001):
            fps.update(None)
        assert 100.1 == round(fps.get_metrics()[0].value, 1)


def test_trace_metric_draw():
    m = inf_tracers.TraceMetric('title', 100.0, 100.0)
    mock = Mock()
    m.draw(mock)
    mock.draw_speedometer.assert_called_once_with(m)


def test_parse_axr_stats():
    logs = '''\
idx:0:00000000 beg:00060ab0804b6738 inp:000004d9 out:01276f65 pat:0000658a krn:000068c8 tot:01313c98
idx:0:00000001 beg:00060ab0817d9029 inp:000001e4 out:005f4234 pat:00002e7f krn:00002f94 tot:00618628
idx:0:00000002 beg:00060ab081df62e0 inp:000000af out:005dc76f pat:00001bdf krn:00001c8d tot:005f7f12
idx:0:00000003 beg:00060ab0823f623e inp:000000a7 out:005d19ba pat:00003597 krn:00003655 tot:00664e57
idx:0:00000004 beg:00060ab082a6072f inp:000000ba out:005dd881 pat:000035e5 krn:0000369a tot:00600699
idx:0:00000005 beg:00060ab08306631c inp:000000ad out:005dc3d3 pat:00002fde krn:000030b6 tot:006033b2
'''
    stats = inf_tracers.parse_axr_stats(logs)
    stats = {k: np.median(v) for k, v in stats.items()}
    assert stats == {
        'Host': 0.006348013,
        'Memcpy-in': 1.805e-07,
        'Kernel': 1.31895e-05,
        'Memcpy-out': 0.006148088,
        'Patch Parameters': 1.29865e-05,
    }
