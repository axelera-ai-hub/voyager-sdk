# Copyright Axelera AI, 2023
import functools
import io
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
    tracer.initialize_models(mis, config.Board.pcie, {0: frequency}, ['metis-0:1:0'])
    tracer.start_monitoring = Mock(return_value=None)
    mock_triton = Mock()
    reads = [log_content]

    def read(*args, **kwargs):
        try:
            return reads.pop(0)
        except IndexError:
            return b''

    mock_triton.read = read
    tracer._tritons = {(0, 0): mock_triton}
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
    s = inf_tracers.Statistics()
    assert s.min == float('inf')
    assert s.max == float('-inf')
    assert s.mean == 0.0
    assert s.median == 0.0
    assert s.std == 0.0
    assert s.sample_count == 0


def test_statistics():
    s = inf_tracers.Statistics()
    for v in [100, 200, 100, 200, 100, 200]:
        s.update(v)
    assert s.max == 200.0
    assert s.min == 100.0
    assert s.mean == 150.0
    assert s.median == 150.0
    assert s.std == 50.0
    assert s.sample_count == 6


def test_statistic_overflow_buffer():
    s = inf_tracers.Statistics(max_samples=6)
    for v in [100, 200, 100, 200, 100, 200, 50, 150, 100]:
        s.update(v)
    assert s.max == 200.0
    assert s.min == 50.0
    assert round(s.mean, 1) == 133.3
    assert s.median == 125.0
    assert round(s.std, 1) == 55.3
    assert s.sample_count == 9


def test_statistic_as_metric():
    s = inf_tracers.Statistics(max_samples=6)
    for v in [100, 200, 100, 200, 100, 200]:
        s.update(v)
    m = s.as_metric('key', 'title', 'unit', 1.0)
    assert m.key == 'key'
    assert m.title == 'title'
    assert m.min == 100.0
    assert m.median == 150.0
    assert m.mean == 150.0
    assert m.value == 150.0
    assert m.max == 200.0
    assert round(m.std, 1) == 50.0
    assert m.text_report() == '150.0unit (min:100.0 max:200.0 σ:50.0 x̄:150.0)unit'


def test_display_tracers():
    m0 = Mock(title='Tracer0', text_report=Mock(return_value='funky report'))
    m1 = Mock(title='Tracer   1', text_report=Mock(return_value='lovely jubbly'))
    tracers = [Mock(get_metrics=Mock(return_value=[m0, m1]))]
    f = io.StringIO()
    inf_tracers.display_tracers(tracers, f)
    assert (
        f.getvalue()
        == '''\
Tracer0    : funky report
Tracer   1 : lovely jubbly
'''
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
