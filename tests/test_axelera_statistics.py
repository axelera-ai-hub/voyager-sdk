#!/usr/bin/env python
# Copyright Axelera AI, 2024

import os
import pathlib
import tempfile
from unittest.mock import patch

from axelera.app import statistics

GOLD_SRC = pathlib.Path(__file__).parent / "example_gst_tracer_log.txt"


class MockMetric:
    def __init__(self, value, title):
        self.value = value
        self.title = title

    def get_metrics(self):
        return [self]


PREFIX_NO_INF = '''\
===========================================================================
Element                                         Latency(us)   Effective FPS
===========================================================================
qtdemux0                                                 15        64,194.9
typefind                                                 22        44,309.3
h265parse0                                               55        17,906.0
capsfilter0                                             231         4,317.5
vaapidecode0                                            724         1,380.9
capsfilter1                                              29        33,918.7
vaapipostproc0                                          262         3,808.8
axinplace-addstreamid0                                   34        29,082.2
input_tee                                                34        29,188.0
convert_in                                              271         3,677.5
axinplace0                                               29        33,906.7
axinplace1                                               39        25,013.8
axtransform-resizeletterbox0                            778         1,284.8
axtransform-totensor0                                   199         5,025.0
axinplace-normalize0                                    305         3,271.1
axtransform-padding0                                  1,846           541.7'''
PREFIX_SHORT = (
    PREFIX_NO_INF
    + '''
inference-task0-YOLOv5s-relu-COCO2017                17,719            56.4'''
)


SUFFIX = '''\
decoder_task0                                        17,094            58.5
axinplace-nms0                                           70        14,100.7
===========================================================================
End-to-end average measurement                                          0.0
==========================================================================='''


def test_format_table_host_and_metis():
    tracers = [MockMetric(200, 'Host'), MockMetric(150, 'Metis')]
    got = statistics.format_table(GOLD_SRC, tracers, statistics.Plain)
    assert (
        got
        == f'''\
{PREFIX_SHORT}
 └─ Metis                                             6,666           150.0
 └─ Host                                              5,000           200.0
{SUFFIX}'''
    )


def test_format_table_host():
    tracers = [MockMetric(200, 'Host')]
    got = statistics.format_table(GOLD_SRC, tracers, statistics.Plain)
    assert (
        got
        == f'''\
{PREFIX_SHORT}
 └─ Host                                              5,000           200.0
{SUFFIX}'''
    )


def test_format_table_metis():
    tracers = [MockMetric(150, 'Metis')]
    got = statistics.format_table(GOLD_SRC, tracers, statistics.Plain)
    assert (
        got
        == f'''\
{PREFIX_SHORT}
 └─ Metis                                             6,666           150.0
{SUFFIX}'''
    )


def test_format_table_no_tracers():
    tracers = []
    got = statistics.format_table(GOLD_SRC, tracers, statistics.Plain)
    assert (
        got
        == f'''\
{PREFIX_SHORT}
{SUFFIX}'''
    )


def test_format_table_tracers_present_but_no_value(caplog):
    tracers = [MockMetric(0, 'Host'), MockMetric(0, 'Metis')]
    got = statistics.format_table(GOLD_SRC, tracers, statistics.Plain)
    assert (
        got
        == f'''\
{PREFIX_SHORT}
{SUFFIX}'''
    )
    assert [r.message for r in caplog.records] == [
        'Unable to determine Host metrics',
        'Unable to determine Metis metrics',
    ]


def test_format_table_long_inference_task():
    long_gold_src = GOLD_SRC.read_text().replace('COCO2017', 'COCO2017-really-really-long')
    tracers = []
    with patch.object(pathlib.Path, 'read_text', return_value=long_gold_src):
        got = statistics.format_table(GOLD_SRC, tracers, statistics.Plain)
    assert (
        got
        == f'''\
{PREFIX_NO_INF}
inference-task0-YOLOv5s-relu-COCO2017-really-really-long
                                                     17,719            56.4
{SUFFIX}'''
    )


def test_format_empty_table():
    tracers = []
    with patch.object(pathlib.Path, 'read_text', return_value=''):
        got = statistics.format_table(GOLD_SRC, tracers, statistics.Plain)
    assert (
        got
        == '''\
===========================================================================
Element                                         Latency(us)   Effective FPS
===========================================================================
===========================================================================
End-to-end average measurement                                          0.0
==========================================================================='''
    )


def test_initialise_logging():
    with patch.object(tempfile, 'NamedTemporaryFile') as mock:
        mock.return_value.name = '/some/fake_log_file'
        with patch.dict(os.environ, {}, clear=True):
            got = statistics.initialise_logging()
            assert os.environ['GST_DEBUG'] == 'GST_TRACER:7'
            assert os.environ['GST_TRACERS'] == 'latency(flags=element)'
            assert os.environ['GST_DEBUG_FILE'] == '/some/fake_log_file'
    mock.assert_called_once_with(mode='w')
    assert got[1] == pathlib.Path('/some/fake_log_file')
    assert hasattr(got[0], 'name')
