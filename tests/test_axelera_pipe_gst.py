# Copyright Axelera AI, 2023
# Construct GStreamer application pipeline
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from axelera.app import config, gst_builder, operators, pipe, pipeline
from axelera.app.pipe import FrameEvent, gst, gst_helper, io


def _sorted(elements):
    return sorted(elements, key=lambda e: e.get_name())


def _get_gst():
    Gst = pytest.importorskip("gi.repository.Gst")
    if not Gst.is_initialized():
        Gst.init(None)
    return Gst


@pytest.mark.parametrize('num_sinks', [0, 1, 2, 4])
def test_iteration_appsinks(num_sinks):
    Gst = _get_gst()
    pipeline = Gst.Pipeline()
    videosrc = Gst.ElementFactory.make('videotestsrc', 'source')
    pipeline.add(videosrc)
    appsinks = []
    for n in range(num_sinks):
        element = Gst.ElementFactory.make('appsink', f'sink{n}')
        pipeline.add(element)
        appsinks.append(element)

    assert [videosrc] == gst_helper.list_all_by_element_factory_name(pipeline, 'videotestsrc')
    assert _sorted(appsinks) == _sorted(
        gst_helper.list_all_by_element_factory_name(pipeline, 'appsink')
    )


def _simple_pipeline():
    Gst = _get_gst()
    pipeline = Gst.Pipeline()
    src = Gst.ElementFactory.make('videotestsrc', 'src')
    q1 = Gst.ElementFactory.make('queue', 'q1')
    q2 = Gst.ElementFactory.make('queue', 'q2')
    sink = Gst.ElementFactory.make('fakesink', 'sink')
    for e in (src, q1, q2, sink):
        pipeline.add(e)
    assert src.link(q1)
    assert q1.link(q2)
    assert q2.link(sink)
    return pipeline, src, sink


def test_iter_upstream():
    pipeline, _, sink = _simple_pipeline()
    upstream = list(gst_helper._iter_upstream(sink))
    assert [e.get_name() for e in upstream] == ['q2', 'q1', 'src']


def test_iter_downstream():
    pipeline, src, _ = _simple_pipeline()
    downstream = list(gst_helper._iter_downstream(src))
    assert [e.get_name() for e in downstream] == ['q1', 'q2', 'sink']


def _expected_rtsp_input_builder(source_id_offset=0):
    exp = gst_builder.Builder(None, None, 4, 'auto')
    exp.rtspsrc(
        {'user-id': '', 'user-pw': ''},
        location='rtsp://localhost:8554/test',
        latency=500,
        connections={'stream_%u': f'rtspcapsfilter{source_id_offset}.sink'},
    )
    exp.capsfilter(
        {'caps': 'application/x-rtp,media=video'}, name=f'rtspcapsfilter{source_id_offset}'
    )
    exp.decodebin(
        {'expose-all-streams': False, 'force-sw-decoders': False},
        caps='video/x-raw(ANY)',
        connections={'src_%u': f'decodebin-link{source_id_offset}.sink'},
    )
    # exp.queue(name='queue_in0')
    exp.axinplace(
        lib='libinplace_addstreamid.so',
        mode='meta',
        options=f'stream_id:{source_id_offset}',
        name=f'decodebin-link{source_id_offset}',
    )
    return exp


def test_build_input_with_normal_input():
    pipein = io.SinglePipeInput('gst', config.Source('rtsp://localhost:8554/test'))
    builder = gst_builder.Builder(None, None, 4, 'auto')
    task = pipeline.AxTask('task0', operators.Input())
    pipe.gst._build_input_pipeline(builder, task, pipein)
    exp = _expected_rtsp_input_builder()
    exp.queue(connections={'src': 'inference-task0.sink_%u'})
    assert list(builder) == list(exp)


def test_build_input_with_normal_input_with_sourceid():
    pipein = io.SinglePipeInput('gst', config.Source('rtsp://localhost:8554/test'), source_id=4)
    builder = gst_builder.Builder(None, None, 4, 'auto')
    task = pipeline.AxTask('task0', operators.Input())
    pipe.gst._build_input_pipeline(builder, task, pipein)
    exp = _expected_rtsp_input_builder(4)
    exp.queue(connections={'src': 'inference-task0.sink_%u'})
    assert list(builder) == list(exp)


def test_save_axnet_files():
    elements = [
        {
            'instance': 'axinferencenet',
            'name': 'task0',
            'model': '/cwd/build/modelA.json',
            'p0_options': 'something:a;classlabels_file:bob;;mode:meta',
        },
        {
            'instance': 'axinferencenet',
            'name': 'task1',
            'model': '/abs/build/modelB.json',
            'p0_options': 'something:a;classlabels_file:bob',
        },
        {
            'instance': 'axinferencenet',
            'name': 'task2',
            'model': '/cwd/build/modelC.json',
            'p0_options': 'classlabels_file:bob;mode:meta',
        },
        {
            'instance': 'axinferencenet',
            'name': 'task3',
            'model': '/cwd/build/modelD.json',
            'p0_options': 'classlabels_file:bob',
        },
    ]
    task_names = ['model0', 'model1', 'model2', 'model3']
    with patch.object(os, 'getcwd', return_value='/cwd'):
        with patch.object(Path, 'write_text') as m:
            gst._save_axnet_files(elements, task_names, Path('/abs'))
        m.assert_has_calls(
            [
                call('model=build/modelA.json\np0_options=something:a;mode:meta'),
                call('model=/abs/build/modelB.json\np0_options=something:a'),
                call('model=build/modelC.json\np0_options=mode:meta'),
                call('model=build/modelD.json\np0_options='),
            ]
        )


def test_handle_pair_validation():
    """Test the pair validation handling logic in GstPipe."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from axelera.app.meta import pair_validation
    from axelera.app.pipe import gst
    from axelera.app.pipe.gst import GstPipe

    # Patch the __init__ method to avoid needing actual dependencies
    with patch.object(GstPipe, '__init__', return_value=None):
        # Create a mock GstPipe object with is_pair_validation=True
        pipe = GstPipe()
        pipe.is_pair_validation = True
        pipe._cached_ax_meta = None

        # Create metadata instances
        ax_meta = gst.meta.AxMeta('id1')
        pv_meta = pair_validation.PairValidationMeta()
        ax_meta.add_instance('task1', pv_meta)

        # Create embeddings - using 2D arrays to match the shape expected by PairValidationMeta
        embeddings1 = np.array([[0.1, 0.2, 0.3, 0.4]])
        embeddings2 = np.array([[0.5, 0.6, 0.7, 0.8]])

        # Create a GstMetaInfo that simulates what we'd get from inference
        meta_info = MagicMock()
        meta_info.task_name = 'task1'
        meta_info.meta_type = pair_validation.PairValidationMeta

        # Create mock task_meta with results
        task_meta = MagicMock()
        task_meta.results = [embeddings1]

        # Create decoded_meta dictionary
        decoded_meta = {meta_info: task_meta}

        # First call to _handle_pair_validation should store the first result and return ax_meta
        cached_meta = pipe._handle_pair_validation(ax_meta, decoded_meta)
        assert cached_meta is not None, "First call should return the metadata for caching"
        assert (
            len(cached_meta.get_instance('task1', pair_validation.PairValidationMeta).results) == 1
        )
        assert np.array_equal(
            cached_meta.get_instance('task1', pair_validation.PairValidationMeta).results[0],
            embeddings1,
        )

        # Update decoded meta with second embedding
        task_meta.results = [embeddings2]

        # Second call to _handle_pair_validation with second embedding should return None
        # indicating processing is complete
        result = pipe._handle_pair_validation(cached_meta, decoded_meta)
        assert result is None, "Second call should return None to indicate pair is complete"

        # Verify both embeddings were collected
        assert (
            len(cached_meta.get_instance('task1', pair_validation.PairValidationMeta).results) == 2
        )
        assert np.array_equal(
            cached_meta.get_instance('task1', pair_validation.PairValidationMeta).results[0],
            embeddings1,
        )
        assert np.array_equal(
            cached_meta.get_instance('task1', pair_validation.PairValidationMeta).results[1],
            embeddings2,
        )


def test_gstpipe_loop_with_pair_validation():
    """Test the _loop method's handling of pair validation within GstPipe."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    from axelera.app.meta import pair_validation
    from axelera.app.pipe import gst
    from axelera.app.pipe.gst import GstPipe, GstStream

    # Patch the __init__ method to avoid needing actual dependencies
    with patch.object(GstPipe, '__init__', return_value=None):
        # Create a mock GstPipe object
        pipe = GstPipe()
        pipe._cached_ax_meta = None
        pipe._meta_assembler = MagicMock()
        pipe.task_graph = MagicMock()
        pipe._stop_event = MagicMock()
        pipe._stop_event.is_set.side_effect = [False, False, True]  # Run loop twice then exit
        pipe.pipeout = MagicMock()
        pipe._on_event = MagicMock()

        # Create metadata instances
        ax_meta1 = gst.meta.AxMeta('id1')
        pv_meta1 = pair_validation.PairValidationMeta()
        ax_meta1.add_instance('task1', pv_meta1)

        ax_meta2 = gst.meta.AxMeta('id2')
        pv_meta2 = pair_validation.PairValidationMeta()
        ax_meta2.add_instance('task1', pv_meta2)

        # Create embeddings - using 2D arrays to match the shape expected by PairValidationMeta
        embeddings1 = np.array([[0.1, 0.2, 0.3, 0.4]])
        embeddings2 = np.array([[0.5, 0.6, 0.7, 0.8]])

        # Create two frames with metadata
        frame1 = MagicMock()
        frame1.stream_id = 0
        frame1.meta = ax_meta1
        frame2 = MagicMock()
        frame2.stream_id = 0
        frame2.meta = ax_meta2

        # Mock decoded metadata
        meta_info = MagicMock()
        meta_info.task_name = 'task1'
        meta_info.meta_type = pair_validation.PairValidationMeta
        meta_info.master = None  # Set master to None to avoid validation errors

        task_meta1 = MagicMock()
        task_meta1.results = [embeddings1]
        decoded_meta1 = {meta_info: task_meta1}

        task_meta2 = MagicMock()
        task_meta2.results = [embeddings2]
        decoded_meta2 = {meta_info: task_meta2}

        # Create a mock stream that yields frame and decoded metadata pairs
        pipe._stream = stream = MagicMock(spec=GstStream)
        evt1 = FrameEvent(result=frame1)
        evt2 = FrameEvent(result=frame2)
        stream.__iter__.return_value = [(evt1, decoded_meta1), (evt2, decoded_meta2)]

        # Test the _loop method's handling of pair validation
        with patch.object(
            pipe, '_handle_pair_validation', wraps=pipe._handle_pair_validation
        ) as mock_handle:
            pipe._loop()

            # Verify _handle_pair_validation was called twice
            assert mock_handle.call_count == 2

            # First call should use frame1.meta and first decoded metadata
            assert mock_handle.call_args_list[0][0][0] == frame1.meta
            assert mock_handle.call_args_list[0][0][1] == decoded_meta1

            # Second call should use cached metadata and second decoded metadata
            assert mock_handle.call_args_list[1][0][1] == decoded_meta2


# ---------------------------------------------------------------------------
# Tests for _verify_pipeline_plugins and its helpers
# ---------------------------------------------------------------------------

SIMPLE_PIPELINE = [
    {'instance': 'filesrc', 'name': 'src0', 'location': '/dev/null'},
    {'instance': 'decodebin', 'name': 'dec0'},
    {'instance': 'axinplace', 'name': 'ip0', 'lib': 'libinplace_addstreamid.so'},
    {'instance': 'axtransform', 'name': 'tx0', 'lib': 'libtransform_colorconvert.so'},
    {
        'instance': 'axinferencenet',
        'name': 'inf0',
        'model': 'model.json',
        'preprocess0_lib': 'libtransform_resize.so',
        'preprocess0_options': 'width:640',
        'postprocess0_lib': 'libdecode_yolov5.so',
        'postprocess0_options': 'meta_key:task0',
    },
    {'instance': 'appsink', 'name': 'sink0'},
]


class TestCollectGstPlugins:
    def test_extracts_unique_instances(self):
        pipeline = [
            {'instance': 'filesrc', 'name': 'a'},
            {'instance': 'decodebin', 'name': 'b'},
            {'instance': 'filesrc', 'name': 'c'},
        ]
        assert gst._collect_gst_plugins(pipeline) == ['decodebin', 'filesrc']

    def test_skips_elements_without_instance(self):
        pipeline = [{'instance': 'appsink'}, {'name': 'orphan'}]
        assert gst._collect_gst_plugins(pipeline) == ['appsink']

    def test_empty_pipeline(self):
        assert gst._collect_gst_plugins([]) == []

    def test_full_pipeline(self):
        result = gst._collect_gst_plugins(SIMPLE_PIPELINE)
        assert result == [
            'appsink',
            'axinferencenet',
            'axinplace',
            'axtransform',
            'decodebin',
            'filesrc',
        ]


class TestCollectAxPlugins:
    def test_collects_lib_from_axinplace_and_axtransform(self):
        pipeline = [
            {'instance': 'axinplace', 'lib': 'libinplace_addstreamid.so'},
            {'instance': 'axtransform', 'lib': 'libtransform_colorconvert.so'},
        ]
        assert gst._collect_ax_plugins(pipeline) == [
            'libinplace_addstreamid.so',
            'libtransform_colorconvert.so',
        ]

    def test_collects_lib_from_axdecode(self):
        pipeline = [{'instance': 'axdecode', 'lib': 'libdecode_yolov5.so'}]
        assert gst._collect_ax_plugins(pipeline) == ['libdecode_yolov5.so']

    def test_collects_pre_post_process_libs_from_axinferencenet(self):
        pipeline = [
            {
                'instance': 'axinferencenet',
                'preprocess0_lib': 'libtransform_resize.so',
                'preprocess0_options': 'width:640',
                'preprocess1_lib': 'libinplace_normalize.so',
                'postprocess0_lib': 'libdecode_yolov5.so',
            }
        ]
        result = gst._collect_ax_plugins(pipeline)
        assert result == [
            'libdecode_yolov5.so',
            'libinplace_normalize.so',
            'libtransform_resize.so',
        ]

    def test_ignores_non_lib_keys_on_axinferencenet(self):
        pipeline = [
            {
                'instance': 'axinferencenet',
                'model': 'model.json',
                'preprocess0_lib': 'libtransform_resize.so',
                'preprocess0_options': 'width:640',
            }
        ]
        assert gst._collect_ax_plugins(pipeline) == ['libtransform_resize.so']

    def test_ignores_elements_without_lib(self):
        pipeline = [
            {'instance': 'axinplace'},
            {'instance': 'filesrc'},
        ]
        assert gst._collect_ax_plugins(pipeline) == []

    def test_deduplicates(self):
        pipeline = [
            {'instance': 'axinplace', 'lib': 'libinplace_addstreamid.so'},
            {'instance': 'axinplace', 'lib': 'libinplace_addstreamid.so'},
        ]
        assert gst._collect_ax_plugins(pipeline) == ['libinplace_addstreamid.so']

    def test_empty_pipeline(self):
        assert gst._collect_ax_plugins([]) == []

    def test_full_pipeline(self):
        result = gst._collect_ax_plugins(SIMPLE_PIPELINE)
        assert result == [
            'libdecode_yolov5.so',
            'libinplace_addstreamid.so',
            'libtransform_colorconvert.so',
            'libtransform_resize.so',
        ]


class TestResolveLib:
    def test_absolute_path_exists_and_loads(self, tmp_path):
        lib = tmp_path / 'libfake.so'
        lib.touch()
        with patch.object(ctypes, 'CDLL', return_value=MagicMock()) as mock_cdll:
            result = gst._resolve_lib(str(lib), [])
        mock_cdll.assert_called_once_with(str(lib))
        assert result == lib

    def test_absolute_path_not_found(self, tmp_path):
        result = gst._resolve_lib('/no/such/libfake.so', [])
        assert result is None

    def test_relative_found_in_search_dirs(self, tmp_path):
        lib = tmp_path / 'libfake.so'
        lib.touch()
        with patch.object(ctypes, 'CDLL', return_value=MagicMock()):
            result = gst._resolve_lib('libfake.so', [tmp_path])
        assert result == lib

    def test_relative_not_found_in_search_dirs(self, tmp_path):
        result = gst._resolve_lib('libfake.so', [tmp_path])
        assert result is None

    def test_relative_found_but_dlopen_fails(self, tmp_path):
        lib = tmp_path / 'libbroken.so'
        lib.touch()
        with patch.object(ctypes, 'CDLL', side_effect=OSError('bad ELF')):
            result = gst._resolve_lib('libbroken.so', [tmp_path])
        assert result is None

    def test_searches_dirs_in_order(self, tmp_path):
        d1 = tmp_path / 'first'
        d2 = tmp_path / 'second'
        d1.mkdir()
        d2.mkdir()
        (d2 / 'libfoo.so').touch()
        with patch.object(ctypes, 'CDLL', return_value=MagicMock()):
            result = gst._resolve_lib('libfoo.so', [d1, d2])
        assert result == d2 / 'libfoo.so'


class TestFindMissingGstPlugins:
    def test_all_present(self):
        pytest.importorskip("gi.repository.Gst")
        # queue and fakesink are always available in any GStreamer install
        assert gst._find_missing_gst_plugins(['queue', 'fakesink']) == []

    def test_missing_detected(self):
        pytest.importorskip("gi.repository.Gst")
        result = gst._find_missing_gst_plugins(['queue', 'nonexistent_element_xyz'])
        assert result == ['nonexistent_element_xyz']


class TestFindMissingAxPlugins:
    def test_all_found(self, tmp_path):
        (tmp_path / 'libfoo.so').touch()
        (tmp_path / 'libgstaxstreamer.so').touch()
        gst._get_search_dirs_for_ax_plugins.cache_clear()
        with patch.object(ctypes, 'CDLL', return_value=MagicMock()):
            with patch.dict(os.environ, {'GST_PLUGIN_PATH': str(tmp_path)}, clear=False):
                gst._get_search_dirs_for_ax_plugins.cache_clear()
                result = gst._find_missing_ax_plugins(['libfoo.so'])
        gst._get_search_dirs_for_ax_plugins.cache_clear()
        assert result == []

    def test_missing_reported(self, tmp_path):
        gst._get_search_dirs_for_ax_plugins.cache_clear()
        with patch.dict(os.environ, {'GST_PLUGIN_PATH': str(tmp_path)}, clear=False):
            # No libgstaxstreamer.so, so it falls back; and our lib won't be found either
            gst._get_search_dirs_for_ax_plugins.cache_clear()
            result = gst._find_missing_ax_plugins(['libno_such_plugin.so'])
        gst._get_search_dirs_for_ax_plugins.cache_clear()
        assert result == ['libno_such_plugin.so']


class TestReportMissingPlugins:
    def test_logs_standard_gst_missing(self):
        with patch.object(gst.LOG, 'error') as mock_err:
            gst._report_missing_plugins(['decodebin', 'fakesink'], [])
        messages = ' '.join(str(c) for c in mock_err.call_args_list)
        assert 'decodebin' in messages
        assert 'fakesink' in messages
        assert 'GStreamer' in messages

    def test_logs_axelera_gst_missing(self):
        with patch.object(gst.LOG, 'error') as mock_err:
            gst._report_missing_plugins(['axinplace'], [])
        messages = ' '.join(str(c) for c in mock_err.call_args_list)
        assert 'axinplace' in messages
        assert 'make' in messages

    def test_logs_ax_plugins_missing(self):
        gst._get_search_dirs_for_ax_plugins.cache_clear()
        with patch.object(gst, '_get_search_dirs_for_ax_plugins', return_value=[Path('/fake')]):
            with patch.object(gst.LOG, 'error') as mock_err:
                gst._report_missing_plugins([], ['libinplace_foo.so'])
        messages = ' '.join(str(c) for c in mock_err.call_args_list)
        assert 'libinplace_foo.so' in messages
        assert '/fake' in messages

    def test_logs_both_kinds(self):
        gst._get_search_dirs_for_ax_plugins.cache_clear()
        with patch.object(gst, '_get_search_dirs_for_ax_plugins', return_value=[]):
            with patch.object(gst.LOG, 'error') as mock_err:
                gst._report_missing_plugins(['axinplace'], ['libfoo.so'])
        messages = ' '.join(str(c) for c in mock_err.call_args_list)
        assert 'axinplace' in messages
        assert 'libfoo.so' in messages


class TestVerifyPipelinePlugins:
    def test_no_errors_when_all_present(self):
        """When nothing is missing, _report_missing_plugins should not be called."""
        with patch.object(gst, '_find_missing_gst_plugins', return_value=[]) as mock_gst:
            with patch.object(gst, '_find_missing_ax_plugins', return_value=[]) as mock_ax:
                with patch.object(gst, '_report_missing_plugins') as mock_report:
                    gst._verify_pipeline_plugins(SIMPLE_PIPELINE)
        mock_gst.assert_called_once()
        mock_ax.assert_called_once()
        mock_report.assert_not_called()

    def test_calls_report_on_missing_gst_plugin(self):
        with patch.object(gst, '_find_missing_gst_plugins', return_value=['bad_element']):
            with patch.object(gst, '_find_missing_ax_plugins', return_value=[]):
                with patch.object(gst, '_report_missing_plugins') as mock_report:
                    gst._verify_pipeline_plugins(SIMPLE_PIPELINE)
        mock_report.assert_called_once_with(['bad_element'], [])

    def test_calls_report_on_missing_ax_plugin(self):
        with patch.object(gst, '_find_missing_gst_plugins', return_value=[]):
            with patch.object(gst, '_find_missing_ax_plugins', return_value=['libmissing.so']):
                with patch.object(gst, '_report_missing_plugins') as mock_report:
                    gst._verify_pipeline_plugins(SIMPLE_PIPELINE)
        mock_report.assert_called_once_with([], ['libmissing.so'])

    def test_calls_report_on_both_missing(self):
        with patch.object(gst, '_find_missing_gst_plugins', return_value=['axinplace']):
            with patch.object(gst, '_find_missing_ax_plugins', return_value=['libx.so']):
                with patch.object(gst, '_report_missing_plugins') as mock_report:
                    gst._verify_pipeline_plugins(SIMPLE_PIPELINE)
        mock_report.assert_called_once_with(['axinplace'], ['libx.so'])

    def test_passes_correct_plugin_lists_to_finders(self):
        pipeline = [
            {'instance': 'filesrc', 'name': 'src0'},
            {'instance': 'axinplace', 'name': 'ip0', 'lib': 'libinplace_addstreamid.so'},
            {'instance': 'appsink', 'name': 'sink0'},
        ]
        with patch.object(gst, '_find_missing_gst_plugins', return_value=[]) as mock_gst:
            with patch.object(gst, '_find_missing_ax_plugins', return_value=[]) as mock_ax:
                gst._verify_pipeline_plugins(pipeline)
        mock_gst.assert_called_once_with(['appsink', 'axinplace', 'filesrc'])
        mock_ax.assert_called_once_with(['libinplace_addstreamid.so'])

    def test_empty_pipeline_no_errors(self):
        with patch.object(gst, '_find_missing_gst_plugins', return_value=[]) as mock_gst:
            with patch.object(gst, '_find_missing_ax_plugins', return_value=[]) as mock_ax:
                with patch.object(gst, '_report_missing_plugins') as mock_report:
                    gst._verify_pipeline_plugins([])
        mock_gst.assert_called_once_with([])
        mock_ax.assert_called_once_with([])
        mock_report.assert_not_called()
