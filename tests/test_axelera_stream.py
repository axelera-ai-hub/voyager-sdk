# Copyright Axelera AI, 2023
import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest

from axelera.app import config, stream
from axelera.app.pipe import FrameEvent, FrameResult, PipeManager


@pytest.fixture
def mock_pipe_mgr():
    mock_pipe_mgr = Mock(spec=PipeManager)
    mock_pipe_mgr.hardware_caps = Mock()
    mock_pipe_mgr.number_of_frames = 100
    mock_pipe_mgr.is_single_image.return_value = False
    mock_pipe_mgr.evaluator = None
    mock_pipe_mgr.eval_mode = False
    mock_pipe_mgr.sources = {0: 'rtsp://localhost:8554/test'}
    return mock_pipe_mgr


MOCK_EVENT = FrameEvent.from_result(FrameResult(None, [], {'some': 'results'}))


def _create_inference_stream(mock_pipe_mgr, **kwargs):
    with patch.object(stream, '_create_manager', return_value=mock_pipe_mgr):
        return stream.create_inference_stream(
            network='wibble',
            sources=[config.Source('rtsp://wobble')],
            **kwargs,
        )


@pytest.fixture
def inference_stream(mock_pipe_mgr):
    return _create_inference_stream(mock_pipe_mgr)


def test_inference_stream_initialization(mock_pipe_mgr):
    s = _create_inference_stream(mock_pipe_mgr)
    assert s.timeout == 5
    assert s.manager is mock_pipe_mgr
    mock_pipe_mgr.setup_callback.assert_called_once()


# Test cases start here
@pytest.mark.parametrize(
    'stream_frames, requested_frames, exp_len',
    [
        (0, 10, 10),
        (10, 10, 10),
        (20, 10, 10),
        (0, 0, 0),
        (10, 0, 10),
    ],
)
def test_inference_stream_len(mock_pipe_mgr, stream_frames, requested_frames, exp_len):
    mock_pipe_mgr.number_of_frames = stream_frames
    s = _create_inference_stream(mock_pipe_mgr, frames=requested_frames)
    assert len(s) == exp_len


def test_inference_stream_one_result(inference_stream, mock_pipe_mgr):
    inference_stream._feed_result(mock_pipe_mgr, MOCK_EVENT)
    inference_stream.stop()
    assert list(inference_stream) == []


def test_inference_stream_one_result_with_multiple_stops(inference_stream, mock_pipe_mgr):
    inference_stream._feed_result(mock_pipe_mgr, MOCK_EVENT)
    inference_stream.stop()
    inference_stream.stop()
    assert list(inference_stream) == []


def test_inference_stream_no_results(inference_stream):
    inference_stream.stop()
    assert list(inference_stream) == []


def test_inference_stream_timeout(mock_pipe_mgr):
    inference_stream = _create_inference_stream(mock_pipe_mgr, timeout=0.1)  # 0.1 second timeout
    # Put item in queue after 0.2s
    threading.Timer(0.2, inference_stream._feed_result, (mock_pipe_mgr, MOCK_EVENT)).start()
    threading.Timer(0.4, inference_stream.stop, ()).start()
    with pytest.raises(RuntimeError, match='Timeout for querying an inference'):
        list(inference_stream)


def test_inference_stream_is_single_image(inference_stream, mock_pipe_mgr):
    assert inference_stream.is_single_image() == mock_pipe_mgr.is_single_image()


def test_inference_stream_collects_evaluation(inference_stream, mock_pipe_mgr):
    frame_result = FrameResult(np.zeros((16, 10, 3), np.uint8), None, 'prediction_metadata')
    event = FrameEvent.from_result(frame_result)
    eop = FrameEvent.from_end_of_pipeline(0, 'End of pipeline')
    mock_pipe_mgr.evaluator = Mock()
    inference_stream._feed_result(mock_pipe_mgr, event)
    inference_stream._feed_result(mock_pipe_mgr, eop)
    assert [frame_result] == list(inference_stream)
    mock_pipe_mgr.evaluator.append_new_sample.assert_called_with(frame_result.meta)


@patch('io.StringIO')
@patch('axelera.app.stream.LOG')
def test_inference_stream_report_summary(
    mock_log, mock_string_io, inference_stream, mock_pipe_mgr
):
    mock_log.info = Mock()
    mock_pipe_mgr.evaluator = Mock()
    inference_stream._pipelines = [mock_pipe_mgr]
    mock_output = mock_string_io.return_value
    mock_output.getvalue.return_value = 'line1\nline2\nline3\n'
    inference_stream._report_summary(mock_pipe_mgr.evaluator)

    mock_pipe_mgr.evaluator.write_metrics.assert_called_once_with(mock_output)
    mock_log.info.assert_called_once_with('line1\nline2\nline3')


def test_interrupt_handler_exits_loop(mock_pipe_mgr, monkeypatch):
    s = _create_inference_stream(mock_pipe_mgr)
    s.stop()

    # Start the stream in a separate thread to simulate real operation
    thread = threading.Thread(target=lambda: list(s))
    thread.start()
    thread.join(timeout=1)

    assert s._interrupt_raised is True
    # Ensure the thread has exited, indicating the loop has been exited
    assert not thread.is_alive()
