# Copyright Axelera AI, 2024

import os
import unittest.mock as mock
from unittest.mock import patch

import pytest

from axelera.app import config
from axelera.app.pipe import manager


def test_create_manager_from_model_name():
    with patch.dict(os.environ, {'AXELERA_FRAMEWORK': "."}):
        network_name = "mc-yolov5s-v7-coco"
        expected_path = "ax_models/model_cards/yolo/object_detection/yolov5s-v7-coco.yaml"
        output = manager._get_real_path_if_path_is_model_name(network_name)
        assert output == expected_path


def test_task_render_proxy():
    """Test the TaskProxy class which allows task-specific render settings."""
    mock_task = mock.MagicMock()
    mock_task.name = 'detections'

    mock_pipeline = mock.MagicMock()
    mock_pipeline.nn.tasks = [mock_task]

    # PipeOutput owns the render config and exposes delegation methods
    render_config = config.RenderConfig(detections=config.TaskRenderConfig())
    mock_pipeout = mock.MagicMock()
    mock_pipeout.get_render_config.return_value = render_config
    mock_pipeout.set_task_render.side_effect = (
        lambda task_name, show_annotations, show_labels: render_config.set_task(
            task_name, show_annotations, show_labels
        )
    )

    pipe_manager = mock.MagicMock()
    pipe_manager._pipeline = mock_pipeline
    pipe_manager.pipeout = mock_pipeout

    proxy = manager.TaskProxy(pipe_manager, 'detections')
    result = proxy.set_render(show_annotations=False, show_labels=False)

    mock_pipeout.set_task_render.assert_called_once_with('detections', False, False)

    task_settings = render_config.get('detections')
    assert task_settings is not None
    assert task_settings.show_annotations is False
    assert task_settings.show_labels is False
    assert result is proxy


def test_pipe_manager_get_attribute_for_task():
    """Test that PipeManager's __getattr__ returns a TaskProxy for tasks."""
    mock_pipeline = mock.MagicMock()
    mock_task = mock.MagicMock()
    mock_task.name = 'detections'
    mock_pipeline.nn.tasks = [mock_task]

    class TestPipeManager(manager.PipeManager):
        def __init__(self):
            self._pipeline = mock_pipeline
            self._pipeout = mock.MagicMock()

    pipe_mgr = TestPipeManager()
    task_proxy = pipe_mgr.detections

    assert isinstance(task_proxy, manager.TaskProxy)
    assert task_proxy.task_name == 'detections'


def test_pipe_manager_set_render():
    """Test PipeManager's set_render method affects all tasks."""
    mock_pipeline = mock.MagicMock()
    mock_task1 = mock.MagicMock()
    mock_task1.name = 'task1'
    mock_task2 = mock.MagicMock()
    mock_task2.name = 'task2'
    mock_pipeline.nn.tasks = [mock_task1, mock_task2]

    # PipeOutput owns the render config and exposes delegation methods
    render_config = config.RenderConfig(
        task1=config.TaskRenderConfig(),
        task2=config.TaskRenderConfig(),
    )
    mock_pipeout = mock.MagicMock()
    mock_pipeout.get_render_config.return_value = render_config
    mock_pipeout.set_task_render.side_effect = (
        lambda task_name, show_annotations, show_labels: render_config.set_task(
            task_name, show_annotations, show_labels
        )
    )

    class TestPipeManager(manager.PipeManager):
        def __init__(self):
            self._pipeline = mock_pipeline
            self._pipeout = mock_pipeout

    pipe_mgr = TestPipeManager()
    pipe_mgr.set_render(show_annotations=False, show_labels=False)

    assert mock_pipeout.set_task_render.call_count == 2

    result_render_config = pipe_mgr.pipeout.get_render_config()
    for task_name in ['task1', 'task2']:
        task_settings = result_render_config.get(task_name)
        assert task_settings is not None
        assert task_settings.show_annotations is False
        assert task_settings.show_labels is False


def test_set_render_config_extra_keys():
    class DummyTask:
        def __init__(self, name):
            self.name = name
            self.task_render_config = config.TaskRenderConfig()

    nn = type('NN', (), {})()
    nn.tasks = [DummyTask('a'), DummyTask('b')]
    render_config = config.RenderConfig(
        a=config.TaskRenderConfig(), b=config.TaskRenderConfig(), c=config.TaskRenderConfig()
    )
    with pytest.raises(ValueError, match=r'render_config.*not in nn.tasks'):
        manager._set_render_config(nn, render_config)


def test_set_render_config_missing_keys():
    class DummyTask:
        def __init__(self, name):
            self.name = name
            self.task_render_config = config.TaskRenderConfig()

    nn = type('NN', (), {})()
    nn.tasks = [DummyTask('a'), DummyTask('b')]
    render_config = config.RenderConfig(a=config.TaskRenderConfig())
    result = manager._set_render_config(nn, render_config)
    assert 'b' in result._config
    assert isinstance(result._config['b'], config.TaskRenderConfig)


def test_set_render_config_none():
    class DummyTask:
        def __init__(self, name):
            self.name = name

    nn = type('NN', (), {})()
    nn.tasks = [DummyTask('a'), DummyTask('b')]
    # Use keyword-based initialization to pre-register tasks
    render_config = config.RenderConfig(a=config.TaskRenderConfig(), b=config.TaskRenderConfig())
    result = manager._set_render_config(nn, render_config)
    assert set(result._config.keys()) == {'a', 'b'}
