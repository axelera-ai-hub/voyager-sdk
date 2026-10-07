# Copyright Axelera AI, 2025

from pathlib import Path
import re

import pytest

from axelera.app import environ


def get(**kwargs):
    return environ.Environment({f'AXELERA_{k.upper()}': v for k, v in kwargs.items()})


def test_framework():
    assert get().framework == Path.cwd()
    assert get(framework='bob').framework == Path('bob')


def test_build_root():
    assert get().build_root == Path.cwd() / 'build'
    assert get(framework='bob').build_root == Path('bob/build')
    assert get(build_root='bob').build_root == Path('bob')


def test_data_root():
    assert get().data_root == Path.cwd() / 'data'
    assert get(framework='bob').data_root == Path('bob/data')
    assert get(data_root='bob').data_root == Path('bob')


def test_exported_root():
    assert get().exported_root == Path.cwd() / 'exported'
    assert get(framework='bob').exported_root == Path('bob/exported')
    assert get(exported_root='bob').exported_root == Path('bob')


def test_max_compiler_cores():
    assert get().max_compiler_cores == 1
    assert get(max_compiler_cores='2').max_compiler_cores == 2


def test_configure_board():
    assert get().configure_board == '1'
    assert get(configure_board='bob').configure_board == 'bob'


def test_videoflip():
    assert get().videoflip == ''
    assert get(videoflip='horizontal').videoflip == 'horizontal'


def test_render_low_latency_streams():
    assert get().render_low_latency_streams == []
    assert get(render_low_latency_streams='1').render_low_latency_streams == [1]
    assert get(render_low_latency_streams='1,2').render_low_latency_streams == [1, 2]


def test_render_fps():
    assert get().render_fps == 15
    assert get(render_fps='1').render_fps == 1
    assert get(render_fps='16').render_fps == 16


def test_render_scale():
    assert get().render_font_scale == 1.0
    assert get(render_font_scale='0.5').render_font_scale == 0.5
    assert get(render_font_scale='2.0').render_font_scale == 2.0


def test_render_line_width():
    assert get().render_line_width == 1
    assert get(render_line_width='1').render_line_width == 1
    assert get(render_line_width='3').render_line_width == 3


def test_render_buffer_status():
    assert get().render_show_buffer_status is False
    assert get(render_show_buffer_status='1').render_show_buffer_status is True
    assert get(render_show_buffer_status='true').render_show_buffer_status is True
    assert get(render_show_buffer_status='True').render_show_buffer_status is True


def test_render_show_fps():
    assert get().render_show_fps is False
    assert get(render_show_fps='1').render_show_fps is True


def test_render_queue_size():
    assert get().render_queue_size == 1
    assert get(render_queue_size='1').render_queue_size == 1
    assert get(render_queue_size='3').render_queue_size == 3


def test_use_dmabuf():
    assert environ.UseDmaBuf.INPUTS in get().use_dmabuf
    assert environ.UseDmaBuf.OUTPUTS in get().use_dmabuf
    assert environ.UseDmaBuf.INPUTS not in get(use_dmabuf='0').use_dmabuf
    assert environ.UseDmaBuf.OUTPUTS not in get(use_dmabuf='0').use_dmabuf
    assert environ.UseDmaBuf.INPUTS in get(use_dmabuf='1').use_dmabuf
    assert environ.UseDmaBuf.OUTPUTS not in get(use_dmabuf='1').use_dmabuf
    assert environ.UseDmaBuf.INPUTS not in get(use_dmabuf='2').use_dmabuf
    assert environ.UseDmaBuf.OUTPUTS in get(use_dmabuf='2').use_dmabuf
    assert environ.UseDmaBuf.INPUTS in get(use_dmabuf='3').use_dmabuf
    assert environ.UseDmaBuf.OUTPUTS in get(use_dmabuf='3').use_dmabuf
    with pytest.raises(ValueError):
        get(use_dmabuf='4').use_dmabuf


def test_use_double_buffer():
    assert get().use_double_buffer is True
    assert get(use_double_buffer='0').use_double_buffer is False


def test_torch_device():
    assert get().torch_device == ''
    assert get(torch_device='cuda').torch_device == 'cuda'


def test_rtsp_protocol():
    assert get().rtsp_protocol == 'all'
    assert get(rtsp_protocol='tcp').rtsp_protocol == 'tcp'
    assert get(rtsp_protocol='udp').rtsp_protocol == 'udp'
    assert get(rtsp_protocol='all').rtsp_protocol == 'all'


def test_help():
    assert get().help == False
    assert get(help='1').help == True


def test_show_help():
    got = get().show_help()
    got_names = re.findall(r'\b(AXELERA_\w+)', got)
    assert set(got_names) == set(environ.ALL_VARS)
    assert (
        '''

AXELERA_RENDER_SHOW_BUFFER_STATUS (default:0)
  Set to 1 to show the current render queue buffer on the display.

'''
        in got
    )
    assert (
        '''
AXELERA_RENDER_QUEUE_SIZE (default:1)
  Depth of the render queue buffer.

  This is the number of frames that can be queued for rendering, this helps reduce'''
        in got
    )
