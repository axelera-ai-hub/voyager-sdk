# Copyright Axelera AI, 2024
import numpy as np
import pytest

from axelera import types
from axelera.app import display, display_cv, inf_tracers


def test_null_app_with_data():
    with display.App(visible=False) as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        def t():
            wnd.show(i, meta)
            wnd.close()

        app.start_thread(t)
        app.run()
        assert wnd.is_closed == True


def test_wrong_type_image():
    with display.App(visible=False) as app:
        wnd = app.create_window('test', (640, 480))
        i = np.zeros((480, 640, 3), np.uint8)
        meta = object()

        def t():
            with pytest.raises(TypeError, match='Expected axelera.types.Image'):
                wnd.show(i, meta)
            wnd.close()

        app.start_thread(t)
        app.run()
        assert wnd.is_closed == True


class MockApp(display.App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.received = []

    def _create_new_window(self, q, title, size):
        return title

    def _destroy_all_windows(self):
        pass

    def _run(self, interval=1 / 30):
        while 1:
            self._create_new_windows()
            new = display_cv._read_new_data(self._wnds, self._queues)
            if new is display.SHUTDOWN:
                return
            self.received.extend(new)
            if self.has_thread_completed:
                return


def test_mock_window_creation():
    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        app.start_thread(wnd.close)
        app.run()
        assert wnd.is_closed == True
    assert app.received == []


def test_mock_window_creation_with_data():
    with MockApp('other') as app:
        wnd = app.create_window('test', (640, 480))
        i = types.Image.fromarray(np.zeros((480, 640, 3), np.uint8))
        meta = object()

        def t():
            wnd.show(i, meta)

        app.start_thread(t)
        app.run()
        assert wnd.is_closed == True
    assert app.received == [('test', display._Frame(0, i, meta))]


def test_speedometer_metrics():
    m = display.SpeedometerMetrics((1000, 2000), 0)
    assert m.top_left == (50, 50)
    assert m.radius == 100
    assert m.needle_radius == 80
    assert m.center == (150, 150)
    assert m.diameter == 200
    assert m.text_offset == 40
    assert m.text_size == 28


@pytest.mark.parametrize(
    'metric, text, needle_pos',
    [
        (inf_tracers.TraceMetric('k', 'title', 0.0, 99.0), '  0.0', 90 + 45),
        (inf_tracers.TraceMetric('k', 'title', 10.0, 100.0), '  10', 162.0),
        (inf_tracers.TraceMetric('k', 'title', 99.0, 110.0), '  99', 18.0),
        (inf_tracers.TraceMetric('k', 'title', 100.1, 110.0), ' 100', 20.7),
        (inf_tracers.TraceMetric('k', 'title', 10.0, 100.0, '%'), '  10%', 162.0),
        (inf_tracers.TraceMetric('k', 'title', 25.0, 50.0, '%'), ' 25.0%', 270.0),
        (inf_tracers.TraceMetric('k', 'title', 99.0, 100.0, '%'), '  99%', 42.3),
    ],
)
def test_text_and_needle(metric, text, needle_pos):
    assert text == display.calculate_speedometer_text(metric)
    assert needle_pos == round(display.calculate_speedometer_needle_pos(metric), 1)


def test_meta_cache():
    mc = display.MetaCache()
    meta0_0 = {'a': 10, 'b': 20, '__fps__': 30}
    meta1_0 = {'a': 11, 'b': 21}
    meta0_1 = {'__fps__': 31}
    meta0_2 = {'a': 110, 'b': 120, '__fps__': 32}
    meta1_2 = {'a': 111, 'b': 121}
    meta0_3 = {'__fps__': 33}
    assert (False, meta0_0) == mc.get(0, meta0_0)
    assert (False, meta1_0) == mc.get(1, meta1_0)
    assert (True, {'a': 10, 'b': 20, '__fps__': 31}) == mc.get(0, meta0_1)
    assert (True, {'a': 11, 'b': 21}) == mc.get(1, None)
    assert (False, {'a': 110, 'b': 120, '__fps__': 32}) == mc.get(0, meta0_2)
    assert (False, {'a': 111, 'b': 121}) == mc.get(1, meta1_2)
    assert (True, {'a': 110, 'b': 120, '__fps__': 33}) == mc.get(0, meta0_3)
    assert (True, {'a': 111, 'b': 121}) == mc.get(1, None)


class A:
    pass


@pytest.mark.parametrize(
    'prop,value,exp,got',
    [
        ('title', 0, 'str', 'int'),
        ('grayscale', '0', 'float | int | bool', 'str'),
        ('grayscale', A(), 'float | int | bool', 'A'),
        ('bbox_label_format', 0, 'str', 'int'),
    ],
)
def test_options_invalid_type(caplog, prop, value, exp, got):
    opts = display.Options()
    opts.update(**{prop: value})
    assert f'Expected {exp} for Options.{prop}, but got {got}' in caplog.text


@pytest.mark.parametrize(
    'props,err',
    [
        (['titel'], 'Options.titel'),
        (['badger', 'mushroom'], 'Options.badger, mushroom'),
    ],
)
def test_options_invalid_prop(caplog, props, err):
    with caplog.at_level('INFO'):
        opts = display.Options()
        opts.update(**dict.fromkeys(props, 0))
        s = 's' if ',' in err else ''
        assert f'Unsupported option{s} : {err}' in caplog.text


@pytest.mark.parametrize(
    'prop,value,got',
    [
        ('title', 'test', 'test'),
        ('grayscale', 0.5, 0.5),
        ('grayscale', 0, 0.0),
        ('grayscale', True, 1.0),
        ('bbox_label_format', 'test', 'test'),
        ('bbox_label_format', '', ''),
    ],
)
def test_options_valid(prop, value, got):
    opts = display.Options()
    opts.update(**{prop: value})
    assert getattr(opts, prop) == got
