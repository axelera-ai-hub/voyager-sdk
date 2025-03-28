#!/usr/bin/env python
# Copyright Axelera AI, 2024

import collections
import os
import pathlib
import re
import tempfile
from typing import List, Type

from . import inf_tracers, logging_utils


class Colouring:
    BOLD = ""
    ITALIC = ""
    UNDERLINE = ""
    INVERSE = ""
    RESET = ""


class Ansi(Colouring):
    BOLD = "\x1b[1m"
    ITALIC = "\x1b[3m"
    UNDERLINE = "\x1b[4m"
    INVERSE = "\x1b[7m"
    RESET = "\x1b[0m"


class Plain(Colouring):
    pass


_ANGLE = "└─"
_ELEMENT = re.compile(r'.*element=\(string\)([^,]+),.*time=\([^)]+\)(\d+).*')
_SPACES = re.compile(f'([\\s{_ANGLE}]*)(.*?)(\\s*)$')

_contains = ['queue']
_starts = ['tensor-aggregator', 'agg', 'qagg', 'avdec_', 'inference-funnel']
LOG = logging_utils.getLogger(__name__)


def _tabulate(element_times, end_to_end_fps, colouring):
    def wrap(s, fmt):
        if fmt:
            if m := _SPACES.match(s):
                return f'{m.group(1)}{fmt}{m.group(2)}{colouring.RESET}{m.group(3)}'
        return s

    def format_row(element, us, fps, fmt=''):
        if len(element) > max_widths[0]:
            return wrap(element, fmt) + '\n' + format_row('', us, fps, fmt)
        return (' ' * gap).join(
            [
                wrap(f'{element:{max_widths[0]}}', fmt),
                wrap(f'{us:>{max_widths[1]}}', fmt),
                wrap(f'{fps:>{max_widths[2]}}', fmt),
            ]
        )

    name_width = 45
    gap = 3
    line_char = '='

    titles = ["Element", "Latency(us)", "Effective FPS"]
    elements = [
        [
            el,
            f'{int(x[0]):,}',
            x[1] if isinstance(x[1], str) else f'{round(x[1],1):,}',
            x[2] if len(x) > 2 else '',
        ]
        for el, x in element_times
    ]
    end = ["End-to-end average measurement", "", f'{round(end_to_end_fps,1):,}', colouring.BOLD]
    rows = [titles] + elements + [end]
    max_widths = [len(max(col, key=len)) for col in zip(*rows)]
    max_widths[0] = name_width
    line = line_char * (sum(max_widths) + gap * (len(max_widths) - 1))
    lines = [line, format_row(*rows[0]), line]
    lines += [format_row(*row) for row in rows[1:-1]]
    lines += [line, format_row(*rows[-1]), line]
    return '\n'.join(lines)


def _skip_element(element, new_inference):
    for contains in _contains:
        if contains in element:
            return True
    for start in _starts:
        if element.startswith(start):
            return True
    if new_inference and element.startswith('inference-task') and ':' not in element:
        return True
    return False


def _element_batch_size(element):
    # really to answer this properly we need to know if it is batch or not.
    m = re.match(r'inference-task\d+-batch(\d+)', element)
    return int(m.group(1)) if m else 1


def _find_aipu_element(elements, prefix="inference-"):
    """
    Find the index of the first element that starts with the specified prefix.
    """

    for ix, name in enumerate(elements):
        if name.endswith(':inference'):
            return ix
    for ix, name in enumerate(elements):
        if name.startswith(prefix) and ':' not in name:
            return ix
    raise ValueError(f"No element starting with '{prefix}' found.")


def initialise_logging():
    '''Configure gstreamer to log timing information to a temporary file.

    returns (tempfile, path)
    '''
    log_file = tempfile.NamedTemporaryFile(mode='w')
    os.environ['GST_DEBUG'] = 'GST_TRACER:7'
    os.environ['GST_TRACERS'] = 'latency(flags=element)'
    os.environ['GST_DEBUG_FILE'] = log_file.name
    return log_file, pathlib.Path(log_file.name)


def determine_element_fps(element, latency_us):
    '''Determine the effective FPS of an element given its latency.

    element: name of the element
    latency_us: latency in microseconds
    '''
    if element.endswith('latency'):
        return 'n/a'
    return inf_tracers.fps_from_latency(latency_us)


def is_using_new_inference(elements):
    for element, _ in elements:
        if element.endswith(':inference'):
            return True
    return False


def format_table(
    log_file_path: pathlib.Path,
    tracers: List[inf_tracers.Tracer],
    colouring: Type[Colouring] = Ansi,
):
    '''Produce a table of statistics from the gstreamer event logs.

    log_file: path to gstreamer log file, return
    tracers: list of tracers used to collect metrics
    colouring: an object with attributes for formatting the output
    '''
    text = log_file_path.read_text()
    matches = re.findall(_ELEMENT, text)
    element_times = collections.defaultdict(list)
    new_inference = is_using_new_inference(matches)
    for element, time in matches:
        if not _skip_element(element, new_inference):
            element_times[element].append(int(time))
    order = [element for element in element_times.keys()]
    element_times = {element: times[2:] for element, times in element_times.items()}
    element_times = {
        element: sum(times) / len(times) / 1000
        for element, times in element_times.items()
        if times
    }
    try:
        tracer_insert_pos_ix = _find_aipu_element(order) + 1
    except ValueError:
        tracer_insert_pos_ix = len(order)
    for k in element_times:
        element_times[k] /= _element_batch_size(k)
    formattings = {}
    tracers_by_title = {t.title: t for t in tracers}
    for tracer_title in ('Host', 'Metis'):
        if tracer := tracers_by_title.get(tracer_title):
            metrics = tracer.get_metrics()  # always returns one (!)
            if not metrics or not metrics[0].value:
                LOG.warning(f"Unable to determine {tracer_title} metrics")
                continue
            filter_breakdown = f' {_ANGLE} {tracer_title}'
            element_times[filter_breakdown] = 1e6 / metrics[0].value
            formattings[filter_breakdown] = colouring.BOLD + colouring.INVERSE
            order.insert(tracer_insert_pos_ix, filter_breakdown)

    element_times = {
        element: [us, determine_element_fps(element, us)] for element, us in element_times.items()
    }
    for k, v in formattings.items():
        element_times[k].append(v)
    ordered = [(element, element_times.get(element, (0.0, 0.0))) for element in order]
    e2e = 0.0
    if e2e_tracer := tracers_by_title.get('End-to-end'):
        e2e = e2e_tracer.get_metrics()[0].value
    return _tabulate(ordered, e2e, colouring)
