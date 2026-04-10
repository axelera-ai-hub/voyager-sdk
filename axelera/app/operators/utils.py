# Copyright Axelera AI, 2023
# Utils for operators

import difflib
from pathlib import Path
import re
import tempfile

from axelera import types
from axelera.app import logging_utils, utils

from .context import PipelineContext

LOG = logging_utils.getLogger(__name__)


def _label_enum_to_int(labels: utils.FrozenIntEnumMeta, label: str) -> int:
    try:
        return str(getattr(labels, utils.ident(label)).value)
    except AttributeError:
        valid = sorted(labels.__members__.keys())
        best = difflib.get_close_matches(label, valid, n=5)
        suggestion = f", did you mean one of: {', '.join(best)}?" if best else ""
        raise ValueError(f"Label '{label}' not found in {labels.__name__}{suggestion}.") from None


def build_class_sieve(label_filter, labels):
    if not label_filter:
        return []
    if isinstance(label_filter, str):
        LOG.warning(f"Label filter string must be parsed first, ignoring: {label_filter}")
        return []
    if not hasattr(label_filter, "__getitem__"):
        raise ValueError(f"Label filter must be subscriptable, got {type(label_filter)}")
    if all(isinstance(label, int) for label in label_filter):
        return [str(label) for label in label_filter]
    if all(isinstance(label, str) for label in label_filter):
        if all(label.isdigit() for label in label_filter):
            return label_filter
        if labels is None:
            raise ValueError(f"No labels provided, cannot filter {label_filter}")
        if isinstance(labels, utils.FrozenIntEnumMeta):
            return [_label_enum_to_int(labels, label) for label in label_filter]
        else:
            return [str(labels.index(label)) for label in label_filter]
    raise ValueError("Type of label filter not supported, must be all int or all str")


def add_alpha_channel(format: str) -> str:
    if not isinstance(format, str):
        format = format.name
    if format.lower() in ['rgb', 'bgr']:
        return f'{format.lower()}a'
    elif format.lower() == 'gray':
        return 'gray8'
    return f'{format.lower()}'


def insert_color_convert(gst, format, vaapi=False, opencl=False, opencv=True):
    color_format = add_alpha_channel(format)
    if bool(opencl) is True:
        gst.axtransform(lib="libtransform_colorconvert_cl.so", options=f'format:{color_format}')
    elif bool(vaapi) is True:
        # For grayscale, use videoconvert instead of vaapipostproc
        if color_format.lower() == 'gray8':
            gst.videoconvert()
            gst.capsfilter(caps=f'video/x-raw,format={color_format}')
            gst.axinplace()
        else:
            gst.vaapipostproc(format=f'{color_format}')
            gst.videoconvert()
            gst.axinplace()
    elif bool(opencv) is True:
        gst.axtransform(lib="libtransform_colorconvert.so", options=f'format:{color_format}')
    else:
        gst.videoconvert()
        gst.capsfilter(caps=f'video/x-raw,format={color_format.upper()}')


def inspect_resize_status(context: PipelineContext):
    if context.resize_status != types.ResizeMode.ORIGINAL:
        msg = "Pipeline contains multiple resize operators, "
        msg += "this may lead to unexpected results."
        LOG.debug(msg)


def create_tmp_chars(chars):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as t:
        t.write('\n'.join(' ' if c == ' ' else (c if c != '' else 'e') for c in chars))
        return Path(t.name)


def create_tmp_labels(labels):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as t:
        try:
            t.write('\n'.join(e.name for e in labels))
        except AttributeError:
            t.write('\n'.join(labels))
        return Path(t.name)


def parse_labels_filter(filter):
    '''
    Parse various input styles for label filter/exclude lists into a consistent format.

    Valid inputs:
      - None, '', [], * -> []
      - 'label' -> ['label']
      - 'label1,label2' -> ['label1', 'label2']
      - 'label1, label2' -> ['label1', 'label2']
      - ' label1 ; label2 ' -> ['label1', 'label2']
      - ['label1', 'label2'] -> ['label1', 'label2']
      etc.

    If `filter` is a $$Variable, it is returned as is.
    '''
    if isinstance(filter, str):
        if filter == '*':
            return []
        if filter.startswith('$$'):
            return filter
        strings = [x for x in re.split(r'\s*[,;]\s*', filter.strip()) if x]
        if all(x.isdigit() for x in strings):
            return [int(x) for x in strings]
        return [x.strip() for x in strings]
    if isinstance(filter, list):
        if all(isinstance(x, int) for x in filter):
            return filter
        return [x.strip() for x in filter]
    return []


def label_exclude_to_label_filter(labels, label_exclude):
    """
    Convert a label exclude list to a label filter list.

    Args:
        labels (list | FrozenIntEnumMeta): List of labels or label enum.
        label_exclude (list): List of labels to exclude.

    Returns:
        list: List of labels that are not in the label exclude list.
    """
    if isinstance(labels, utils.FrozenIntEnumMeta):
        return [x.name for x in labels if x.name not in label_exclude]
    return [x for x in labels if x not in label_exclude]
