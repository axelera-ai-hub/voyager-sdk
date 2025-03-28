# Copyright Axelera AI, 2024
# Utils for operators

from pathlib import Path
import tempfile

from axelera import types
from axelera.app import logging_utils

from .context import PipelineContext

LOG = logging_utils.getLogger(__name__)


def build_class_sieve(label_filter, labels_file):
    if isinstance(label_filter, str):
        # We need to handle the where we have a label sentinel
        return []
    try:
        with open(labels_file) as f:
            labels = [x.strip() for x in f.readlines()]

        sieve = []
        for x in label_filter:
            for i, label in enumerate(labels):
                if x == label and not str(i) in sieve:
                    sieve.append(str(i))
                    break
            else:
                LOG.warning(
                    f'Attempting to filter label "{x}" which does not exist the labels file'
                )
        return sieve
    except FileNotFoundError:
        LOG.warning(f'Attempting to filter labels from file "{labels_file}" which does not exist')
        return []


def insert_color_convert(gst, vaapi, opencl, format, **kwargs):
    if bool(opencl) is True:
        gst.axtransform(
            lib="libtransform_colorconvert.so", options=f'format:{format.lower()}', **kwargs
        )
    elif bool(vaapi) is True:
        gst.vaapipostproc(format=format.lower())
        gst.axinplace(**kwargs)
    else:
        gst.videoconvert()
        gst.capsfilter(caps=f'video/x-raw,format={format.upper()}', **kwargs)


def inspect_resize_status(context: PipelineContext):
    if context.resize_status != types.ResizeMode.ORIGINAL:
        msg = f"Please ensure that there is only one Resize or Letterbox operator in the "
        msg += f"pipeline. The current resize status is {context.resize_status} already."
        raise ValueError(msg)


def create_tmp_labels(labels):
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as t:
        try:
            t.write('\n'.join(e.name for e in labels))
        except AttributeError:
            t.write('\n'.join(labels))
        return Path(t.name)
