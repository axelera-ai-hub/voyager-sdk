# Copyright Axelera AI, 2025
# Access environment variable configuration switches in a consistent way
from __future__ import annotations

import dataclasses
import enum
import os
from pathlib import Path
import re
import sys
import textwrap
import typing

# DO NOT IMPORT ANYTHING FROM AXELERA.APP HERE


class UseDmaBuf(enum.Flag):
    '''Bitwise flag to determine if dmabufs should be used for inputs and/or outputs.'''

    INPUTS = 1
    OUTPUTS = 2


_vars = []


@dataclasses.dataclass
class _Var:
    name: str  # name of the environment variable
    doc: str  # documentation for the environment variable
    default: str  # default value for the environment variable (as a string not the converted type)

    @property
    def env_name(self):
        '''The name of the environment variable as AXELERA_UPPER.'''
        return f'AXELERA_{self.name.upper()}'


def _get_str_converter(env_type):
    if env_type is bool:
        return lambda v: v.lower() in ('1', 'true', 'yes')
    if env_type == list[int]:
        return lambda v: [] if v == '' else [int(x) for x in v.split(',')]
    if env_type in (Path, str, int, float):
        return env_type
    if issubclass(env_type, enum.Flag):
        return lambda v: env_type(int(v))
    raise TypeError(f"Unsupported type for var : {env_type}")


def _convert_to_default(value):
    if isinstance(value, list):
        return ','.join(map(_convert_to_default, value))
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


def _var(f):
    '''Decorator to define an environment variable min the Environment class.

    The docstring is used as the AXELERA_HELP output.
    The return type annotation is used to perform the conversion (see _get_str_converter).
    The return value is used as the default and is converted to a string.

    The return value/default may refer to other configuration variables
    (eg. $AXELERA_FRAMEWORK/build) but may not refer to system environment variables.
    '''
    default = _convert_to_default(f(None))
    env_type = typing.get_type_hints(f)['return']
    converter = _get_str_converter(env_type)
    var = _Var(f.__name__, f.__doc__ or '', default)

    def getter(self):
        v = self._environ.get(var.env_name, var.default)
        # expand any locally defined variables, e.g. `$AXELERA_FRAMEWORK/build
        v = self.expandvars(v)
        return converter(v)

    _vars.append(var)
    return property(getter)


class Environment:
    '''Access global configuration values through a standardized and consistent interface.

    Variables are read from the environent on access, not at module initialisation time.
    '''

    def __init__(self, environ=os.environ):
        self._environ = environ

    def expandvars(self, value: str) -> str:
        '''Expand $AXELERA_* variables using the current environment configuration.

        Only variables matching $AXELERA_<NAME> are expanded; other shell variables
        are left unchanged. Use os.path.expandvars for general shell variable expansion.
        '''
        return re.sub(
            r'\$AXELERA_([A-Z_]+)\b', lambda m: str(getattr(self, m.group(1).lower())), value
        )

    @_var
    def help(self) -> bool:
        '''Set to true to show help and exit.'''
        return False

    @_var
    def framework(self) -> Path:
        '''Root of application framework.

        path to dir containing ax_models and ax_datasets inference.py
        Some places assume that axelera/app can also be found here but we should discourage that as
        we should rely on the package being found by importlib as standard
        '''
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    @_var
    def build_root(self) -> Path:
        '''Location to compile/deploy networks. Location of datasets directory.

        Used by deploy.py and inference.py.
        '''
        return '$AXELERA_FRAMEWORK/build'

    @_var
    def data_root(self) -> Path:
        '''Location of datasets directory. Used by deploy.py and inference.py.'''
        return '$AXELERA_FRAMEWORK/data'

    @_var
    def exported_root(self) -> Path:
        '''Location to save zip of models compiled with --export'''
        return '$AXELERA_FRAMEWORK/exported'

    @_var
    def llm_root(self) -> Path:
        '''Location of LLM directory.'''
        return '$AXELERA_FRAMEWORK/llm'

    @_var
    def max_compiler_cores(self) -> int:
        '''Determine fall back batch size

        If a model yaml does not specify the batch size, the compiler will use this value as the
        maximum batch size.

        When the max_compiler_cores of a model is 1 (the most common case), then models will always
        be compiled with a batch size of 1, and when executed on multiple cores it will be
        inferenced by creating multiple instances of the model.

        When the max_compiler_cores of a model is greater than 1, then the model will be compiled
        for min(--aipu-cores, AXELERA_MAX_COMPILER_CORES) cores.  For example if the
        max_compiler_cores for a model is 2 and the --aipu-cores is 4, then the model will be
        compiled to execute as a batch on 2 cores, but when inferencing for 4 cores then 2
        instances of the model will be created.

        In the more common case of max_compiler_cores for a model is 4, then the model will be
        compiled for that many cores, but whem inferenced it will have just a single instance
        created.
        '''
        return 1

    @_var
    def configure_board(self) -> str:
        '''Prevent or override setting of clock profile, and mvm limitation.

        0 = Do not set clock profile
        1 = (default) set the clock profile

        Or a 2 tuple, comma separated: [clock],[mvm_limitation]. Any part may be
        missing. For example ,90 will override mvm limitation but not change the core clock,
        or 400 will set core clock but not change mvm.

        clock:int : set clock profile to N.
                    N should be one of 100, 200, 400, 600, 800 today, but other values may be
                    supported in the future.
        mvm_limitation:int : set the runtime mvm limitation, as % from 1-100 (the hardware actually
                             supports ~65 increments so approx 1.5% increments are used, the
                             nearest will be selected.)
        '''
        return '1'

    @_var
    def videoflip(self) -> str:
        '''Set to a gstreamer GstVideoFlipMethod value to add a videoflip in the video stream.

        This might be used to invery a camera which has been mounted upside down. The value can be
        given as a numerical value or the lower case name below.

        ######################## #################################################
        Value                    Description
        ######################## #################################################
        (0) none                 Identity (no rotation)
        (1) clockwise            Rotate clockwise 90 degrees
        (2) rotate-180           Rotate 180 degrees
        (3) counterclockwise     Rotate counter-clockwise 90 degrees
        (4) horizontal-flip      Flip horizontally
        (5) vertical-flip        Flip vertically
        (6) upper-left-diagonal  Flip across upper left/lower right diagonal
        (7) upper-right-diagonal Flip across upper right/lower left diagonal
        (8) automatic            Select flip method based on image-orientation tag
        ######################## #################################################

        If empty no videoflip will be added.
        '''
        return ''

    @_var
    def render_fps(self) -> int:
        '''Rate at which the OpenGL window should be updated.'''
        return 15

    @_var
    def render_font_scale(self) -> float:
        '''In OpenGL rendering, scale the label size (1.0 == default, 2.0 double size).'''
        return 1.0

    @_var
    def render_line_width(self) -> int:
        '''In OpenGL rendering, control the width of the line used in boxes (linux only).'''
        return 1

    @_var
    def render_show_fps(self) -> bool:
        '''Set to 1 to show the actual redraw rate of the onscreen in the OpenGL window.'''
        return False

    @_var
    def render_queue_size(self) -> int:
        '''Depth of the render queue buffer.

        This is the number of frames that can be queued for rendering, this helps reduce jitter
        effects in live video, but it increases latency. For example a queue depth of 30 on an
        30 fps camera will result in 1s of extra latency.

        In practice the render queue is not necessary because the AxInferenceNet performs round
        robin buffer pull on its input pads and so this naturally reduces jitter.
        '''
        return 1

    @_var
    def render_keypoint_skeletons(self) -> bool:
        return 1

    @_var
    def render_keypoints(self) -> bool:
        return 1

    @_var
    def render_box_keypoints(self) -> bool:
        return 0

    @_var
    def render_low_latency_streams(self) -> list[int]:
        '''List of stream ids to not have a render queue.

        For example for usb devices on stream id 0 and 3 then set this to "0,3".

        Since the default render queue size is 1, this will not have an affect unless the render
        queue is configured to be greater than 1.
        '''
        return []

    @_var
    def render_show_buffer_status(self) -> bool:
        '''Set to 1 to show the current render queue buffer on the display.'''
        return False

    @_var
    def render_speedometers_on_saved_outputs(self) -> bool:
        '''When saving output images/videos, whether to include performance metrics (speedometers).

        If set to false, only the visualization elements configured in the render_config will be shown.
        If set to true, all visualization elements, including performance metrics, will be shown.

        Usage:
        - Set to disable: export AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS=0
        - Default: true (shows all visualization elements)

        Accepts: 1, true, yes (case-insensitive) for true; all other values for false.
        '''
        return True

    @_var
    def render_bbox_class(self) -> bool:
        '''Whether to show class labels when drawing bounding boxes.

        When enabled, bounding boxes will display the predicted class name (e.g., "duck", "car").
        When disabled, only the bounding box outline and confidence score (if enabled) are shown.

        Usage:
        - Set to disable: export AXELERA_RENDER_BBOX_CLASS=0
        - Default: true (shows class labels)

        Accepts: 1, true, yes (case-insensitive) for true; all other values for false.
        '''
        return True

    @_var
    def render_bbox_score(self) -> bool:
        '''Whether to show confidence scores when drawing bounding boxes.

        When enabled, bounding boxes will display the confidence score (e.g., "0.85", "0.92").
        When disabled, only the bounding box outline and class label (if enabled) are shown.

        Usage:
        - Set to disable: export AXELERA_RENDER_BBOX_SCORE=0
        - Default: true (shows confidence scores)

        Accepts: 1, true, yes (case-insensitive) for true; all other values for false.
        '''
        return True

    # this makes the main instance of Environment behave like a module,
    # e.g. config.env.UseDmaBuf.INPUTS
    UseDmaBuf = globals()['UseDmaBuf']

    @_var
    def use_dmabuf(self) -> UseDmaBuf:
        '''Enable dmabufs for inputs/outputs to model.

        0 - no dmabufs
        1 - inputs only
        2 - outputs only
        3 - inputs and outputs (unless batch is in use, in which case output dmabufs will not be used)
        '''
        return 3

    @_var
    def use_double_buffer(self) -> bool:
        '''Enable double buffering in the inference pipeline.

        This improves performance but at present also increases latency.
        '''
        return True

    @_var
    def use_cl_double_buffer(self) -> bool:
        '''Enable OpenCL double buffering in the inference pipeline.

        Enabling double buffering in the OpenCL pipeline will use a second buffer to increase
        utilisation of the GPU.

        This improves performance but at present also increases latency. It also complicates
        the interpretation of measurement of per-element measued using --show-stas, so when
        inspecting --show-stats it is recommended to disable this option.
        '''
        return True

    @_var
    def torch_device(self) -> str:
        '''The device to use for torch operations.

        Can be 'cuda', 'cpu', 'mps' depending on what backends are available on the platform.
        '''
        return ''

    @_var
    def s3_available(self) -> str:
        # set to 1 or subset to enable internal s3 access for axelera developers, there is no
        # docstring here because users cannot make of this feature.s
        return '0'

    @_var
    def inference_mock(self) -> str:
        '''Replace the Metis based inference with a mock accelerator.

        With `save-<directory>`, a Metis device must be connected and the inference will be run and
        each set of inputs to the model will have a checksum and outputs will be saved to a
        directory along with a shapes.txt describing the shapes of the input and output tensors.

        Note: output dmabufs are not currently supported, so you must disable them.
        Note: the saved output files can be very large, 100 output tensors for yolov5 are for
              example approximately 2GiB.

        In loading mode, the inference pipeline will read frames from the file specified by the
        path.  Matching checksums will be used to select specific output tensors.  If no matching
        checksum is found an all zeros tensor will be used.

        It is also possible to 'load' mock inference data without saving it, but in this case you
        must specify the shapes of the input and output tensors by creating a file <dir>/shapes.txt.
        For example:

        $ export AXELERA_USE_DMABUF=1
        $ mkdir tdata
        $ echo 1x644x656x4,1x40x40x256,1x20x20x256,1x80x80x256 > tdata/shapes.txt
        $ AXELERA_INFERENCE_MOCK=load-tdata ./inference.py yolov5s-v7-coco dataset --frames=100

        In this case we must tell the mock object what shapes the inputs amd outputs are. The
        inference will run returning all zeros for the outputs.

        If zero data is not suitable, you can collect output data using the save option:

        $ export AXELERA_USE_DMABUF=1
        $ mkdir tdata
        $ AXELERA_INFERENCE_MOCK=save-tdata ./inference.py yolov5s-v7-coco dataset --frames=100
        $ AXELERA_INFERENCE_MOCK=load-tdata ./inference.py yolov5s-v7-coco dataset --frames=100

        The load command can also be used to mock the Metis at a specific frame rate (default: 500fps/core)

        $ AXELERA_INFERENCE_MOCK=load-tdata@100 ./inference.py yolov5s-v7-coco dataset --frames=100

        Will run the inference at approximately 100fps per core.
        '''
        return ''

    @_var
    def opengl_backend(self) -> str:
        '''The version of the OpenGL backend to use. Given in the format "[gl/gles],major,minor",

        For example, "gl,4,5" for OpenGL 4.5 or "gles,3,2" for OpenGL ES 3.2.
        Defaults to "gl,3,3" for OpenGL 3.3.
        '''
        return 'gl,3,3'

    @_var
    def rtsp_protocol(self) -> str:
        '''The network protocol to use for RTSP streams.

        Can be 'tcp', 'udp', or 'all'.
        With 'all', it enables udp-mcast+udp+tcp protocols.
        '''
        return 'all'

    def show_help(self) -> str:
        lines = [
            'NOTE: the following environment variables are subject to change and are for advanced',
            'usage to override defaults. They are not intended for general use.',
            '',
        ]
        for var in _vars:
            if var.doc:
                doclines = var.doc.splitlines()
                if len(doclines) > 1:
                    dedented = textwrap.dedent('\n'.join(doclines[1:]))
                    doc = doclines[0] + '\n' + textwrap.indent(dedented, '  ').rstrip()
                else:
                    doc = doclines[0]

                lines.extend([f"{var.env_name} (default:{var.default or '(empty)'})\n  {doc}", ''])

        return '\n'.join(lines)


Environment.DEFAULTS = Environment({})

ALL_VARS = [v.env_name for v in _vars if v.doc]
'''List of all environment variables.'''

env = Environment()
if env.help:
    print(env.show_help())
    sys.exit(1)
