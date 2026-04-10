#!/usr/bin/env python
# Copyright Axelera AI, 2024

'''
This example runs all 4 streams and then processes a command list to add, remove, pause, and
unpause streams.  It shows how the pipeline can be modified to add and remove streams whislt the
pipeline is running.
'''

import itertools
import sys
import threading
import time

from axelera.app import config, inf_tracers, logging_utils
from axelera.app.display import App
from axelera.app.stream import create_inference_stream

LOG = logging_utils.getLogger(__name__)
NETWORK = 'yolov8s-coco'

# format: off
commands = [
    ('sleep', 5),  #  PLAY      PAUSED   REMOVED
    ('remove', 0),  #  1,2,3                0
    ('remove', 1),  #  2,3                 0,1
    ('pause', 2),  #  3         2         0,1
    ('sleep', 5),  #
    # ('pause', 3),  #            2,3       0,1
    ('sleep', 5),  #
    ('add', 0),  #  0         2       1
    ('pause', 3),  # 0           2,3       1
    ('sleep', 5),  #
    ('add', 1),  #  0,1       2,3
    ('resume', 2),  #  0,1,2     3
    ('sleep', 5),  #
    ('pause', 0),  #  1,2       0,3
    ('pause', 1),  #  2         0,1,3
    ('sleep', 5),  #
    ('resume', 0),  #  0,2       1,3
    ('sleep', 5),  #
    ('resume', 3),  #  0,2,3     1
    ('resume', 1),  #  0,1,2,3
    ('sleep', 5),  #
    ('remove', 3),  #  0,1,2                 3
    ('sleep', 5),  #
    ('remove', 2),  #  0,1                 2,3
    ('sleep', 5),  #
    ('remove', 1),  #  0                 1,2,3
    ('add', 1),  #  0,1       2,3
    ('add', 2),  #  0,1,2     3
    ('add', 3),  #  0,1,2,3
    ('check',),  # all streams enabled and present again, so we can start at the beginning
]
# format: on


def _short_source(source):
    loc = str(source.location)
    return loc[:50] + '...' if len(loc) > 50 else loc


def control_func(window, stream):
    stopped = {}
    for stream_id, source in stream.sources.items():
        window.options(stream_id, title=f"{_short_source(source)} : PLAYING")
    for cmdn, (cmd, *args) in zip(itertools.count(), itertools.cycle(commands)):
        if cmd == 'sleep':
            time.sleep(*args)
            continue
        LOG.debug(f"#{cmdn} Command: {cmd} {args}")
        if cmd in ('pause', 'resume'):
            playing_streams = set(stream.get_stream_select())
            (stream_id,) = args
            source = stream.sources[stream_id]
            if cmd == 'pause' and stream_id in playing_streams:
                playing_streams.remove(stream_id)
            elif cmd == 'resume' and stream_id not in playing_streams:
                playing_streams.add(stream_id)
            else:
                state = 'playing' if stream_id in playing_streams else 'paused'
                LOG.warning(f"Stream {stream_id} : {source} is already {state}")
                continue
            stream.stream_select(playing_streams)
            state = 'playing' if stream_id in playing_streams else 'paused'
        elif cmd in ('add', 'remove'):
            (stream_id,) = args
            existing_source = stream.sources.get(stream_id)
            if cmd == 'add':
                if existing_source is not None:
                    LOG.warning(f"Stream {stream_id} is already playing as {existing_source}")
                    continue
                source = stopped.pop(stream_id)
                stream.add_source(source, stream_id)
                state = 'added'
                window.open_source(stream_id)
            else:
                stream.remove_source(stream_id)
                stopped[stream_id] = existing_source
                state = 'removed'
                window.close_source(stream_id, reopen=False)
        elif cmd == 'check':
            if set(stream.get_stream_select()) == set(stream.sources.keys()):
                LOG.debug("All streams are present and playing")
            else:
                LOG.warning("The command list has not restored the starting state")
            continue
        else:
            LOG.error(f"Unknown command {cmd}")
            continue
        window.options(stream_id, title=f"{_short_source(source)} : {state.upper()}")
        play = set(stream.get_stream_select())
        pause = set(stream.sources.keys()) - play
        fmt = lambda streams: ' '.join(str(i) for i in sorted(streams)).ljust(10)
        desc = f"#{cmdn:<4d}: {cmd} {stream_id}"
        LOG.info(f"{desc:<20s} Playing {fmt(play)} Paused {fmt(pause)} Removed {fmt(stopped)}")


def main(window, stream):
    control = threading.Thread(
        target=control_func, args=(window, stream), name="ControlThread", daemon=True
    )
    control.start()
    for frame_result in stream:
        if frame_result.image:
            window.show(frame_result.image, frame_result.meta, frame_result.stream_id)

        if window.is_closed:
            break


if __name__ == '__main__':
    parser = config.create_inference_argparser(
        default_network=NETWORK, description='Perform inference on an Axelera platform'
    )
    args = parser.parse_args()
    if args.pipe != 'gst':
        sys.exit("Only gst pipe is supported for this example")
    tracers = inf_tracers.create_tracers('core_temp', 'end_to_end_fps', pipe_type=args.pipe)
    if args.sources == ['rtsp']:
        # For simplicity allow sources to be just `rtsp` and use a local rtsp server
        args.sources = [f'rtsp://127.0.0.1:8554/{n}' for n in range(4)]
    if len(args.sources) < 4:
        sys.exit("At least 4 sources are required for this example")
    stream = create_inference_stream(
        network=args.network,
        sources=args.sources,
        pipe_type=args.pipe,
        log_level=logging_utils.get_config_from_args(args).console_level,
        hardware_caps=config.HardwareCaps.from_parsed_args(args),
        tracers=tracers,
        specified_frame_rate=30,
        device_selector=args.devices,
        aipu_cores=args.aipu_cores,
    )
    try:
        with App(renderer=args.display, opengl=stream.hardware_caps.opengl) as app:
            wnd = app.create_window("Stream select demo", args.window_size)
            app.start_thread(main, (wnd, stream), name='InferenceThread')
            app.run()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        stream.stop()
