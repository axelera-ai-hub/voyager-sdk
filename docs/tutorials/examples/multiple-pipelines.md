---
title: "Multiple Pipelines with Dynamic Switching"
---
# Multiple Pipelines with Dynamic Switching

This example demonstrates how to run multiple inference pipelines simultaneously and dynamically swap them at runtime. It starts four pipelines with different YOLOv8 variants (detection, pose estimation, segmentation) on RTSP sources, then periodically removes and re-adds pipelines with different networks, showing hot-swapping of models without restarting the application.

Use this for multi-camera deployments where you need to change models on the fly, or for testing different network configurations against the same video sources.

## What you'll learn

- How to create a shared inference stream and add multiple pipelines dynamically
- How to remove pipelines by index or by reference at runtime
- How to cycle through different network configurations on the same source
- How to use argparse integration for hardware capabilities, display, and logging
- How to work with performance tracers across multiple pipelines

## Prerequisites

- Voyager SDK installed and activated
- Four RTSP video sources running at `rtsp://127.0.0.1:8554/0` through `/3`
- GStreamer pipeline support (`pipe_type='gst'`)

## Source

> [!TIP]
> **Download**
> This example is included in the SDK at `examples/multiple_pipelines.py`.


```python
#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Extended app with additional config, showing advanced usage of metadata
import argparse
import itertools
import time

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

SWITCH_INTERVAL = 5  # seconds
LOG = logging_utils.getLogger(__name__)

parser = argparse.ArgumentParser(description="Multiple pipelines demo")
config.HardwareCaps.add_to_argparser(parser)
config.add_display_arguments(parser)
logging_utils.add_logging_args(parser)
args = parser.parse_args()

hwcaps = config.HardwareCaps.from_parsed_args(args)
framework = config.env.framework
tracers = inf_tracers.create_tracers('end_to_end_fps', 'latency')

stream = create_inference_stream(
    logging_config=logging_utils.get_config_from_args(args),
    hardware_caps=hwcaps,
)

networks = ['yolov8n-coco', 'yolov8npose-coco', 'yolov8nseg-coco', 'yolov8s-coco']
sources = [f'rtsp://127.0.0.1:8554/{n}' for n in range(4)]
_networks = itertools.cycle(networks)


def add_new_pipeline(stream, network, new_location):
    return stream.add_pipeline(
        network=network,
        sources=[new_location],
        pipe_type='gst',
        allow_hardware_codec=False,
        tracers=tracers,
        rtsp_latency=50,
    )


def remove_pipeline(wnd, stream, index) -> str:
    new_net = next(_networks)
    old = stream.pipelines[index]
    sid, source = next(iter(old.sources.items()))
    old_location = source.location
    net = old.network.name
    wnd.text(
        '20px, 10%',
        f"Switching pipeline {net} to network: {new_net}",
        fadeout_from=time.time() + 2.0,
        stream_id=sid,
    )
    if (index & 1) == 0:
        # demonstrate removing by index
        stream.remove_pipeline(index)
    else:
        # demonstrate removing by pipeline manager
        stream.remove_pipeline(old)
    return old_location


def main(window, stream):
    for i, pipeline in enumerate(stream.pipelines):
        for src_id, src in pipeline.sources.items():
            title = f"Pipeline {i}: {pipeline.network.name} ({src.location})"
            window.options(src_id, title=title)
            print(src_id, title)
    last_switch = time.time()
    parked_source = None
    pipeline_to_switch = 0
    next(_networks)  # advance one so we don't repeat the first network immediately
    for frame_result in stream:
        now = time.time()
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
        if now - last_switch > SWITCH_INTERVAL:
            pipeline_to_switch = (pipeline_to_switch + 1) % len(stream.pipelines)
            old = stream.pipelines[pipeline_to_switch]
            old_name = old.network.name
            LOG.info(f"Removing {old_name} : source_id %s : %s", *next(iter(old.sources.items())))
            parked_source = remove_pipeline(window, stream, pipeline_to_switch)
            network = next(_networks)
            LOG.info(f"Adding {network} for {parked_source}")
            p = add_new_pipeline(stream, network, parked_source)
            LOG.info(" -> New source_id : %s : %s", *next(iter(p.sources.items())))
            last_switch = now

        if window.is_closed:
            break


if __name__ == '__main__':
    for s in sources:
        add_new_pipeline(stream, next(_networks), s)

    with display.App(
        render=args.display,
        opengl=stream.hardware_caps.opengl,
        buffering=not stream.is_single_image(),
    ) as app:
        wnd = app.create_window("Multiple pipeline demo", args.window_size)
        app.start_thread(main, (wnd, stream), name='InferenceThread')
        app.run()
    stream.stop()
```

## Key concepts

**Dynamic pipeline management** is the core feature demonstrated here. The stream is initially created empty (no network or sources), and pipelines are added at runtime with `stream.add_pipeline()`. Each pipeline gets its own network, source, and configuration. Pipelines can be removed by index (`stream.remove_pipeline(index)`) or by reference (`stream.remove_pipeline(old)`), and the freed source can immediately be reassigned to a new pipeline with a different model.

**Argparse integration** is built into the SDK configuration classes. `config.HardwareCaps.add_to_argparser(parser)`, `config.add_display_arguments(parser)`, and `logging_utils.add_logging_args(parser)` add standardized command-line flags for hardware acceleration, display settings, and log verbosity. This gives your application a consistent CLI interface without manual argument definitions.

**The cycling pattern** uses `itertools.cycle` to rotate through a list of network variants (YOLOv8 detection, pose, segmentation, and a larger YOLOv8s). Every `SWITCH_INTERVAL` seconds, one pipeline is torn down and rebuilt with the next network in the cycle. A fadeout text overlay notifies the user which pipeline is being switched, demonstrating the `window.text()` API with a timed fadeout.

---
