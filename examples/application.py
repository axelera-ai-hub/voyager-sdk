#!/usr/bin/env python
from axelera.app import config, display
from axelera.app.stream import create_inference_stream

stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=[
        str(config.env.framework / "media/traffic1_1080p.mp4"),
        str(config.env.framework / "media/traffic2_1080p.mp4"),
    ],
)


def main(window, stream):
    window.options(0, title="Traffic 1")
    window.options(1, title="Traffic 2")
    VEHICLE = ('car', 'truck', 'motorcycle')
    center = lambda box: ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
        for veh in frame_result.detections:
            if veh.is_a(VEHICLE):
                print(
                    f"{veh.label.name} {veh.track_id}: {center(veh.history[0])} → {center(veh.history[-1])} @ stream {frame_result.stream_id}"
                )


with display.App(visible=True) as app:
    wnd = app.create_window("Business logic demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run(interval=1 / 10)
stream.stop()
