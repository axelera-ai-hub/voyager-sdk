#!/usr/bin/env python
# Copyright Axelera AI, 2024
import os
import sys

if __name__ == '__main__':
    # Application Framework is not a package, so add it to the path to import it
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from axelera.app import config, create_inference_stream, display

stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=[
        config.env.framework / "media/traffic1_1080p.mp4",
        config.env.framework / "media/traffic2_1080p.mp4",
    ],
)


def main(window, stream):
    window.options(0, title="Traffic 1")
    window.options(1, title="Traffic 2")
    VEHICLE = ('car', 'truck', 'motorcycle')
    center = lambda box: ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
        for veh in frame_result.pedestrian_and_vehicle_tracker:
            print(
                f"{veh.label.name} {veh.track_id}: {center(veh.history[0])} → {center(veh.history[-1])} @ stream {frame_result.stream_id}"
            )

        if window.is_closed:
            break


with display.App(renderer=True) as app:
    wnd = app.create_window("Business logic demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
stream.stop()
