#!/usr/bin/env python
# Copyright Axelera AI, 2024
# Sample demo application of counting vehicles crossing a line
# The tracker metadata usage is demonstrated in this example
# FIXME: change this example as async mode and remove from .gitattributes
import os
import sys
import cv2

if __name__ == '__main__':
    # Application Framework is not a package, so add it to the path to import it
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from axelera import types
from axelera.app import config, logging_utils
from axelera.app.display import App
from axelera.app.stream import create_inference_stream

network = "yolov5m-v7-coco-tracker"
source = config.env.framework / "media/traffic1_480p.mp4"

vehicles = [2, 5, 7]  # car, bus, truck

# InferenceStream constructor compiles all models and deploys the pipeline
stream = create_inference_stream(
    network=network,
    sources=[source],
    pipe_type='gst',
    log_level=logging_utils.INFO,
)


def main(window, stream):
    mid_line_start, mid_line_slope, mid_line_intercept = None, None, None

    def is_below_line(point):
        return (
            point[1] > (mid_line_slope * point[0] + mid_line_intercept)
            if mid_line_slope != float('inf')
            else point[0] > mid_line_start[0]
        )

    n_frames = 90  # Number of frames to cache crossed cars
    crossed_car_up, crossed_car_down = 0, 0
    crossed_ids_up_the_last_nframes, crossed_ids_down_the_last_nframes = [], []
    already_counted_ids = set()  # To track cars that have already been counted

    up = window.text('10px, 50px', 'Vehicles Crossed Up: 0', color=(255, 165, 0, 255), stream_id=0)
    down = window.text(
        '10px, 100px', 'Vehicles Crossed Down: 0', color=(255, 165, 0, 255), stream_id=0
    )

    frame_count = 0
    for frame_result in stream:
        frame_count += 1  # Increment frame counter for each new frame

        image = frame_result.image.asarray(types.ColorFormat.BGR).copy()
        if mid_line_slope is None or mid_line_intercept is None:
            height, width, _ = image.shape
            mid_line_start = (0, (3 * height) // 4)
            mid_line_end = (width, (3 * height) // 4)
            if (mid_line_end[0] - mid_line_start[0]) != 0:
                mid_line_slope = (mid_line_end[1] - mid_line_start[1]) / (
                    mid_line_end[0] - mid_line_start[0]
                )
                mid_line_intercept = mid_line_start[1] - (mid_line_slope * mid_line_start[0])
            else:
                # Handle vertical line case
                mid_line_slope = float('inf')
                mid_line_intercept = mid_line_start[0]

        cv2.line(image, mid_line_start, mid_line_end, (0, 255, 0), 2)

        detections = [
            v for v in frame_result.pedestrian_and_vehicle_tracker if v.class_id in vehicles
        ]
        for veh in detections:
            if veh.track_id in already_counted_ids:
                continue  # Skip if this car ID has already been counted

            if len(veh.history) > 1:
                the_last_bbox_center = (
                    (veh.history[-1][0] + veh.history[-1][2]) / 2,
                    (veh.history[-1][1] + veh.history[-1][3]) / 2,
                )
                the_oldest_bbox_center = (
                    (veh.history[0][0] + veh.history[0][2]) / 2,
                    (veh.history[0][1] + veh.history[0][3]) / 2,
                )

                if is_below_line(the_oldest_bbox_center) and not is_below_line(
                    the_last_bbox_center
                ):
                    crossed_car_down += 1
                    already_counted_ids.add(veh.track_id)
                    crossed_ids_down_the_last_nframes.append((veh.track_id, frame_count))
                elif not is_below_line(the_oldest_bbox_center) and is_below_line(
                    the_last_bbox_center
                ):
                    crossed_car_up += 1
                    already_counted_ids.add(veh.track_id)
                    crossed_ids_up_the_last_nframes.append((veh.track_id, frame_count))

        # Filter out IDs that are older than n_frames
        crossed_ids_up_the_last_nframes = [
            (id, frame)
            for id, frame in crossed_ids_up_the_last_nframes
            if frame_count - frame <= n_frames
        ]
        crossed_ids_down_the_last_nframes = [
            (id, frame)
            for id, frame in crossed_ids_down_the_last_nframes
            if frame_count - frame <= n_frames
        ]

        up["text"] = (
            f"Vehicles Crossed Up: {crossed_car_up} ({', '.join([str(id) for id, _ in crossed_ids_up_the_last_nframes])})"
        )
        down["text"] = (
            f"Vehicles Crossed Down: {crossed_car_down} ({', '.join([str(id) for id, _ in crossed_ids_down_the_last_nframes])})"
        )
        window.show(
            types.Image.fromarray(image, types.ColorFormat.BGR),
            frame_result.meta,
            frame_result.stream_id,
        )

        if window.is_closed:
            break


with App(renderer=True, opengl=stream.hardware_caps.opengl) as app:
    wnd = app.create_window("Directional Line Cross Count", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
