#!/usr/bin/env python
# Copyright Axelera AI, 2025
import os
import sys
import time

from axelera.types import ColorFormat, Image
import cv2

if __name__ == '__main__':
    # Application Framework is not a package, so add it to the path to import it
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from axelera.app import config, display
from axelera.app.stream import create_inference_stream

NETWORK = "yolov8s-coco"
SOURCES = [
    config.env.framework / "media/traffic1_720p.mp4",
    config.env.framework / "media/traffic2_720p.mp4",
]

TITLE = "Axelera Python Data Reader Example"
TITLE_FONT = 39

LOGO0 = os.path.join(
    config.env.framework, "axelera/app/render_assets/axelera-ai-logo-logo-only.png"
)
LOGO1 = os.path.join(config.env.framework, "axelera/app/render_assets/voyager-sdk-logo-white.png")
LOGO2 = os.path.join(
    config.env.framework, "axelera/app/render_assets/axelera-ai-logo-text-only.png"
)


def frame_reader(src):
    cap = cv2.VideoCapture(src)
    n = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if n % 2 == 0:
                yield frame
            else:
                # Every other frame is returned as an axelera.types.Image, which
                # supports PIL, numpy and gstreamer buffers
                # The types may be used together as shown here
                yield Image.fromarray(frame, ColorFormat.BGR)
            n += 1
    finally:
        cap.release()


stream = create_inference_stream(network=NETWORK, sources=[frame_reader(s) for s in SOURCES])


def main(window, stream):
    window.options(
        -1,
        title=TITLE,
        title_position="50%, 0%",
        title_size=TITLE_FONT,
        title_anchor_x="center",
    )

    logo0 = wnd.image('52%, 88%', LOGO0, anchor_x='right', anchor_y='center', scale=0.4)
    logo1 = wnd.image('52%, 88%', LOGO1, anchor_x='left', anchor_y='center', scale=0.45)
    logo2 = wnd.image(
        '52%, 88%', LOGO2, anchor_x='left', anchor_y='center', scale=0.4, fadeout_from=0.0
    )
    supported = logo0 and logo1 and logo2
    start = last_logo_switch = time.time()
    period = 10.0

    for frame_result in stream:
        now = time.time()
        if supported and ((now - last_logo_switch) > period):
            logo1.hide(now, 1.0)
            logo2.show(now, 1.0)
            logo1, logo2 = logo2, logo1
            last_logo_switch = now

        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
        count = sum(x.is_car for x in frame_result.detections)
        print(f"Detected {count} cars\r", end="")

        if window.is_closed:
            break

    duration = time.time() - start
    fps = stream.frames_executed / duration
    print(' ' * 80)
    print(f"{stream.frames_executed} frames in {duration:.1f} seconds (FPS: {fps:.1f})")


with display.App(renderer=True) as app:
    wnd = app.create_window(TITLE, (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
stream.stop()
