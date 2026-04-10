#!/usr/bin/env python
# Copyright Axelera AI, 2025
import alive_progress
import cv2
import numpy as np
import os
import sys

if __name__ == '__main__':
    # Application Framework is not a package, so add it to the path to import it
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from axelera.app import config
from axelera.app.stream import create_inference_stream


def postprocess_yolov8(
    data, shape, orig_w, orig_h, model_w=640, model_h=640, conf_threshold=0.25, letterboxed=True
):
    # YOLOv8 output: (1, 84, 8400) => (batch, channels, num_anchors)
    # Each anchor: [x, y, w, h, score_0, ..., score_79]
    # We'll use only the first batch
    while data.ndim > 3:
        data = np.squeeze(data, axis=1)
    shape = data.shape
    num_classes = shape[1] - 4
    num_anchors = shape[2]
    detections = []
    for i in range(num_anchors):
        x = data[0, 0, i]
        y = data[0, 1, i]
        w = data[0, 2, i]
        h = data[0, 3, i]
        scores = data[0, 4:, i]
        class_id = np.argmax(scores)
        score = scores[class_id]
        if score > conf_threshold:
            # Convert from center x, y, w, h to x1, y1, x2, y2
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            # Map to original image coordinates
            if letterboxed:
                # Calculate scale and padding
                r = min(model_w / orig_w, model_h / orig_h)
                new_w, new_h = int(orig_w * r), int(orig_h * r)
                pad_w, pad_h = (model_w - new_w) // 2, (model_h - new_h) // 2

                # Undo letterbox
                x1 = (x1 - pad_w) / r
                y1 = (y1 - pad_h) / r
                x2 = (x2 - pad_w) / r
                y2 = (y2 - pad_h) / r
            else:
                # Simple resize
                x1 = x1 * orig_w / model_w
                y1 = y1 * orig_h / model_h
                x2 = x2 * orig_w / model_w
                y2 = y2 * orig_h / model_h

            detections.append((x1, y1, x2, y2, class_id, float(score)))
    return detections


def render_detections(image, detections, labels=None):
    if labels is None:
        labels = [f"object_{i}" for i in range(80)]
    for x1, y1, x2, y2, class_id, score in detections:
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))
        cv2.rectangle(image, pt1, pt2, (255, 255, 0), 2)
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        text = f"{label} {score:.2f}"
        cv2.putText(
            image, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
        )


def main(window, stream):
    display_w, display_h = 640, 360  # or any size you prefer
    for frame_result in alive_progress.alive_it(stream):
        tensor_wrapper = frame_result.meta['detections']
        tensor = tensor_wrapper.tensors[0]  # numpy array
        bgr_img = frame_result.image.asarray('BGR')
        # Resize image first for faster processing and display
        bgr_img_small = cv2.resize(bgr_img, (display_w, display_h))
        orig_h, orig_w = bgr_img_small.shape[:2]
        detections = postprocess_yolov8(
            tensor, tensor.shape, orig_w, orig_h, model_w=640, model_h=640, letterboxed=True
        )
        render_detections(bgr_img_small, detections)
        cv2.imshow('Detections', bgr_img_small)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Detections', cv2.WND_PROP_VISIBLE) < 1:
            break


if __name__ == '__main__':
    stream = create_inference_stream(
        network="yolov8n-output-tensor",
        sources=[
            str(config.env.framework / "media/traffic1_1080p.mp4"),
        ],
    )
    try:
        main(None, stream)
    finally:
        stream.stop()
