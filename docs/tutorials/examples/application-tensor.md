---
title: "Raw Tensor Output with Custom Postprocessing"
---
# Raw Tensor Output with Custom Postprocessing

This example shows how to access raw output tensors from the neural network and perform your own postprocessing in Python. Instead of using the SDK's built-in detection pipeline, it retrieves the YOLOv8 output tensor as a NumPy array, decodes bounding boxes manually, and renders them with OpenCV.

Use this when you need full control over postprocessing logic, want to implement a custom decoder, or are working with a model whose output format is not yet supported by the built-in task handlers.

## What you'll learn

- How to use a network configuration that outputs raw tensors (`yolov8n-output-tensor`)
- How to extract NumPy tensors from frame result metadata
- How to implement YOLOv8 postprocessing (anchor decoding, letterbox correction, NMS)
- How to render detections manually with OpenCV

## Prerequisites

- Voyager SDK installed and activated
- OpenCV (`cv2`) and NumPy installed
- Sample media files available in `media/` (included with the SDK)

## Source

> [!TIP]
> **Download**
> This example is included in the SDK at `examples/application_tensor.py`.


```python
#!/usr/bin/env python
# Copyright Axelera AI, 2025
import cv2
import numpy as np

from axelera.app import config, display
from axelera.app.stream import create_inference_stream

stream = create_inference_stream(
    network="yolov8n-output-tensor",
    sources=[
        str(config.env.framework / "media/traffic1_1080p.mp4"),
    ],
)


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
        cv2.rectangle(image, pt1, pt2, (0, 255, 255), 2)
        label = labels[class_id] if class_id < len(labels) else str(class_id)
        text = f"{label} {score:.2f}"
        cv2.putText(
            image, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )


def main(window, stream):
    display_w, display_h = 640, 360  # or any size you prefer
    for frame_result in stream:
        tensor_wrapper = frame_result.meta['detections']
        tensor = tensor_wrapper.tensors[0]  # numpy array
        rgb_img = frame_result.image.asarray()
        # Resize image first for faster processing and display
        rgb_img_small = cv2.resize(rgb_img, (display_w, display_h))
        orig_h, orig_w = rgb_img_small.shape[:2]
        detections = postprocess_yolov8(
            tensor, tensor.shape, orig_w, orig_h, model_w=640, model_h=640, letterboxed=True
        )
        bgr_img = cv2.cvtColor(rgb_img_small, cv2.COLOR_RGB2BGR)
        render_detections(bgr_img, detections)
        cv2.imshow('Detections', bgr_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


with display.App(renderer=False) as app:
    app.start_thread(main, (None, stream), name='InferenceThread')
    app.run()
stream.stop()
```

## Key concepts

**Raw tensor access** is enabled by using a network configuration that ends with `-output-tensor` (here `yolov8n-output-tensor`). Instead of producing parsed detections, the pipeline places raw NumPy arrays into `frame_result.meta['detections'].tensors`. This gives you direct access to the model's output for custom decoding.

**YOLOv8 postprocessing** is implemented in the `postprocess_yolov8` function. The model outputs a tensor of shape `(1, 84, 8400)` where each of the 8400 anchors contains 4 box coordinates plus 80 class scores. The function converts center-format boxes to corner-format, applies confidence thresholding, and corrects for letterbox padding to map coordinates back to the original image dimensions.

**Manual rendering with OpenCV** replaces the SDK's built-in renderer. Notice that `display.App(renderer=False)` disables the SDK renderer entirely, and the code uses `cv2.imshow` and `cv2.rectangle` directly. This is the pattern to follow when you need pixel-level control over the output visualization or are integrating with an existing OpenCV-based pipeline.

---
