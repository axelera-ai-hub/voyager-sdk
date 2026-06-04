---
title: "Image Classification with Generator Input"
---
# Image Classification with Generator Input

This example demonstrates image classification using a ResNet18 model with ImageNet classes. It also shows how to feed images into the pipeline using a Python generator instead of video files or camera streams, which is useful for batch processing or custom image sources.

Use this when you need to classify individual images, process a batch of files, or integrate the SDK with a custom data source.

## What you'll learn

- How to use a generator function as an image source for the inference stream
- How to access classification results including top-1 and top-k predictions
- How to read confidence scores and class labels from classification metadata
- How to validate inference results programmatically

## Prerequisites

- Voyager SDK installed and activated
- OpenCV (`cv2`) installed
- Sample classification images in `examples/images/` (included with the SDK)

## Source

> [!TIP]
> **Download**
> This example is included in the SDK at `examples/classification_example.py`.


```python
#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Example application showing how to access classification meta data using application pipeline
# This example also shows how to use a generator to pass images to the pipeline
import cv2

from axelera.app import config
from axelera.app.stream import create_inference_stream

images = [
    ('apple.jpg', 'granny_smith'),
    ('orange.jpg', 'orange'),
    ('banana.jpg', 'banana'),
]

queued = []
got_labels = []
expected_labels = []


def reader():
    for path, expected in images:
        img = cv2.imread(f'{config.env.framework}/examples/images/{path}')
        if img is None:
            print(f"Failed to read image {path}.jpg")
            continue
        queued.append((path, expected))
        yield img


stream = create_inference_stream(network="resnet18-imagenet", sources=[reader()])

for result in stream:
    image_name, expected_label = queued.pop(0)
    m = result.classifications[0]
    expected_labels.append(expected_label)
    got_labels.append(m.label.name)
    print(f"Image {image_name} is classified as {m.label.name} with {m.score:.2f}% confidence")
    for topn, x in enumerate(m.topk[1:], 1):
        print(f"  or alternative {topn}: {x.label.name} with {x.score:.2f}% confidence")
stream.stop()

if got_labels == expected_labels:
    print("All images classified correctly!")
else:
    exit(f"Some images were misclassified:\n{got_labels=}\n{expected_labels=}")
```

## Key concepts

**Generator-based input** is a powerful alternative to file or camera sources. The `reader()` function yields OpenCV images one at a time, and passing `[reader()]` as the `sources` list tells the SDK to pull frames from the generator. This pattern works for any iterable that produces image arrays, making it easy to integrate with custom data loaders, web APIs, or in-memory image buffers.

**Classification metadata** is accessed through `result.classifications`, which returns a list of classification outputs. Each entry has a `label` (with a `.name` attribute), a `score` (confidence percentage), and a `topk` list containing the top-N alternative predictions ranked by confidence. This makes it straightforward to implement top-1 accuracy checks or display ranked alternatives.

The example also demonstrates a practical validation pattern: it compares predicted labels against expected labels and exits with an error if any misclassification occurs. This is useful for automated testing and quality assurance of model deployments.

---
