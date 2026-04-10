#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Example application showing how to access classification meta data using application pipeline
# This example also shows how to use a generator to pass images to the pipeline
import cv2
import os
import sys

if __name__ == '__main__':
    # Application Framework is not a package, so add it to the path to import it
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


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
