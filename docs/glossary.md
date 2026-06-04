# Glossary

Plain-language definitions for terms used throughout the Voyager SDK documentation. If you encounter a term that isn't here, let us know on the [Community forum](https://community.axelera.ai/).

---

## A

### AIPU
AI Processing Unit. The dedicated chip on Metis hardware that performs AI inference. Analogous to how a GPU handles graphics, the AIPU handles AI workloads.

### Accuracy (mAP)
See [mAP](#map).

### Activation (environment)
Running `source venv/bin/activate` to set up your terminal session with the correct paths and libraries for the SDK. Must be done in every new terminal. See [Install the SDK](user-guides/sdk-install.md).

### AxRuntime
The low-level C API for directly controlling Metis devices — loading models, running inference, managing memory. Used when you need full control rather than the higher-level pipeline tools.

## B

### Beta
A tag applied to SDK features that are **tested and functional** but still growing. Beta features are safe to use in development and are expected to evolve in future releases. Compare with [Experimental](#experimental).

### Benchmarking
Measuring how fast and how accurately a model runs on your hardware. Throughput (FPS) measures speed. mAP measures accuracy. See [Measure Accuracy](tutorials/measure-accuracy.md).

### BSP
Board Support Package. The firmware and operating system image for the Metis Compute Board. Must be flashed before using the board.

## C

### Cascaded pipeline
Running multiple models in sequence — the output of one feeds into the next. For example: a face detector finds faces, then a recognition model identifies who they are.

### Classification
A type of AI task. Given an image, the model answers "what is this?" (e.g., "cat", "car", "building"). It tells you *what* but not *where*. Compare with [Object detection](#object-detection).

### COCO
Common Objects in Context. A widely-used dataset containing 80 categories of everyday objects (people, cars, dogs, chairs, etc.). Used to train and evaluate object detection models. When you see `coco` in a model name like `yolov5s-v7-coco`, it means the model was trained on this dataset.

### Compilation (model)
The process of converting a model into a format the Metis AIPU can execute. This happens automatically the first time you run a model and takes a few minutes. The compiled result is cached, so subsequent runs start immediately.

## D

### Dataset
A collection of labelled images used to train or evaluate AI models. Common datasets include [COCO](#coco) (objects), [ImageNet](#imagenet) (categories), and VOC (objects). The `dataset` source in `inference.py` runs the model's default validation dataset.

### Deploy
In this SDK, "deploy" means preparing a model for execution on Metis hardware — compiling it and generating the pipeline code. This is not the same as deploying software to a server.

### Device throughput
The maximum inference speed the Metis AIPU can achieve, measured in FPS. This number shows what the hardware is capable of, independent of other bottlenecks like camera framerate or CPU processing.

### Experimental
A tag applied to SDK features that are **early-stage with limited testing**. Experimental features may change significantly or be removed in future releases. Use with caution in production. Compare with [Beta](#beta).

## F

### FPS
Frames Per Second. How many images the system processes each second. Higher is better. See [System throughput](#system-throughput) and [Device throughput](#device-throughput).

## G

### GStreamer
An open-source multimedia framework that handles video capture, processing, and display. The SDK uses GStreamer under the hood to build video pipelines. You don't need to know GStreamer to use the SDK, but advanced users can build custom GStreamer pipelines with Axelera plugins.

## H

### Haar cascade
An older computer vision technique for detecting objects (especially faces) using hand-crafted feature patterns. Faster but significantly less accurate than modern deep learning models like [YOLO](#yolo). Part of OpenCV's classic toolkit.

## I

### ImageNet
A large-scale image dataset with 1,000 categories (dog breeds, car types, plants, household objects, etc.). Used to train and evaluate classification models. When you see `imagenet` in a model name, it was trained on this dataset.

### Inference
The act of running a trained AI model on input data to get results. You give it a video frame, it tells you what objects are in it (detection) or what the image shows (classification). This is what the Metis hardware accelerates.

Not to be confused with "interference."

### InferenceStream
A Python/C++ API object provided by the SDK that wraps a complete pipeline. You create one, point it at a video source, and iterate over the results. The simplest way to integrate Metis inference into your own application.

### inference.py
The SDK's command-line tool for running models, evaluating performance, and benchmarking accuracy. Takes a model name and a source as arguments. See [inference.py reference](reference/tools/inference-py.md).

## M

### mAP
Mean Average Precision. The standard metric for measuring object detection accuracy. It considers both whether objects are found (recall) and whether detections are correct (precision). Ranges from 0 to 1, where higher is better. A model with mAP 0.56 correctly detects and locates objects about 56% of the time.

### Metis
The name of Axelera AI's AI Processing Unit (AIPU) chip family. Available in M.2, PCIe, and Compute Board form factors.

### Model
A trained neural network that performs a specific AI task. Models are identified by name in the SDK (e.g., `yolov5s-v7-coco`). The [Model Zoo](#model-zoo) contains pre-optimized models ready to use.

### Model Zoo
The collection of pre-trained, pre-optimized models shipped with the Voyager SDK. These cover common tasks like object detection, classification, and pose estimation. Browse them using `inference.py` or the Model Zoo reference page.

## O

### Object detection
A type of AI task. Given an image, the model answers "what is in this image *and where*?" It draws bounding boxes around detected objects. Compare with [Classification](#classification).

### ONNX
Open Neural Network Exchange. A standard file format for AI models, like PDF is for documents. Models from different frameworks (PyTorch, TensorFlow) can be exported to ONNX and then compiled for Metis hardware.

### Operator
A single processing step in a pipeline. Examples: resize an image, run inference, draw bounding boxes, decode detection results. Operators chain together to form the complete pipeline from input to output.

## P

### Pipeline
The complete processing chain from input to output. A typical pipeline: camera captures a frame, the frame is resized, inference runs on the Metis AIPU, results are decoded, bounding boxes are drawn, the frame is displayed. The SDK builds this pipeline automatically from a YAML description.

### Post-processing
Steps that happen after inference — interpreting the raw model output into usable results. For object detection, this includes decoding bounding box coordinates and filtering by confidence. Handled automatically by the pipeline.

### Pre-processing
Steps that happen before inference — preparing the input for the model. Typically involves resizing, normalizing pixel values, and converting color formats. Handled automatically by the pipeline.

## Q

### Quantization
Making a model smaller and faster by using less precise numbers. Instead of 32-bit floating point, the model uses 8-bit integers. Like rounding to fewer decimal places — a small accuracy trade-off for a large speed gain. The Metis AIPU is optimized for quantized models.

## R

### ResNet
Residual Network. A family of image classification models. The number indicates depth: ResNet-50 has 50 layers. Deeper models are more accurate but slower. Commonly trained on [ImageNet](#imagenet).

### RTSP
Real-Time Streaming Protocol. A network protocol for streaming video from IP cameras. The SDK accepts RTSP URLs as video sources. Format: `rtsp://user:password@host:port/path`.

## S

### Source
The input to a pipeline — where the video frames come from. Can be a USB camera (`usb:0`), a video file (`media/video.mp4`), an RTSP stream (`rtsp://...`), or a validation dataset (`dataset`). See [Video Sources](tutorials/video-sources.md).

### System throughput
The end-to-end speed of the complete pipeline, measured in FPS. This includes everything: video capture, pre-processing, inference, post-processing, and display. This is the number that matters for real-world performance.

## T

### Throughput
See [FPS](#fps), [System throughput](#system-throughput), [Device throughput](#device-throughput).

## V

### Voyager SDK
Axelera AI's complete software stack for developing and deploying AI applications on Metis hardware. Includes drivers, compiler, runtime, evaluation tools, APIs, and a model zoo.

## Y

### YAML
A human-readable configuration format used to describe pipelines in the SDK. You define which model to use, what pre/post-processing to apply, and the SDK compiles it into executable code. YAML files live in the `ax_models/` directory.

### YOLO
"You Only Look Once." A family of fast object detection models that identify and locate objects in images in a single pass through the network. The numbers indicate the version (v5, v7, v8 — higher is generally newer and better). The letter after the number indicates the model size:
- **n** (nano) — fastest, least accurate
- **s** (small) — good balance
- **m** (medium) — more accurate, slower
- **l** (large) — high accuracy
- **x** (extra large) — highest accuracy, slowest

So `yolov5s-v7-coco` means: YOLO version 5, small size, variant 7, trained on the COCO dataset.
