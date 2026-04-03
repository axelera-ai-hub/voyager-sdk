# Axelera Model Zoo — Quick Reference

Reference with: @docs/model_zoo.md
Source: Voyager SDK v1.3+ (June 2025), community.axelera.ai

---

## How to use a model
```bash
# From CLI
inference.py yolo11s-coco-onnx dataset --pipe=torch-aipu

# From Python wrapper (this project)
from axelera.inference import detect
result = detect("file:///tmp/image.jpg", model="yolo11s-coco-onnx")
```

Model names follow the pattern: `{architecture}-{dataset}-{format}`
e.g. `yolov8n-coco-onnx`, `resnet50-imagenet`, `yolo11s-coco-onnx`

---

## Object detection (26 models)

| Model | Resolution | Format | Notes |
|-------|-----------|--------|-------|
| yolov8n-coco-onnx | 640x640 | ONNX | recommended default, fast |
| yolov8s-coco-onnx | 640x640 | PyTorch, ONNX | |
| yolov8m-coco-onnx | 640x640 | PyTorch, ONNX | |
| yolov8l-coco-onnx | 640x640 | PyTorch, ONNX | highest accuracy YOLO8 |
| yolo11n-coco-onnx | 640x640 | ONNX | |
| yolo11s-coco-onnx | 640x640 | ONNX | good default balance |
| yolo11m-coco-onnx | 640x640 | ONNX | |
| yolo11l-coco-onnx | 640x640 | ONNX | |
| yolo11x-coco-onnx | 640x640 | ONNX | largest, highest accuracy |
| yolov5n-v7-coco | 640x640 | PyTorch, ONNX | |
| yolov5s-coco | 640x640 | PyTorch, ONNX | |
| yolov5m-v7-coco | 640x640 | PyTorch, ONNX | |
| yolov5l-v7-coco | 640x640 | PyTorch, ONNX | |
| yolov7-coco | 640x640 | PyTorch, ONNX | |
| yolov9t-coco | 640x640 | ONNX | |
| yolov9s-coco | 640x640 | ONNX | |
| yolov9m-coco | 640x640 | ONNX | |
| yolov9c-coco | 640x640 | ONNX | |
| ssd-mobilenetv1-coco | 300x300 | ONNX | smallest/fastest |
| ssd-mobilenetv2-coco | 300x300 | ONNX | |
| yolox-s-coco | 640x640 | ONNX | |
| yolox-m-coco | 640x640 | ONNX | |

---

## Image classification (18 models)

| Model | Resolution | Notes |
|-------|-----------|-------|
| resnet50-imagenet | 224x224 | recommended default |
| resnet18-imagenet | 224x224 | fastest |
| resnet34-imagenet | 224x224 | |
| resnet101-imagenet | 224x224 | |
| resnet152-imagenet | 224x224 | |
| mobilenetv2-imagenet | 300x300 | |
| mobilenetv4-small-imagenet | 224x224 | |
| mobilenetv4-medium-imagenet | 224x224 | |
| mobilenetv4-large-imagenet | 384x384 | |
| efficientnet-b0-imagenet | 224x224 | |
| efficientnet-b1-imagenet | 224x224 | |
| efficientnet-b2-imagenet | 224x224 | |
| efficientnet-b3-imagenet | 224x224 | |
| efficientnet-b4-imagenet | 224x224 | highest accuracy classifier |
| squeezenet1.0-imagenet | 224x224 | |
| squeezenet1.1-imagenet | 224x224 | |

---

## Instance segmentation (5 models)

| Model | Resolution | Notes |
|-------|-----------|-------|
| yolov8n-seg-coco-onnx | 640x640 | recommended default |
| yolov8s-seg-coco-onnx | 640x640 | |
| yolov8l-seg-coco-onnx | 640x640 | highest accuracy |
| yolo11n-seg-coco-onnx | 640x640 | |
| yolo11l-seg-coco-onnx | 640x640 | |

---

## Keypoint detection / pose (5 models)

| Model | Resolution | Notes |
|-------|-----------|-------|
| yolov8n-pose-coco-onnx | 640x640 | |
| yolov8s-pose-coco-onnx | 640x640 | |
| yolov8l-pose-coco-onnx | 640x640 | |
| yolo11n-pose-coco-onnx | 640x640 | |
| yolo11l-pose-coco-onnx | 640x640 | |

---

## Specialised models

| Model | Task | Notes |
|-------|------|-------|
| retinaface-resnet50 | Face detection | 840x840 |
| retinaface-mobilenet0.25 | Face detection | 640x640, faster |
| osnet-x1.0 | Person re-ID | 256x128 |
| lprnet | License plate recognition | 24x94 |
| fastdepth | Depth estimation | 224x224 |
| real-esrgan-x4plus | Super resolution | 128x128 input |
| unet-fcn | Semantic segmentation | 256x256–512x1024 |

---

## Defaults used in this project

| Tool | Default model | Reason |
|------|--------------|--------|
| detect_objects | yolo11s-coco-onnx | Best speed/accuracy balance for general CV |
| classify_image | resnet50-imagenet | Most widely benchmarked classifier |
| segment_image | yolov8n-seg-coco-onnx | Fastest segmentation model |
