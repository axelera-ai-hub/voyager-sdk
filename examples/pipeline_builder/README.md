# Pipeline Builder API Examples

Standalone, runnable demos showing how to build inference pipelines with
`axelera.runtime` (`op`, `cv`, `display`). Each script is self-contained and
depends only on `axelera.runtime` + `argparse`, so users can copy a file and
start from it without pulling in the smoke-test scaffolding.

## Demos

| Script                            | What it does                                                                                                             | Model                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------ |
| `classification.py`               | ImageNet top-5 classification with standard ImageNet preprocessing.                                                      | `squeezenet1.0-imagenet.axm`                     |
| `detection.py`                    | YOLOv8 object detection on COCO (80 classes); NMS + `DetectedObject` output.                                             | `yolov8n-coco.axm`                               |
| `detection_batched.py`            | YOLOv8 COCO detection driven by batched `pipeline.batch()` calls, stepping batch size 1→128 to show flexible batching.   | `yolov8n-coco.axm`                               |
| `detection_stream.py`             | YOLOv8 COCO detection driven by a pipelined `pipeline.stream()` over the input source (video-oriented).                  | `yolov8n-coco.axm`                               |
| `detection_vary_confidence.py`    | Same as `detection.py`, but varies `confidence_threshold` every 60 frames to show runtime parameter tuning.              | `yolov8n-coco.axm`                               |
| `nms_free_detection.py`           | NMS-free YOLO detection (yolo26); `decode_detections` filters by confidence, so no separate NMS stage is needed.         | `yolo26n-coco-onnx.axm`                          |
| `pose_detection.py`               | YOLOv8 human pose estimation with 17 COCO keypoints.                                                                     | `yolov8npose-coco.axm`                           |
| `segmentation.py`                 | YOLOv8 instance segmentation with prototype-based mask prediction (uses `par` + `itemgetter` for explicit tuple flow).   | `yolov8nseg-coco.axm`                            |
| `depth_estimation.py`             | Monocular depth estimation with FastDepth (NYU Depth V2 -- works best on indoor scenes).                                 | `fastdepth-nyudepthv2-onnx.axm`                  |
| `obb.py`                          | YOLO11n oriented-bounding-box detection on DOTA (15 classes: plane, ship, vehicle, ...).                                 | `yolo11n-obb-dotav1-onnx.axm`                    |
| `tracking.py`                     | Multi-object tracking with state lifecycle, detection correlation, and class filtering.                                  | `yolov8n-coco.axm`                               |
| `tracking_with_classification.py` | Detection → filtering → tracking → per-track classification. Shows that a tracked object can drive a downstream cascade. | `yolov8n-coco.axm`, `squeezenet1.0-imagenet.axm` |

## Downloading the required `.axm` files

The demos expect their `.axm` files under `~/.cache/axelera/runtime2/`. Run the
helper script in this directory to fetch every `.axm` referenced by the demos
above:

```bash
python examples/pipeline_builder/download_axm.py
```

The script scans `examples/pipeline_builder/*.py` for `{MODEL_DIR}/<stem>.axm`
references, then calls `axdownloadmodel --axm <stem>` for each one.
Already-present files are skipped.

```bash
# Just list the .axm stems the demos reference -- no downloads
python examples/pipeline_builder/download_axm.py --list

# For .axm not in the public catalog, fall back to compiling locally with
# `yolo export model=<stem>.pt format=axelera` (needs ultralytics +
# axelera-devkit; can take several minutes per model).
python examples/pipeline_builder/download_axm.py --deploy-missing
```

## Running a demo

```bash
# Image or video input is auto-detected by file extension.  The window
# stays open after processing by default so a still-image result is
# inspectable; press Q (or close the window) to exit.
python examples/pipeline_builder/detection.py path/to/input.jpg

# Display backend selection (default is auto-detect):
python examples/pipeline_builder/detection.py path/to/input.mp4 --display opencv
python examples/pipeline_builder/detection.py path/to/input.mp4 --display none

# Exit as soon as the input ends (useful for batch/video runs).
python examples/pipeline_builder/detection.py path/to/input.mp4 --no-wait
```

All scripts share the same CLI surface:

```
positional:
  input                  Image or video file to process

options:
  --display {none,opencv,console,iterm2,auto}
  --window-width INT             (default 800)
  --window-height INT            (default 500)
  -w, --wait / --no-wait         Keep the window open until the user closes it
                                 (default: --wait)
```
