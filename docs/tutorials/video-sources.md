# Video Sources

The SDK supports USB cameras, RTSP streams, video files, and multiple simultaneous inputs. This page covers how to configure each.

## Quickstart

```bash
source venv/bin/activate

# USB camera
./inference.py yolov5s-v7-coco usb:0

# Video file
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4

# RTSP stream
./inference.py yolov5s-v7-coco rtsp://192.168.1.100:8554/stream1

# Multiple sources
./inference.py yolov5s-v7-coco usb:0 usb:1 media/traffic1_1080p.mp4
```

---

## Prerequisites

- SDK installed and environment activated (see [Install the SDK](../user-guides/sdk-install.md))
- At least one input source available (camera, video file, or RTSP stream)

> [!CAUTION]
> **Before every session**
> ```bash
> source venv/bin/activate
> ```


## USB cameras

USB cameras are enumerated at `/dev/video0`, `/dev/video1`, etc. Use the shorthand `usb:0`, `usb:1`.

```bash
./inference.py yolov5s-v7-coco usb:0
```

### Set resolution and framerate

Use the format `usb:N:WIDTHxHEIGHT@FPS`:

```bash
./inference.py yolov5s-v7-coco usb:0:1920x1080@30
```

### Check supported formats

Install `v4l-utils` to inspect camera capabilities:

```bash
sudo apt-get install v4l-utils
v4l2-ctl --device /dev/video0 --list-formats-ext
```

> [!NOTE]
> **Multiple sensors**
> Many USB cameras expose multiple sensors (RGB, infrared, etc.). The RGB sensor is almost always the first listed. If you have two cameras, the second camera's RGB sensor may be at `usb:4` rather than `usb:1`, depending on how many sensors the first camera exposes.


> [!TIP]
> **Consistent ordering**
> Linux may enumerate USB cameras in a different order after each boot. To ensure consistent ordering, plug cameras in sequence after the system has booted.


## RTSP cameras

Use the RTSP URL directly:

```bash
./inference.py yolov5s-v7-coco rtsp://192.168.1.100:8554/stream1
```

The general format is `rtsp://<user>:<password>@<host>:<port>/<path>`. Omit user and password if the stream does not require authentication.

> [!NOTE]
> You must have an RTSP server running and accessible on your network before using this source type. The SDK connects to the stream as a client — it does not create an RTSP server. If you see a "Could not open resource" or timeout error, check that the server is running and the URL is correct.


## Video files

Specify the file path:

```bash
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4
```

### Throttle playback framerate

You can append `@FPS` to a video path to limit the playback rate:

```bash
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4@10
```

This throttles playback to 10 FPS using a wall-clock sleep per frame. It will not speed up playback beyond what the pipeline can deliver — if the pipeline is already slower than the requested FPS, the sleep is skipped. The timing is approximate (not GStreamer clock-synchronized).

The SDK ships with sample videos in the `media/` directory. If these are missing, download them:

```bash
deactivate
./install.sh --media
source venv/bin/activate
```

## Validation dataset (`dataset`)

`dataset` is a special source that runs inference against the model's official validation dataset — the same data used to measure the accuracy figures in the [Model Zoo](../reference/models/model-zoo.md). Use it to verify or reproduce published accuracy numbers.

```bash
./inference.py yolov5s-v7-coco dataset --no-display
```

The SDK downloads the validation data automatically if it is not already present. For most models (COCO, ImageNet) the download is automatic. A small number of datasets require manual download due to licensing restrictions — the SDK prints an error with the expected path and download instructions if this applies.

> [!NOTE]
> `dataset` loads individual images rather than a video stream, so the reported FPS will be lower than with a camera or video file. This is expected — the bottleneck is image loading, not the AIPU.


See [Measure Accuracy](measure-accuracy.md) for the full workflow.

---

## Multiple sources

List multiple sources to run inference on all of them simultaneously:

```bash
./inference.py yolov5s-v7-coco usb:0 usb:1 usb:2
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4 media/traffic2_720p.mp4
```

You can mix source types:

```bash
./inference.py yolov5s-v7-coco usb:0 media/traffic1_1080p.mp4
```

## Source reference

| Source type | Syntax | Example |
|-------------|--------|---------|
| USB camera | `usb:N` | `usb:0` |
| USB with resolution | `usb:N:WxH@FPS` | `usb:0:1920x1080@30` |
| RTSP stream | `rtsp://...` | `rtsp://192.168.1.100:8554/1` |
| Video file | `path/to/file` | `media/traffic1_1080p.mp4` |
| Validation dataset | `dataset` | `dataset` (see [above](#validation-dataset-dataset)) |

> [!NOTE]
> **Application integration**
> These same source strings work in the Python API. Pass them to `InferenceStream` exactly as you would to `inference.py`. See the [inference.py reference](../reference/tools/inference-py.md) for all available options.


---

## Next steps

- [Measure accuracy](measure-accuracy.md) — benchmark a model against its validation dataset
- [inference.py](../reference/tools/inference-py.md) — full command reference for all source options
