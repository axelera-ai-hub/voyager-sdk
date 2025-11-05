# Jetson guide: reliable MP4/H.264 video with GStreamer

On NVIDIA Jetson devices, GStreamer decodebin may fail to autoplug the required parser for MP4 (avc1) H.264 streams, causing caps negotiation errors like "capsfilter not-negotiated" or qtdemux streaming errors.

This guide shows how to enable a reliable demux/parse/decode chain in Voyager SDK and validate your setup.

## TL;DR

If you see negotiation errors when playing MP4/H.264 files, enable the explicit chain:

```bash
AXELERA_GST_EXPLICIT_PARSE=1 ./inference.py <model-yaml-or-alias> <video.mp4> --aipu-cores 4
```

This switches VIDEO_FILE inputs from decodebin to an explicit pipeline:

- filesrc → qtdemux(video_%u) → h264parse → decoder → (nvvidconv if HW) → capsfilter(video/x-raw,format=NV12) → …

Notes:
- Software decode (avdec_h264) is used unless you enable hardware decoders in your run.
- Hardware decode (nvv4l2decoder) outputs NVMM memory; we convert to system memory via nvvidconv + caps.

## Why this helps

- MP4 containers often carry H.264 in avc1 format; many hardware decoders require byte-stream format.
- decodebin doesn’t always insert h264parse for avc1 on Jetson, leading to not-negotiated errors.
- Inserting qtdemux + h264parse explicitly ensures the stream is converted for the decoder.

## Verify your environment

Optional checks (diagnostic only):

```bash
# Confirm NVIDIA decoder and converter are present
gst-inspect-1.0 nvv4l2decoder | grep -E 'Sink|Src|NVMM' -n || true
gst-inspect-1.0 nvvidconv | head -n 20 || true

# See codec info (H.264 avc1) for your video
ffprobe -hide_banner -v error -select_streams v:0 -show_entries stream=codec_name,profile,codec_tag_string -of default=nk=1:nw=1 <video.mp4>
```

## Run examples

- Software decode (recommended for stability):

```bash
AXELERA_GST_EXPLICIT_PARSE=1 ./inference.py yolov8s-coco-onnx ./media/traffic3_480p.mp4 --aipu-cores 4 --no-display --frames 30
```

- Hardware decode (optional; may depend on Jetson image/plugins):

```bash
AXELERA_GST_EXPLICIT_PARSE=1 ./inference.py yolov8s-coco-onnx ./media/traffic3_1080p.mp4 --aipu-cores 4
```

If you encounter instability with the NVIDIA plugins, try the software decode first to confirm decoding and negotiation are okay.

## Behavior details

- Default behavior remains decodebin for backward compatibility and testing.
- Setting AXELERA_GST_EXPLICIT_PARSE=1 enables the explicit chain for VIDEO_FILE inputs only.
- RTSP/USB/HLS paths are unchanged.

## Troubleshooting

- Still seeing not-negotiated?
  - Ensure the env var is set in the same shell invocation as inference.
  - Try the software path first (don’t enable hardware decoder flags).
  - Check that your GStreamer has h264parse and qtdemux elements.
- HW decode crashes or times out?
  - Use software decode as a baseline; investigate Jetson plugin versions (nvv4l2decoder/nvvidconv) and caps.

## Contributing upstream

- The change is opt-in to avoid breaking existing golden YAMLs.
- Propose a PR with these changes and include brief logs and pipeline snippets demonstrating the fix on Jetson.
- Optionally add a CLI flag in the future (e.g., `--explicit-parse`) to replace the env var.
