---
title: "Pipeline"
---

# Pipeline

How the Voyager SDK inference pipeline is structured — from raw video frame to structured results. This section covers the YAML-based GStreamer pipeline. For the Pythonic alternative, see [Pipeline Builder](../pipeline-builder/index.md).

| Page | What it covers |
|------|---------------|
| [Pipeline Basics](pipeline-basics.md) | How a pipeline is defined in YAML: tasks, operators, the GStreamer runtime, and how data flows from input to output. |
| [Model Formats](model-formats.md) | The three file types in a compiled model: ONNX (input), `.axmodel` (compiled binary), and `model.json` (pipeline descriptor). |
| [GStreamer Operators](gst-operators.md) | Reference for all built-in operators: resize, normalize, decode (YOLO, SSD, classification), NMS, tracker, and utilities. |
