---
title: "Code Examples"
---
# Code Examples

Complete, runnable examples from the Voyager SDK. Each example is a standalone script you can use as a starting point for your own applications.

## Python Examples

| Example | Description |
|---------|-------------|
| [Basic Application](application.md) | Minimal detection loop with display — the "hello world" of Voyager |
| [Extended Application](application-extended.md) | Hardware caps, frame rate control, temperature monitoring |
| [Tensor Access](application-tensor.md) | Direct tensor output for custom postprocessing |
| [Classification](classification.md) | Image classification with generator-based input |
| [Cross-Line Counter](cross-line-count.md) | Vehicle counting using tracker metadata |
| [Remote Monitor](remote-monitor.md) | TCP broadcast server for remote cross-line monitoring |

## C++ Examples (AxInferenceNet)

| Example | Description |
|---------|-------------|
| [Basic Inference](axinferencenet-basic.md) | C++ detection loop using AxInferenceNet API |
| [Cascaded Pipeline](axinferencenet-cascaded.md) | Multi-model C++ pipeline (detect → classify) |
| [Tensor Access](axinferencenet-tensor.md) | Raw tensor output in C++ for custom processing |

## Running the examples

All examples are included in the SDK under `examples/`. To run a Python example:

```bash
cd $AXELERA_FRAMEWORK
python examples/application.py
```

To build and run the C++ examples:

```bash
cd $AXELERA_FRAMEWORK
make examples
./build/examples/axinferencenet/axinferencenet_example
```

## See also

- [First Inference](../../user-guides/first-inference.md) — guided walkthrough of your first model run
- [Run Inference in Python](../run-inference-in-python.md) — deeper dive into the Python API
- [Video Sources](../video-sources.md) — configuring cameras, RTSP streams, and files

---
