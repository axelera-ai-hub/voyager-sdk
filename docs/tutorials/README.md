# Tutorials

In-depth guides for advanced SDK workflows.

| Tutorial | What you'll do |
|----------|---------------|
| [Video Sources](video-sources.md) | Connect USB cameras, RTSP streams, and video files as pipeline inputs. |
| [Run Inference in Python (Low level API)](axruntime-python/axruntime-python-quickstart.md) | Build a Python application that uses the low level axelera-runtime API to read inference results frame by frame. |
| [Run Inference in Python (High Level API)](run-inference-in-python.md) | Build a Python application that reads inference results frame by frame using a high performance gstreamer optimised pipeline. |
| [Measure Accuracy](measure-accuracy.md) | Run a model against a validation dataset and get mAP or top-1 accuracy scores. |
| [Deploy Custom Weights](custom-weights.md) | Use your own trained weights with a Model Zoo architecture, through a YAML pipeline and `deploy.py`. |
| [Cascaded Pipelines](cascaded-pipelines.md) | Chain two models together: run a classifier on regions detected by a detector, through a YAML pipeline. For the Pipeline Builder Python API, see the [Cascade Tutorial](pipeline-builder/cascade-tutorial.md). |
| [Custom Model Architecture](custom-model.md) | Deploy a completely new model architecture with a custom decoder and dataset adapter. |
| [Compile and Run an LLM](llm-compile-and-run.md) | Quantize, compile and run HuggingFace causal-LMs (Llama, Qwen3) on Metis. |
| **Pipeline Builder, step 1** [Pipeline Builder Quickstart](pipeline-builder/quickstart.md) | Compile a YOLO model and run your first Python pipeline on a Metis device. |
| **Pipeline Builder, step 2** [Model Compilation](pipeline-builder/model-compilation.md) | Compile an Ultralytics, ONNX or PyTorch model to an `.axm` for the Pipeline Builder. |
| **Pipeline Builder, step 3** [Cascade Tutorial](pipeline-builder/cascade-tutorial.md) | Run a second model on each detection of the first, then trim, track and nest cascades. |
| **Pipeline Builder, step 4** [Coordinate System Tutorial](pipeline-builder/coordinate-system-tutorial.md) | Map boxes, keypoints and rotated boxes from any cascade level back to the original frame. |
| **Pipeline Builder, step 5** [Profiling](pipeline-builder/profiling.md) | Read a pipeline's per-operator report and find what limits throughput. |
| **AxScript, step 1** [Compile and Run an AxScript Model](axscript/quickstart.md) | Build one of the AxScript cards this SDK ships, run it on a Metis device, and read what it prints, before changing anything. |
| **AxScript, step 2** [Serve an AxScript LLM through HuggingFace, or Your Own Front End](axscript/llm-huggingface.md) | Load a built text or vision-language artifact with the Auto classes and your own sampling parameters, or drive the same engine from a front end of your own. |
| **AxScript, step 3** [Build an AxScript Model from Your Own Weights](axscript/custom-weights.md) | Point a vision, text or vision-language card at your fine-tune, your output head, and your calibration data. |

The Pipeline Builder reference (pipeline overview, model formats, postambles, API) is at [Pipeline Builder](../reference/pipeline-builder/README.md).
