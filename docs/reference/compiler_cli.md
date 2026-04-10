![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Compiler CLI

- [Compiler CLI](#compiler-cli)
  - [Basic Compilation](#basic-compilation)
  - [Generating and Using a Custom Configuration](#generating-and-using-a-custom-configuration)
  - [Quantize Only](#quantize-only)
  - [Models with Dynamic Shapes](#models-with-dynamic-shapes)
  - [Using Real Images for Calibration](#using-real-images-for-calibration)
  - [Reusing CLI Arguments from a JSON File](#reusing-cli-arguments-from-a-json-file)
  - [All Available Options](#all-available-options)
  - [Compilation Artifacts](#compilation-artifacts)
  - [Compilation errors](#compilation-errors)

The Compiler CLI provides a command-line interface to compile and quantize models using the Axelera Compiler. It supports both ONNX models and pre-quantized models in the form of a manifest file. Model quantization can be performed using real or randomized image data.

## Basic Compilation

The simplest way to compile a model is:

```bash
axcompile -i /path/to/input/model.onnx -o /path/to/output/directory
```

This will:

* Compile the ONNX model using **default compiler configuration**.
* Save all output artifacts to the specified directory.

## Generating and Using a Custom Configuration

You can generate a default compiler configuration file with:

```bash
axcompile --generate-config /path/to/output/directory
```
This saves `default_conf.json` to the output directory. You can modify this file and use it later:

```bash
axcompile -i /path/to/input/model.onnx --conf /path/to/configuration/file -o /path/to/output/directory
```

The list of all compiler configurations can be found in [`Compiler Configurations`](/docs/reference/compiler_configs_full.md).

## Quantize Only

To perform quantization only and skip full compilation:

```bash
axcompile -i /path/to/input/model.onnx --quantize-only -o /path/to/output/directory
```

This produces `quantized_model_manifest.json`, which you can later pass to compile:

```bash
axcompile -i /path/to/quantized/model/quantized_model_manifest.json -o /path/to/output/directory
```

## Models with Dynamic Shapes

If your input model has dynamic input shapes, use --input-shape to provide a static shape that will be used for compilation and during inference.

```bash
axcompile -i /path/to/input/model.onnx --input-shape 1,3,224,224 -o /path/to/output/directory
```

## Using Real Images for Calibration

To use real images for calibration instead of random data:

```bash
axcompile -i /path/to/input/model.onnx \
  --input-shape 1,224,224,3 \
  --imageset /path/to/images \
  --transform /path/to/preprocess_transform.py \
  --input-data-layout NHWC \
  --color-format BGR \
  --imreader-backend OPENCV \
  -o /path/to/output/directory
```
Specify the color format of the input images using the --color-format option. Supported formats: `RGB` (default), `BGR` and `GRAY`.

Select the library used to read input images with the --imreader-backend option. Supported backends:
* `PIL` – Uses the Pillow library (default).
* `OPENCV` – Uses OpenCV for image loading. May offer better performance and more consistent handling of some image formats.

Your `preprocess_transform.py` file must define:

```python
def get_preprocess_transform(image: PIL.Image.Image | np.ndarray) -> torch.Tensor:
    ...
    return tensor
```

This function will be applied to each image during calibration. It must be implemented to match:
* The model’s expected input preprocessing steps (e.g., resizing, normalization).
* The selected --color-format (e.g., RGB, BGR, or GRAY).
* The selected --imreader-backend (e.g., PIL or OPENCV), as it determines the image object type (PIL.Image.Image or np.ndarray) and initial color channel ordering.

Make sure your transformation logic correctly interprets the input image based on the backend and converts it to a properly shaped and normalized PyTorch tensor that the model expects.

## Reusing CLI Arguments from a JSON File

Every compilation automatically generates a `cli_args.json` file in the output directory. This file contains all command-line arguments used during that compilation and can be reused in future runs.

To reuse a previously saved argument set:

```bash
axcompile -i new_model.onnx --cli-args /path/to/previous_run/cli_args.json --output /new/output/dir
```

Note: Any CLI arguments passed in the current invocation will override values from the --cli-args file.

## All Available Options

To see a full list of available flags:

```bash
axcompile --help
```

## Compilation Artifacts

Compilation artifacts are saved to the output directory in the following structure:

```bash
.
├── cli_args.json
├── conf.json
├── input_model
│   └── fp32_model.onnx
├── quantized_model
│   ├── quantized_model_manifest.json
│   ├── quantized_model.json
│   ├── quantized_model.txt
│   └── report.json
├── compiled_model
│   ├── manifest.json
│   ├── model.json
│   ├── quantized_model.json
│   ├── kernel_function.c 
│   ├── pool_l2_const.bin
│   └── report.json
├── compilation_report.json
└── compilation_log.txt
```

| File                            | Description                                                                                                                            |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `cli_args.json`                 | Dump of CLI arguments used at this compilation. Can be reused as input for another compilation.                                        |
| `conf.json`                     | Compiler configuration used at compilation                                                                                             |
| `input_model`                   | Directory that contains a copy of the input model                                                                                      |
| `quantized_model`               | Artifacts of model quantization. Contains the manifest of the quantized model that can be used as input for compilation.               |
| `compiled_model`                | Artifacts of model compilation. Contains the manifest of the compiled model that can be used as input for inference.                   |
| `compilation_report.json`       | Compilation status report. In case of failed compilation store error message and information at which compilation step network failed. |
| `compilation_log.txt`           | Full log output of the compilation process.                                                                                            |

## Compilation errors

The `compilation_report.json` file includes a status field indicating the outcome of the compilation. Below is a list of possible statuses and their meanings:

| Compilation Status       | Description                                                                                   |
|--------------------------|-----------------------------------------------------------------------------------------------|
| SUCCEEDED                | Compilation (quantization and/or lowering) process succeeded.                                 |
| INITIALIZE_CLI_ERROR     | Error during initialization of CLI.                                                           |
| ONNX_GRAPH_CLEANER_ERROR | Error during ONNX graph cleaning by ONNXGraphCleaner.                                         |
| QTOOLS_ERROR             | Error during quantization with QTools.                                                        |
| GRAPH_EXPORTER_ERROR     | Error during graph simplification and exporting to torchscript with GraphExporter.            |
| TVM_FRONTEND_ERROR       | Error during consuming of simplified graph with TVM PyTorch frontend.                         |
| TOP_LEVEL_QUANTIZE_ERROR | Error during `top_level.quantize()` call that is not covered by other statuses.               |
| LOWER_FRONTEND_ERROR     | Error during `axelera.compiler.pipeline.frontend.LowerFrontend` or `LowerHostEmulation` call. |
| FRONTEND_TO_MIDEND_ERROR | Error during `axelera.compiler.conversions.frontend_to_midend.FrontendToMidend` call.         |
| LOWER_MIDEND_ERROR       | Error during `axelera.compiler.pipeline.midend.LowerMidend` call.                             |
| MIDEND_TO_TIR_ERROR      | Error during `axelera.compiler.conversions.midend_to_tir.MidendToTIR` call.                   |
| LOWER_TIR_ERROR          | Error during `axelera.compiler.pipeline.backend.LowerTIR` call.                               |
| TIR_TO_RUNTIME_ERROR     | Error during `axelera.compiler.conversions.tir_to_runtime.TIRToRuntime` call.                 |
| TOP_LEVEL_LOWER_ERROR    | Error during `top_level.lower()` call that is not covered by other statuses.                  |
