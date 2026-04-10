![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Deploy

- [Deploy](#deploy)
  - [Configuration and Advanced Usage](#configuration-and-advanced-usage)
    - [--build-root ](#--build-root-)
    - [--data-root ](#--data-root-)
    - [--model , --models-only, --pipeline-only](#--model----models-only---pipeline-only)
    - [--mode ](#--mode-)
    - [--pipe `<type>`](#--pipe-type)
    - [--export](#--export)
    - [--aipu-cores `<core-count>`](#--aipu-cores-core-count)
    - [-num-cal-images ](#-num-cal-images-)

Axelera's deployment tool is used to and compile and deploy a given network, for the required
system and pipeline. After deployment, the network can be used for inference or as part
of an application.

Typical usage of the deploy tool is as follows:

```bash
./deploy.py <network-name>
```

Where `<network-name>` is the name of the network to deploy. Only one network may be
provided for deployment at a time.

Deploying a network for the first time may take some time, typically minutes, depending on
the system configuration. After deployment, the network may be reused in applications without
redeploying.

After deployment, new files will be created in a specified build directory (by default, a
`build/` directory in your current directory, but this can be configured. See advanced
usage). The file path gives information about the network deployment. It will typically
look like `build/<network-name>/<model-name>/model_info.json`.

As we can see from this path, the top level is the name of the network. Under each
network, each individual model in the network will be deployed into its own folder,
denoted by the model name. Finally, within each model, there will be a `model_info.json`
file, and other model-specific requirements where relevant.

## Configuration and Advanced Usage

Numerous configurations can be made to the deployment by using command line arguments. All
options may be listed via the `./deploy.py --help` command, but the most common ones can be found below:

### --build-root <path>

This configuration can be used to overwrite the default build directory to deploy to. A path
can be provided to deploy networks to somewhere other than `build/` in your current directory.

### --data-root <path>

Networks will typically require the corresponding datasets for their models to be present to
compile and deploy. If these datasets are downloaded somewhere other than `data/` in your
current directory, this configuration can be used to point to it.

### --model <model>, --models-only, --pipeline-only

Typically, a deployment run will deploy the entire network. This will involve compiling all
the models and deploying the pipeline. This collection of configurations can be used to only
perform specific parts of the deployment run.

`--model <model>` can be used to only compile the specified model within the network, and
no further deployment will take place.

When using `--models-only`, all models in the network will be compiled, but the pipeline will
not be deployed.

`--pipeline-only` can be used to only deploy the pipeline, without compiling any of the models.
This can be useful if the models are pre-compiled, or if the Axelera compiler is unavailable.

### --mode <mode>

This configuration is used to specify which mode the model(s) in the network should be deployed
with. Which option to use depends on whether your model is already quantized or not. There are
three options:

`QUANTIZE`, which will quantize the model. The pipeline will not be deployed in this mode.

`QUANTCOMPILE`, is the same as `QUANTIZE`, but the model will also be compiled.

`PREQUANTIZED`, is the default option. This checks if a prequantized model is available, first locally
in the build directory, then in the network YAML if there is a prequantized_url to download from. If no
prequantized model is available, then one is generated locally. After prequantization is done via one
of these methods, the model is compiled.

### --pipe `<type>`

The pipe argument can be used to specify which pipeline type to deploy for. The options are the
`torch` pipeline for a complete PyTorch-based pipeline that uses ONNXRuntime if the model is in ONNX format,
`gst` for a C/C++ pipeline using GStreamer with the core model on AIPU, and `torch-aipu` for a PyTorch
pipeline with the core model offloaded to AIPU. GStreamer will always use the AIPU.

The default pipeline is GStreamer (`gst`)

### --export

This configuration can be used to create a zip containing the deployed/quantized model in addition
to the regular output. The zip is saved to the `exported/` directory. For `QUANTIZE` mode, the zip
is named `<model>-prequantized.zip`. For other modes, it is named `<model>.zip`.

### --aipu-cores `<core-count>`

This configuration can be used to specify how many of the AIPU cores the deployed network will be able to be
run with. By default, this will be all the cores available (currently 4). This command can
be used to restrict the cores to use to fewer, such as `1`, `2` or `3`.

Note this configuration can only be used if the network has a single model. If there are multiple models, the
AIPU cores to use for each model needs to be specified in the network YAML.


### -num-cal-images <count>

This configuration specifies the number of images used to quantize the models in the network. By default, 200 images are used. We encourage you to experiment with 100 to 400 images, as typically, 100 images can yield good results, but sometimes 200, 300, or 400 images can give better results, depending on the dataset.

## What Happens During Deployment

When you run `./deploy.py <network-name>`, the following process occurs:

### 1. Calibration Data Preparation

**Purpose:** Prepare calibration images for the quantization process

**Process:**
1. Load model YAML configuration
2. Parse `preprocess:` section (Resize, Normalize, etc.)
3. Load calibration images from dataset (default: 200 images)
4. Apply YAML preprocessing to calibration images
5. Preprocessed images are ready for compilation phase

**Example YAML:**
```yaml
preprocess:
  - letterbox:
      width: 640
      height: 640
  - torch-totensor:
  - normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

**Key point:** YAML preprocessing is ONLY used during deployment to prepare calibration data. It is NOT part of the runtime inference pipeline.

---

### 2. Compilation Phase (Automatic)

**Purpose:** Compile model for Axelera hardware with optimal accuracy

The compilation phase consists of three sub-steps that happen automatically:

#### 2.1 Graph Partitioning
- Compiler analyzes ONNX graph structure
- Identifies operations better suited for CPU vs AIPU
- Extracts preamble (preprocessing before AIPU) → saves `preamble.onnx`
- Extracts postamble (postprocessing after AIPU) → saves `postamble.onnx`
- **This extraction is automatic and CANNOT be disabled**

#### 2.2 Quantization
- Apply preamble to preprocessed calibration data
- Use calibration data to convert model from float32 to int8
- Calibrate quantization parameters (scales, zero_points) to achieve best accuracy
- Optimize for minimal accuracy loss

#### 2.3 Hardware Compilation
- Compile quantized model for Metis AIPU
- Optimize for hardware constraints and performance
- Generate binary model file (model.axm)

**Key point:** Preamble is extracted AND used during quantization. The `handle_*` flags in the inference section control whether preamble/postamble are USED at runtime, not whether they're extracted or used during compilation.

---

### 3. Save Artifacts

**Output directory structure:**
```
build/<network-name>/<model-name>/
├── manifest.json                    # Model metadata
├── quantized/
│   ├── model.axm                    # Compiled AIPU model
│   ├── preamble.onnx               # CPU preprocessing (if extracted)
│   ├── postamble.onnx              # CPU postprocessing (if extracted)
│   └── quantization_params.json    # Scales, zero_points, padding
└── model_info.json                  # Model configuration
```

**manifest.json includes:**
- Model input/output shapes
- Quantization parameters (scales, zero_points)
- Padding information (for hardware alignment)
- Paths to preamble.onnx and postamble.onnx (if extracted)
- Reference to compiled model file

---

## Next Steps: Running Inference

After deployment completes, your model is ready to use. The deployment process has created all necessary artifacts (quantized model, preamble.onnx, postamble.onnx, etc.).

**To run your deployed model:**
- **[Inference Tool Usage](inference.md)** - How to run `./inference.py` and CLI options
- **[Inference Configuration Guide](inference_configuration.md)** - How to configure runtime behavior using `handle_*` flags in your YAML

**Key distinction:**
- **Deployment** (this guide) creates the compiled model artifacts
- **Inference** (above guides) runs the model and controls runtime behavior

The `inference:` section in your YAML (with `handle_*` flags) is IGNORED during deployment - it only affects runtime behavior when you run inference.

## See Also

- **[Compiler Configuration](compiler_configs.md)** - How to configure compiler settings via TOML files or inline YAML
