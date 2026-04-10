![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Compiler Configuration

- [Compiler Configuration](#compiler-configuration)
  - [Overview](#overview)
  - [Configuration Methods](#configuration-methods)
    - [Method 1: Inline Configuration](#method-1-inline-configuration)
    - [Method 2: TOML Configuration Files (Recommended)](#method-2-toml-configuration-files-recommended)
  - [Using SDK-Provided TOML Configs](#using-sdk-provided-toml-configs)
    - [Discovering Available Configs](#discovering-available-configs)
    - [Using SDK Configs with Custom Models](#using-sdk-configs-with-custom-models)
  - [Creating Custom TOML Configs](#creating-custom-toml-configs)
  - [Configuration Precedence](#configuration-precedence)
  - [Multi-Core Configuration](#multi-core-configuration)

## Overview

Compiler configuration parameters control how your model is compiled for Metis hardware. These settings are specified in the `extra_kwargs` section of your model's YAML file.

There are two ways to provide compiler settings:
1. **Inline configuration** - Embed settings directly in YAML (legacy approach)
2. **TOML configuration files** - Reference external TOML files (recommended)

## Configuration Methods

### Method 1: Inline Configuration

Embed compiler settings directly in the YAML using `compilation_config`:

```yaml
models:
  model-name:
    class: AxONNXModel
    extra_kwargs:
      compilation_config:
        quantization_scheme: per_tensor_min_max
        ignore_weight_buffers: false
        aipu_cores_used: 1
        resources_used: 0.25
```

**When to use:** Quick experiments or model-specific settings that won't be reused.

A full list of all configuration parameters is available in the [Full Compiler Config Reference](/docs/reference/compiler_configs_full.md).

### Method 2: TOML Configuration Files (Recommended)

Reference an external TOML file using `compiler_config_file`:

```yaml
models:
  model-name:
    class: AxONNXModel
    extra_kwargs:
      compiler_config_file: yolo11n.toml
```

**Benefits:**
- **Reusable** across multiple models with the same architecture
- **Pre-tested** SDK-provided configs for common model architectures
- **Cleaner YAMLs** with separated concerns
- **Flexible** - can override individual settings with inline `compilation_config` if needed

**When to use:** Production deployments, custom models based on known architectures, or when sharing configurations across models.

## Using SDK-Provided TOML Configs

The SDK includes pre-tested TOML configurations for common model architectures. These are installed with the `axelera_compiler` wheel in the `axelera/compiler/config/models/` directory.

### Discovering Available Configs

**List all available SDK-provided TOML configs:**

```bash
python -c "import axelera.compiler.config as c; from pathlib import Path; \
  for f in (Path(c.__file__).parent / 'models').glob('*.toml'): print(f.name)"
```

Example output:
```
yolo11n.toml
yolo11s.toml
yolo11.toml
yolov8s.toml
...
```

### Using SDK Configs with Custom Models

If you've trained your own model using a standard architecture (like YOLO11n, ResNet, etc.), you can use the SDK's pre-tested configuration for that architecture.

**Example: Custom YOLO11n model with your own weights**

```yaml
models:
  my-custom-yolo11n:
    class: AxONNXModel
    weight_path: my_custom_weights.onnx              # Your trained weights
    extra_kwargs:
      compiler_config_file: yolo11n.toml  # SDK-provided config
```

The SDK's model YAML files already reference the most appropriate TOML config for each model. You can use these as templates when deploying custom models based on the same architecture.

**Path resolution order (for filename-only references like `yolo11n.toml`):**
1. Relative to your YAML file location (local custom configs take precedence)
2. Compiler's `axelera/compiler/config/models/` directory (SDK-provided configs)

Relative paths with directories (e.g., `configs/my.toml`) resolve relative to the YAML file. Absolute paths are used as-is.

## Creating Custom TOML Configs

For models with unique architectures or when you need to tune compiler settings, create custom TOML configuration files.

**Option 1: Copy and modify an existing SDK TOML**

```bash
# Copy an SDK TOML to your project
python -c "import axelera.compiler.config as c; from pathlib import Path; import shutil; \
  src = Path(c.__file__).parent / 'models' / 'yolo11n.toml'; \
  shutil.copy(src, 'my-custom-config.toml')"

# Edit my-custom-config.toml with your settings
```

Then reference it in your YAML:
```yaml
extra_kwargs:
  compiler_config_file: my-custom-config.toml  # Relative to YAML file
```

**Option 2: Convert existing inline config to TOML**

```bash
python tools/json_to_toml_config.py --from-yaml ax_models/model_cards/my-model.yaml
```

This creates `my-model.toml` with your current inline settings from the YAML.

**Option 3: Create TOML from scratch**

Create a `.toml` file with compiler settings:

```toml
# my-config.toml
quantization_scheme = "per_tensor_min_max"
ignore_weight_buffers = false
aipu_cores_used = 1
resources_used = 0.25
```

See [Full Compiler Config Reference](/docs/reference/compiler_configs_full.md) for all available parameters.

## Configuration Precedence

When both TOML and inline configs are present, settings are merged with this priority (highest to lowest):

1. **Inline YAML** `compilation_config` (highest - always wins)
2. **TOML file** via `compiler_config_file`
3. **Default values** from CompilerConfig

**Example: Override specific TOML settings**

```yaml
models:
  my-model:
    class: AxONNXModel
    extra_kwargs:
      compiler_config_file: base-config.toml
      compilation_config:
        quantization_scheme: per_tensor_histogram  # Overrides TOML value
```

This pattern is useful when you want to use a base configuration but tweak specific parameters for experimentation.

## Multi-Core Configuration

By default, models are compiled for single-core execution. For multi-core execution, there are two approaches:

### Batch-1 Multi-Core (Lower Latency)

Each core runs independently with constrained resources. This allows asynchronous frame dispatch to available cores, resulting in **lower latency** since cores begin execution immediately without waiting for other inputs.

**4-core Batch-1 configuration:**
```toml
aipu_cores_used = 1
resources_used = 0.25  # 1.0 / 4 cores
```

The runtime instantiates the model **4 times** (once per core). The `axrunmodel` and `axinferencenet` libraries handle multi-core dispatch automatically. See [axrunmodel.md](/docs/reference/axrunmodel.md) and [axinferencenet.md](/docs/reference/axinferencenet.md) for details.

**Trade-off:** Each core has access to only a proportion of on-chip memory, which may limit what the compiler can optimize per core.

### Batch-X Multi-Core

Multiple cores execute in a co-dependent way, processing a batch of inputs jointly. The compiler can optimize memory allocation across all cores together.

**4-core Batch-4 configuration:**
```toml
aipu_cores_used = 4
resources_used = 1.0
```

The runtime instantiates the model **once** for all 4 cores, executing at the granularity of 4 inputs jointly.

**Trade-off:** Higher latency since all cores must wait for a full batch. Whether this improves throughput depends on the model - smaller models tend to benefit more from batching, while larger models may see better throughput with Batch-1.

### Mixed Multi-Core Pipelines

For pipelines with multiple models, you can allocate different core counts to different models:

```yaml
# Model A uses 3 cores
model-a:
  extra_kwargs:
    compilation_config:
      aipu_cores_used: 3
      resources_used: 0.75  # 3 × 0.25

# Model B uses 1 core
model-b:
  extra_kwargs:
    compilation_config:
      aipu_cores_used: 1
      resources_used: 0.25  # 1 × 0.25
```

**Constraints:**
- `resources_used` must be a multiple of `0.25`
- Sum of all `resources_used` across models must be <= `1.0`

Note: The `max_compiler_cores` parameter also affects multi-core compilation. When deploying with `--deploy-cores N`, the framework uses `max_compiler_cores` to constrain how many cores the compiler targets. This interacts with `aipu_cores_used` and `resources_used` - see `deploy.py --help` for details.

### Choosing Between Batch-1 and Batch-X

| Metric | Batch-1 Multi-Core | Batch-X Multi-Core |
|--------|-------------------|-------------------|
| **Latency** | Lower (cores run independently) | Higher (cores wait for full batch) |
| **Throughput** | Often higher for larger models | Can be higher for smaller models |
| **Use Case** | Real-time applications, low-latency requirements | Smaller models where batch throughput helps |

## See Also

- [Deploy Reference](/docs/reference/deploy.md) - Deployment command-line options
- [Full Compiler Config Reference](/docs/reference/compiler_configs_full.md) - All configuration parameters
- [axrunmodel.md](/docs/reference/axrunmodel.md) - Low-level multi-core runtime
- [axinferencenet.md](/docs/reference/axinferencenet.md) - High-level multi-core inference
