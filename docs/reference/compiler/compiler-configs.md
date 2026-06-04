---
title: "Compiler Configuration"
---
# Compiler Configuration

How to configure the compiler for model deployments, including the recommended TOML-based configuration and multi-core options.

---

## Configuration methods

There are two ways to provide compiler settings:

### Method 1: Inline YAML (legacy)

Embed compiler settings directly in `compilation_config`:

```yaml
models:
  my-model:
    class: AxUltralyticsYOLO
    extra_kwargs:
      compilation_config:
        aipu_cores_used: 4
        resources_used: 1.0
```

### Method 2: TOML configuration files (recommended)

Reference an external TOML file using `compiler_config_file`:

```yaml
models:
  my-model:
    class: AxONNXModel
    extra_kwargs:
      compiler_config_file: yolo11n.toml
```

Benefits:
- **Reusable** across multiple models with the same architecture
- **Pre-tested** SDK-provided configs for common architectures
- **Cleaner YAMLs** with separated concerns

### SDK-provided TOML configs

The SDK ships pre-tested configs in `axelera/compiler/config/models/`. List them:

```bash
python -c "import axelera.compiler.config as c; from pathlib import Path; \
  for f in (Path(c.__file__).parent / 'models').glob('*.toml'): print(f.name)"
```

Use an SDK config with your custom model:

```yaml
models:
  my-custom-yolo11n:
    class: AxONNXModel
    weight_path: my_custom_weights.onnx
    extra_kwargs:
      compiler_config_file: yolo11n.toml
```

**Path resolution:** filename-only references check your YAML's directory first, then the SDK's config directory. Relative paths resolve from the YAML location. Absolute paths are used as-is.

### Creating custom TOML configs

```toml
# my-config.toml
quantization_scheme = "per_tensor_min_max"
ignore_weight_buffers = false
aipu_cores_used = 1
resources_used = 0.25
```

Or convert an existing inline config:
```bash
python tools/json_to_toml_config.py --from-yaml ax_models/model_cards/my-model.yaml
```

### Configuration precedence

When both TOML and inline configs are present:

1. **Inline YAML** `compilation_config` (highest — always wins)
2. **TOML file** via `compiler_config_file`
3. **Default values** from CompilerConfig

This lets you use a base TOML and override specific settings for experiments:

```yaml
extra_kwargs:
  compiler_config_file: base-config.toml
  compilation_config:
    quantization_scheme: per_tensor_histogram  # overrides TOML value
```

---

## Multi-core modes

By default, a model is compiled for a single core. To use more cores, choose one of two approaches:

### Batch-1 mode (multiple instances)

Compile for one core with constrained resources, then the runtime instantiates the model multiple times — one instance per core. Cores are independent, which gives **lower latency** (each core can start a new frame immediately).

```yaml
compilation_config:
  aipu_cores_used: 1
  resources_used: 0.25    # = 1/4 of memory for 4-core execution
```

The runtime (axinferencenet or InferenceStream) automatically instantiates 4 copies and dispatches frames across them.

### Batch-4 mode (shared memory)

Compile for 4 cores sharing all available on-chip memory. The compiler can optimize memory allocation across all cores jointly, which gives **higher throughput** on some models — but each inference step requires all 4 cores simultaneously, so latency is higher.

```yaml
compilation_config:
  aipu_cores_used: 4
  resources_used: 1.0
```

### Choosing between modes

| | Batch-1 | Batch-4 |
|--|---------|---------|
| Latency | Lower | Higher |
| Throughput | Good | Potentially higher |
| Flexibility | Can split cores across models | All 4 cores tied to one model |

When in doubt, start with Batch-1. Batch-4 is worth trying on compute-heavy models where memory bandwidth is the bottleneck.

### Splitting cores across models in a cascade

When a pipeline has multiple models, allocate cores so that `resources_used` values sum to ≤ 1.0:

```yaml
# Model A: 3 cores
compilation_config:
  aipu_cores_used: 3
  resources_used: 0.75

# Model B: 1 core
compilation_config:
  aipu_cores_used: 1
  resources_used: 0.25
```

Each `resources_used` value must be a multiple of 0.25.

---

## Python API: CompilerConfig

When using the [Compiler Python API](compiler-api.md), pass the same settings via `CompilerConfig`:

```python
from axelera.compiler import CompilerConfig

# Batch-1, 4 cores
config = CompilerConfig(aipu_cores=1, resources=0.25)

# Batch-4, 4 cores
config = CompilerConfig(aipu_cores=4, resources=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aipu_cores` | int | all available | Number of AIPU cores targeted |
| `resources` | float | `1.0` | Fraction of on-chip memory to use (must be multiple of 0.25) |
| `ptq_scheme` | string | `"per_tensor_histogram"` | Quantization scheme: `per_tensor_min_max` or `per_tensor_histogram` |
| `save_error_artifact` | bool | `False` | Keep intermediate files when compilation fails |

Full list of all `CompilerConfig` properties: see `compiler_configs_full.md` in the SDK reference (`docs/reference/`).

---

## See also

- [Compiler Python API](compiler-api.md) — programmatic compilation with `CompilerConfig`
- [Compiler CLI](compiler-cli.md) — command-line compilation with `--aipu-cores`
- [axrunmodel](../tools/axrunmodel.md) — verify multi-core performance after compilation
