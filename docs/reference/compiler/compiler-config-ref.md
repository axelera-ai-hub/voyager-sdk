---
title: "CompilerConfig Reference"
---
# CompilerConfig Reference

Complete property listing for `axelera.compiler.CompilerConfig`. For usage examples and the compilation workflow, see [Compiler Python API](compiler-api.md).

```python
from axelera.compiler import CompilerConfig

config = CompilerConfig()
config.aipu_cores_used = 4
config.multicore_mode = "batch"
```

---

## Properties

### Quantization

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `compiler_mode` | `CompilerMode` | `"quantize_and_lower"` | Controls what the compiler does: quantize only, lower only, or both. |
| `quantization_scheme` | `QuantizationScheme` | `"per_tensor_histogram"` | Quantization algorithm for activations and weights. |
| `quantize_dw_channel_wise` | boolean | `false` | Quantize depthwise convolutions channel-wise instead of per-tensor. |
| `onnx_opset_version` | integer ≥ 17 | `17` | ONNX opset version used during PyTorch → ONNX conversion. |
| `quantization_debug` | boolean | `false` | Dump model after quantization for accuracy debugging. |
| `remove_io_quantization` | boolean | `false` | Remove quantize/dequantize ops from graph inputs and outputs. |
| `quantized_graph_export` | boolean | `true` | Export the quantized graph to a JSON file. |

### Graph transformations

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `apply_pword_padding` | boolean | `true` | Pad input channels to a multiple of PWORD (hardware requirement). |
| `rewrite_concat_to_resadd` | boolean | `true` | Convert concatenation ops to pad+shift+add (no native concat on hardware). |
| `rewrite_dense_to_conv2d` | boolean | `true` | Canonicalise `nn.dense` to `conv2d` with 1×1 kernel. |
| `remove_io_padding_and_layout_transform` | boolean | `true` | Remove padding/transpose ops from the graph when the SDK pipeline handles them. Leave `true` unless running without the SDK preprocessing elements. |
| `apply_arithmetic_simplification` | boolean | `true` | Fast-math simplifications. May not preserve numerics exactly. |
| `simplify_mac_before_after_lut` | boolean | `true` | Optimize multiply-add ops around LUT operations. May not preserve numerics exactly. |
| `validate_operators` | boolean | `true` | Check operator compatibility with hardware before compilation. |

### Graph cleaner

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `run_graph_cleaner` | boolean | `true` | Run the ONNX graph cleaner to remove pre/post-processing ops that belong on the host. |
| `graph_cleaner_split_pre_post_processing` | boolean | `true` | Split pre- and post-processing into separate passes in the cleaner. |
| `graph_cleaner_condition` | `GraphCleanerCondition` or null | `null` | Condition used to identify the split point. |
| `graph_cleaner_node` | `GraphCleanerNode` or null | `null` | Node type the condition applies to. |
| `graph_cleaner_threshold` | integer | `0` | Threshold value for the condition. |
| `graph_cleaner_dump_core_onnx` | string or null | `null` | Filename to save the core ONNX model after cleaning (for debugging). |
| `graph_cleaner_dump_full_opt_onnx` | string or null | `null` | Filename to save the full optimized ONNX model after cleaning. |
| `remove_layout_transform_from_preamble` | boolean | `false` | Remove transpose/reshape from model preamble for NHWC layouts. |

### Multi-core and scheduling

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `aipu_cores_used` | integer 1–4 | `1` | Number of AIPU cores to compile for. |
| `multicore_mode` | `MulticoreMode` | `"multiprocess"` | How cores share work. See [Multi-Core Configuration](compiler-configs.md). |
| `pipeline_spatial_tiles` | boolean | `true` | Software-pipeline tasks across height tiles. |
| `pipeline_channel_tiles` | boolean | `true` | Software-pipeline tasks across output channel tiles. |
| `inter_operator_async` | boolean | `true` | Asynchronous scheduling between sequential operators. |
| `use_list_scheduler` | boolean | `false` | Use resource-aware list scheduler for async scheduling. |
| `unroll_prologue_epilogue` | boolean | `false` | Unroll prologue/epilogue loops to expose more parallelism. |
| `use_hw_tokens` | boolean | `true` | Use hardware tokens for synchronisation. |
| `group_ifdw_tasks` | boolean | `false` | Hoist IFDW operations earlier for better IMC weightset utilization. |
| `double_buffer` | boolean | `true` | Double-buffer host↔device data transfers to hide latency. |
| `host_processes_used` | integer | `1` | Number of host processes for execution. |

### Tiling and memory planning

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `max_memplan_attempts` | integer ≥ 1 | `5` | Max iterations for the memory planner to find a valid configuration. |
| `max_tiling_attempts` | integer ≥ 1 | `8` | Max attempts to tile an operator to fit in memory. |
| `force_h_tiling` | integer or null | `null` | Force a specific height-dimension tiling factor. |
| `force_oc_tiling` | integer or null | `null` | Force a specific output-channel tiling factor. |
| `tiling_depth` | integer ≥ 1 | `1` | Max tiling depth. Set to `6` to enable depth-first scheduling; `1` disables it. |
| `dfs_search_constraint` | integer or null | `null` | Limit depth-first search space. Set to `1` for large networks. |

### Memory hierarchy

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `enable_buffer_promotion` | boolean | `true` | Allow buffers to be promoted from DDR → L2 → L1 for faster access. |
| `split_buffer_promotion` | boolean | `false` | Split promotion into two passes (L1 and L2) with DFS scheduling in between. |
| `io_memory_pool` | string | `"global.ddr"` | Initial home for I/O buffers. One of: `global.ddr`, `global.l2`, `global.l1`. |
| `constant_memory_pool` | string | `"global.ddr"` | Initial home for constant buffers. |
| `workspace_memory_pool` | string | `"global.ddr"` | Initial home for workspace buffers. |
| `l1_constraint` | integer or null | `null` | Max L1 memory the compiler may use (bytes). |
| `l2_constraint` | integer or null | `null` | Max L2 memory the compiler may use (bytes). |
| `ddr_constraint` | integer or null | `null` | Max DDR memory the compiler may use (bytes). |
| `l1_size_used` | integer | `4194304` | L1 memory available to the compiler (bytes). Default: 4 MB. |
| `l2_size_used` | integer | `33554432` | L2 memory available to the compiler (bytes). Default: 32 MB. |
| `ddr_size_used` | integer | `1073741824` | DDR memory available to the compiler (bytes). Default: 1 GB. |
| `use_sysdma` | boolean | `false` | Use System DMA for DDR→L2 weight transfers (Core DMA handles L2→L1). |
| `dma_dual_channel` | boolean | `true` | Enable dual-channel DMA optimization. |
| `page_memory` | boolean | `true` | Apply paging to reduce L2/DDR fragmentation. |
| `elf_in_ddr` | boolean | `true` | Store ELF file in DDR rather than L2. |
| `stream_tasklist` | boolean | `true` | Stream tasklist chunks into AIPU memory on-the-fly. |
| `dpu_constants_home` | string | `"global.l2"` | Where DPU constants reside. One of: `global.ddr`, `global.l2`. |

### IMC and MVM

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `mvm_utilization_limit` | float 0.125–1.0 | `1.0` | Fraction of MVM array active MACs. Reduce to lower power consumption. |
| `enable_icr` | boolean | `true` | In-Core Replication for layers with small output-channel counts. |
| `icrx_force` | integer | `-1` | Force a specific ICR factor. `-1` = automatic. |
| `icrx_parallel_block_threshold` | integer 1–4 | `4` | Maximum number of parallel IMC blocks for which ICR is still applied. |
| `icrx_max_factor` | integer 2–8 | `8` | Maximum ICR factor along the X image direction. |
| `enable_swicr` | boolean | `true` | Subword ICR for first layers with small input-channel counts. |
| `imc_double_buffer_pipeline` | boolean | `false` | Double-buffer weight loading in software-pipelined sections. |
| `imc_double_buffer_sequential` | boolean | `false` | Double-buffer weight loading in sequential IR sections. |
| `dpu_allocation_algorithm` | `DPUAllocationAlgorithm` | `"try_all"` | Register-allocation algorithm for the DPU vector unit. |
| `softmax_neutral_value` | float | `-100000.0` | Padding value for softmax inputs. Chosen so the softmax LUT outputs zero for padding elements without affecting non-padding numerics. |

### Output and paths

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `output_dir` | path string | — | Directory for compiler outputs and deployment artifacts. |
| `remove_output_dir` | boolean | `false` | Clean the output directory before compilation. |
| `model_name` | string | `""` | Model name used in logging and to select model-specific optimization presets. |
| `save_error_artifact` | boolean | `false` | Save a ZIP archive with the lowered model and error messages on failure. |
| `randomize_onnx_model` | boolean | `true` | Randomise cached ONNX model weights before saving the error artifact. |

### Hardware and runtime

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `frequency` | integer 20 MHz–800 MHz | `800000000` | Device clock frequency in Hz. |
| `resources_used` | float (0, 1] | `1.0` | Fraction of memory resources the compiled model may use. |
| `input_dmabuf` | boolean | `false` | Use DMA for input data transfer. |
| `output_dmabuf` | boolean | `false` | Use DMA for output data transfer. |

### Profiling and debugging

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `profiling_levels` | list of `ProfilingLevel` | `[]` | Profiling trace levels to enable. Enabling multiple simultaneously can reduce accuracy. |
| `profiling_drop_percentile` | float | `0.25` | Drop top and bottom N% of profiling samples to remove outliers. |
| `trace_tvm_passes` | boolean | `false` | Trace TVM pass execution (start/end times, parent-child relationships). Output: `pass_dependency_graph.json`. |
| `propagate_span_information` | boolean | `true` | Propagate source-span information through the compiler. |
| `model_debug_save_dir` | path string | — | Directory to save the quantized/optimized model for debugging. |
| `quantization_debug` | boolean | `false` | Dump the quantized model for accuracy measurement and debugging. |

### Advanced — internal paths

These are resolved automatically by the SDK. Override only if your installation is non-standard or you are running the compiler outside the SDK environment.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `compiler_dir` | path string | — | Root directory of the compiler package, used to locate internal resources and dependencies. |
| `runtime_dir` | path string | — | Directory for the Axelera runtime. |
| `device_dir` | path string | — | Directory for device resources. |

### Advanced — memory layout constants

Hardware memory map values. **Do not change** unless directed by Axelera support — incorrect values will prevent models from running.

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `l1_size_reserved` | integer | `524288` | L1 memory reserved for the system (bytes). Default: 512 KB. |
| `l2_size_reserved` | integer | `1245184` | L2 memory reserved for the system (bytes). |
| `l2_size_reserved_tasklist` | integer | `1048576` | L2 memory reserved per core for the tasklist (bytes). Default: 1 MB. |
| `ddr_size_max` | integer | `1073741824` | Total DDR memory size (bytes). Default: 1 GB. |
| `ddr_size_reserved` | integer | `33554432` | DDR memory reserved for the system (bytes). Default: 32 MB. |
| `l1_virtual_address` | integer | `206175207424` | Virtual address of L1 memory. Copied from `mmap_config.h`. |
| `l1_core0_physical_address` | integer | `402653184` | Physical address of L1 memory in core 0. Copied from `memorymap.h`. Required because the AIPU simulator does not support virtual memory. |
| `dpu_instructions_home` | string | `"default"` | Where DPU instructions are placed. One of: `"default"`, `"l2"`. |
| `dwpu_instructions_home` | string | `"default"` | Where DWPU instructions are placed. One of: `"default"`, `"l2"`. |
| `ignore_weight_buffers` | boolean | `true` | Exclude weight buffers when determining tiling factors during memory scheduling. |

---

## Enum definitions

### `CompilerMode`

| Value | Description |
|-------|-------------|
| `"quantize_and_lower"` | Quantize the model then compile to hardware binary (default). |
| `"quantize_only"` | Quantize only — produces a model that runs on CPU for accuracy validation. |
| `"lower_only"` | Compile a pre-quantized model to hardware binary without re-quantizing. |

### `MulticoreMode`

| Value | Description |
|-------|-------------|
| `"multiprocess"` | Each core runs a separate OS process (default). |
| `"multithread"` | Cores share a process with multiple threads. |
| `"batch"` | Different items in a batch run on different cores simultaneously. |
| `"cooperative"` | Cores cooperate on a single inference. |
| `"pipeline"` | Model is split across cores as a pipeline stage. |

See [Multi-Core Configuration](compiler-configs.md) for when to use each mode.

### `QuantizationScheme`

| Value | Description |
|-------|-------------|
| `"per_tensor_histogram"` | Activations quantized per-tensor with histogram observer; weights per-channel with min-max (default). |
| `"per_tensor_min_max"` | Activations quantized per-tensor with min-max observer; weights per-channel with min-max. |
| `"hybrid_per_tensor_per_channel"` | Activations per-tensor (histogram), except depth-wise convolution inputs (per-channel min-max); weights per-channel min-max. |

### `DPUAllocationAlgorithm`

| Value | Description |
|-------|-------------|
| `"try_all"` | Try all algorithms in sequence until one succeeds (default). |
| `"graph"` | Graph-coloring register allocator. |
| `"lazy"` | Lazy (greedy) allocator. |
| `"backjump_recursive"` | Backtracking recursive allocator. |

### `GraphCleanerCondition`

| Value | Description |
|-------|-------------|
| `"maximum_weight_tensor_size"` | Split on the node with the largest weight tensor. |
| `"maximum_weight_tensor_first_dimension_size"` | Split on the node with the largest first weight dimension. |

### `GraphCleanerNode`

| Value | Description |
|-------|-------------|
| `"MatMul"` | Apply the condition to MatMul nodes. |
| `"Gemm"` | Apply the condition to Gemm nodes. |
| `"Clip"` | Apply the condition to Clip nodes. |

### `ProfilingLevel`

Trace-line type identifiers used in the profiling output file.

`[?]` · `[B]` · `[PB]` · `[PE]` · `[K]` · `[M]` · `[T]`
