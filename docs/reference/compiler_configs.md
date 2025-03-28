![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Compiler configuration parameters

- [Compiler configuration parameters](#compiler-configuration-parameters)
  - [Quantization config (`quantization_config`)](#quantization-config-quantization_config)
  - [Backend config (`backend_config`)](#backend-config-backend_config)

The compiler configuration parameters can be found in the `compilation_config` portion of `extra_kwargs` for each model in its corresponding YAML file. Subcategories of parameters are denoted either by `quantization_config` or `backend_config`, leading to a hierarchy which looks as follows:

```
models:
  model-name:
    class: XYZ
    ...
    extra_kwargs:
      compilation_config:
        quantization_config:
          ...
        backend_config:
          ...
```

## Quantization config (`quantization_config`)
| Parameter                     | Type   | Description |
| ----------------------------- | ------ | ----------- |
| `ptq_scheme`                  | string | This parameter determines which post-training quantization (PTQ) strategy to use. It can be one of the following three options:<ul><li>`per_tensor_histogram`: Applies per-tensor quantization with histogram observers.</li><li>`per_tensor_min_max`: Applies per-tensor quantization with min-max observers.</li><li>`hybrid_per_tensor_per_channel`: Combines per-channel quantization (when applicable) with per-tensor quantization as a fallback for unsupported cases.</li></ul> The default setting is `per_tensor_histogram` which should only be changed based on recommendations made by appropriate Axelera AI representatives. |

## Backend config (`backend_config`)
| Parameter                     | Type  | Description |
| ----------------------------- | ----- | ----------- |
| `elf_in_ddr`                  | bool  | Place the ELF binary with compiled neural network into DDR memory. This increases the available L2 memory for intermediate data and hence can result in increased performance. For very small networks, which do not utilize the full memory to begin with, it can be beneficial to set this to `false`. The regular case should be `true`.        |
| `in_core_replication`         | bool  | Replicate weights inside the in-memory-compute (IMC) array to maximise array utilisation. This increases performance but comes at the expense of larger weights. It should always be set to `true` unless otherwise advised e.g. due to very network-specific restrictions.        |
| `l2_reserved_nbytes_tasklist` | uint  | Amount of memory to reserve in L2 for the ELF binary with compiled neural network. If `elf_in_ddr` is set to `true`, this must be set to `0`. Otherwise, a good size is 1MB, i.e. `1048576`.        |
| `page_memory`                 | bool  | Activate static memory paging to allow for more fine-grained placement of buffers across the memory hierarchy. For memory intensive networks in particular, this can give a noticeable performance boost and should always be set to `true`.        |
| `tiling_depth`                | uint  | If set to `1`, this feature is disabled. If set to a value greater than `1`, depth-first tiling is enabled which allows computation of cones of data along the depth-dimension of a vision network. The result is a reduction in intermediate memory accesses, leading to increased performance. The value of `6` is recommended as the default value if this feature is turned on. The provided value is equal to the maximum number of subsequent operators to be fused. This feature must only be turned on for selected networks based on recommendations made by appropriate Axelera AI representatives.        |
