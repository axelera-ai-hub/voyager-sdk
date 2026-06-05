---
title: "axelera.runtime API"
---
# axelera.runtime API

## Introduction

The axelera.runtime API provides low-level control over Metis accelerators through a hierarchical object model. All objects are owned by a Context, which manages device connections and model instances. Models run quantized INT8 inference requiring explicit padding, quantization, depadding, and dequantization transformations.

**Object Hierarchy:**
```
Context (root, manages all resources)
├── Model (loaded model metadata and weights)
└── Connection (reserved hardware resources)
    └── ModelInstance (executable model on hardware)
```

When Context is released, all child objects are automatically released.

**Quantized Inference Pipeline:**
```
Float32 Input
    ↓ Quantization
INT8 Quantized
    ↓ Padding
INT8 Padded
    ↓ Inference (Metis AIPU)
INT8 Padded Output
    ↓ Depadding
INT8 Unpadded
    ↓ Dequantization
Float32 Output
```

For detailed tutorials and examples, see the [python axruntime documentation](../../tutorials/axruntime-python/axruntime-python-quickstart.md).

## Objects

### *class* axelera.runtime.Object

Abstract base class for all objects in the runtime.

All objects created by the runtime are owned by the Context that created
them.  When the Context is released, all objects created by it are also
released.

#### context

Reference to the [`Context`](#class-axeleraruntimecontext) that owns this object.

#### release()

Release the object and all its children.

### *class* axelera.runtime.Context

Bases: [`Object`](#class-axeleraruntimeobject)

The Context object is the root object of the runtime.

A Context is used to create and manage all other objects in the runtime. When Context is released, all child objects (Models, Connections, ModelInstances) are automatically released. This parent-child ownership model ensures proper resource cleanup.

Best practice is to use the context manager pattern:

```python
with axr.Context() as context:
    devices = context.list_devices()
    # Resources automatically released when exiting with block
```

Alternatively, you can call release() explicitly to release the context and all its children:

```python
context = axr.Context()
try:
    # Use context
finally:
    context.release()  # Ensures cleanup even if exception occurs
```

#### configure_device(device, \*\*kwargs)

Configure device clock speeds and utilization. Default settings are sufficient for most applications.

**Note:** MVM utilization must be configured for some models on Metis M.2. Check the compiler settings for your model, or set utilization here to apply to all models.

Returns True if the configuration setting is complete, False if it is pending. Configuration changes are asynchronous - use [`device_ready()`](#device_readydevice) to poll for completion.

**Example** (changing configuration on two devices):

```python
res0 = context.configure_device(device0, clock_profile=1000)
res1 = context.configure_device(device1, clock_profile=1000)
while not res0 or not res1:
    time.sleep(0.05)
    res0 = context.device_ready(device0)
    res1 = context.device_ready(device1)
```

Valid properties are:

| Property                 | Default    | Description                            |
|--------------------------|------------|----------------------------------------|
| clock_profile            | 800        | Device clock profile in MHz            |
| clock_profile_core_0-3   | 800        | Per-core clock profile in MHz          |
| mvm_utilisation_core_0-3 | 100        | Per-core MVM utilisation as percentage |
* **Return type:**
  `bool`

#### device_connect(device=None, num_sub_devices=1, \*\*kwargs)

Connect to one or more AIPU cores on a device.

This reserves the AIPU cores exclusively for this process. Other processes cannot use these cores until Connection.release() is called.

The returned Connection object is used to load models and run them on the reserved cores.

**Parameters:**
* `device` - Device to connect to:
  * `None`: Auto-select available device (default)
  * `int`: Device index (e.g., `0` for first device)
  * `str`: Device name (e.g., `"metis-0:4:0"`)
* `num_sub_devices` - Number of AIPU cores to reserve (1-4). Should match model batch_size.

* **Return type:**
  [`Connection`](#class-axeleraruntimeconnection)

**Example:**
```python
batch_size = model.inputs()[0].shape[0]  # Read from model
conn = context.device_connect(device=None, num_sub_devices=batch_size)
```

#### device_ready(device)

Returns True if the configuration setting is complete, False if it is pending.

* **Return type:**
  `bool`

#### list_devices()

List all devices on the system.

* **Return type:**
  `list`[[`DeviceInfo`](#class-axeleraruntimedeviceinfo)]

**Example:**
```python
devices = context.list_devices()
print(f"Found {len(devices)} device(s)")
for i, device in enumerate(devices):
    print(f"  Device {i}: {device.name}, {device.subdevice_count} cores")
```

Use device index, name string, or None (auto-select) with [`device_connect()`](#device_connectdevicenone-num_sub_devices1-kwargs).

#### load_model(path)

Load a compiled model from a file.

The path points to `model.json`. The model directory should also contain `manifest.json` (quantization metadata) and compiled weights. The returned Model can be loaded onto multiple Connection objects using Connection.load_model_instance().

* **Return type:**
  [`Model`](#class-axeleraruntimemodel)

**Example:**
```python
model = context.load_model("build/yolo11s-coco-onnx/model.json")
input_infos = model.inputs()
output_infos = model.outputs()
batch_size = input_infos[0].shape[0]
```

#### read_device_configuration(device)

Read all available configuration properties of the device.

* **Return type:**
  `dict`[`str`, `str`]

#### release()

Release the context and all its children.

This function is called automatically when Context is used as a context
manager.

### *class* axelera.runtime.Connection

Bases: [`Object`](#class-axeleraruntimeobject)

A connection to one or more AIPU cores on a device.

#### load_model_instance(model, \*\*kwargs)

Load a model onto the AIPU cores and create a ModelInstance.

Each ModelInstance is tied to its Connection and should only be used by one thread. For parallel execution, create multiple Connection/ModelInstance pairs.

Valid kwargs are:

| Property         |   Default | Description                                                         |
|------------------|-----------|---------------------------------------------------------------------|
| aipu_cores       |         0 | L2 memory allocation. Each core = 25% of device L2. Should match num_sub_devices. |
| num_sub_devices  |         0 | Number of AIPU cores for this instance. Should match model batch_size. |
| input_dmabuf     |         0 | True if the input arguments are dmabuf file descriptors             |
| device_profiling |         0 | True to enable device profiling                                     |
| host_profiling   |         0 | True to enable host profiling                                       |
| output_dmabuf    |         0 | True if the output arguments are dmabuf file descriptors            |
| double_buffer    |         0 | Enable DMA pipelining for higher throughput (adds 2-frame latency). See [Double Buffering guide](../../tutorials/axruntime-python/double-buffering.md). |
| elf_in_ddr       |         1 | True if the model was compiled with elf_in_ddr as True.             |
* **Return type:**
  [`ModelInstance`](#class-axeleraruntimemodelinstance)

**Example:**
```python
batch_size = model.inputs()[0].shape[0]
num_instances = 4 // batch_size  # 4 cores per Metis device

instances = []
for i in range(num_instances):
    conn = context.device_connect(device=None, num_sub_devices=batch_size)
    instance = conn.load_model_instance(
        model,
        num_sub_devices=batch_size,
        aipu_cores=batch_size
    )
    instances.append(instance)
```

### *class* axelera.runtime.Model

Bases: [`Object`](#class-axeleraruntimeobject)

A model object that can be loaded onto multiple Connection objects.

#### properties

* `preamble_graph` - Relative path to a preamble ONNX file containing preprocessing operations. Empty if no preprocessing was extracted. If present, execute this ONNX graph before AIPU inference. Path is relative to model directory.

* `postamble_graph` - Relative path to a postamble ONNX file containing postprocessing operations. Empty if no postprocessing was extracted. If present, execute this ONNX graph after AIPU inference (depad, transpose, and dequantize first). Path is relative to model directory. See [Postamble Processing guide](../../tutorials/axruntime-python/postamble-processing.md).

* `input_tensor_layout` - Always NHWC. AIPU outputs are also NHWC. Transpose to NCHW if needed for ONNX postamble graphs.


#### inputs()

Return information about the input tensors to the model.

* **Return type:**
  `list`[[`TensorInfo`](#class-axeleraruntimetensorinfo)]

Use this to get tensor shapes, quantization parameters, and padding information for preprocessing. See [`TensorInfo`](#class-axeleraruntimetensorinfo) for details.

**Example:**
```python
input_infos = model.inputs()
batch_size = input_infos[0].shape[0]
# Allocate input buffers
inputs = [np.zeros(info.shape, info.dtype) for info in input_infos]
```

#### outputs()

Return information about the output tensors of the model.

* **Return type:**
  `list`[[`TensorInfo`](#class-axeleraruntimetensorinfo)]

Use this to allocate output buffers and get quantization parameters for postprocessing.

**Example:**
```python
output_infos = model.outputs()
# Allocate output buffers
outputs = [np.zeros(info.shape, info.dtype) for info in output_infos]
```

### *class* axelera.runtime.ModelInstance

Bases: [`Object`](#class-axeleraruntimeobject)

A model instance that has been loaded onto a Connection object. Each ModelInstance should only be used by one thread.

#### run(inputs, outputs)

Execute inference synchronously. This call blocks until the AIPU completes.

For maximum throughput, create one worker thread per ModelInstance. While one thread waits for inference, the OS can schedule other workers. See [Basic Usage Tutorial - Worker Threads](../../tutorials/axruntime-python/basic-usage-tutorial.md#step-6-running-inference-with-worker-threads).

**Parameters:**
* `inputs` - List of input numpy arrays (or dmabuf file descriptors if `input_dmabuf=True`). Arrays are modified in-place and can be reused across multiple run() calls.
* `outputs` - Pre-allocated output numpy arrays (or dmabuf file descriptors if `output_dmabuf=True`). Results are written in-place.

**Buffer reuse:** The same input and output buffers can be reused for all inferences. Allocate once during initialization.

On failure, an exception is raised.

* **Return type:**
  `None`

**Example:**
```python
# Pre-allocate buffers once
inputs = [np.zeros(info.shape, info.dtype) for info in model.inputs()]
outputs = [np.zeros(info.shape, info.dtype) for info in model.outputs()]

# Reuse for all inferences
for image in images:
    inputs[0][:] = preprocess(image)  # Copy data into buffer
    instance.run(inputs, outputs)     # Blocking call
    result = postprocess(outputs[0])
```

## Types

### *class* axelera.runtime.BoardType

```
class BoardType(enum.Enum):
    alpha_pcie = 0
    alpha_m2 = 1
    pcie = 2
    m2 = 3
    devboard = 4
    sbc = 5
    unknown = 6
```

### *class* axelera.runtime.DeviceInfo

Information about an available Axelera device.

Returned by [`Context.list_devices()`](#list_devices). Use the `name` field to target a specific device with [`device_connect()`](#device_connectdevicenone-num_sub_devices1-kwargs).

- `name: str` <br>_Device name (e.g., "metis-0:3:0" or "metis-0:4:0"). Use this with `device_connect(device=name)`._

- `subdevice_count: int` <br>_Number of AIPU cores on the device. For Metis this is 4._

- `max_memory: int` <br>_Maximum memory available on the device._

- `in_use: bool` <br>_Whether any cores are in use._

- `in_use_by: str` <br>_Username and process ID using the device, comma separated._

**Note:** In current SDK version, `max_memory`, `in_use`, and `in_use_by` fields are not populated.
    
- `board_type: BoardType` <br>_The board type of the device._
    
- `firmware_version: str` <br>_The firmware version of the device, for example v1.1.0-rc5-2-g1234567._
    
- `board_revision: int` <br>_The board revision of the device._
    
- `flashed_firmware_version: str` <br>_The version of the firmware stored in flash memory of the device._
    
- `board_controller_firmware_version: str` <br>_The board controller firmware version on the device._

- `board_controller_board_type: str` <br>_The board controller board type._


### *class* axelera.runtime.TensorInfo

Information about a tensor input/output.

TensorInfo provides tensor shape, quantization parameters, and padding information needed for preprocessing and postprocessing. Use `unpadded_shape` for preprocessing (resize target dimensions), and `shape` for buffer allocation (includes hardware padding).

**Input preprocessing pipeline:**
```python
# 1. Quantize: float32 → int8
quantized = np.round((normalized / input_info.scale) + input_info.zero_point)
quantized = quantized.clip(-128, 127).astype(np.int8)

# 2. Pad: Add hardware alignment (skip batch dimension)
padded = np.pad(quantized, input_info.padding[1:],
                mode='constant', constant_values=input_info.zero_point)
```

**Output postprocessing pipeline:**
```python
# 1. Depad: Remove alignment padding
depadded = output[tuple(slice(b, -e if e else None)
                        for b, e in output_info.padding)]

# 2. Dequantize: int8 → float32
result = (depadded.astype(np.float32) - output_info.zero_point) * output_info.scale
```

- `shape: tuple[int, ...]` <br>_Full tensor shape including hardware padding. Use this for buffer allocation: `np.zeros(info.shape, info.dtype)`._

- `dtype: np.dtype = np.int8` <br>_Data type of the tensor. Always `np.int8` for quantized inference._

- `name: str = ''` <br>_The name of the tensor._

- `padding: list[tuple[int, int]]` <br>_Padding per dimension as `[(start0, end0), (start1, end1), ...]`. Format matches `numpy.pad`. For inputs, typically only channel dimension (index 3) has padding. For input preprocessing, use `padding[1:]` to skip batch dimension._

- `scale: float = 1.0` <br>_Quantization scale factor. Input: `quantized = (float / scale) + zero_point`. Output: `float = (int8 - zero_point) * scale`._

- `zero_point: int = 0` <br>_Quantization zero point. Use this as the constant value when padding (not zero)._

**Properties**

- `size: int` <br>_The size of the tensor in bytes._

- `unpadded_shape: tuple[int, ...]` <br>_Logical shape without padding. Use this for preprocessing resize target dimensions._

## Exceptions

### *exception* axelera.runtime.ConnectionError

Raised when device connection fails. Check device availability with [`list_devices()`](#list_devices).

### *exception* axelera.runtime.DeviceInUse

Raised when attempting to connect to AIPU cores already reserved by another process.

### *exception* axelera.runtime.IncompatibleDevice

Raised when model requires hardware features not available on the device.

### *exception* axelera.runtime.InternalError

Raised for internal runtime errors.

### *exception* axelera.runtime.InvalidArgument

Raised for invalid parameters (e.g., requesting more cores than available, invalid buffer shapes).

### *exception* axelera.runtime.InvalidConfiguration

Raised when [`configure_device()`](#configure_devicedevice-kwargs) receives invalid configuration values.

### *exception* axelera.runtime.NotImplemented

Raised when using features not yet implemented in current SDK version.

### *exception* axelera.runtime.Pending

Raised when operation is still in progress. Poll with [`device_ready()`](#device_readydevice) for configuration operations.

### *exception* axelera.runtime.UnknownError

Raised for unexpected or unknown errors. 
