![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# axelera.runtime documentation

- [axelera.runtime documentation](#axeleraruntime-documentation)
  - [Objects](#objects)
    - [*class* axelera.runtime.Object](#class-axeleraruntimeobject)
      - [context](#context)
      - [release()](#release)
    - [*class* axelera.runtime.Context](#class-axeleraruntimecontext)
      - [configure\_device(device, \*\*kwargs)](#configure_devicedevice-kwargs)
      - [device\_connect(device=None, num\_sub\_devices=1, \*\*kwargs)](#device_connectdevicenone-num_sub_devices1-kwargs)
      - [device\_ready(device)](#device_readydevice)
      - [list\_devices()](#list_devices)
      - [load\_model(path)](#load_modelpath)
      - [read\_device\_configuration(device)](#read_device_configurationdevice)
      - [release()](#release-1)
    - [*class* axelera.runtime.Connection](#class-axeleraruntimeconnection)
      - [load\_model\_instance(model, \*\*kwargs)](#load_model_instancemodel-kwargs)
    - [*class* axelera.runtime.Model](#class-axeleraruntimemodel)
      - [inputs()](#inputs)
      - [outputs()](#outputs)
    - [*class* axelera.runtime.ModelInstance(obj, context)](#class-axeleraruntimemodelinstance)
      - [run(inputs, outputs)](#runinputs-outputs)
  - [Types](#types)
    - [*class* axelera.runtime.BoardType](#class-axeleraruntimeboardtype)
    - [*class* axelera.runtime.DeviceInfo](#class-axeleraruntimedeviceinfo)
    - [*class* axelera.runtime.TensorInfo](#class-axeleraruntimetensorinfo)
  - [Exceptions](#exceptions)
    - [*exception* axelera.runtime.ConnectionError](#exception-axeleraruntimeconnectionerror)
    - [*exception* axelera.runtime.DeviceInUse](#exception-axeleraruntimedeviceinuse)
    - [*exception* axelera.runtime.IncompatibleDevice](#exception-axeleraruntimeincompatibledevice)
    - [*exception* axelera.runtime.InternalError](#exception-axeleraruntimeinternalerror)
    - [*exception* axelera.runtime.InvalidArgument](#exception-axeleraruntimeinvalidargument)
    - [*exception* axelera.runtime.InvalidConfiguration](#exception-axeleraruntimeinvalidconfiguration)
    - [*exception* axelera.runtime.NotImplemented](#exception-axeleraruntimenotimplemented)
    - [*exception* axelera.runtime.Pending](#exception-axeleraruntimepending)
    - [*exception* axelera.runtime.UnknownError](#exception-axeleraruntimeunknownerror)

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

A Context is used to create and manage all other objects in the runtime.
Normally you would create it in a with statement, like this

```default
with axr.Context() as context:
    devices = context.list_devices()
```

Alternatively, you can call release() to release the context and all its
children.

#### configure_device(device, \*\*kwargs)

Returns True if the configuration setting is complete, False if it is pending.

For example to change a configuration on two devices:

```default
res0 = context.configure_device(device0, clock_profile=1000)
res1 = context.configure_device(device1, clock_profile=1000)
while not res0 or not res1:
    time.sleep(0.05)
    res0 = context.device_ready(device0)
    res1 = context.device_ready(device1)
```

Valid properties are :

| Property                 | Default    | Description                            |
|--------------------------|------------|----------------------------------------|
| ddr_size                 | 0x40000000 | Size of DDR memory in bytes            |
| clock_profile            | 800        | Device clock profile in MHz            |
| clock_profile_core_0-3   | 800        | Per-core clock profile in MHz          |
| mvm_utilisation_core_0-3 | 100        | Per-core MVM utilisation as percentage |
* **Return type:**
  `bool`

#### device_connect(device=None, num_sub_devices=1, \*\*kwargs)

Connect to one or more sub-devices.

This reserves the sub-devices so that other processes cannot use them.

The returned Connection object is used to load models and run them on the
sub-devices.

* **Return type:**
  [`Connection`](#class-axeleraruntimeconnection)

#### device_ready(device)

Returns True if the configuration setting is complete, False if is pending.

* **Return type:**
  `bool`

#### list_devices()

List all devices on the system.

* **Return type:**
  `list`[[`DeviceInfo`](#class-axeleraruntimedeviceinfo)]

#### load_model(path)

Load a model from a file.

The returned model can be loaded onto multiple Connection objects using
Connection.load_model_instance().

* **Return type:**
  [`Model`](#class-axeleraruntimemodel)

#### read_device_configuration(device)

Read all available configiration properties of the device.

* **Return type:**
  `dict`[`str`, `str`]

#### release()

Release the context and all its children.

This function is called automatically when Context is used as acontext
manager.

### *class* axelera.runtime.Connection

Bases: [`Object`](#class-axeleraruntimeobject)

A connection to one or more sub-devices.

#### load_model_instance(model, \*\*kwargs)

Load a model onto the sub-devices.

Valid kwargs are :

| Property         |   Default | Description                                                         |
|------------------|-----------|---------------------------------------------------------------------|
| aipu_cores       |         0 | Amount of L2 resources to allocate for the model, set to batch size |
| num_sub_devices  |         0 | Number of sub-devices to use, set to batch size of the model        |
| input_dmabuf     |         0 | True if the input arguments are dmabuf file descriptors             |
| device_profiling |         0 | True to enable device profiling                                     |
| host_profiling   |         0 | True to enable host profiling                                       |
| output_dmabuf    |         0 | True if the output arguments are dmabuf file descriptors            |
| double_buffer    |         0 | True to enable double buffering                                     |
| elf_in_ddr       |         1 | True if the model was compiled with elf_in_ddr as True.             |
* **Return type:**
  [`ModelInstance`](#class-axeleraruntimemodelinstance)

### *class* axelera.runtime.Model

Bases: [`Object`](#class-axeleraruntimeobject)

A model object that can be loaded onto a Connection object.

#### properties

* `preamble_graph` Relative path to a preamble ONNX file containing the initial nodes of the model. These nodes are removed by the compiler and executed on the host.
* `postamble_graph` Relative path to a postamble ONNX file containing the final nodes of the model. These nodes are removed by the compiler and executed on the host.
* `input_tensor_layout`  Always NHWC in this version.


#### inputs()

Return information about the input tensors to the model.

* **Return type:**
  `list`[[`TensorInfo`](#class-axeleraruntimetensorinfo)]

#### outputs()

Return information about the output tensors of the model.

* **Return type:**
  `list`[[`TensorInfo`](#class-axeleraruntimetensorinfo)]

### *class* axelera.runtime.ModelInstance

Bases: [`Object`](#class-axeleraruntimeobject)

A model instance that has been loaded onto a Connection object.

#### run(inputs, outputs)

Run the model instance.

If the model instance was created with input_dmabuf=True then inputs
must be a list of file descriptors.  Otherwise it should be a list of
numpy arrays.

If the model instance was created with output_dmabuf=True then outputs
must be a list of file descriptors.  Otherwise it should be a list of
numpy arrays.

On failure, an exception is raised.

* **Return type:**
  `None`

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

The result of enumeraing the available Axelera devices.

DeviceInfo is also used to indicate which device to configure, read configuration, and select desired device.

- `name: str` <br>_The name of the device. For example 'metis-0:3:0'.
    
- `subdevice_count: int` <br>_The number of subdevices on the device, for metis this is 4._
    
- `max_memory: int` <br>_The maximum memory available on the device.  Note in the current implementation this field is not populated and will always be 0._
    
- `in_use: bool` <br>_The number of subdevices in use. Note in the current implementation this field is not populated and will always be 0._
    
- `in_use_by: str` <br>_The username and process id of the user(s) using the device, comma separated. Note in the current implementation this field is not populated and will always be 0._
    
- `board_type: BoardType` <br>_The board type of the device._
    
- `firmware_version: str` <br>_The firmware version of the device, for example v1.1.0-rc5-2-g1234567._
    
- `board_revision: int` <br>_The board revision of the device._
    
- `flashed_firmware_version: str` <br>_The version of the firmware stored in flash memory of the device._
    
- `board_controller_firmware_version: str` <br>_The board controller firmware version on the device._

- `board_controller_board_type: str` <br>_The board controller board type._


### *class* axelera.runtime.TensorInfo

Information about a tensor input/output.

This includes quantization and padding information if manifest.json was found
in the model.

For example to quantize and pad a tensor:

```python
>>> input = TensorInfo((1, 230, 240, 3), padding=[(0, 0), (3, 3), (3, 13), (0, 1)])
>>> src = np.zeros(input.unpadded_shape, dtype=np.float32)
>>> quant = np.round((src / input.scale) + input.zero_point).clip(-128, 127).astype(np.int8)
>>> padded = np.pad(quant, input.padding, constant_values=input.zero_point)
```

To depad and dequantize a tensor:
```
>>> output = TensorInfo((1, 1, 1, 1024), padding=[(0, 0), (0, 0), (0, 0), (0, 24)])
>>> out = np.zeros(output.shape, dtype=np.int8)
>>> depadded = out[tuple(slice(b, -e if e else None) for b, e in output.padding)]
>>> dequant = (depadded.astype(np.float32) - output.zero_point) * output.scale
```

- `shape: tuple[int, ...]` <br>_The shape of the tensor._

- `dtype: np.dtype = np.int8` <br>_The data type of the tensor._

- `name: str = ''` <br>_The name of the tensor._

- `padding: list[tuple[int, int]]` <br>_Amount of padding on the tensor. As list of [(start0, end0), (start1, end1), ...  (see numpy.pad)_

- `scale: float = 1.0` <br>_scale for quantization/dequantization._

- `zero_point: int = 0` <br>_zero-point for quantization/dequantization._

**Properties**

- `size: int` <br>_The size of the tensor in bytes._

- `unpadded_shape: tuple[int, ...]` <br>_The shape of the tensor without padding._

## Exceptions

### *exception* axelera.runtime.ConnectionError

### *exception* axelera.runtime.DeviceInUse

### *exception* axelera.runtime.IncompatibleDevice

### *exception* axelera.runtime.InternalError

### *exception* axelera.runtime.InvalidArgument

### *exception* axelera.runtime.InvalidConfiguration

### *exception* axelera.runtime.NotImplemented

### *exception* axelera.runtime.Pending

### *exception* axelera.runtime.UnknownError
