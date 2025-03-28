![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# API Reference

- [API Reference](#api-reference)
  - [Header files](#header-files)
  - [File include/axruntime/axruntime.h](#file-includeaxruntimeaxruntimeh)
  - [Structures and Types](#structures-and-types)
  - [Functions](#functions)
  - [Structures and Types Documentation](#structures-and-types-documentation)
    - [struct `axrArgument`](#struct-axrargument)
    - [enum `axrBoardType`](#enum-axrboardtype)
    - [typedef `axrConnection`](#typedef-axrconnection)
    - [typedef `axrContext`](#typedef-axrcontext)
    - [struct `axrDeviceInfo`](#struct-axrdeviceinfo)
    - [enum `axrLogLevel`](#enum-axrloglevel)
    - [typedef `axrModel`](#typedef-axrmodel)
    - [typedef `axrModelInstance`](#typedef-axrmodelinstance)
    - [typedef `axrObject`](#typedef-axrobject)
    - [typedef `axrProperties`](#typedef-axrproperties)
    - [enum `axrResult`](#enum-axrresult)
    - [constant `AXR_MAX_TENSOR_DIMS`](#constant-axr_max_tensor_dims)
    - [enum `AXR_TENSOR_DATA_TYPE`](#enum-axr_tensor_data_type)
    - [struct `axrTensorInfo`](#struct-axrtensorinfo)
    - [typedef `axr_logger_fn`](#typedef-axr_logger_fn)
  - [Functions Documentation](#functions-documentation)
    - [function `axr_api_version`](#function-axr_api_version)
    - [function `axr_board_type_string`](#function-axr_board_type_string)
    - [function `axr_configure_device`](#function-axr_configure_device)
    - [function `axr_create_context`](#function-axr_create_context)
    - [function `axr_create_properties`](#function-axr_create_properties)
    - [function `axr_del_property`](#function-axr_del_property)
    - [function `axr_destroy`](#function-axr_destroy)
  - [Object](#object)
    - [function `axr_device_connect`](#function-axr_device_connect)
    - [function `axr_device_ready`](#function-axr_device_ready)
    - [function `axr_error_string`](#function-axr_error_string)
    - [function `axr_get_context`](#function-axr_get_context)
    - [function `axr_get_model_input`](#function-axr_get_model_input)
    - [function `axr_get_model_output`](#function-axr_get_model_output)
    - [function `axr_get_model_properties`](#function-axr_get_model_properties)
    - [function `axr_get_property`](#function-axr_get_property)
    - [function `axr_last_error`](#function-axr_last_error)
    - [function `axr_last_error_string`](#function-axr_last_error_string)
    - [function `axr_list_devices`](#function-axr_list_devices)
    - [function `axr_list_properties`](#function-axr_list_properties)
    - [function `axr_load_model`](#function-axr_load_model)
    - [function `axr_load_model_instance`](#function-axr_load_model_instance)
    - [function `axr_num_model_inputs`](#function-axr_num_model_inputs)
    - [function `axr_num_model_outputs`](#function-axr_num_model_outputs)
    - [function `axr_read_device_configuration`](#function-axr_read_device_configuration)
    - [function `axr_run_model_instance`](#function-axr_run_model_instance)
    - [function `axr_set_logger`](#function-axr_set_logger)
    - [function `axr_set_property`](#function-axr_set_property)
    - [function `axr_tensor_size`](#function-axr_tensor_size)
  - [File include/axruntime/axruntime.hpp](#file-includeaxruntimeaxruntimehpp)
  - [Structures and Types](#structures-and-types-1)
  - [Structures and Types Documentation](#structures-and-types-documentation-1)
    - [struct `axr::object_deleter`](#struct-axrobject_deleter)

## Header files

- [include/axruntime/axruntime.h](#file-includeaxruntimeaxruntimeh)
- [include/axruntime/axruntime.hpp](#file-includeaxruntimeaxruntimehpp)

## File include/axruntime/axruntime.h


## Structures and Types

| Type | Name |
| ---: | :--- |
| struct | [**axrArgument**](#struct-axrargument) <br>_An input or output argument to a model._ |
| enum  | [**axrBoardType**](#enum-axrboardtype)  <br> |
| struct | [**axrConnection**](#typedef-axrconnection)  <br> |
| struct | [**axrContext**](#typedef-axrcontext)  <br> |
| struct | [**axrDeviceInfo**](#struct-axrdeviceinfo) <br> |
| enum  | [**axrLogLevel**](#enum-axrloglevel)  <br> |
| struct | [**axrModel**](#typedef-axrmodel)  <br> |
| struct | [**axrModelInstance**](#typedef-axrmodelinstance)  <br> |
| struct | [**axrObject**](#typedef-axrobject)  <br> |
| struct | [**axrProperties**](#typedef-axrproperties)  <br> |
| enum  | [**axrResult**](#enum-axrresult)  <br> |
| enum  | [**AXR_TENSOR_DATA_TYPE**](#enum-axr_tensor_data_type)  <br> |
| struct | [**axrTensorInfo**](#struct-axrtensorinfo) <br> |
| function pointer type | [**axr\_logger\_fn**](#typedef-axr_logger_fn)  <br> |

## Functions

| Type | Name |
| ---: | :--- |
|  unsigned int | [**axr\_api\_version**](#function-axr_api_version) () <br>_Get the version of the API._ |
|  const char \* | [**axr\_board\_type\_string**](#function-axr_board_type_string) (axrBoardType result) <br> |
|  axrResult | [**axr\_configure\_device**](#function-axr_configure_device) (axrContext \*context, [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, axrProperties \*properties) <br>_Configure a device._ |
|  axrContext \* | [**axr\_create\_context**](#function-axr_create_context) () <br>_The context object is the root object for the library._ |
|  axrProperties \* | [**axr\_create\_properties**](#function-axr_create_properties) (axrContext \*context, const char \*initial\_properties) <br>_Create a new properties object._ |
|  void | [**axr\_del\_property**](#function-axr_del_property) (axrProperties \*properties, const char \*key) <br>_Remove key from properties._ |
|  void | [**axr\_destroy**](#function-axr_destroy) (const axrObject \*obj) <br>_Destroy any non-value object created by the library._ |
|  axrConnection \* | [**axr\_device\_connect**](#function-axr_device_connect) (axrContext \*context, [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, size\_t num\_sub\_devices, axrProperties \*properties) <br>_Connect to a device._ |
|  axrResult | [**axr\_device\_ready**](#function-axr_device_ready) (axrContext \*context, [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_Check if a device is ready to be connected to after configuration._ |
|  const char \* | [**axr\_error\_string**](#function-axr_error_string) (axrResult result) <br>_convert axrResult enum to a string. The string is statically allocated and should not be freed. Never returns NULL._ |
|  axrContext \* | [**axr\_get\_context**](#function-axr_get_context) (axrObject \*obj) <br>_Get the context that owns the object._ |
|  [**axrTensorInfo**](#struct-axrtensorinfo) | [**axr\_get\_model\_input**](#function-axr_get_model_input) (axrModel \*model, size\_t n) <br>_Get the shape and dtype of the input tensor n._ |
|  [**axrTensorInfo**](#struct-axrtensorinfo) | [**axr\_get\_model\_output**](#function-axr_get_model_output) (axrModel \*model, size\_t n) <br>_Get the shape and dtype of the output tensor n._ |
|  const char \* | [**axr\_get\_property**](#function-axr_get_property) (axrProperties \*properties, const char \*key) <br>_Get the value of a property._ |
|  axrResult | [**axr\_last\_error**](#function-axr_last_error) (const axrObject \*c) <br>_get the last error that occurred on the context._ |
|  const char \* | [**axr\_last\_error\_string**](#function-axr_last_error_string) (const axrObject \*c) <br>_get the last error string that occurred on the context. The string is valid until the next call to an axr function in this thread._ |
|  size\_t | [**axr\_list\_devices**](#function-axr_list_devices) (axrContext \*context, [**axrDeviceInfo**](#struct-axrdeviceinfo) \*\*devices) <br>_Enumerate devices available to the context._ |
|  const char \* | [**axr\_list\_properties**](#function-axr_list_properties) (axrProperties \*properties) <br>_Get a list of keys in the properties object._ |
|  axrModel \* | [**axr\_load\_model**](#function-axr_load_model) (axrContext \*context, const char \*path) <br>_Parse a model from a file._ |
|  axrModelInstance \* | [**axr\_load\_model\_instance**](#function-axr_load_model_instance) (axrConnection \*connection, axrModel \*model, axrProperties \*properties) <br>_Load a model onto a connected device._ |
|  size\_t | [**axr\_num\_model\_inputs**](#function-axr_num_model_inputs) (axrModel \*model) <br>_Get the number of inputs to the model._ |
|  size\_t | [**axr\_num\_model\_outputs**](#function-axr_num_model_outputs) (axrModel \*model) <br>_Get the number of outputs to the model._ |
|  axrProperties \* | [**axr\_get\_model\_properties**](#function-axr_get_model_properties) (axrModel \*model) <br>_Get further information about the model emitted by the compiler._ |
|  axrProperties \* | [**axr\_read\_device\_configuration**](#function-axr_read_device_configuration) (axrContext \*context, [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_Read the configuration of a device._ |
|  axrResult | [**axr\_run\_model\_instance**](#function-axr_run_model_instance) (axrModelInstance \*instance, [**axrArgument**](#struct-axrargument) \*inputs, size\_t num\_inputs, [**axrArgument**](#struct-axrargument) \*output, size\_t num\_outputs) <br>_Run one frame the model instance._ |
|  void | [**axr\_set\_logger**](#function-axr_set_logger) (axrContext \*context, axrLogLevel level, axr\_logger\_fn logger, void \*arg) <br> |
|  void | [**axr\_set\_property**](#function-axr_set_property) (axrProperties \*properties, const char \*key, const char \*value) <br>_Set the value of a property._ |
|  size\_t | [**axr\_tensor\_size**](#function-axr_tensor_size) (const [**axrTensorInfo**](#struct-axrtensorinfo) \*info) <br>_Get the size of a tensor in bytes._ |


## Structures and Types Documentation

### struct `axrArgument`

_An input or output argument to a model._

Variables:

-  `void * ptr`  <br>_Pointer to the data if the data is in host memory, or NULL if the data is in a DMABuf._

-  `int fd`  <br>_File descriptor for a DMA Buf. Or 0 or -1 if the data is in host memory._


-  `size_t offset`  <br>_Offset within the buffer that the data starts._
**Note:** in this version this is not supported, it must be zero. 

-  `size_t size`  <br>_This is optional. If 0 it will be ignored, if given it will be checked against the expected tensor size._

### enum `axrBoardType`

```c
enum axrBoardType {
    AXR_BOARD_METIS_UNUSED1,
    AXR_BOARD_METIS_UNUSED2,
    AXR_BOARD_METIS_OMEGA_PCIE,
    AXR_BOARD_METIS_OMEGA_M2,
    AXR_BOARD_METIS_UNUSED3,
    AXR_BOARD_METIS_UNUSED3,
    AXR_BOARD_METIS_MAX
};
```

### typedef `axrConnection`

```c
typedef struct axrConnection axrConnection;
```

### typedef `axrContext`

```c
typedef struct axrContext axrContext;
```

### struct `axrDeviceInfo`


Variables:

-  `char name[256]`  <br>_The name of the device, for example "metis-0:3:0"._

-  `size_t subdevice_count`  <br>_The number of subdevices on the device, for metis this is 4._

-  `size_t max_memory`  <br>_The maximum memory available on the device._<br>**Note:** 
in the current implementation this field is not populated and will always be 0.

-  `int in_use`  <br>_The number of subdevices in use._<br>**Note:**
in the current implementation this field is not populated and will always be 0.

-  `char in_use_by[256]`  <br>_The username and process id of the user(s) using the device, comma separated._<br>**Note:**
in the current implementation this field is not populated and will always be empty.

-  `axrBoardType board_type`  <br>_The board type of the device._

-  `char firmware_version[256]`  <br>_The firmware version of the device, for example v1.1.0-rc5-2-g1234567._

-  `int board_revision`  <br>_The board revision of the device._

-  `char flashed_firmware_version[256]`  <br>_The version of the firmware stored in flash memory of the device._

-  `char board_controller_firmware_version[256]`  <br>_The board controller firmware version on the device._

-  `char board_controller_board_type[256]`  <br>_The board controller board type._

### enum `axrLogLevel`

```c
enum axrLogLevel {
    AXR_LOG_TRACE,
    AXR_LOG_LOG,
    AXR_LOG_DEBUG,
    AXR_LOG_INFO,
    AXR_LOG_FIXME,
    AXR_LOG_WARNING,
    AXR_LOG_ERROR
};
```

### typedef `axrModel`

```c
typedef struct axrModel axrModel;
```

### typedef `axrModelInstance`

```c
typedef struct axrModelInstance axrModelInstance;
```

### typedef `axrObject`

```c
typedef struct axrObject axrObject;
```

### typedef `axrProperties`

```c
typedef struct axrProperties axrProperties;
```

### enum `axrResult`

```c
enum axrResult {
    AXR_SUCCESS,
    AXR_ERROR_RUNTIME_ERROR,
    AXR_ERROR_VALUE_ERROR,
    AXR_ERROR_INVALID_ARGUMENT,
    AXR_ERROR_CONNECTION_ERROR,
    AXR_ERROR_DEVICE_IN_USE,
    AXR_ERROR_INCOMPATIBLE_DEVICE,
    AXR_ERROR_INVALID_CONFIGURATION,
    AXR_ERROR_NOT_IMPLEMENTED,
    AXR_ERROR_INTERNAL_ERROR,
    AXR_ERROR_PENDING_ERROR,
    AXR_ERROR_UNKNOWN_ERROR
};
```

### constant `AXR_MAX_TENSOR_DIMS`

This is used to give a fixed size to the axrTensorInfo, and determines the maximum
dimension of a tensor.

```c
#define AXR_TENSOR_DATA_TYPE 8
```

### enum `AXR_TENSOR_DATA_TYPE`

```c
enum AXR_TENSOR_DATA_TYPE {
  AXR_UNSIGNED = 0,
  AXR_SIGNED = 1,
  AXR_FLOAT = 2,
};
```

### struct `axrTensorInfo`


Variables:

  `size_t dims[AXR_MAX_TENSOR_DIMS]`  <br>_The dimensions of the tensor, dims[0:ndims] can be considered the shape._

  `size_t ndims`  <br>_The number of dimensions in the tensor._

  `size_t bits`  <br>_The number of bits per element._

  `size_t type`  <br>_The format of the tensor, one of AXR_TENSOR_DATA_TYPE._

  `char name[256]`  <br>_The name of the tensor referenced in the onnx model._

  `size_t padding[AXR_MAX_TENSOR_DIMS][2]`  <br>_Amount of padding on the tensor. As list of [(start0, end0), (start1, end1), ...  (see numpy.pad).  For example for NHWC this is
  ```
   (Nbefore, Nafter), (Hbefore, Hafter), (Wbefore, Wafter), (Cbefore, Cafter) = padding[:ndims* 2]
  ```

  `double scale`  <br>_Scale for quantization/dequantization._

  `int zero_point`  <br>_Zero-point for quantization/dequantization._


### typedef `axr_logger_fn`

```c
typedef void(* axr_logger_fn) (void *arg, axrLogLevel level, const char *msg);
```


## Functions Documentation

### function `axr_api_version`

_Get the version of the API._
```c
AXR_EXPORT unsigned int axr_api_version ()
```


**Returns:**

the version of the API.

This allows a client to check the version of the API at runtime to ensure compatibility. Minor version changes may include additions to the API, while major version changes may include breaking changes.

### function `axr_board_type_string`

```c
AXR_EXPORT const char * axr_board_type_string (
    axrBoardType result
)
```

### function `axr_configure_device`

_Configure a device._
```c
AXR_EXPORT axrResult axr_configure_device (
    axrContext *context,
    axrDeviceInfo *device,
    axrProperties *properties
)
```


**Parameters:**


* `context` the context
* `device` the device to configure
* `properties` the options to change. Currently supported options are clock\_profile:int - the clock profile to use, frequency in MHz. ddr\_size:int - the DDR size to use, in bytes.


**Returns:**

AXR\_SUCCESS if the device was configured successfully.
<br >AXR\_ERROR\_INVALID\_CONFIGURATION if a configuration name is invalid.
<br >AXR\_ERROR\_NOT\_SUPPORTED if a configuration name is valid but not supported on this hw.
<br >AXR\_ERROR\_VALUE\_ERROR if a configuration name is valid but the value is invalid.


If the change requires time to take effect, AXR\_ERROR\_PENDING\_ERROR is returned. The user should poll the device\_ready function to check if the device is ready before any other operations are performed on that device. For example, to configure multiple devices::


### function `axr_create_context`

_The context object is the root object for the library._
```c
AXR_EXPORT axrContext * axr_create_context ()
```

Create a new context.

**Returns:**

 @return a valid context (never NULL).

 The context is used to create all other objects and it owns all the objects created from it, as well as providing access to some global services like errors and logging.

### function `axr_create_properties`

_Create a new properties object._
```c
AXR_EXPORT axrProperties * axr_create_properties (
    axrContext *context,
    const char *initial_properties
)
```


**Parameters:**


* `context` the context
* `initial_properties` a string of ; separated key=value pairs.


**Returns:**

a valid properties object (never NULL unless context is NULL)
### function `axr_del_property`

_Remove key from properties._
```c
AXR_EXPORT void axr_del_property (
    axrProperties *properties,
    const char *key
)
```

### function `axr_destroy`

_Destroy any non-value object created by the library._
```c
void axr_destroy (
    const axrObject *obj
)
```


## Object




**Parameters:**


* `obj` the object to destroy.

This function is only required to be called on the context object, all other objects will be destroyed when the context is destroyed. However other objects can be destroyed sooner if necessary. For example to load a different model onto the target you should first destroy the model instance.
### function `axr_device_connect`

_Connect to a device._
```c
AXR_EXPORT axrConnection * axr_device_connect (
    axrContext *context,
    axrDeviceInfo *device,
    size_t num_sub_devices,
    axrProperties *properties
)
```


**Parameters:**


* `context` the context
* `device` the device to connect to (or NULL to connect to the first available device)
* `num_sub_devices` number of sub devices to connect to
* `properties` options to use when connecting to the device
### function `axr_device_ready`

_Check if a device is ready to be connected to after configuration._
```c
AXR_EXPORT axrResult axr_device_ready (
    axrContext *context,
    axrDeviceInfo *device
)
```


**Parameters:**


* `context` the context
* `device` the device to check


**Returns:**

AXR\_SUCCESS if the device is ready to be connected to, otherwise an AXR\_ERROR\_PENDING\_ERROR if the device is still configuring.
### function `axr_error_string`

_convert axrResult enum to a string. The string is statically allocated and should not be freed. Never returns NULL._
```c
AXR_EXPORT const char * axr_error_string (
    axrResult result
)
```

### function `axr_get_context`

_Get the context that owns the object._
```c
axrContext * axr_get_context (
    axrObject *obj
)
```


**Parameters:**


* `obj` the object


**Returns:**

the context that owns the object, or NULL if the object is NULL

### function `axr_get_model_input`

_Get the shape and dtype of the input tensor n._
```c
AXR_EXPORT axrTensorInfo axr_get_model_input (
    axrModel *model,
    size_t n
)
```


**Parameters:**


* `model` the model
* `n` the input tensor index, if out of range AXR\_ERROR\_INVALID\_ARGUMENT is set and the return ndims is 0

### function `axr_get_model_output`

_Get the shape and dtype of the output tensor n._
```c
AXR_EXPORT axrTensorInfo axr_get_model_output (
    axrModel *model,
    size_t n
)
```


**Parameters:**


* `model` the model
* `n` the output tensor index, if out of range AXR\_ERROR\_INVALID\_ARGUMENT is set and the return ndims is 0.


### function `axr_get_model_properties`

Get further information about the model emitted by the compiler.

```c
AXR_EXPORT axrProperties *axr_get_model_properties(
    axrModel *model
)
```

**Parameters:**
* `model` the model

**Details**

This function returns a set of properties that describe the model,
The properties available are subject to change. Some keys that might be included are:

| Key                 | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| preamble_graph      | Relative path to a preamble ONNX file containing the initial nodes of the model. These nodes are removed by the compiler and executed on the host. |
| postamble_graph     | Relative path to a postamble ONNX file containing the final nodes of the model. These nodes are removed by the compiler and executed on the host. |
| input_tensor_layout | Always NHWC in this version.                                                |

**Returns:**

An `axrProperties` object that contains the properties.

### function `axr_get_property`

_Get the value of a property._
```c
AXR_EXPORT const char * axr_get_property (
    axrProperties *properties,
    const char *key
)
```


**Parameters:**


* `properties` the properties object
* `key` the key to set


**Returns:**

the value of the property, or NULL if the property does not exist
### function `axr_last_error`

_get the last error that occurred on the context._
```c
AXR_EXPORT axrResult axr_last_error (
    const axrObject *c
)
```

### function `axr_last_error_string`

_get the last error string that occurred on the context. The string is valid until the next call to an axr function in this thread._
```c
AXR_EXPORT const char * axr_last_error_string (
    const axrObject *c
)
```

### function `axr_list_devices`

_Enumerate devices available to the context._
```c
AXR_EXPORT size_t axr_list_devices (
    axrContext *context,
    axrDeviceInfo **devices
)
```


**Parameters:**


* `context` the context
* `devices` [out] a pointer to receive a an array of [**axrDeviceInfo**](#struct-axrdeviceinfo) pointers. If NULL then the number of devices is returned. The returned array is owned by the context and should not be freed by the caller. It is valid until the next call to axr\_list\_devices.


**Returns:**

the number of devices available to the context, or 0 if no devices are available or an error occurred. Check axr\_last\_error for details.


Example:
````cpp
axrDeviceInfo* devices = nullptr
size_t num_devices = axr_list_devices(context, &devices);
for (size_t n = 0; n != num_devices; ++n) {
   printf("Device %s\n", devices[n]->name);
}
````
### function `axr_list_properties`

_Get a list of keys in the properties object._
```c
AXR_EXPORT const char * axr_list_properties (
    axrProperties *properties
)
```


**Returns:**

semi-colon separated list of properties keys on the object
### function `axr_load_model`

_Parse a model from a file._
```c
AXR_EXPORT axrModel * axr_load_model (
    axrContext *context,
    const char *path
)
```


**Parameters:**


* `context` the context
* `path` the path to the model file return null if the model could not be created if non-null the model creation was successful
### function `axr_load_model_instance`

_Load a model onto a connected device._
```c
AXR_EXPORT axrModelInstance * axr_load_model_instance (
    axrConnection *connection,
    axrModel *model,
    axrProperties *properties
)
```


**Parameters:**


* `connection` the connected device
* `model` the model to load
* `properties` the properties to use when loading the model, or NULL to use the default set of properties.
### function `axr_num_model_inputs`

_Get the number of inputs to the model._
```c
AXR_EXPORT size_t axr_num_model_inputs (
    axrModel *model
)
```


**Parameters:**


* `model` the model
### function `axr_num_model_outputs`

_Get the number of outputs to the model._
```c
AXR_EXPORT size_t axr_num_model_outputs (
    axrModel *model
)
```


**Parameters:**


* `model` the model
### function `axr_read_device_configuration`

_Read the configuration of a device._
```c
AXR_EXPORT axrProperties * axr_read_device_configuration (
    axrContext *context,
    axrDeviceInfo *device
)
```


**Parameters:**


* `context` the context
* `device` the device to read the configuration from


**Returns:**

the configuration of the device, or NULL if the device is not valid.
### function `axr_run_model_instance`

_Run one frame the model instance._
```c
AXR_EXPORT axrResult axr_run_model_instance (
    axrModelInstance *instance,
    axrArgument *inputs,
    size_t num_inputs,
    axrArgument *output,
    size_t num_outputs
)
```


**Parameters:**


* `instance` the model instance
* `inputs/num_inputs` an array of input arguments
* `output/num_outputs` an array of output arguments


**Returns:**

AXR\_SUCCESS if the model was run successfully, otherwise an error code
### function `axr_set_logger`

```c
AXR_EXPORT void axr_set_logger (
    axrContext *context,
    axrLogLevel level,
    axr_logger_fn logger,
    void *arg
)
```


brief Set a logger and the logging level for the context

**Parameters:**


* `context` the context
* `logger` a function that will be called with log messages
* `level` the minimum level of log message to forward to the logger
* `arg` an argument that will be passed to the logger function
### function `axr_set_property`

_Set the value of a property._
```c
AXR_EXPORT void axr_set_property (
    axrProperties *properties,
    const char *key,
    const char *value
)
```


**Parameters:**


* `properties` the properties object
* `key` the key to set
* `value` the value to set
### function `axr_tensor_size`

_Get the size of a tensor in bytes._
```c
AXR_EXPORT size_t axr_tensor_size (
    const axrTensorInfo *info
)
```



## File include/axruntime/axruntime.hpp





## Structures and Types

| Type | Name |
| ---: | :--- |
| struct | [**object\_deleter**](#struct-axrobject_deleter) <br> |



## Structures and Types Documentation

### struct `axr::object_deleter`
