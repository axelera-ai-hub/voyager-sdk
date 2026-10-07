---
title: "AxRuntime C/C++ API"
---

# AxRuntime C/C++ API

The low-level C API for direct AIPU access, callable from C or C++. This is the
same object model exposed by the [axelera.runtime Python API](axruntime-py.md) —
context, connection, model, model instance — with manual lifetime management.

Use this API when you are integrating Metis into an existing C or C++
application and need fine-grained control over device selection, buffer
management, and multi-instance execution. If you want a ready-made C++
inference pipeline with preprocessing, decoding and multi-stream support, start
with [AxInferenceNet](axinferencenet.md) instead.

## Header files

The headers are installed by the runtime wheel (`axelera_runtime`, pulled in by
`pip install axelera-rt` — see the [SDK installation
guide](../../user-guides/sdk-install.md)) into the `axelera/include` directory
of your virtual environment:

| Header | Contents |
|--------|----------|
| [`axruntime/axruntime.h`](#file-axruntimeh) | The C API: all types and functions. |
| [`axruntime/axruntime.hpp`](#file-axruntimehpp) | An optional, experimental C++ convenience wrapper providing RAII handles. |

Link against `libaxruntime`, installed alongside them at
`axelera/lib/libaxruntime.so`. Including `axruntime.h` from C++ automatically
pulls in `axruntime.hpp`.

## Versioning

The API carries a version number that a client can check at runtime against the
version it compiled against. This page describes the API version defined by
`AXR_API_VERSION` in the header you have installed; at the time of writing that
is **3.3**.

```c
#define AXR_API_VERSION_MAJOR 0x0003u
#define AXR_API_VERSION_MINOR 0x0003u
#define AXR_API_VERSION ((AXR_API_VERSION_MAJOR << 16) | AXR_API_VERSION_MINOR)
```

The version is a 32-bit integer with the major version in the upper 16 bits and
the minor version in the lower 16 bits. The major version is incremented when
there are breaking changes, and the minor version when there are non-breaking
changes. Adding a new function results in a minor version change; changing an
enum or a function signature results in a major version change. Major changes
require a re-compilation and may require changes to your code.

```c
if (axr_api_version() != AXR_API_VERSION) {
  fprintf(stderr, "API version mismatch\n");
  return 1;
}
```

## File axruntime.h

### Structures and Types

| Type | Name |
| ---: | :--- |
| struct | [**axrArgument**](#struct-axrargument) <br>_An input or output argument to a model._ |
| struct | [**axrDeviceInfo**](#struct-axrdeviceinfo) <br>_Description of a device available to the context._ |
| struct | [**axrTensorInfo**](#struct-axrtensorinfo) <br>_Shape, format, padding and quantization of a model input or output._ |
| enum | [**axrResult**](#enum-axrresult) <br>_Result and error codes._ |
| enum | [**axrLogLevel**](#enum-axrloglevel) <br>_Host log levels._ |
| enum | [**axrBoardType**](#enum-axrboardtype) <br>_The board a device is hosted on._ |
| enum | [**axrHWGeneration**](#enum-axrhwgeneration) <br>_The hardware generation of a board._ |
| enum | [**AXR\_TENSOR\_DATA\_TYPE**](#enum-axr_tensor_data_type) <br>_Element format of a tensor._ |
| typedef | [**axrObject**](#opaque-object-types) <br>_The base type of all library objects._ |
| typedef | [**axrContext**](#opaque-object-types) <br>_The root object for the library._ |
| typedef | [**axrConnection**](#opaque-object-types) <br>_A connection to a device._ |
| typedef | [**axrModel**](#opaque-object-types) <br>_A parsed model._ |
| typedef | [**axrModelInstance**](#opaque-object-types) <br>_A model loaded onto a connected device._ |
| typedef | [**axrProperties**](#opaque-object-types) <br>_A set of key/value pairs._ |
| typedef | [**axr\_file\_handle\_t**](#typedef-axr_file_handle_t) <br>_Platform-specific handle for a dmabuf._ |
| function pointer type | [**axr\_logger\_fn**](#typedef-axr_logger_fn) <br>_Host log callback._ |
| function pointer type | [**axr\_device\_logger\_fn**](#typedef-axr_device_logger_fn) <br>_Device log streaming callback._ |
| constant | [**AXR\_MAX\_TENSOR\_DIMS**](#constant-axr_max_tensor_dims) <br>_Maximum rank of a tensor._ |
| macro | [**AXR\_OBJECT**](#macro-axr_object) <br>_Cast any object to_ `axrObject`_._ |

### Functions

#### Library and objects

| Type | Name |
| ---: | :--- |
| unsigned int | [**axr\_api\_version**](#function-axr_api_version) (void) <br>_Get the version of the API._ |
| void | [**axr\_destroy**](#function-axr_destroy) (const axrObject \*obj) <br>_Destroy any non-value object created by the library._ |
| axrContext \* | [**axr\_get\_context**](#function-axr_get_context) (axrObject \*obj) <br>_Get the context that owns the object._ |
| const char \* | [**axr\_error\_string**](#function-axr_error_string) (axrResult result) <br>_Convert an_ `axrResult` _to a string._ |
| axrResult | [**axr\_last\_error**](#function-axr_last_error) (const axrObject \*c) <br>_Get the last error that occurred on the context._ |
| const char \* | [**axr\_last\_error\_string**](#function-axr_last_error_string) (const axrObject \*c) <br>_Get the last error string that occurred on the context._ |
| size\_t | [**axr\_tensor\_size**](#function-axr_tensor_size) (const [**axrTensorInfo**](#struct-axrtensorinfo) \*info) <br>_Get the size of a tensor in bytes._ |
| const char \* | [**axr\_board\_type\_string**](#function-axr_board_type_string) (axrBoardType result) <br>_Convert an_ `axrBoardType` _to a string._ |
| axrHWGeneration | [**axr\_hw\_generation\_from\_board\_type**](#function-axr_hw_generation_from_board_type) (const axrBoardType type) <br>_Get the hardware generation for a board type._ |

#### Context

| Type | Name |
| ---: | :--- |
| axrContext \* | [**axr\_create\_context**](#function-axr_create_context) (void) <br>_Create a new context._ |
| void | [**axr\_set\_logger**](#function-axr_set_logger) (axrContext \*context, axrLogLevel level, axr\_logger\_fn logger, void \*arg) <br>_Set a logger and logging level for the context._ |

#### Devices

| Type | Name |
| ---: | :--- |
| size\_t | [**axr\_list\_devices**](#function-axr_list_devices) (axrContext \*context, [**axrDeviceInfo**](#struct-axrdeviceinfo) \*\*devices) <br>_Enumerate devices available to the context._ |
| size\_t | [**axr\_device\_l2\_size**](#function-axr_device_l2_size) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_Get the size of the runtime-reserved L2 region._ |
| axrResult | [**axr\_configure\_device**](#function-axr_configure_device) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, const axrProperties \*properties) <br>_Configure a device._ |
| axrResult | [**axr\_device\_ready**](#function-axr_device_ready) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_Check if a device is ready after configuration._ |
| axrProperties \* | [**axr\_read\_device\_configuration**](#function-axr_read_device_configuration) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_Read the live configuration of a device._ |
| axrProperties \* | [**axr\_read\_saved\_config**](#function-axr_read_saved_config) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_Read the persisted configuration for a device._ |
| axrResult | [**axr\_write\_saved\_config**](#function-axr_write_saved_config) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, const char \*key, const char \*value) <br>_Set a key in a device's saved configuration._ |
| axrResult | [**axr\_remove\_saved\_config**](#function-axr_remove_saved_config) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, const char \*key) <br>_Remove a key from a device's saved configuration._ |
| axrConnection \* | [**axr\_device\_connect**](#function-axr_device_connect) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, size\_t num\_sub\_devices, const axrProperties \*properties) <br>_Connect to a device._ |

#### Device logging

| Type | Name |
| ---: | :--- |
| const char \* | [**axr\_device\_list\_log\_text\_sources**](#function-axr_device_list_log_text_sources) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_List available text log sources._ |
| const char \* | [**axr\_device\_list\_log\_data\_sources**](#function-axr_device_list_log_data_sources) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_List available binary log sources._ |
| axrResult | [**axr\_device\_set\_log\_level**](#function-axr_device_set_log_level) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, const char \*source, const char \*config) <br>_Set the log configuration for a source._ |
| const char \* | [**axr\_device\_peek\_log**](#function-axr_device_peek_log) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, const char \*source) <br>_Non-destructively read the latest log message._ |
| const char \* | [**axr\_device\_get\_log\_stats**](#function-axr_device_get_log_stats) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, const char \*source) <br>_Get statistics about a log source._ |
| axrResult | [**axr\_device\_clear\_log**](#function-axr_device_clear_log) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, const char \*source) <br>_Clear buffered logs for a source._ |
| axrResult | [**axr\_device\_start\_logger**](#function-axr_device_start_logger) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device, const char \*sources, axr\_device\_logger\_fn logger, void \*arg, const axrProperties \*kwargs) <br>_Start streaming logs from device sources._ |
| void | [**axr\_device\_stop\_logger**](#function-axr_device_stop_logger) (axrContext \*context, const [**axrDeviceInfo**](#struct-axrdeviceinfo) \*device) <br>_Stop streaming logs._ |

#### Properties

| Type | Name |
| ---: | :--- |
| axrProperties \* | [**axr\_create\_properties**](#function-axr_create_properties) (axrContext \*context, const char \*initial\_properties) <br>_Create a new properties object._ |
| const char \* | [**axr\_list\_properties**](#function-axr_list_properties) (axrProperties \*properties) <br>_Get a list of keys in the properties object._ |
| const char \* | [**axr\_get\_property**](#function-axr_get_property) (axrProperties \*properties, const char \*key) <br>_Get the value of a property._ |
| void | [**axr\_set\_property**](#function-axr_set_property) (axrProperties \*properties, const char \*key, const char \*value) <br>_Set the value of a property._ |
| void | [**axr\_del\_property**](#function-axr_del_property) (axrProperties \*properties, const char \*key) <br>_Remove a key from properties._ |

#### Models

| Type | Name |
| ---: | :--- |
| axrModel \* | [**axr\_load\_model**](#function-axr_load_model) (axrContext \*context, const char \*path) <br>_Parse a model from a file._ |
| size\_t | [**axr\_num\_model\_inputs**](#function-axr_num_model_inputs) (axrModel \*model) <br>_Get the number of inputs to the model._ |
| [**axrTensorInfo**](#struct-axrtensorinfo) | [**axr\_get\_model\_input**](#function-axr_get_model_input) (axrModel \*model, size\_t n) <br>_Get details of the input tensor n._ |
| size\_t | [**axr\_num\_model\_outputs**](#function-axr_num_model_outputs) (axrModel \*model) <br>_Get the number of outputs to the model._ |
| [**axrTensorInfo**](#struct-axrtensorinfo) | [**axr\_get\_model\_output**](#function-axr_get_model_output) (axrModel \*model, size\_t n) <br>_Get details of the output tensor n._ |
| axrProperties \* | [**axr\_get\_model\_properties**](#function-axr_get_model_properties) (axrModel \*model) <br>_Get further information about the model emitted by the compiler._ |
| axrModelInstance \* | [**axr\_load\_model\_instance**](#function-axr_load_model_instance) (axrConnection \*connection, axrModel \*model, axrProperties \*properties) <br>_Load a model onto a connected device._ |
| axrResult | [**axr\_run\_model\_instance**](#function-axr_run_model_instance) (axrModelInstance \*instance, [**axrArgument**](#struct-axrargument) \*inputs, size\_t num\_inputs, [**axrArgument**](#struct-axrargument) \*output, size\_t num\_outputs) <br>_Run one frame through the model instance._ |

#### Error reporting

| Type | Name |
| ---: | :--- |
| bool | [**axr\_activate\_error\_reporter**](#function-axr_activate_error_reporter) (void) <br>_Activate the error reporter._ |
| bool | [**axr\_trigger\_error\_reporter**](#function-axr_trigger_error_reporter) (void) <br>_Explicitly trigger the error reporter._ |

### Structures and Types Documentation

#### struct `axrArgument`

_An input or output argument to a model._

The data field is a pointer to the data, which can be in host memory, or
alternatively it can be a reference to a dmabuf file descriptor and an offset
within it. Only one of `fd` or `ptr` can be valid (`nullptr` for `ptr`, `0` or
`-1` for `fd`).

```c
typedef struct axrArgument {
  void *ptr;
  axr_file_handle_t fd;
  size_t offset;
  size_t size;
} axrArgument;
```

Variables:

- `void * ptr` <br>_Pointer to the data if the data is in host memory, or NULL if the data is in a dmabuf._

- `axr_file_handle_t fd` <br>_File descriptor for a dmabuf, or 0 or -1 if the data is in host memory._

- `size_t offset` <br>_Offset within the buffer at which the data starts._<br>**Note:** in this version this is not supported, it must be zero.

- `size_t size` <br>_Optional. If 0 it will be ignored; if given it will be checked against the expected tensor size._

#### struct `axrDeviceInfo`

_Description of a device available to the context._

Returned by [axr_list_devices](#function-axr_list_devices). The array is owned
by the context.

Variables:

- `char name[256]` <br>_The name of the device, for example "metis-0:3:0"._

- `size_t subdevice_count` <br>_The number of subdevices on the device; for Metis this is 4._

- `size_t max_memory` <br>_The maximum memory available on the device._<br>**Note:** in the current implementation this field is not populated and will always be 0.

- `int in_use` <br>_The number of subdevices in use._<br>**Note:** in the current implementation this field is not populated and will always be 0.

- `char in_use_by[256]` <br>_The username and process id of the user(s) using the device, comma separated._<br>**Note:** in the current implementation this field is not populated and will always be empty.

- `axrBoardType board_type` <br>_The board type of the device._

- `char firmware_version[256]` <br>_The firmware version of the device, for example v1.1.0-rc5-2-g1234567._

- `int board_revision` <br>_The board revision of the device._

- `char flashed_firmware_version[256]` <br>_The version of the firmware stored in flash memory of the device._

- `char board_controller_firmware_version[256]` <br>_The board controller firmware version on the device._

- `char board_controller_board_type[256]` <br>_The board controller board type._

**Note:** the size and layout of this structure are part of the ABI. Values
that would otherwise be added here are exposed as functions instead — see
[axr_device_l2_size](#function-axr_device_l2_size).

#### struct `axrTensorInfo`

_Shape, format, padding and quantization of a model input or output._

Returned by [axr_get_model_input](#function-axr_get_model_input) and
[axr_get_model_output](#function-axr_get_model_output). See the
[axelera.runtime](axruntime-py.md) documentation for `TensorInfo` for example
NumPy code that applies the quantization and padding information.

Variables:

- `size_t dims[AXR_MAX_TENSOR_DIMS]` <br>_The dimensions of the tensor; dims[0:ndims] can be considered the shape._

- `size_t ndims` <br>_The number of dimensions in the tensor._

- `size_t bits` <br>_The number of bits per element._

- `size_t type` <br>_The format of the tensor, one of [AXR\_TENSOR\_DATA\_TYPE](#enum-axr_tensor_data_type)._

- `char name[256]` <br>_The name of the tensor referenced in the model._

- `size_t padding[AXR_MAX_TENSOR_DIMS][2]` <br>_Amount of padding on the tensor, as a list of (start, end) pairs (see_ `numpy.pad`_). For example for NHWC this is:_
  ```
  (Nbefore, Nafter), (Hbefore, Hafter), (Wbefore, Wafter), (Cbefore, Cafter) = padding[:ndims * 2]
  ```
  _Fewer than_ `ndims` _pairs may be meaningful; any trailing entries are zero. A tensor stored at a higher rank than its logical shape — for example an axis split across the array — carries no padding on those extra dimensions._

- `double scale` <br>_Scale for quantization/dequantization._<br>**Note:** this is NaN when the tensor uses per-channel quantization, which is not represented here; the consumer performs dequantization itself.

- `int zero_point` <br>_Zero-point for quantization/dequantization._

#### enum `axrResult`

_Result and error codes._

```c
typedef enum axrResult {
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
  AXR_ERROR_UNKNOWN_ERROR,
} axrResult;
```

Use [axr_error_string](#function-axr_error_string) to convert a code to a
human-readable string.

#### enum `axrLogLevel`

_Host log levels, used by [axr_set_logger](#function-axr_set_logger)._

```c
typedef enum axrLogLevel {
  AXR_LOG_TRACE,
  AXR_LOG_LOG,
  AXR_LOG_DEBUG,
  AXR_LOG_INFO,
  AXR_LOG_FIXME,
  AXR_LOG_WARNING,
  AXR_LOG_ERROR,
} axrLogLevel;
```

#### enum `axrBoardType`

_The board a device is hosted on._

```c
typedef enum axrBoardType {
  AXR_BOARD_METIS_ALPHA_PCIE,
  AXR_BOARD_METIS_ALPHA_M2,
  AXR_BOARD_METIS_OMEGA_PCIE,
  AXR_BOARD_METIS_OMEGA_M2,
  AXR_BOARD_METIS_OMEGA_DEVBOARD,
  AXR_BOARD_METIS_OMEGA_SBC,
  AXR_BOARD_EUROPA_DEVBOARD,
  AXR_BOARD_EUROPA_PCIE,
  AXR_BOARD_MAX = 8,
} axrBoardType;
```

Use [axr_board_type_string](#function-axr_board_type_string) to convert a value
to a string.

#### enum `axrHWGeneration`

_The hardware generation of a board._

```c
typedef enum axrHWGeneration {
  AXR_HW_GENERATION_METIS,
  AXR_HW_GENERATION_EUROPA,
  AXR_HW_GENERATION_MAX = 2,
} axrHWGeneration;
```

Derive this from a board type with
[axr_hw_generation_from_board_type](#function-axr_hw_generation_from_board_type).

#### enum `AXR_TENSOR_DATA_TYPE`

_Element format of a tensor, reported in the_ `type` _field of
[axrTensorInfo](#struct-axrtensorinfo)._

```c
typedef enum AXR_TENSOR_DATA_TYPE {
  AXR_UNSIGNED = 0,
  AXR_SIGNED = 1,
  AXR_FLOAT = 2,
} AXR_TENSOR_DATA_TYPE;
```

#### Opaque object types

These are opaque handles. Each is created by the library and destroyed with
[axr_destroy](#function-axr_destroy); all of them can be cast to `axrObject`
with the [AXR_OBJECT](#macro-axr_object) macro.

```c
typedef struct axrObject axrObject;
typedef struct axrModel axrModel;
typedef struct axrConnection axrConnection;
typedef struct axrContext axrContext;
typedef struct axrModelInstance axrModelInstance;
typedef struct axrProperties axrProperties;
```

| Type | Created by | Represents |
|------|-----------|------------|
| `axrObject` | — | The base type of all objects; accepts the error and context accessors. |
| `axrContext` | [axr_create_context](#function-axr_create_context) | The root object; owns every object created from it. |
| `axrConnection` | [axr_device_connect](#function-axr_device_connect) | A connection to a device. |
| `axrModel` | [axr_load_model](#function-axr_load_model) | A model parsed from a file, not yet on a device. |
| `axrModelInstance` | [axr_load_model_instance](#function-axr_load_model_instance) | A model loaded onto a connected device and ready to run. |
| `axrProperties` | [axr_create_properties](#function-axr_create_properties) | A set of key/value pairs used to configure the calls above. |

#### typedef `axr_file_handle_t`

_A platform-specific file handle type for containing a dmabuf._

```c
#ifdef _WIN32
  typedef HANDLE axr_file_handle_t;
#else
  typedef int axr_file_handle_t;
#endif
```

#### typedef `axr_logger_fn`

_Host log callback, registered with [axr_set_logger](#function-axr_set_logger)._

```c
typedef void (*axr_logger_fn)(void *arg, axrLogLevel level, const char *msg);
```

#### typedef `axr_device_logger_fn`

_Device log streaming callback, registered with
[axr_device_start_logger](#function-axr_device_start_logger)._

```c
typedef void (*axr_device_logger_fn)(void *arg, axrLogLevel level,
                                     const char *source, const char *msg,
                                     size_t msg_size);
```

**Parameters:**

* `arg` the user-provided argument passed to the logger
* `level` log level of the message
* `source` originating log source name
* `msg` pointer to the message buffer (**not** null-terminated)
* `msg_size` size of the message buffer in bytes

The callback is called from a background thread owned by the context.

**Note:** in the current implementation all messages are reported as
`AXR_LOG_INFO`.

#### constant `AXR_MAX_TENSOR_DIMS`

This gives a fixed size to [axrTensorInfo](#struct-axrtensorinfo) and
determines the maximum rank of a tensor.

```c
#define AXR_MAX_TENSOR_DIMS 8
```

#### macro `AXR_OBJECT`

Casts any of the [opaque object types](#opaque-object-types) to `axrObject`, so
it can be passed to [axr_destroy](#function-axr_destroy),
[axr_last_error](#function-axr_last_error) and
[axr_get_context](#function-axr_get_context). In C++ this resolves to the
type-checked `axr::to_object` helper.

```c
#ifdef __cplusplus
#define AXR_OBJECT(obj) axr::to_object(obj)
#else
#define AXR_OBJECT(obj) ((axrObject *)obj)
#endif
```

### Functions Documentation

#### Library and objects

##### function `axr_api_version`

_Get the version of the API._

```c
AXR_EXPORT unsigned int axr_api_version(void)
```

**Returns:** the version of the API.

This allows a client to check the version of the API at runtime to ensure
compatibility — see [Versioning](#versioning).

##### function `axr_destroy`

_Destroy any non-value object created by the library._

```c
AXR_EXPORT void axr_destroy(const axrObject *obj)
```

**Parameters:**

* `obj` the object to destroy

This function is only required to be called on the context object; all other
objects will be destroyed when the context is destroyed. However, other objects
can be destroyed sooner if necessary. For example, to load a different model
onto the target you should first destroy the model instance.

##### function `axr_get_context`

_Get the context that owns the object._

```c
AXR_EXPORT axrContext *axr_get_context(axrObject *obj)
```

**Parameters:**

* `obj` the object

**Returns:** the context that owns the object, or NULL if the object is NULL.

##### function `axr_error_string`

_Convert an_ `axrResult` _enum to a string._

```c
AXR_EXPORT const char *axr_error_string(axrResult result)
```

The string is statically allocated and should not be freed. Never returns NULL.

##### function `axr_last_error`

_Get the last error that occurred on the context._

```c
AXR_EXPORT axrResult axr_last_error(const axrObject *c)
```

##### function `axr_last_error_string`

_Get the last error string that occurred on the context._

```c
AXR_EXPORT const char *axr_last_error_string(const axrObject *c)
```

The string is valid until the next call to an `axr` function in this thread.

##### function `axr_tensor_size`

_Get the size of a tensor in bytes._

```c
AXR_EXPORT size_t axr_tensor_size(const axrTensorInfo *info)
```

##### function `axr_board_type_string`

_Convert an_ `axrBoardType` _to a string._

```c
AXR_EXPORT const char *axr_board_type_string(axrBoardType result)
```

The string is statically allocated and should not be freed. Never returns NULL.

##### function `axr_hw_generation_from_board_type`

_Get the hardware generation for a given board type._

```c
AXR_EXPORT axrHWGeneration axr_hw_generation_from_board_type(
    const axrBoardType type
)
```

#### Context

##### function `axr_create_context`

_Create a new context._

```c
AXR_EXPORT axrContext *axr_create_context(void)
```

**Returns:** a valid context (never NULL).

The context is the root object for the library. It is used to create all other
objects and it owns all the objects created from it, as well as providing access
to some global services like errors and logging.

##### function `axr_set_logger`

_Set a logger and the logging level for the context._

```c
AXR_EXPORT void axr_set_logger(
    axrContext *context,
    axrLogLevel level,
    axr_logger_fn logger,
    void *arg
)
```

**Parameters:**

* `context` the context
* `level` the minimum level of log message to forward to the logger
* `logger` a function that will be called with log messages
* `arg` an argument that will be passed to the logger function

#### Devices

##### function `axr_list_devices`

_Enumerate devices available to the context._

```c
AXR_EXPORT size_t axr_list_devices(
    axrContext *context,
    axrDeviceInfo **devices
)
```

**Parameters:**

* `context` the context
* `devices` [out] a pointer to receive an array of [axrDeviceInfo](#struct-axrdeviceinfo) pointers. If NULL then only the number of devices is returned. The returned array is owned by the context and should not be freed by the caller. It is valid until the next call to `axr_list_devices`.

**Returns:** the number of devices available to the context, or 0 if no devices
are available or an error occurred. Check
[axr_last_error](#function-axr_last_error) for details.

Example:

```cpp
axrDeviceInfo *devices = nullptr;
size_t num_devices = axr_list_devices(context, &devices);
for (size_t n = 0; n != num_devices; ++n) {
  printf("Device %s\n", devices[n].name);
}
```

##### function `axr_device_l2_size`

_Get the size in bytes of the runtime-reserved L2 memory region on a device._

```c
AXR_EXPORT size_t axr_device_l2_size(
    axrContext *context,
    const axrDeviceInfo *device
)
```

**Parameters:**

* `context` the context
* `device` the device to query, as returned by [axr_list_devices](#function-axr_list_devices)

**Returns:** the L2 region size in bytes, or 0 on error (check
[axr_last_error](#function-axr_last_error)).

This is exposed as a function rather than a field of
[axrDeviceInfo](#struct-axrdeviceinfo) so that adding it does not change the
size or layout of that public structure, preserving ABI compatibility for
existing clients.

##### function `axr_configure_device`

_Configure a device._

```c
AXR_EXPORT axrResult axr_configure_device(
    axrContext *context,
    const axrDeviceInfo *device,
    const axrProperties *properties
)
```

**Parameters:**

* `context` the context
* `device` the device to configure
* `properties` the options to change

Supported options are:

| Property | Default | Description |
|----------|---------|-------------|
| `reboot` | 0 | If 1, reboot the device before applying the configuration. |
| `clock_profile` | 800 | Device clock profile in MHz. |
| `clock_profile_core_0` ... `clock_profile_core_7` | 800 | Per-core clock profile in MHz. |
| `mvm_utilisation_core_0` ... `mvm_utilisation_core_7` | 100 | Per-core MVM utilisation as a percentage. |
| `device_firmware` | 0 | 1 to check and reload the firmware if necessary, `force` to force a reload. |
| `power_limit` | 0.0 | Power limit in watts before throttling is applied. Set to 0 or `off` to disable throttling (the limit is set to the maximum). |
| `sw_throttling` | device setting | Software thermal throttling, as `temp_threshold:temp_hysteresis[:mvm_limit]`. |
| `hw_throttling` | device setting | Hardware thermal throttling, same format as `sw_throttling`. |
| `freq_downscaling` | device setting | Thermal frequency downscaling, same format as `sw_throttling`. |
| `pvt_warning_threshold` | device setting | PVT sensor warning threshold, in degrees Celsius. |
| `persist` | 1 | A reserved control key rather than a device property. Set it to 0 or `false` to apply the configuration to the live device without writing the persistent keys to the device config file. |

`power_limit` is the only persistent key: it is written to the device config
file and restored on the next `axr_configure_device` call, unless `persist=0`
is given. Every other key applies to the live device only.

**Returns:**

* `AXR_SUCCESS` if the device was configured successfully
* `AXR_ERROR_INVALID_CONFIGURATION` if a configuration name is not recognised, or its value is not valid for that name
* `AXR_ERROR_INCOMPATIBLE_DEVICE` if a configuration name is recognised but the device does not support it, for example `power_limit` on a board without power-limit control
* `AXR_ERROR_VALUE_ERROR` if a value cannot be parsed

If the change requires time to take effect, `AXR_ERROR_PENDING_ERROR` is
returned. The caller should poll
[axr_device_ready](#function-axr_device_ready) to check whether the device is
ready before any other operations are performed on that device. For example, to
configure multiple devices:

```cpp
auto *props = axr_create_properties(context, "clock_profile=1000");
auto res0 = axr_configure_device(context, device0, props);
auto res1 = axr_configure_device(context, device1, props);
while (res0 == AXR_ERROR_PENDING_ERROR || res1 == AXR_ERROR_PENDING_ERROR) {
  usleep(50000);
  res0 = axr_device_ready(context, device0);
  res1 = axr_device_ready(context, device1);
}
```

##### function `axr_device_ready`

_Check if a device is ready to be connected to after configuration._

```c
AXR_EXPORT axrResult axr_device_ready(
    axrContext *context,
    const axrDeviceInfo *device
)
```

**Parameters:**

* `context` the context
* `device` the device to check

**Returns:** `AXR_SUCCESS` if the device is ready to be connected to, otherwise
`AXR_ERROR_PENDING_ERROR` if the device is still configuring.

##### function `axr_read_device_configuration`

_Read the configuration of a device._

```c
AXR_EXPORT axrProperties *axr_read_device_configuration(
    axrContext *context,
    const axrDeviceInfo *device
)
```

**Parameters:**

* `context` the context
* `device` the device to read the configuration from

**Returns:** the configuration of the device, or NULL if the device is not
valid.

##### function `axr_read_saved_config`

_Read the saved (persisted) configuration for a device._

```c
AXR_EXPORT axrProperties *axr_read_saved_config(
    axrContext *context,
    const axrDeviceInfo *device
)
```

**Parameters:**

* `context` the context
* `device` the device whose saved config to read

Reads the device config file and returns the key=value pairs from the device's
section verbatim — every key as written, with no filtering. This is a pure
filesystem read keyed by the device name; it does **not** open or query the
device, unlike
[axr_read_device_configuration](#function-axr_read_device_configuration).

**Returns:** the saved properties (empty if the file or section is missing), or
NULL on a hard error (check [axr_last_error](#function-axr_last_error)). The
caller owns the result and must free it with
[axr_destroy](#function-axr_destroy).

##### function `axr_write_saved_config`

_Set a key=value pair in a device's saved configuration._

```c
AXR_EXPORT axrResult axr_write_saved_config(
    axrContext *context,
    const axrDeviceInfo *device,
    const char *key,
    const char *value
)
```

**Parameters:**

* `context` the context
* `device` the device whose saved config to write
* `key` the configuration key
* `value` the configuration value

Writes under the device's section, preserving all other sections and keys. The
file and parent directories are created as needed, and the write is atomic
(temp file plus rename). No validation is performed on the key. This is a pure
filesystem operation and does not touch the device.

**Returns:** `AXR_SUCCESS`, or an error — for example if the config path cannot
be resolved.

##### function `axr_remove_saved_config`

_Remove a key from a device's saved configuration._

```c
AXR_EXPORT axrResult axr_remove_saved_config(
    axrContext *context,
    const axrDeviceInfo *device,
    const char *key
)
```

**Parameters:**

* `context` the context
* `device` the device whose saved config to modify
* `key` the configuration key to remove

A no-op if the file, section, or key is absent. The write is atomic.

**Returns:** `AXR_SUCCESS`, or an error if the config path cannot be resolved.

##### function `axr_device_connect`

_Connect to a device._

```c
AXR_EXPORT axrConnection *axr_device_connect(
    axrContext *context,
    const axrDeviceInfo *device,
    size_t num_sub_devices,
    const axrProperties *properties
)
```

**Parameters:**

* `context` the context
* `device` the device to connect to, or NULL to connect to the first available device
* `num_sub_devices` number of sub-devices to connect to
* `properties` options to use when connecting to the device

The device runtime firmware will be checked and updated if necessary. This can
be skipped to speed up connection by setting `device_firmware_check=0` in the
properties.

Valid properties are:

| Property | Default | Description |
|----------|---------|-------------|
| `device_firmware_check` | 1 | If 1, check the firmware version of the device and reload if necessary. |

Any other property is reserved for internal use.

#### Device logging

These functions read and stream logs produced on the device itself. They are
independent of [axr_set_logger](#function-axr_set_logger), which forwards logs
produced by the host library.

##### function `axr_device_list_log_text_sources`

_List available text log sources on a device._

```c
AXR_EXPORT const char *axr_device_list_log_text_sources(
    axrContext *context,
    const axrDeviceInfo *device
)
```

**Parameters:**

* `context` the context
* `device` the device to query

**Returns:** a pointer to a statically managed C string containing a
semicolon-separated list of text sources. Valid until the next `axr` call in
this thread.

##### function `axr_device_list_log_data_sources`

_List available binary/data log sources on a device._

```c
AXR_EXPORT const char *axr_device_list_log_data_sources(
    axrContext *context,
    const axrDeviceInfo *device
)
```

**Parameters:**

* `context` the context
* `device` the device to query

**Returns:** a pointer to a statically managed C string containing a
semicolon-separated list of data sources. Valid until the next `axr` call in
this thread.

##### function `axr_device_set_log_level`

_Set the log configuration for a device log source._

```c
AXR_EXPORT axrResult axr_device_set_log_level(
    axrContext *context,
    const axrDeviceInfo *device,
    const char *source,
    const char *config
)
```

**Parameters:**

* `context` the context
* `device` the device for which to set the log level/config
* `source` the log source name to configure, as reported by [axr_device_list_log_text_sources](#function-axr_device_list_log_text_sources) or [axr_device_list_log_data_sources](#function-axr_device_list_log_data_sources)
* `config` the configuration string

**Returns:** `AXR_SUCCESS` on success, otherwise an error code indicating
failure.

##### function `axr_device_peek_log`

_Non-destructively read the latest available log message from a device source._

```c
AXR_EXPORT const char *axr_device_peek_log(
    axrContext *context,
    const axrDeviceInfo *device,
    const char *source
)
```

**Parameters:**

* `context` the context
* `device` the device to read logs from
* `source` the log source name to peek

**Returns:** a pointer to a statically managed C string with the latest message,
or NULL if none is available. Valid until the next `axr` call in this thread.

##### function `axr_device_get_log_stats`

_Get statistics about a device log source._

```c
AXR_EXPORT const char *axr_device_get_log_stats(
    axrContext *context,
    const axrDeviceInfo *device,
    const char *source
)
```

**Parameters:**

* `context` the context
* `device` the device to query
* `source` the log source name to get stats for

**Returns:** a pointer to a statically managed C string containing the stats,
never NULL on success. The format is subject to change.

##### function `axr_device_clear_log`

_Clear buffered logs for a device source._

```c
AXR_EXPORT axrResult axr_device_clear_log(
    axrContext *context,
    const axrDeviceInfo *device,
    const char *source
)
```

**Parameters:**

* `context` the context
* `device` the device whose logs to clear
* `source` the log source name to clear

**Returns:** `AXR_SUCCESS` on success, otherwise an error code.

##### function `axr_device_start_logger`

_Start streaming logs from one or more device sources._

```c
AXR_EXPORT axrResult axr_device_start_logger(
    axrContext *context,
    const axrDeviceInfo *device,
    const char *sources,
    axr_device_logger_fn logger,
    void *arg,
    const axrProperties *kwargs
)
```

**Parameters:**

* `context` the context
* `device` the device to stream logs from
* `sources` comma-separated list of source names to subscribe to
* `logger` callback to receive log messages, see [axr_device_logger_fn](#typedef-axr_device_logger_fn)
* `arg` user-provided argument to pass to the callback
* `kwargs` additional configuration options for the logger. There are no keyword arguments at present, so this can be NULL.

The logger callback is called from a background thread. The thread is owned by
the context and will be stopped when the context is destroyed or when
[axr_device_stop_logger](#function-axr_device_stop_logger) is called. Only one
logger can be configured per device.

**Returns:** `AXR_SUCCESS` on success, otherwise an error code.

##### function `axr_device_stop_logger`

_Stop streaming logs previously started with
[axr_device_start_logger](#function-axr_device_start_logger)._

```c
AXR_EXPORT void axr_device_stop_logger(
    axrContext *context,
    const axrDeviceInfo *device
)
```

**Parameters:**

* `context` the context
* `device` the device to stop streaming from

This function blocks until the background logging thread has exited. If no
logging thread is running it returns immediately.

#### Properties

##### function `axr_create_properties`

_Create a new properties object._

```c
AXR_EXPORT axrProperties *axr_create_properties(
    axrContext *context,
    const char *initial_properties
)
```

**Parameters:**

* `context` the context
* `initial_properties` a string of newline-separated key=value pairs

For integer parameters the value should be passed as decimal. For boolean
parameters use 0/1.

**Returns:** a valid properties object (never NULL unless `context` is NULL).

##### function `axr_list_properties`

_Get a list of keys in the properties object._

```c
AXR_EXPORT const char *axr_list_properties(axrProperties *properties)
```

**Returns:** a semicolon-separated list of property keys on the object.

##### function `axr_get_property`

_Get the value of a property._

```c
AXR_EXPORT const char *axr_get_property(
    axrProperties *properties,
    const char *key
)
```

**Parameters:**

* `properties` the properties object
* `key` the key to read

**Returns:** the value of the property, or NULL if the property does not exist.

##### function `axr_set_property`

_Set the value of a property._

```c
AXR_EXPORT void axr_set_property(
    axrProperties *properties,
    const char *key,
    const char *value
)
```

**Parameters:**

* `properties` the properties object
* `key` the key to set
* `value` the value to set

##### function `axr_del_property`

_Remove a key from properties._

```c
AXR_EXPORT void axr_del_property(
    axrProperties *properties,
    const char *key
)
```

#### Models

##### function `axr_load_model`

_Parse a model from a file._

```c
AXR_EXPORT axrModel *axr_load_model(
    axrContext *context,
    const char *path
)
```

**Parameters:**

* `context` the context
* `path` the path to the model file

**Returns:** the parsed model, or NULL if the model could not be created.

##### function `axr_num_model_inputs`

_Get the number of inputs to the model._

```c
AXR_EXPORT size_t axr_num_model_inputs(axrModel *model)
```

##### function `axr_get_model_input`

_Get details of the input tensor n._

```c
AXR_EXPORT axrTensorInfo axr_get_model_input(
    axrModel *model,
    size_t n
)
```

**Parameters:**

* `model` the model
* `n` the input tensor index. If out of range, `AXR_ERROR_INVALID_ARGUMENT` is set and the returned `ndims` is 0.

##### function `axr_num_model_outputs`

_Get the number of outputs to the model._

```c
AXR_EXPORT size_t axr_num_model_outputs(axrModel *model)
```

##### function `axr_get_model_output`

_Get details of the output tensor n._

```c
AXR_EXPORT axrTensorInfo axr_get_model_output(
    axrModel *model,
    size_t n
)
```

**Parameters:**

* `model` the model
* `n` the output tensor index. If out of range, `AXR_ERROR_INVALID_ARGUMENT` is set and the returned `ndims` is 0.

##### function `axr_get_model_properties`

_Get further information about the model emitted by the compiler._

```c
AXR_EXPORT axrProperties *axr_get_model_properties(axrModel *model)
```

**Parameters:**

* `model` the model

This function returns a set of properties that describe the model. The
properties available are subject to change. The keys in this version are:

| Key | Description |
|-----|-------------|
| `preamble_graph` | Relative path to a preamble ONNX file containing the initial nodes of the model. These nodes are removed by the compiler and executed on the host. |
| `postamble_graph` | Relative path to a postamble ONNX file containing the final nodes of the model. These nodes are removed by the compiler and executed on the host. |
| `input_tensor_layout` | Always NHWC in this version. |
| `graphs` | The graphs contained in the model. |
| `metadata` | Model metadata recorded by the compiler. |
| `manifest_version` | Version of the model manifest. |
| `model_lib_file` | Relative path to the compiled model library. |
| `quantized_model_file` | Relative path to the quantized model. |
| `input_shapes_original` | Input shapes before any padding or reshaping applied by the compiler. |
| `output_shapes_original` | Output shapes before any padding or reshaping applied by the compiler. |

**Returns:** an `axrProperties` object that contains the properties.

##### function `axr_load_model_instance`

_Load a model onto a connected device._

```c
AXR_EXPORT axrModelInstance *axr_load_model_instance(
    axrConnection *connection,
    axrModel *model,
    axrProperties *properties
)
```

**Parameters:**

* `connection` the connected device
* `model` the model to load
* `properties` the properties to use when loading the model, or NULL to use the default set of properties

Valid properties are:

| Property | Default | Description |
|----------|---------|-------------|
| `aipu_cores` | 1 | L2 resources to allocate for the model; set to the batch size. |
| `num_sub_devices` | 1 | Number of sub-devices to use; set to the batch size of the model. |
| `input_dmabuf` | 0 | True if the input arguments are dmabuf file descriptors. |
| `output_dmabuf` | 0 | True if the output arguments are dmabuf file descriptors. |
| `device_profiling` | 1 | True to enable device profiling. This is enabled by default; pass 0 to turn it off. |
| `host_profiling` | 0 | True to enable host profiling. |
| `double_buffer` | 0 | True to enable double buffering. |
| `stream_tasklist` | 1 | True to stream the task list to the device. |
| `elf_in_ddr` | 1 | True if the model was compiled with `elf_in_ddr` as True. |
| `async_mode` | 0 | True to run the model through the non-blocking executor, so that several frames can be in flight on one instance. Supported only for batch-size-1 models. The completion-callback entry point itself is internal and not part of the stable ABI; the supported way to use this path is [AxInferenceNet](axinferencenet.md), which selects it automatically. |
| `max_inflight` | 0 | Maximum concurrent in-flight workloads per instance. Used only when `async_mode` is set; 0 lets the runtime choose. |
| `max_pending` | 0 | Maximum submitted-but-not-completed workloads per instance. Used only when `async_mode` is set; 0 lets the runtime choose. |

Any key that is not in this table is rejected with
`AXR_ERROR_INVALID_ARGUMENT`. Boolean properties accept the values `1`, `0`,
`true` and `false` (case-insensitive, surrounding whitespace ignored). Any
other value is rejected with `AXR_ERROR_VALUE_ERROR` or
`AXR_ERROR_INVALID_ARGUMENT`.

##### function `axr_run_model_instance`

_Run one frame through the model instance._

```c
AXR_EXPORT axrResult axr_run_model_instance(
    axrModelInstance *instance,
    axrArgument *inputs,
    size_t num_inputs,
    axrArgument *output,
    size_t num_outputs
)
```

**Parameters:**

* `instance` the model instance
* `inputs`, `num_inputs` an array of input arguments
* `output`, `num_outputs` an array of output arguments

**Returns:** `AXR_SUCCESS` if the model was run successfully, otherwise an error
code.

#### Error reporting

##### function `axr_activate_error_reporter`

_Activate the error reporter._

```c
AXR_EXPORT bool axr_activate_error_reporter(void)
```

This forks a child process that monitors the parent process. If the parent dies
unexpectedly, or if the parent calls
[axr_trigger_error_reporter](#function-axr_trigger_error_reporter), the child
process generates a report containing as much information as possible about the
host environment and the device state at the time of reporting.

Calling this more than once while the reporter is active has no effect.

##### function `axr_trigger_error_reporter`

_Explicitly trigger the error reporter._

```c
AXR_EXPORT bool axr_trigger_error_reporter(void)
```

This tells the child process to generate a report. Use it when the parent
process detects an error situation but has not crashed.

## File axruntime.hpp

**Note:** this header is an experimental C++ convenience wrapper and is subject
to change. The C API in [axruntime.h](#file-axruntimeh) is the stable interface.

It is included automatically when `axruntime.h` is compiled as C++, and provides
RAII handles so that library objects are destroyed on scope exit rather than by
explicit [axr_destroy](#function-axr_destroy) calls.

### Structures and Types

| Type | Name |
| ---: | :--- |
| struct | [**axr::object\_deleter**](#struct-axrobject_deleter) <br>_Deleter that calls_ `axr_destroy`_._ |
| typedef | [**axr::ptr**](#typedef-axrptr) <br>_A_ `unique_ptr` _alias for library objects._ |
| function | [**axr::to\_object**](#function-axrto_object) <br>_Type-checked cast to_ `axrObject *`_._ |
| function | [**axr::to\_ptr**](#function-axrto_ptr) <br>_Wrap a raw object pointer in an_ `axr::ptr`_._ |

### Structures and Types Documentation

#### struct `axr::object_deleter`

_A deleter that calls [axr_destroy](#function-axr_destroy) on any library
object._

```cpp
struct object_deleter {
  template <typename T>
  void operator()(T *ptr) const;
};
```

#### typedef `axr::ptr`

_A_ `std::unique_ptr` _alias that owns a library object and destroys it with
[axr_destroy](#function-axr_destroy)._

```cpp
template <typename T>
using ptr = std::unique_ptr<T, object_deleter>;
```

Example:

```cpp
axr::ptr<axrContext> context = axr::to_ptr(axr_create_context());
axr::ptr<axrModel> model = axr::to_ptr(axr_load_model(context.get(), path));
// both are destroyed in reverse order on scope exit
```

#### function `axr::to_object`

_Casts any library object to_ `axrObject *`_, with a compile-time check that the
argument is one of the library object types._

```cpp
template <typename T>
axrObject *to_object(T *obj);
```

This is what the [AXR_OBJECT](#macro-axr_object) macro resolves to in C++.

#### function `axr::to_ptr`

_Wraps a raw object pointer in an [axr::ptr](#typedef-axrptr)._

```cpp
template <typename T>
ptr<T> to_ptr(T *p);
```

## See also

* [axelera.runtime API](axruntime-py.md) — the Python binding of this same object model, with worked tutorials.
* [AxInferenceNet C++ Reference](axinferencenet.md) — the higher-level C++ inference pipeline, with preprocessing, decoding and multi-stream support.
* [AxInferenceNet C++ Example](../../tutorials/examples/axinferencenet-basic.md) — a runnable C++ example.
