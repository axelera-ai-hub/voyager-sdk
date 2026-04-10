![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Temperature Monitoring and Thermal Management Guide

- [Temperature Monitoring and Thermal Management Guide](#temperature-monitoring-and-thermal-management-guide)
  - [1.1. Temperature Monitoring Tools](#11-temperature-monitoring-tools)
    - [1.1.1. During Inference Execution](#111-during-inference-execution)
    - [1.1.2. Using axlogdevice in a terminal](#112-using-axlogdevice-in-a-terminal)
    - [1.1.3. In Your Application Code](#113-in-your-application-code)
    - [1.1.4. Using axmonitor](#114-using-axmonitor)
  - [1.2. Temperature Settings](#12-temperature-settings)
    - [1.2.1. Considerations and Throttling](#121-considerations-and-throttling)
    - [1.2.2. Default Temperature Settings](#122-default-temperature-settings)
  - [1.3. Configuring Temperature Throttling](#13-configuring-temperature-throttling)
    - [1.3.1. Setting Software Throttling](#131-setting-software-throttling)
  - [1.4. Safety Mechanisms](#14-safety-mechanisms)
    - [1.4.1. Warning Temperature](#141-warning-temperature)
    - [1.4.2. Shutdown Temperature](#142-shutdown-temperature)
    - [1.4.3. Frequency Downscaling](#143-frequency-downscaling)
  - [1.5. Temperature Specifications](#15-temperature-specifications)
    - [1.5.1. Operating Range](#151-operating-range)
- [2. Power Management Guide](#2-power-management-guide)
  - [2.1. Overview](#21-overview)
    - [2.1.1. Power Limiter](#211-power-limiter)
  - [2.2. Power Settings](#22-power-settings)
    - [2.2.1. Considerations and Control](#221-considerations-and-control)
  - [2.3. Configuring the Power Limit](#23-configuring-the-power-limit)
    - [2.3.1. Setting the Power Limit](#231-setting-the-power-limit)
  - [2.4. Enabling and Disabling Power Control](#24-enabling-and-disabling-power-control)
  - [2.5. Hardware Support](#25-hardware-support)
    - [2.5.1. Supported Boards](#251-supported-boards)

## 1.1. Temperature Monitoring Tools

You can monitor the chip temperature using any of these methods:

### 1.1.1. During Inference Execution

When running inference using `inference.py`, the temperature is automatically displayed in the output:

```bash
$ ./inference.py yolov8s-coco-onnx media/traffic2_480p.mp4 --no-display
INFO    : Core Temp : 39.0°C
```

This shows the maximum temperature among the 5 internal temperature sensors in Metis.

### 1.1.2. Using axlogdevice in a terminal

To get detailed temperature logs with timestamps from all 5 Metis internal temperature sensors (1x temperature sensor outside of AIPU cores silicon area and 4x temperature sensors for each AIPU core):

```bash
$ axlogdevice --slog-level inf:collector --slog
[04:58:54.012,603] <inf> collector: core_temps=[35,34,34,35,34]
```

The five values in `core_temps` correspond to: board sensor, core 0, core 1, core 2, core 3.

### 1.1.3. In Your Application Code

To monitor temperature in your application:

```python
from axelera.app import inf_tracers
tracers = inf_tracers.create_tracers('core_temp')
stream = create_inference_stream(
    ...
    tracers=tracers,
)
core_temp = stream.get_all_metrics()['core_temp']
print(f"Core temp is {core_temp.value}".center(90, '='))
```

### 1.1.4. Using axmonitor

You can monitor device temperatures in real-time using the `axmonitor` tool, which provides a visual interface for monitoring core temperatures and other device metrics. For more details, see the [axmonitor documentation](/docs/tutorials/axmonitor.md).

## 1.2. Temperature Settings

### 1.2.1. Considerations and Throttling

Temperature throttling in Metis devices is primarily achieved through two mechanisms:

1. **MVM Maximum Utilization Throttling**: The primary method of thermal management involves limiting the maximum percentage utilization of the Metis In-Memory-Compute block, which performs Matrix-Vector-Multiplications (MVM). When temperature thresholds are exceeded, the system reduces the MVM maximum utilization to maintain safe operating temperatures.

2. **Frequency Scaling**: As a secondary mechanism, the system can reduce the chip frequency when temperatures approach critical thresholds, providing an additional layer of thermal protection.

The throttling mechanism operates as follows:
- When temperature exceeds threshold T (°C), MVM utilization is limited to L%
- When temperature decreases by H hysteresis degrees (°C), the MVM utilization limit is removed

The temperature used for throttling is the maximum temperature across all Metis internal temperature sensors, and throttling is applied uniformly to all AIPU Cores in Metis.

The following table in section 1.2.2 details the specific temperature thresholds and parameters used for these thermal management mechanisms.

### 1.2.2. Default Temperature Settings

The following table shows all default temperature settings in Metis:

| Type | Parameter | Default Value | User Configurable | Notes |
|------|-----------|---------------|-------------------|-------|
| Software Throttling | T<sub>s</sub> | 200°C (Effectively Disabled) | Yes | Software-based thermal throttling threshold for temperature-constrained environments |
| | H<sub>s</sub> | 10°C | Yes | |
| | L<sub>s</sub> | 10% | Yes | |
| Hardware Throttling | T<sub>h</sub> | 105°C | No | Backup mechanism if warning signal is unused |
| | H<sub>h</sub> | 10°C | No | |
| | L<sub>h</sub> | 1% | No | |
| Safety Guards | Warning (T<sub>j</sub>) | 95°C | No |  |
| | Shutdown (T<sub>j</sub>) | 120°C | No | Set below absolute maximum (125°C) |
| | Freq Downscaling Start | 110°C | No | Activates after hardware throttling |

> [!NOTE]
> All temperatures in the table above refer to silicon junction temperatures (T<sub>j</sub>), which represent the temperature at the silicon die level. These temperatures are typically higher than the package or ambient temperatures.


## 1.3. Configuring Temperature Throttling

### 1.3.1. Setting Software Throttling

You can configure software throttling using the axdevice command line interface:

```bash
$ axdevice --set-sw-throttling=T:H:L
```

Where:
- T: Temperature threshold in Celsius
- H: Hysteresis in Celsius
- L: Throttle rate as percentage

Example: To set temperature threshold to 100°C, hysteresis to 5°C, and throttle rate to 10%:
```bash
$ axdevice --set-sw-throttling=100:5:10
```

To view current settings:
```bash
$ axdevice -v
```

> [!IMPORTANT]
> - Settings apply to all AIPU Cores
> - Settings do not persist across device reboots or firmware reloads
> - After exiting throttling mode, MVM utilization returns to its pre-throttling value

## 1.4. Safety Mechanisms

### 1.4.1. Warning Temperature

- Generates a log entry when reached
- Configurable via:
```bash
$ axdevice --set-pvt-warning-threshold 85
```
- Does not persist across reboots

### 1.4.2. Shutdown Temperature

> [!WARNING]
> - Fixed at default value, as in the table above (non-configurable)
> - Triggers board controller to disable all regulators
> - Requires full power cycle to recover

### 1.4.3. Frequency Downscaling

- Reduces chip frequency by 100 MHz every second while temperature remains above default threshold (minimum 200 MHz)
- Increases chip frequency by 100 MHz when temperature drops 5°C below the default threshold (e.g., 110°C results in 700 MHz, but reaching 105°C returns frequency to 800 MHz)
- Checks every second if temperature is within 5°C of Shutdown threshold
- Operates independently of MVM-based throttling

## 1.5. Temperature Specifications

### 1.5.1. Operating Range

For PCIe and M.2 boards (REV1.1, SDK v1.3.0):
- Operating range: [-20°C, +70°C]
- No loss of function, performance, or lifetime expected in this range

> [!NOTE]
> The temperature operating range specified above refers to ambient temperature, not junction temperature as shown in the table earlier in this document.

# 2. Power Management Guide

This guide explains how to configure and manage power consumption on Metis-based products using the power control loop.

## 2.1. Overview

### 2.1.1. Power Limiter

The Metis power limiter is a closed-loop PID (Proportional-Integral-Derivative) controller that continuously measures the board power consumption via the on-board INA236 power sensor and dynamically adjusts the MVM (Matrix-Vector-Multiplication) utilization across all AIPU cores to keep average power within a configured limit.

The control loop operates as follows:

1. **Measure**: reads instantaneous power from the INA236 sensor at ~200 Hz
2. **Compute**: calculates PID error relative to the configured power limit, applying a low-pass filter to smooth setpoint transitions
3. **Actuate**: maps the PID output to an MVM utilization percentage and applies it uniformly to all AIPU cores

This mechanism trades off peak compute throughput for predictable power envelope, which is essential in thermally or electrically constrained environments such as M.2 form-factor boards.

## 2.2. Power Settings

### 2.2.1. Considerations and Control

Power control in Metis is achieved exclusively through MVM utilization throttling: when measured power exceeds the configured limit, the controller reduces the maximum MVM utilization percentage on all AIPU cores; when power drops below the limit, utilization is gradually restored.

Key characteristics:

- **Sensor**: INA236 current/power monitor on the board power rail
- **Actuator**: per-core MVM utilization limit (same mechanism used by thermal throttling)
- **Scope**: applied uniformly to all AIPU cores in the device

## 2.3. Configuring the Power Limit

### 2.3.1. Setting the Power Limit

Use the `axdevice` command line interface to set the power limit:

```bash
$ axdevice --set-power-limit LIMIT
```

Where `LIMIT` is an integer representing watts (e.g., `20` represents 20 W).

Example: to limit device power consumption to 20 W:
```bash
$ axdevice --set-power-limit 20
```

Example: to disable the power limiter:
```bash
$ axdevice --set-power-limit 0
```

To view the current device configuration:
```bash
$ axdevice -v
```

## 2.4. Enabling and Disabling Power Control

The power limiter is enabled or disabled through the power limit value:

**To enable power limiting** — set the limit to a value below the interface maximum:
```bash
$ axdevice --set-power-limit 20   # enable, targeting 20.0 W
```

**To disable power limiting** — set the limit to 0:
```bash
$ axdevice --set-power-limit 0   # disable
```

## 2.5. Hardware Support

### 2.5.1. Supported Boards

The power limiter is currently supported on **M.2 MAX** (M.2 Rev2) boards only.

| Board | Form factor | Interface power budget | Support |
|-------|-------------|------------------------|---------|
| M.2 MAX | M.2 | ~23 W | Yes |
| Other | — | — | No |

> [!WARNING]
> Attempting to use `--set-power-limit` on unsupported boards will return an error. The default limits are chosen to match the M.2 interface power budget, ensuring that the power limiter is safe to enable on supported hardware without additional configuration.
