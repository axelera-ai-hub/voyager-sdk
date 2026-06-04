---
title: "Thermal & Power Management"
---
# Thermal & Power Management

How Metis handles temperature and power, how to monitor both, and how to configure throttling and power limits.

---

## Monitoring temperature

### During inference

`inference.py` shows peak core temperature at runtime:

```bash
$ ./inference.py yolov8s-coco-onnx media/traffic.mp4 --no-display
INFO    : Core Temp : 39.0°C
```

This reports the maximum temperature across all 5 internal sensors.

### From your application

Use the `core_temp` tracer (see [InferenceStream API](../reference/apis/inference-stream.md#tracers)):

```python
from axelera.app import inf_tracers

tracers = inf_tracers.create_tracers('core_temp')
stream = create_inference_stream(..., tracers=tracers)

temp = stream.get_all_metrics()['core_temp'].value
print(f"Core temperature: {temp}°C")
```

### Detailed sensor readout

`axlogdevice` reports all 5 sensors (1 outside the AIPU core area, 4 per-core):

```bash
axlogdevice --slog-level inf:collector --slog
# [04:58:54.012,603] <inf> collector: core_temps=[35,34,34,35,34]
```

### axmonitor

[axmonitor](axmonitor.md) provides a real-time dashboard with temperature graphs and threshold indicators.

---

## How throttling works

Metis uses two mechanisms to manage heat:

**1. MVM utilization throttling** (primary)

When temperature exceeds a threshold T, the Matrix-Vector-Multiplication block is capped at L% utilization. This reduces compute intensity and brings temperature down. When temperature drops by H hysteresis degrees, the limit is lifted.

**2. Frequency scaling** (secondary)

If temperature approaches the hardware throttling threshold, the chip clock is reduced by 100 MHz per second (minimum 200 MHz). The clock recovers by 100 MHz when temperature drops 5°C below the threshold.

---

## Default temperature settings

All temperatures below are silicon junction temperatures (T_j) — higher than ambient or package temperatures.

| Type | Parameter | Default | Configurable | Notes |
|------|-----------|---------|--------------|-------|
| Software throttling | Threshold T_s | 200°C | Yes | Effectively disabled by default |
| | Hysteresis H_s | 10°C | Yes | |
| | MVM limit L_s | 10% | Yes | |
| Hardware throttling | Threshold T_h | 105°C | No | Backup if warning signal unused |
| | Hysteresis H_h | 10°C | No | |
| | MVM limit L_h | 1% | No | |
| Safety — warning | T_j warning | 95°C | No | Generates a log entry |
| Safety — shutdown | T_j shutdown | 120°C | No | Disables all regulators; requires power cycle |
| Frequency scaling | Starts at | 110°C | No | After hardware throttling activates |

---

## Configuring software throttling

Use `axdevice` to set custom throttling thresholds:

```bash
# Format: --set-sw-throttling=T:H:L
# T = temperature threshold (°C)
# H = hysteresis (°C)
# L = MVM utilization limit (%)

axdevice --set-sw-throttling=100:5:10
```

This example throttles to 10% MVM utilization above 100°C, and removes the limit when temperature drops to 95°C.

View current settings:
```bash
axdevice -v
```

> [!NOTE]
> Throttling settings do not persist across device reboots or firmware reloads. Re-apply after each system start if needed.


---

## Safety mechanisms

**Warning (95°C):** Logs an entry. Configurable threshold:
```bash
axdevice --set-pvt-warning-threshold 85
```
Does not persist across reboots.

**Shutdown (120°C):** Fixed. Triggers the board controller to disable all power regulators. Requires a full power cycle to recover — not just a reboot.

**Frequency downscaling (110°C):** Reduces clock by 100 MHz/second while above threshold (minimum 200 MHz). Returns +100 MHz when temperature drops 5°C below the threshold. Runs independently of MVM throttling.

---

## Operating range

For PCIe and M.2 boards (REV1.1):

| | Value |
|---|---|
| Ambient operating range | −20°C to +70°C |
| Junction operating range | See safety table above |

Performance and lifetime are within specification across the full ambient operating range.

---

## Power management

Metis includes a closed-loop power limiter that keeps board power consumption within a configured budget by dynamically adjusting MVM utilization — the same mechanism used for thermal throttling.

### How it works

The power limiter is a PID controller that:

1. **Measures** instantaneous power from the on-board INA236 sensor at ~200 Hz
2. **Computes** PID error relative to the configured power limit
3. **Adjusts** MVM utilization percentage uniformly across all AIPU cores

When measured power exceeds the limit, utilization is reduced. When power drops below the limit, utilization is gradually restored.

### Setting a power limit

```bash
# Set power limit to 20W
axdevice --set-power-limit 20

# Disable the power limiter
axdevice --set-power-limit 0

# Check current settings
axdevice -v
```

### Hardware support

The power limiter is currently supported on **M.2 MAX** (M.2 Rev2) boards only.

| Board | Form factor | Interface power budget | Supported |
|-------|-------------|------------------------|-----------|
| M.2 MAX | M.2 | ~23 W | Yes |
| Other | — | — | No |

> [!WARNING]
> `--set-power-limit` returns an error on unsupported boards.

> [!NOTE]
> The default limit is 8.5W. This is stable on most hosts, but some low power hosts may
> require a lower limit. Some high powered hosts will benefit from a higher limit delivering
> higher inference performance. Setting the limit higher than a host can handle may result
> in device instability.


---

## See also

- [axmonitor](axmonitor.md) — real-time temperature and power dashboard
- [Performance Metrics](../reference/measurement/performance.md) — measuring throughput under thermal load
- [axdevice](../reference/tools/axdevice.md) — full `axdevice` CLI reference
